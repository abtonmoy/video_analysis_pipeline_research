"""
HMMD vs. Multi-Baseline Benchmarking Framework
================================================
Compares Hierarchical Multi-Modal Deduplication (HMMD) against 7 baseline
keyframe extraction / deduplication techniques.

Usage:
    python benchmark.py --video_dir ./videos --output_dir ./results
    python benchmark.py --video_dir ./videos --output_dir ./results --tests 1 3 8
    python benchmark.py --video_dir ./videos --output_dir ./results --skip_gpu

Requirements:
    pip install opencv-python-headless torch torchvision lpips
    pip install sentence-transformers scikit-learn pandas tqdm imagehash Pillow
"""

import os
import sys
import csv
import time
import random
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Lazy imports for heavy deps
_torch = None
_lpips = None
_SentenceTransformer = None
_KMeans = None
_imagehash = None
_Image = None


def _import_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _import_lpips():
    global _lpips
    if _lpips is None:
        import lpips
        _lpips = lpips
    return _lpips


def _import_sentence_transformer():
    global _SentenceTransformer
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer
        _SentenceTransformer = SentenceTransformer
    return _SentenceTransformer


def _import_kmeans():
    global _KMeans
    if _KMeans is None:
        from sklearn.cluster import KMeans
        _KMeans = KMeans
    return _KMeans


def _import_imagehash():
    global _imagehash, _Image
    if _imagehash is None:
        import imagehash
        from PIL import Image
        _imagehash = imagehash
        _Image = Image
    return _imagehash, _Image


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VLM_COST_PER_1K_TOKENS = 0.015
TOKENS_PER_FRAME_ESTIMATE = 765  # ~typical for 512x512 image in VLM

# Default thresholds
HIST_CORR_THRESH = 0.95
ORB_MATCH_THRESH = 40  # min good matches to declare duplicate
CLIP_COSINE_THRESH = 0.92
LPIPS_THRESH = 0.1
HASH_THRESH = 10  # hamming distance bits
OPTICAL_FLOW_PERCENTILE = 85  # motion magnitude percentile for peaks
KMEANS_K = 10


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class FrameData:
    """Lightweight container for a decoded frame."""
    index: int
    timestamp_s: float
    image: np.ndarray  # BGR uint8


@dataclass
class BenchmarkResult:
    test_name: str
    video_name: str
    total_frames: int
    selected_count: int
    compression_ratio: float
    latency_s: float
    info_density: float  # mean pairwise CLIP distance of selected frames
    estimated_vlm_cost: float
    selected_indices: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Video I/O helpers
# ---------------------------------------------------------------------------
def frame_generator(video_path: str, max_dim: int = 640) -> Generator[FrameData, None, None]:
    """Yield frames one-at-a-time to avoid OOM. Optionally resize for speed."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        yield FrameData(index=idx, timestamp_s=idx / fps, image=frame)
        idx += 1
    cap.release()


def video_metadata(video_path: str) -> Tuple[int, float]:
    """Return (total_frames, fps)."""
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return n, fps


def preload_frames(video_path: str, max_dim: int = 640) -> List[FrameData]:
    """Load all frames into memory. Use only when needed (e.g., random access)."""
    return list(frame_generator(video_path, max_dim))


# ---------------------------------------------------------------------------
# CLIP embedding cache (shared across tests)
# ---------------------------------------------------------------------------
class CLIPCache:
    """Manages a per-video CLIP embedding matrix so we don't re-encode."""

    def __init__(self, model_name: str = "clip-ViT-B-32", device: str = "auto"):
        ST = _import_sentence_transformer()
        torch = _import_torch()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        log.info(f"Loading CLIP model '{model_name}' on {device} …")
        self.model = ST(model_name, device=device)
        self._cache: Dict[str, np.ndarray] = {}

    def get_embeddings(self, video_path: str, frames: List[FrameData], batch_size: int = 64) -> np.ndarray:
        key = video_path
        if key in self._cache:
            return self._cache[key]
        from PIL import Image as PILImage
        imgs = [PILImage.fromarray(cv2.cvtColor(f.image, cv2.COLOR_BGR2RGB)) for f in frames]
        embs = self.model.encode(imgs, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
        self._cache[key] = embs
        return embs

    def info_density(self, embeddings: np.ndarray, indices: List[int]) -> float:
        """Mean pairwise cosine distance among selected frames. Higher = more diverse."""
        if len(indices) < 2:
            return 0.0
        sel = embeddings[indices]
        sim = sel @ sel.T
        n = len(indices)
        # Mean of upper triangle (excluding diagonal)
        total = sim[np.triu_indices(n, k=1)].sum()
        pairs = n * (n - 1) / 2
        mean_sim = total / pairs if pairs > 0 else 1.0
        return float(1.0 - mean_sim)


# ---------------------------------------------------------------------------
# Test implementations
# ---------------------------------------------------------------------------

# ---- Test 1: Uniform Sampling ----
def test_uniform_sampling(video_path: str, target_fps: float = 1.0) -> List[int]:
    total, fps = video_metadata(video_path)
    if target_fps >= fps:
        return list(range(total))
    step = int(round(fps / target_fps))
    return list(range(0, total, max(step, 1)))


# ---- Test 2: Random Sampling ----
def test_random_sampling(video_path: str, k: int) -> List[int]:
    total, _ = video_metadata(video_path)
    k = min(k, total)
    indices = sorted(random.sample(range(total), k))
    return indices


# ---- Test 3: Color Histogram Correlation ----
def test_histogram_dedup(video_path: str, threshold: float = HIST_CORR_THRESH) -> List[int]:
    selected = []
    prev_hist = None
    for frame in frame_generator(video_path):
        hsv = cv2.cvtColor(frame.image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 16, 16], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist = hist.flatten()
        if prev_hist is None:
            selected.append(frame.index)
            prev_hist = hist
            continue
        corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
        if corr < threshold:
            selected.append(frame.index)
            prev_hist = hist
    return selected


# ---- Test 4: ORB Feature Matching ----
def test_orb_dedup(video_path: str, match_thresh: int = ORB_MATCH_THRESH) -> List[int]:
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    selected = []
    prev_des = None
    for frame in frame_generator(video_path):
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        if des is None:
            selected.append(frame.index)
            prev_des = None
            continue
        if prev_des is None:
            selected.append(frame.index)
            prev_des = des
            continue
        matches = bf.match(prev_des, des)
        good = [m for m in matches if m.distance < 50]
        if len(good) < match_thresh:
            selected.append(frame.index)
            prev_des = des
    return selected


# ---- Test 5: Optical Flow (Motion Peaks) ----
def test_optical_flow(video_path: str, percentile: float = OPTICAL_FLOW_PERCENTILE) -> List[int]:
    magnitudes = []
    prev_gray = None
    for frame in frame_generator(video_path):
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            magnitudes.append(0.0)
            prev_gray = gray
            continue
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitudes.append(float(mag.mean()))
        prev_gray = gray

    if not magnitudes:
        return []
    thresh = np.percentile(magnitudes, percentile)
    selected = [i for i, m in enumerate(magnitudes) if m >= thresh or i == 0]
    # Always include first and last
    if len(magnitudes) - 1 not in selected:
        selected.append(len(magnitudes) - 1)
    return sorted(set(selected))


# ---- Test 6: CLIP-Only Deduplication ----
def test_clip_dedup(frames: List[FrameData], embeddings: np.ndarray,
                    threshold: float = CLIP_COSINE_THRESH) -> List[int]:
    selected = [0]
    for i in range(1, len(frames)):
        sim = float(embeddings[i] @ embeddings[selected[-1]])
        if sim < threshold:
            selected.append(i)
    return selected


# ---- Test 7: K-Means Clustering ----
def test_kmeans_clustering(frames: List[FrameData], embeddings: np.ndarray,
                           k: int = KMEANS_K) -> List[int]:
    KMeans = _import_kmeans()
    k = min(k, len(frames))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    selected = []
    for c in range(k):
        cluster_idx = np.where(labels == c)[0]
        dists = np.linalg.norm(embeddings[cluster_idx] - km.cluster_centers_[c], axis=1)
        best = cluster_idx[np.argmin(dists)]
        selected.append(int(best))
    return sorted(selected)


# ---- Test 8: Full HMMD Cascade ----
def test_hmmd(video_path: str, frames: List[FrameData], embeddings: np.ndarray,
              hash_thresh: int = HASH_THRESH,
              lpips_thresh: float = LPIPS_THRESH,
              clip_thresh: float = CLIP_COSINE_THRESH,
              nms_window: int = 5) -> List[int]:
    """
    Three-stage cheap-to-expensive filter + temporal NMS.
    Stage 1: Perceptual hash (pHash) – drop obvious duplicates.
    Stage 2: LPIPS – perceptual distance on survivors.
    Stage 3: CLIP cosine – semantic distance on survivors.
    Stage 4: Non-Maximum Suppression for temporal spacing.
    """
    torch = _import_torch()
    lpips_mod = _import_lpips()
    imagehash, PILImage = _import_imagehash()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Stage 1: Hash voting ---
    log.info("  HMMD Stage 1: Hash voting …")
    hash_survivors = [0]
    prev_phash = imagehash.phash(PILImage.fromarray(cv2.cvtColor(frames[0].image, cv2.COLOR_BGR2RGB)))
    for i in range(1, len(frames)):
        pil = PILImage.fromarray(cv2.cvtColor(frames[i].image, cv2.COLOR_BGR2RGB))
        h = imagehash.phash(pil)
        if abs(h - prev_phash) > hash_thresh:
            hash_survivors.append(i)
            prev_phash = h
    log.info(f"    Hash survivors: {len(hash_survivors)}/{len(frames)}")

    # --- Stage 2: LPIPS ---
    log.info("  HMMD Stage 2: LPIPS filtering …")
    loss_fn = lpips_mod.LPIPS(net="squeeze").to(device)

    def to_lpips_tensor(img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = cv2.resize(img, (64, 64))
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        return t * 2.0 - 1.0  # scale to [-1, 1]

    lpips_survivors = [hash_survivors[0]]
    prev_tensor = to_lpips_tensor(frames[hash_survivors[0]].image)
    for idx in hash_survivors[1:]:
        cur_tensor = to_lpips_tensor(frames[idx].image)
        with torch.no_grad():
            dist = loss_fn(prev_tensor, cur_tensor).item()
        if dist > lpips_thresh:
            lpips_survivors.append(idx)
            prev_tensor = cur_tensor
    log.info(f"    LPIPS survivors: {len(lpips_survivors)}/{len(hash_survivors)}")

    # --- Stage 3: CLIP semantic ---
    log.info("  HMMD Stage 3: CLIP semantic filtering …")
    clip_survivors = [lpips_survivors[0]]
    for idx in lpips_survivors[1:]:
        sim = float(embeddings[idx] @ embeddings[clip_survivors[-1]])
        if sim < clip_thresh:
            clip_survivors.append(idx)
    log.info(f"    CLIP survivors: {len(clip_survivors)}/{len(lpips_survivors)}")

    # --- Stage 4: Temporal NMS ---
    log.info("  HMMD Stage 4: Temporal NMS …")
    if len(clip_survivors) <= 1:
        return clip_survivors

    # Compute a "score" per survivor: CLIP distance from nearest neighbour (higher = more unique)
    scores = []
    for i, idx in enumerate(clip_survivors):
        neighbours = []
        if i > 0:
            neighbours.append(float(1.0 - embeddings[idx] @ embeddings[clip_survivors[i - 1]]))
        if i < len(clip_survivors) - 1:
            neighbours.append(float(1.0 - embeddings[idx] @ embeddings[clip_survivors[i + 1]]))
        scores.append(max(neighbours) if neighbours else 0.0)

    # Suppress low-score frames that are temporally close to higher-score frames
    order = np.argsort(scores)[::-1]
    suppressed = set()
    final = []
    for rank in order:
        idx = clip_survivors[rank]
        if idx in suppressed:
            continue
        final.append(idx)
        # Suppress neighbours within window
        for other_rank, other_idx in enumerate(clip_survivors):
            if other_idx != idx and abs(other_idx - idx) < nms_window:
                suppressed.add(other_idx)

    final = sorted(final)
    log.info(f"    NMS final: {len(final)}/{len(clip_survivors)}")
    return final


# ---------------------------------------------------------------------------
# Saving selected frames
# ---------------------------------------------------------------------------
def save_selected_frames(video_path: str, indices: List[int], out_dir: str, max_dim: int = 640):
    os.makedirs(out_dir, exist_ok=True)
    idx_set = set(indices)
    for frame in frame_generator(video_path, max_dim=max_dim):
        if frame.index in idx_set:
            fname = os.path.join(out_dir, f"frame_{frame.index:06d}.jpg")
            cv2.imwrite(fname, frame.image, [cv2.IMWRITE_JPEG_QUALITY, 90])


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
class HMMDBenchmark:
    def __init__(self, video_dir: str, output_dir: str, skip_gpu: bool = False,
                 tests: Optional[List[int]] = None, save_frames: bool = True):
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.skip_gpu = skip_gpu
        self.tests = tests or list(range(1, 9))
        self.save_frames = save_frames
        self.results: List[BenchmarkResult] = []
        self.clip_cache: Optional[CLIPCache] = None

        os.makedirs(self.output_dir, exist_ok=True)

        # Init CLIP if any GPU test is requested
        gpu_tests = {6, 7, 8}
        if not skip_gpu and gpu_tests & set(self.tests):
            self.clip_cache = CLIPCache()

    def _videos(self) -> List[Path]:
        exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
        vids = sorted(p for p in self.video_dir.iterdir() if p.suffix.lower() in exts)
        if not vids:
            log.warning(f"No videos found in {self.video_dir}")
        return vids

    def _make_result(self, name: str, video: str, total: int, indices: List[int],
                     elapsed: float, embeddings: Optional[np.ndarray] = None) -> BenchmarkResult:
        k = len(indices)
        cr = total / k if k > 0 else float("inf")
        density = 0.0
        if embeddings is not None and k >= 2:
            density = self.clip_cache.info_density(embeddings, indices)
        cost = (k * TOKENS_PER_FRAME_ESTIMATE / 1000.0) * VLM_COST_PER_1K_TOKENS
        return BenchmarkResult(
            test_name=name, video_name=video, total_frames=total,
            selected_count=k, compression_ratio=round(cr, 2),
            latency_s=round(elapsed, 3), info_density=round(density, 5),
            estimated_vlm_cost=round(cost, 4), selected_indices=indices,
        )

    def run(self):
        videos = self._videos()
        for vpath in videos:
            vname = vpath.name
            total, fps = video_metadata(str(vpath))
            log.info(f"\n{'='*60}\nProcessing: {vname} ({total} frames, {fps:.1f} fps)\n{'='*60}")

            # Pre-load frames & embeddings if needed for GPU tests
            frames, embeddings = None, None
            gpu_tests = {6, 7, 8}
            if not self.skip_gpu and gpu_tests & set(self.tests):
                log.info("Pre-loading frames for GPU tests …")
                frames = preload_frames(str(vpath))
                embeddings = self.clip_cache.get_embeddings(str(vpath), frames)

            # ---- Test 1a: Uniform 1 FPS ----
            if 1 in self.tests:
                t0 = time.perf_counter()
                idx = test_uniform_sampling(str(vpath), target_fps=1.0)
                elapsed = time.perf_counter() - t0
                r = self._make_result("Uniform_1FPS", vname, total, idx, elapsed, embeddings)
                self.results.append(r)
                log.info(f"  [Test 1a] Uniform 1FPS → {r.selected_count} frames, {r.latency_s}s")
                if self.save_frames:
                    save_selected_frames(str(vpath), idx,
                                         str(self.output_dir / vname / "uniform_1fps"))

                # Test 1b: Uniform 30 FPS (full data upper bound)
                t0 = time.perf_counter()
                idx30 = test_uniform_sampling(str(vpath), target_fps=30.0)
                elapsed = time.perf_counter() - t0
                r = self._make_result("Uniform_30FPS", vname, total, idx30, elapsed, embeddings)
                self.results.append(r)
                log.info(f"  [Test 1b] Uniform 30FPS → {r.selected_count} frames, {r.latency_s}s")

            # ---- Test 2: Random Sampling ----
            if 2 in self.tests:
                # Match HMMD count if available, else use 1 FPS count
                hmmd_result = next((r for r in self.results
                                    if r.test_name == "HMMD" and r.video_name == vname), None)
                k = hmmd_result.selected_count if hmmd_result else max(1, int(total / fps))
                t0 = time.perf_counter()
                idx = test_random_sampling(str(vpath), k)
                elapsed = time.perf_counter() - t0
                r = self._make_result("Random", vname, total, idx, elapsed, embeddings)
                self.results.append(r)
                log.info(f"  [Test 2] Random → {r.selected_count} frames, {r.latency_s}s")
                if self.save_frames:
                    save_selected_frames(str(vpath), idx,
                                         str(self.output_dir / vname / "random"))

            # ---- Test 3: Histogram ----
            if 3 in self.tests:
                t0 = time.perf_counter()
                idx = test_histogram_dedup(str(vpath))
                elapsed = time.perf_counter() - t0
                r = self._make_result("Histogram", vname, total, idx, elapsed, embeddings)
                self.results.append(r)
                log.info(f"  [Test 3] Histogram → {r.selected_count} frames, {r.latency_s}s")
                if self.save_frames:
                    save_selected_frames(str(vpath), idx,
                                         str(self.output_dir / vname / "histogram"))

            # ---- Test 4: ORB ----
            if 4 in self.tests:
                t0 = time.perf_counter()
                idx = test_orb_dedup(str(vpath))
                elapsed = time.perf_counter() - t0
                r = self._make_result("ORB", vname, total, idx, elapsed, embeddings)
                self.results.append(r)
                log.info(f"  [Test 4] ORB → {r.selected_count} frames, {r.latency_s}s")
                if self.save_frames:
                    save_selected_frames(str(vpath), idx,
                                         str(self.output_dir / vname / "orb"))

            # ---- Test 5: Optical Flow ----
            if 5 in self.tests:
                t0 = time.perf_counter()
                idx = test_optical_flow(str(vpath))
                elapsed = time.perf_counter() - t0
                r = self._make_result("OpticalFlow", vname, total, idx, elapsed, embeddings)
                self.results.append(r)
                log.info(f"  [Test 5] Optical Flow → {r.selected_count} frames, {r.latency_s}s")
                if self.save_frames:
                    save_selected_frames(str(vpath), idx,
                                         str(self.output_dir / vname / "optical_flow"))

            # ---- Test 6: CLIP-Only ----
            if 6 in self.tests and not self.skip_gpu:
                t0 = time.perf_counter()
                idx = test_clip_dedup(frames, embeddings)
                elapsed = time.perf_counter() - t0
                r = self._make_result("CLIP_Only", vname, total, idx, elapsed, embeddings)
                self.results.append(r)
                log.info(f"  [Test 6] CLIP-Only → {r.selected_count} frames, {r.latency_s}s")
                if self.save_frames:
                    save_selected_frames(str(vpath), idx,
                                         str(self.output_dir / vname / "clip_only"))

            # ---- Test 7: K-Means ----
            if 7 in self.tests and not self.skip_gpu:
                t0 = time.perf_counter()
                idx = test_kmeans_clustering(frames, embeddings, k=KMEANS_K)
                elapsed = time.perf_counter() - t0
                r = self._make_result("KMeans", vname, total, idx, elapsed, embeddings)
                self.results.append(r)
                log.info(f"  [Test 7] K-Means → {r.selected_count} frames, {r.latency_s}s")
                if self.save_frames:
                    save_selected_frames(str(vpath), idx,
                                         str(self.output_dir / vname / "kmeans"))

            # ---- Test 8: HMMD ----
            if 8 in self.tests and not self.skip_gpu:
                t0 = time.perf_counter()
                idx = test_hmmd(str(vpath), frames, embeddings)
                elapsed = time.perf_counter() - t0
                r = self._make_result("HMMD", vname, total, idx, elapsed, embeddings)
                self.results.append(r)
                log.info(f"  [Test 8] HMMD → {r.selected_count} frames, {r.latency_s}s")
                if self.save_frames:
                    save_selected_frames(str(vpath), idx,
                                         str(self.output_dir / vname / "hmmd"))

        self._write_csv()

    def _write_csv(self):
        csv_path = self.output_dir / "results.csv"
        rows = []
        for r in self.results:
            rows.append({
                "test_name": r.test_name,
                "video": r.video_name,
                "total_frames": r.total_frames,
                "selected_count": r.selected_count,
                "compression_ratio": r.compression_ratio,
                "latency_s": r.latency_s,
                "info_density": r.info_density,
                "estimated_vlm_cost_usd": r.estimated_vlm_cost,
            })
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        log.info(f"\n{'='*60}\nResults saved to {csv_path}\n{'='*60}")
        print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="HMMD Benchmarking Framework")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing input videos")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results", help="Output directory")
    parser.add_argument("--tests", type=int, nargs="*", default=None,
                        help="Which tests to run (1-8). Default: all.")
    parser.add_argument("--skip_gpu", action="store_true", help="Skip GPU-dependent tests (6, 7, 8)")
    parser.add_argument("--no_save", action="store_true", help="Don't save extracted frame images")
    args = parser.parse_args()

    bench = HMMDBenchmark(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        skip_gpu=args.skip_gpu,
        tests=args.tests,
        save_frames=not args.no_save,
    )
    bench.run()


if __name__ == "__main__":
    main()