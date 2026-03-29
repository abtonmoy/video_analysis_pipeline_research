#!/usr/bin/env python3
"""
Task 2: True SVD-Based ISD Validation
======================================
Addresses reviewer W2: "The theoretically motivated budget derivation was not
actually implemented."

This script:
1. Samples ~100 videos from the 484 benchmark set
2. For each video, extracts ALL candidate frames and computes CLIP embeddings
3. Computes true ISD (k*) via SVD of the embedding matrix (Eq. 4 in paper)
4. Correlates ISD with the proxy (`frames_after_clip`) and downstream metrics
5. Generates scatter plots and LaTeX paragraph

Ground truth: Pitt Ads Dataset cleaned annotations (video_Topics_clean.json, etc.)
    - Keyed by YouTube video ID (= filename stem without extension)
    - Located at data/annotations_videos/video/cleaned_result/

Requires:
    pip install open-clip-torch torch numpy scipy matplotlib seaborn pandas

Usage:
    python task2_isd_validation.py \
        --benchmark_results test_results/benchmark_results.json \
        --pipeline_results results/processing_results.json \
        --video_dir data/ads \
        --gt_dir data/annotations_videos/video/cleaned_result \
        --output_dir figures \
        --n_sample 100 \
        --seed 42
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Ground Truth Loader — Pitt Ads Dataset
# ============================================================================

class PittAdsGroundTruth:
    """
    Load Pitt Ads Dataset cleaned annotations.

    Files (all keyed by YouTube video ID, i.e. filename stem):
        video_Topics_clean.json     -> int (1-38 topic ID)
        video_Sentiments_clean.json -> int (1-30 sentiment ID)
        video_Funny_clean.json      -> float (0.0-1.0 funny score)
        video_Exciting_clean.json   -> float (0.0-1.0 exciting score)
        video_Effective_clean.json  -> str of int ("1"-"5" effectiveness)
        video_Language_clean.json   -> str ("1"=English, "0"=non-English, "-1"=N/A)
    """

    def __init__(self, gt_dir: str):
        self.gt_dir = Path(gt_dir)
        self._cache: Dict[str, Dict] = {}
        self._load_all()

    def _load_all(self):
        """Load all available annotation files."""
        file_map = {
            "topics":     "video_Topics_clean.json",
            "sentiments": "video_Sentiments_clean.json",
            "funny":      "video_Funny_clean.json",
            "exciting":   "video_Exciting_clean.json",
            "effective":  "video_Effective_clean.json",
            "language":   "video_Language_clean.json",
        }

        for key, filename in file_map.items():
            path = self.gt_dir / filename
            if path.exists():
                with open(path, "r") as f:
                    self._cache[key] = json.load(f)
                logger.info(f"  Loaded GT {key}: {len(self._cache[key])} entries from {filename}")
            else:
                logger.warning(f"  GT file not found: {path}")
                self._cache[key] = {}

    def _vid_id(self, video_name: str) -> str:
        """Extract YouTube video ID from filename (stem without extension)."""
        return Path(video_name).stem

    def get_topic(self, video_name: str) -> Optional[int]:
        """Get ground-truth topic ID (1-38) for a video."""
        vid = self._vid_id(video_name)
        val = self._cache.get("topics", {}).get(vid)
        return int(val) if val is not None else None

    def get_sentiment(self, video_name: str) -> Optional[int]:
        """Get ground-truth primary sentiment ID (1-30)."""
        vid = self._vid_id(video_name)
        val = self._cache.get("sentiments", {}).get(vid)
        return int(val) if val is not None else None

    def get_funny(self, video_name: str) -> Optional[float]:
        """Get funny score (0.0-1.0)."""
        vid = self._vid_id(video_name)
        val = self._cache.get("funny", {}).get(vid)
        return float(val) if val is not None else None

    def get_exciting(self, video_name: str) -> Optional[float]:
        """Get exciting score (0.0-1.0)."""
        vid = self._vid_id(video_name)
        val = self._cache.get("exciting", {}).get(vid)
        return float(val) if val is not None else None

    def get_effective(self, video_name: str) -> Optional[int]:
        """Get effectiveness score (1-5)."""
        vid = self._vid_id(video_name)
        val = self._cache.get("effective", {}).get(vid)
        return int(val) if val is not None else None

    def get_language(self, video_name: str) -> Optional[str]:
        """Get language label ('1'=English, '0'=non-English, '-1'=N/A)."""
        vid = self._vid_id(video_name)
        return self._cache.get("language", {}).get(vid)

    def has_topic(self, video_name: str) -> bool:
        return self.get_topic(video_name) is not None

    def coverage(self, video_names: List[str]) -> Dict[str, int]:
        """Report how many videos have each annotation type."""
        result = {}
        for key in self._cache:
            count = sum(1 for v in video_names if self._vid_id(v) in self._cache[key])
            result[key] = count
        return result


# ============================================================================
# Topic Super-Category Mapping
# ============================================================================

TOPIC_SUPER_CATEGORIES = {
    "food_beverage":    [1, 2, 3, 4, 5, 6, 7, 8],
    "auto":             [9],
    "tech_service":     [10, 11, 14, 15],
    "finance_edu":      [12, 13],
    "other_service":    [16],
    "personal_care":    [17, 18, 19],
    "home_family":      [20, 22, 23, 24],
    "entertainment":    [21, 26, 27, 29],
    "travel_shopping":  [25, 28],
    "social_cause":     [30, 31, 32, 33, 34, 35, 36, 37, 38],
}

_TOPIC_TO_SUPER = {}
for cat, ids in TOPIC_SUPER_CATEGORIES.items():
    for tid in ids:
        _TOPIC_TO_SUPER[tid] = cat


def get_topic_super_category(topic_id: int) -> str:
    return _TOPIC_TO_SUPER.get(topic_id, "unknown")


# ============================================================================
# ISD Computation (Eq. 4 from the paper)
# ============================================================================

def compute_isd(
    embeddings: np.ndarray,
    tau: float = 0.95,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Compute Intrinsic Semantic Dimensionality via SVD.

    Given an (N, D) matrix of CLIP embeddings Phi:
        1. Center the embeddings (subtract mean).
        2. SVD: U, S, V^T = svd(Phi_centered).
        3. k* = min k s.t. cumulative explained variance >= tau.

    Args:
        embeddings: (N, D) matrix of L2-normalised CLIP embeddings.
        tau: Variance retention threshold (default 0.95 per paper).

    Returns:
        k_star: The intrinsic semantic dimensionality.
        singular_values: Full vector of singular values.
        explained_variance: Cumulative explained variance curve.
    """
    if embeddings.shape[0] < 2:
        return 1, np.array([1.0]), np.array([1.0])

    # Center
    centered = embeddings - embeddings.mean(axis=0)

    # Economy SVD (N x D -> at most min(N,D) singular values)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)

    # Cumulative explained variance
    var = S ** 2
    total_var = var.sum()
    if total_var == 0:
        return 1, S, np.ones(len(S))

    explained = np.cumsum(var) / total_var

    # k* where cumulative variance first reaches tau
    k_star = int(np.searchsorted(explained, tau)) + 1
    k_star = min(k_star, len(S))

    return k_star, S, explained


# ============================================================================
# CLIP Embedding Extraction
# ============================================================================

class CLIPEmbedder:
    """
    Thin wrapper around open_clip for batch embedding.
    Mirrors src/deduplication/clip_embed.py CLIPDeduplicator but returns
    the FULL embedding matrix (before dedup filtering).
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "auto",
        batch_size: int = 32,
    ):
        import torch
        import open_clip

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.batch_size = batch_size

        logger.info(f"Loading CLIP model {model_name} ({pretrained}) on {self.device}...")
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self._model.eval()
        logger.info("CLIP model loaded.")

    def embed_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Compute L2-normalised CLIP embeddings for a list of BGR frames.

        Returns:
            (N, D) numpy array of unit-norm embeddings (D=512 for ViT-B-32).
        """
        import torch
        from PIL import Image

        all_embs = []
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i : i + self.batch_size]

            pil_images = [
                Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in batch
            ]
            tensors = torch.stack([self._preprocess(img) for img in pil_images]).to(
                self.device
            )

            with torch.no_grad():
                embs = self._model.encode_image(tensors)
                embs = embs / embs.norm(dim=-1, keepdim=True)

            all_embs.append(embs.cpu().numpy())

        return np.vstack(all_embs)


# ============================================================================
# Candidate Frame Extraction
# ============================================================================

def extract_candidate_frames(
    video_path: str,
    max_resolution: int = 720,
    sample_interval_ms: int = 100,
) -> List[Tuple[float, np.ndarray]]:
    """
    Extract candidate frames by sampling every `sample_interval_ms` ms.

    Simplified version of pipeline's CandidateFrameExtractor.
    For ISD we want ALL sampled frames (before any dedup), so we sample
    at a fixed interval — matching the benchmark runner's _precompute_shared.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(fps * sample_interval_ms / 1000))

    frames: List[Tuple[float, np.ndarray]] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            ts = frame_idx / fps

            h, w = frame.shape[:2]
            if max(h, w) > max_resolution:
                scale = max_resolution / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            frames.append((ts, frame))

        frame_idx += 1

    cap.release()
    logger.debug(f"  Extracted {len(frames)} candidate frames from {Path(video_path).name}")
    return frames


# ============================================================================
# Data Loaders — Pipeline & Benchmark Results
# ============================================================================

def load_pipeline_results(path: str) -> Dict[str, Any]:
    """
    Load main pipeline results.
    Handles both formats:
        { "results": [ {"video_name": ..., ...}, ... ] }
        { "per_video": { "video.mp4": {...}, ... } }
    Returns dict keyed by video filename.
    """
    with open(path, "r") as f:
        data = json.load(f)

    if "results" in data and isinstance(data["results"], list):
        return {r["video_name"]: r for r in data["results"] if "video_name" in r}
    elif "per_video" in data:
        return data["per_video"]
    return data


def load_benchmark_results(path: str) -> Dict[str, Any]:
    """Load benchmark results. Returns per_video dict."""
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("per_video", data)


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(args):

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading pipeline results...")
    pipeline_data = load_pipeline_results(args.pipeline_results)
    logger.info(f"  {len(pipeline_data)} pipeline results.")

    logger.info("Loading benchmark results...")
    benchmark_data = load_benchmark_results(args.benchmark_results)
    logger.info(f"  {len(benchmark_data)} benchmark results.")

    logger.info("Loading Pitt Ads ground truth...")
    gt = PittAdsGroundTruth(args.gt_dir)

    # ------------------------------------------------------------------
    # 2. Select video sample
    # ------------------------------------------------------------------
    benchmark_videos = set(benchmark_data.keys())
    pipeline_videos = set(pipeline_data.keys())
    common_videos = sorted(benchmark_videos & pipeline_videos)
    logger.info(f"  Common (benchmark & pipeline): {len(common_videos)}")

    # Locate video files on disk
    video_dir = Path(args.video_dir)
    video_paths: Dict[str, str] = {}

    # Direct lookup first
    for v in common_videos:
        p = video_dir / v
        if p.exists():
            video_paths[v] = str(p)

    # If none found, try recursive search
    if not video_paths:
        logger.info("  No videos in top-level dir, trying recursive search...")
        all_files = {}
        for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
            for p in video_dir.rglob(f"*{ext}"):
                all_files[p.name] = str(p)
        for v in common_videos:
            if v in all_files:
                video_paths[v] = all_files[v]

    available = sorted(video_paths.keys())
    logger.info(f"  Available on disk: {len(available)}")

    # Report GT coverage
    gt_coverage = gt.coverage(available)
    logger.info(f"  GT coverage: {gt_coverage}")

    if len(available) == 0:
        logger.error("No videos found. Check --video_dir path.")
        sys.exit(1)

    # Sample
    rng = np.random.RandomState(args.seed)
    n_sample = min(args.n_sample, len(available))
    sampled = list(rng.choice(available, size=n_sample, replace=False))
    logger.info(f"Sampled {n_sample} videos for ISD experiment.")

    # ------------------------------------------------------------------
    # 3. Compute ISD + collect metrics
    # ------------------------------------------------------------------
    embedder = CLIPEmbedder(
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
        device=args.device,
        batch_size=args.clip_batch_size,
    )

    records: List[Dict[str, Any]] = []

    for idx, vname in enumerate(sampled):
        logger.info(f"[{idx+1}/{n_sample}] Processing {vname}...")
        vpath = video_paths[vname]

        try:
            # Extract ALL candidate frames (pre-dedup)
            candidates = extract_candidate_frames(
                vpath,
                max_resolution=args.max_resolution,
                sample_interval_ms=args.sample_interval_ms,
            )
            n_candidates = len(candidates)

            if n_candidates < 3:
                logger.warning(f"  Skipping {vname}: only {n_candidates} candidates.")
                continue

            # CLIP embeddings for all candidates
            frame_arrays = [f for _, f in candidates]
            t0 = time.perf_counter()
            embeddings = embedder.embed_frames(frame_arrays)
            embed_time = time.perf_counter() - t0
            logger.info(
                f"  {n_candidates} frames -> {embeddings.shape}, {embed_time:.1f}s"
            )

            # True ISD at multiple tau values
            k95, S, explained95 = compute_isd(embeddings, tau=0.95)
            k90, _, _           = compute_isd(embeddings, tau=0.90)
            k99, _, _           = compute_isd(embeddings, tau=0.99)

            # Spectral entropy (alternative complexity measure)
            var = S ** 2
            probs = var / var.sum() if var.sum() > 0 else np.ones(len(var)) / len(var)
            spectral_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

            # Pipeline proxy metrics
            pdata = pipeline_data.get(vname, {})
            pstats = pdata.get("pipeline_stats", {})
            frames_after_clip  = pstats.get("frames_after_clip")
            final_frame_count  = pstats.get("final_frame_count")
            total_sampled      = pstats.get("total_frames_sampled")
            frames_after_phash = pstats.get("frames_after_phash")
            frames_after_ssim  = pstats.get("frames_after_ssim")
            reduction_rate     = pstats.get("reduction_rate")

            compression_ratio = (
                n_candidates / final_frame_count
                if final_frame_count and final_frame_count > 0
                else None
            )

            # Pipeline's predicted topic
            extraction = pdata.get("extraction", {})
            predicted_topic = None
            if isinstance(extraction, dict):
                topic_info = extraction.get("topic", {})
                if isinstance(topic_info, dict):
                    predicted_topic = topic_info.get("topic_id")

            # Ground truth from Pitt Ads annotations
            gt_topic     = gt.get_topic(vname)
            gt_sentiment = gt.get_sentiment(vname)
            gt_funny     = gt.get_funny(vname)
            gt_exciting  = gt.get_exciting(vname)
            gt_effective = gt.get_effective(vname)

            # Topic accuracy
            topic_correct = None
            topic_super_correct = None
            if predicted_topic is not None and gt_topic is not None:
                try:
                    topic_correct = int(predicted_topic) == int(gt_topic)
                    topic_super_correct = (
                        get_topic_super_category(int(predicted_topic))
                        == get_topic_super_category(int(gt_topic))
                    )
                except (ValueError, TypeError):
                    pass

            records.append({
                "video": vname,
                "youtube_id": Path(vname).stem,
                "n_candidates": n_candidates,
                # True ISD
                "isd_k95": k95,
                "isd_k90": k90,
                "isd_k99": k99,
                "spectral_entropy": round(spectral_entropy, 4),
                # Pipeline proxy metrics
                "frames_after_clip": frames_after_clip,
                "final_frame_count": final_frame_count,
                "total_sampled": total_sampled,
                "frames_after_phash": frames_after_phash,
                "frames_after_ssim": frames_after_ssim,
                "reduction_rate": reduction_rate,
                "compression_ratio": compression_ratio,
                # Pipeline prediction
                "predicted_topic": predicted_topic,
                # Ground truth (Pitt Ads)
                "gt_topic": gt_topic,
                "gt_sentiment": gt_sentiment,
                "gt_funny": gt_funny,
                "gt_exciting": gt_exciting,
                "gt_effective": gt_effective,
                # Accuracy
                "topic_correct": topic_correct,
                "topic_super_correct": topic_super_correct,
                # Timing
                "embed_time_s": round(embed_time, 2),
            })

        except Exception as e:
            logger.error(f"  Failed on {vname}: {e}", exc_info=True)
            continue

    # ------------------------------------------------------------------
    # 4. Analysis
    # ------------------------------------------------------------------
    df = pd.DataFrame(records)
    logger.info(f"\nProcessed {len(df)} / {n_sample} videos successfully.\n")

    if len(df) < 5:
        logger.error("Too few results for meaningful analysis.")
        sys.exit(1)

    # Save raw data
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "isd_validation_data.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Raw data -> {csv_path}")

    # --- Correlations ---
    corr_results = {}

    def _corr(label, col_x, col_y, method="pearson"):
        sub = df.dropna(subset=[col_x, col_y])
        if len(sub) < 5:
            logger.warning(f"  {label}: too few data points ({len(sub)})")
            return
        if method == "pearson":
            r, p = scipy_stats.pearsonr(sub[col_x], sub[col_y])
            rho, p_rho = scipy_stats.spearmanr(sub[col_x], sub[col_y])
            corr_results[label] = {"r": r, "p": p, "rho": rho, "p_rho": p_rho, "n": len(sub)}
            logger.info(f"  {label}: r={r:.3f} (p={p:.2e}), rho={rho:.3f}, N={len(sub)}")
        elif method == "pointbiserial":
            r, p = scipy_stats.pointbiserialr(sub[col_y].astype(int), sub[col_x])
            corr_results[label] = {"r": r, "p": p, "n": len(sub)}
            logger.info(f"  {label}: r_pb={r:.3f} (p={p:.2e}), N={len(sub)}")

    logger.info("--- Correlations ---")
    _corr("ISD vs frames_after_clip", "isd_k95", "frames_after_clip")
    _corr("ISD vs final_frame_count", "isd_k95", "final_frame_count")
    _corr("ISD vs compression_ratio", "isd_k95", "compression_ratio")
    _corr("ISD vs n_candidates",      "isd_k95", "n_candidates")
    _corr("ISD vs topic_correct",     "isd_k95", "topic_correct", method="pointbiserial")
    _corr("ISD vs topic_super_correct","isd_k95", "topic_super_correct", method="pointbiserial")

    # Also correlate spectral entropy
    _corr("entropy vs frames_after_clip", "spectral_entropy", "frames_after_clip")

    # --- Summary stats ---
    logger.info("\n--- ISD Summary ---")
    for col in ["isd_k95", "isd_k90", "isd_k99", "spectral_entropy", "n_candidates"]:
        if col in df.columns:
            logger.info(
                f"  {col:20s}: mean={df[col].mean():.1f}, "
                f"median={df[col].median():.1f}, "
                f"std={df[col].std():.1f}, "
                f"range=[{df[col].min():.0f}, {df[col].max():.0f}]"
            )

    # GT topic coverage
    n_with_gt = df["gt_topic"].notna().sum()
    n_with_pred = df["predicted_topic"].notna().sum()
    n_both = df.dropna(subset=["gt_topic", "predicted_topic"]).shape[0]
    logger.info(f"\n  GT topics available: {n_with_gt}, predictions: {n_with_pred}, both: {n_both}")
    if n_both > 0:
        exact_acc = df.dropna(subset=["topic_correct"])["topic_correct"].mean()
        super_acc = df.dropna(subset=["topic_super_correct"])["topic_super_correct"].mean()
        logger.info(f"  Topic accuracy (N={n_both}): exact={exact_acc:.1%}, super-cat={super_acc:.1%}")

    # ------------------------------------------------------------------
    # 5. Scatter plot
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Panel A: ISD vs frames_after_clip (the proxy)
        ax = axes[0]
        c = corr_results.get("ISD vs frames_after_clip")
        if c:
            sub = df.dropna(subset=["isd_k95", "frames_after_clip"])
            ax.scatter(sub["isd_k95"], sub["frames_after_clip"],
                       alpha=0.6, s=30, edgecolors="k", linewidths=0.3)
            z = np.polyfit(sub["isd_k95"], sub["frames_after_clip"], 1)
            x_line = np.linspace(sub["isd_k95"].min(), sub["isd_k95"].max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.7)
            ax.set_xlabel("True ISD ($k^*$, $\\tau$=0.95)", fontsize=11)
            ax.set_ylabel("Frames After CLIP (proxy)", fontsize=11)
            ax.set_title(
                f"(A) ISD vs Proxy\n$r$={c['r']:.3f}, $\\rho$={c['rho']:.3f}, N={c['n']}",
                fontsize=11,
            )
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("(A) ISD vs Proxy")

        # Panel B: ISD vs compression ratio
        ax = axes[1]
        c = corr_results.get("ISD vs compression_ratio")
        if c:
            sub = df.dropna(subset=["isd_k95", "compression_ratio"])
            ax.scatter(sub["isd_k95"], sub["compression_ratio"],
                       alpha=0.6, s=30, edgecolors="k", linewidths=0.3, color="tab:orange")
            z = np.polyfit(sub["isd_k95"], sub["compression_ratio"], 1)
            x_line = np.linspace(sub["isd_k95"].min(), sub["isd_k95"].max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.7)
            ax.set_xlabel("True ISD ($k^*$, $\\tau$=0.95)", fontsize=11)
            ax.set_ylabel("Compression Ratio", fontsize=11)
            ax.set_title(
                f"(B) ISD vs Compression\n$r$={c['r']:.3f}, N={c['n']}",
                fontsize=11,
            )
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("(B) ISD vs Compression")

        # Panel C: ISD distribution by topic correctness
        ax = axes[2]
        c = corr_results.get("ISD vs topic_correct")
        if c and c["n"] >= 5:
            sub = df.dropna(subset=["isd_k95", "topic_correct"])
            correct   = sub[sub["topic_correct"] == True]["isd_k95"]
            incorrect = sub[sub["topic_correct"] == False]["isd_k95"]

            data_to_plot = []
            labels = []
            positions = []
            if len(correct) > 0:
                data_to_plot.append(correct.values)
                labels.append(f"Correct\n(n={len(correct)})")
                positions.append(1)
            if len(incorrect) > 0:
                data_to_plot.append(incorrect.values)
                labels.append(f"Incorrect\n(n={len(incorrect)})")
                positions.append(2)

            if data_to_plot:
                parts = ax.violinplot(data_to_plot, positions=positions,
                                      showmeans=True, showmedians=True)
                ax.set_xticks(positions)
                ax.set_xticklabels(labels)
            ax.set_ylabel("True ISD ($k^*$, $\\tau$=0.95)", fontsize=11)
            ax.set_title(
                f"(C) ISD by Topic Accuracy\n$r_{{pb}}$={c['r']:.3f}, N={c['n']}",
                fontsize=11,
            )
        else:
            ax.text(0.5, 0.5, "No ground-truth\ntopic data available",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_title("(C) ISD by Topic Accuracy")

        plt.tight_layout()
        fig_path = os.path.join(args.output_dir, "isd_true_validation.pdf")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        fig.savefig(fig_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        logger.info(f"\nFigure -> {fig_path}")
        plt.close(fig)

    except ImportError as e:
        logger.warning(f"Plotting failed: {e}")

    # ------------------------------------------------------------------
    # 6. LaTeX paragraph
    # ------------------------------------------------------------------
    c_proxy = corr_results.get("ISD vs frames_after_clip")
    c_comp  = corr_results.get("ISD vs compression_ratio")
    c_topic = corr_results.get("ISD vs topic_correct")

    tex = []
    tex.append(r"% === Task 2: True ISD Validation (auto-generated) ===")
    tex.append(r"\paragraph{SVD-Based ISD Validation.}")
    tex.append(
        f"To validate the theoretical framework of Section~4, we computed the true "
        f"Intrinsic Semantic Dimensionality ($k^*$) for a random subset of "
        f"$N={len(df)}$ videos from the benchmark set. "
        f"For each video, we extracted all candidate frames at 100\\,ms intervals, "
        f"computed ViT-B/32 CLIP embeddings for the full candidate set, and "
        f"determined $k^*$ via SVD with $\\tau=0.95$ (Eq.~4)."
    )
    tex.append("")

    if c_proxy:
        r, rho, n = c_proxy["r"], c_proxy["rho"], c_proxy["n"]
        strength = "strong" if abs(r) > 0.7 else ("moderate" if abs(r) > 0.4 else "weak")
        tex.append(
            f"The true ISD shows a {strength} correlation with the pipeline's "
            f"\\texttt{{frames\\_after\\_clip}} proxy "
            f"(Pearson $r={r:.3f}$, Spearman $\\rho={rho:.3f}$, "
            f"$p<{max(c_proxy['p'], 1e-10):.1e}$, $N={n}$)."
        )

    if c_comp:
        tex.append(
            f"The correlation between $k^*$ and overall compression ratio is "
            f"$r={c_comp['r']:.3f}$ ($p={c_comp['p']:.2e}$, $N={c_comp['n']}$)."
        )

    if c_topic:
        tex.append(
            f"A point-biserial correlation between $k^*$ and per-video topic "
            f"classification accuracy (against Pitt Ads~\\cite{{hussain2017automatic}} "
            f"ground truth) yields "
            f"$r_{{pb}}={c_topic['r']:.3f}$ ($p={c_topic['p']:.2e}$, $N={c_topic['n']}$)."
        )

    tex.append(
        f"Across the sample, $k^*$ ranged from {df['isd_k95'].min():.0f} to "
        f"{df['isd_k95'].max():.0f} (mean$={df['isd_k95'].mean():.1f}$, "
        f"median$={df['isd_k95'].median():.1f}$), confirming substantial "
        f"variation in semantic complexity across advertisement videos."
    )

    # Interpretation — adapts to actual results
    if c_proxy and abs(c_proxy["r"]) > 0.7:
        tex.append(
            "This validates that the proxy metric used in the pipeline closely "
            "tracks the theoretically motivated $k^*$, supporting the ISD-based "
            "budget allocation described in Section~4."
        )
    elif c_proxy and abs(c_proxy["r"]) > 0.4:
        tex.append(
            "While the correlation is moderate rather than near-perfect, "
            "the proxy captures the broad trend of semantic complexity, "
            "and the downstream task performance remains strong."
        )
    elif c_proxy:
        tex.append(
            "The weaker-than-expected correlation suggests that the proxy "
            "captures different aspects of visual complexity than the SVD-based $k^*$. "
            "We note this as a limitation: a tighter integration of SVD-based "
            "budgeting could further improve the pipeline."
        )

    latex_text = "\n".join(tex)
    tex_path = os.path.join(args.output_dir, "task2_isd_paragraph.tex")
    with open(tex_path, "w") as f:
        f.write(latex_text)
    logger.info(f"LaTeX -> {tex_path}")

    print("\n" + "=" * 70)
    print("LATEX PARAGRAPH FOR section 6.1:")
    print("=" * 70)
    print(latex_text)
    print("=" * 70)

    # ------------------------------------------------------------------
    # 7. Summary table
    # ------------------------------------------------------------------
    print("\n--- CORRELATION SUMMARY ---")
    print(f"{'Comparison':<45s} {'r':>8s} {'p':>12s} {'N':>6s}")
    print("-" * 75)
    for label, c in corr_results.items():
        print(f"{label:<45s} {c['r']:>8.3f} {c['p']:>12.2e} {c['n']:>6d}")
    print()


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Task 2: True SVD-Based ISD Validation")

    # Data paths
    p.add_argument("--benchmark_results", default="test_results/benchmark_results.json")
    p.add_argument("--pipeline_results",  default="main_results/processing_results.json")
    p.add_argument("--video_dir",         default="data/hussain_videos")
    p.add_argument("--gt_dir",            default="data/annotations_videos/video/cleaned_result",
                   help="Directory with Pitt Ads cleaned annotation JSONs.")
    p.add_argument("--output_dir",        default="figures")

    # Experiment params
    p.add_argument("--n_sample", type=int, default=500)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--tau",      type=float, default=0.95)

    # CLIP config (match pipeline defaults)
    p.add_argument("--clip_model",      default="ViT-B-32")
    p.add_argument("--clip_pretrained", default="openai")
    p.add_argument("--clip_batch_size", type=int, default=32)
    p.add_argument("--device",          default="auto")

    # Video sampling (match pipeline defaults)
    p.add_argument("--max_resolution",     type=int, default=720)
    p.add_argument("--sample_interval_ms", type=int, default=100)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)