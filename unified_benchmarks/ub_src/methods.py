#!/usr/bin/env python3
"""
Baseline frame selection methods for unified benchmarks.

Implements 7 baseline methods:
1. Uniform 1 FPS sampling
2. Random sampling
3. Histogram-based change detection
4. ORB feature matching
5. Optical flow peaks
6. CLIP-only sequential dedup
7. K-means clustering
"""

import random
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseMethod(ABC):
    """Abstract base class for all frame selection methods."""

    def __init__(self, name: str, requires_gpu: bool = False):
        self.name = name
        self.requires_gpu = requires_gpu

    @abstractmethod
    def select_frames(
        self,
        video_path: str,
        **kwargs
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Select frames from video.

        Args:
            video_path: Path to video file
            **kwargs: Additional arguments (target_k, clip_embeddings, etc.)

        Returns:
            List of (timestamp_seconds, bgr_frame) tuples
        """
        pass

    def run_timed(self, video_path: str, **kwargs) -> Tuple[List[Tuple[float, np.ndarray]], float]:
        """Run method and return frames with timing."""
        import time
        start = time.time()
        frames = self.select_frames(video_path, **kwargs)
        latency = time.time() - start
        return frames, latency


class UniformSampling(BaseMethod):
    """Uniform sampling at 1 FPS (or specified target FPS)."""

    def __init__(self, target_fps: float = 1.0):
        super().__init__(name="uniform_1fps", requires_gpu=False)
        self.target_fps = target_fps

    def select_frames(self, video_path: str, **kwargs) -> List[Tuple[float, np.ndarray]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # Sample every Nth frame to achieve target_fps
        sample_interval = int(round(fps / self.target_fps))
        if sample_interval < 1:
            sample_interval = 1

        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                timestamp = frame_idx / fps
                frames.append((timestamp, frame))

            frame_idx += 1

        cap.release()
        logger.info(f"Uniform sampling: selected {len(frames)} frames from {total_frames}")
        return frames


class RandomSampling(BaseMethod):
    """Random sampling with target frame count."""

    def __init__(self):
        super().__init__(name="random", requires_gpu=False)

    def select_frames(self, video_path: str, **kwargs) -> List[Tuple[float, np.ndarray]]:
        target_k = kwargs.get('target_k')
        all_frames = kwargs.get('all_frames')

        if all_frames is None:
            # Decode all frames at 100ms interval
            all_frames = self._decode_all_frames(video_path)

        if target_k is None or target_k >= len(all_frames):
            return all_frames

        # Random sample
        selected = random.sample(all_frames, target_k)
        # Sort by timestamp
        selected.sort(key=lambda x: x[0])

        return selected

    def _decode_all_frames(self, video_path: str, interval_ms: int = 100) -> List[Tuple[float, np.ndarray]]:
        """Decode frames at specified interval."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval_frames = int(fps * interval_ms / 1000)
        if interval_frames < 1:
            interval_frames = 1

        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval_frames == 0:
                timestamp = frame_idx / fps
                frames.append((timestamp, frame))

            frame_idx += 1

        cap.release()
        return frames


class HistogramDedup(BaseMethod):
    """HSV histogram change detection."""

    def __init__(self, correlation_threshold: float = 0.95):
        super().__init__(name="histogram", requires_gpu=False)
        self.correlation_threshold = correlation_threshold

    def select_frames(self, video_path: str, **kwargs) -> List[Tuple[float, np.ndarray]]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
        prev_hist = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to HSV and compute histogram
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 16, 16], [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            # Compare with previous
            if prev_hist is not None:
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if correlation >= self.correlation_threshold:
                    frame_idx += 1
                    continue  # Skip similar frame

            # Keep this frame
            timestamp = frame_idx / fps
            frames.append((timestamp, frame))
            prev_hist = hist
            frame_idx += 1

        cap.release()
        logger.info(f"Histogram dedup: selected {len(frames)} frames")
        return frames


class ORBDedup(BaseMethod):
    """ORB feature matching for deduplication."""

    def __init__(self, n_features: int = 500, good_match_threshold: int = 40):
        super().__init__(name="orb", requires_gpu=False)
        self.n_features = n_features
        self.good_match_threshold = good_match_threshold
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def select_frames(self, video_path: str, **kwargs) -> List[Tuple[float, np.ndarray]]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
        prev_des = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Compute ORB features
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, des = self.orb.detectAndCompute(gray, None)

            # If no features, auto-keep
            if des is None:
                timestamp = frame_idx / fps
                frames.append((timestamp, frame))
                prev_des = None
                frame_idx += 1
                continue

            # Compare with previous
            if prev_des is not None:
                matches = self.bf.match(prev_des, des)
                good_matches = [m for m in matches if m.distance < 50]

                if len(good_matches) >= self.good_match_threshold:
                    frame_idx += 1
                    continue  # Skip similar frame

            # Keep this frame
            timestamp = frame_idx / fps
            frames.append((timestamp, frame))
            prev_des = des
            frame_idx += 1

        cap.release()
        logger.info(f"ORB dedup: selected {len(frames)} frames")
        return frames


class OpticalFlowPeaks(BaseMethod):
    """Optical flow motion peak detection."""

    def __init__(self, percentile: float = 85.0):
        super().__init__(name="optical_flow", requires_gpu=False)
        self.percentile = percentile

    def select_frames(self, video_path: str, **kwargs) -> List[Tuple[float, np.ndarray]]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # First pass: compute flow magnitudes
        magnitudes = []
        frames_buffer = []
        prev_gray = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
                magnitudes.append((frame_idx, mag, frame))
            else:
                # Always include first frame
                magnitudes.append((frame_idx, float('inf'), frame))

            prev_gray = gray
            frame_idx += 1

        cap.release()

        if len(magnitudes) <= 2:
            return [(idx / fps, frame) for idx, _, frame in magnitudes]

        # Compute threshold
        mags_only = [m for _, m, _ in magnitudes[1:-1]]  # Exclude first/last
        threshold = np.percentile(mags_only, self.percentile) if mags_only else 0

        # Select peaks and first/last
        selected = []
        for idx, mag, frame in magnitudes:
            if idx == 0 or idx == magnitudes[-1][0] or mag >= threshold:
                timestamp = idx / fps
                selected.append((timestamp, frame))

        logger.info(f"Optical flow: selected {len(selected)} frames")
        return selected


class CLIPOnlyDedup(BaseMethod):
    """CLIP embedding sequential deduplication."""

    def __init__(self, cosine_threshold: float = 0.92):
        super().__init__(name="clip_only", requires_gpu=True)
        self.cosine_threshold = cosine_threshold
        self.clip_dedup = None

    def select_frames(self, video_path: str, **kwargs) -> List[Tuple[float, np.ndarray]]:
        clip_embeddings = kwargs.get('clip_embeddings')
        all_frames = kwargs.get('all_frames')

        if all_frames is None:
            raise ValueError("CLIPOnlyDedup requires all_frames in kwargs")

        if clip_embeddings is None:
            # Compute embeddings
            from ub_src.extraction_wrapper import get_clip_deduplicator
            self.clip_dedup = get_clip_deduplicator()
            frame_arrays = [f for _, f in all_frames]
            clip_embeddings = self.clip_dedup.compute_signatures_batch(frame_arrays)

        # Sequential dedup based on cosine similarity
        selected = []
        prev_embedding = None

        for i, (timestamp, frame) in enumerate(all_frames):
            emb = clip_embeddings[i]

            if prev_embedding is not None:
                similarity = np.dot(emb, prev_embedding) / (np.linalg.norm(emb) * np.linalg.norm(prev_embedding))
                if similarity >= self.cosine_threshold:
                    continue  # Skip similar frame

            selected.append((timestamp, frame))
            prev_embedding = emb

        logger.info(f"CLIP-only: selected {len(selected)} frames from {len(all_frames)}")
        return selected


class KMeansClustering(BaseMethod):
    """K-means clustering on CLIP embeddings."""

    def __init__(self, seconds_per_cluster: int = 3):
        super().__init__(name="kmeans", requires_gpu=True)
        self.seconds_per_cluster = seconds_per_cluster

    def select_frames(self, video_path: str, **kwargs) -> List[Tuple[float, np.ndarray]]:
        clip_embeddings = kwargs.get('clip_embeddings')
        all_frames = kwargs.get('all_frames')
        duration = kwargs.get('duration')

        if all_frames is None or duration is None:
            raise ValueError("KMeansClustering requires all_frames and duration in kwargs")

        if clip_embeddings is None:
            from ub_src.extraction_wrapper import get_clip_deduplicator
            clip_dedup = get_clip_deduplicator()
            frame_arrays = [f for _, f in all_frames]
            clip_embeddings = clip_dedup.compute_signatures_batch(frame_arrays)

        # Determine k based on duration
        k = max(5, min(20, int(duration / self.seconds_per_cluster)))
        k = min(k, len(all_frames))  # Can't have more clusters than frames

        if k >= len(all_frames):
            return all_frames

        # Run K-means
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(clip_embeddings)

        # Select frame nearest to each centroid
        selected = []
        for cluster_id in range(k):
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue

            centroid = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(clip_embeddings[cluster_indices] - centroid, axis=1)
            best_idx = cluster_indices[np.argmin(distances)]

            timestamp, frame = all_frames[best_idx]
            selected.append((timestamp, frame))

        # Sort by timestamp
        selected.sort(key=lambda x: x[0])

        logger.info(f"K-means: selected {len(selected)} frames with k={k}")
        return selected


# Registry of all methods
ALL_METHODS = {
    "uniform_1fps": UniformSampling,
    "random": RandomSampling,
    "histogram": HistogramDedup,
    "orb": ORBDedup,
    "optical_flow": OpticalFlowPeaks,
    "clip_only": CLIPOnlyDedup,
    "kmeans": KMeansClustering,
}


def get_method(name: str, **kwargs) -> BaseMethod:
    """Get method instance by name."""
    if name not in ALL_METHODS:
        raise ValueError(f"Unknown method: {name}. Available: {list(ALL_METHODS.keys())}")

    return ALL_METHODS[name](**kwargs)
