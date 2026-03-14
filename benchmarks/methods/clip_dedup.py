"""
CLIP-only deduplication baseline.
Uses CLIP embeddings to select diverse frames.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

from benchmarks.base import BaselineMethod

logger = logging.getLogger(__name__)


class CLIPOnlyDedup(BaselineMethod):
    """
    Select frames using CLIP embedding similarity.

    This is a GPU-dependent method that requires pre-computed CLIP embeddings.
    It selects frames that are most dissimilar to already-selected frames.
    """

    name = "clip_only"
    requires_gpu = True

    def __init__(self, threshold: float = 0.92):
        """
        Args:
            threshold: Cosine similarity threshold for considering frames as duplicates
        """
        self.threshold = threshold

    def select_frames(
        self,
        video_path: str,
        target_k: int,
        clip_embeddings: Optional[np.ndarray] = None,
        all_frames: Optional[List[Tuple[float, np.ndarray]]] = None,
        **kwargs
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Select diverse frames using CLIP embeddings.

        Args:
            video_path: Path to video file
            target_k: Target number of frames to select
            clip_embeddings: Pre-computed CLIP embeddings (N, D) array
            all_frames: All decoded frames as (timestamp, frame_array) tuples

        Returns:
            List of (timestamp, frame_array) tuples for selected frames
        """
        if clip_embeddings is None or all_frames is None:
            logger.warning("CLIP embeddings or frames not available, returning empty")
            return []

        if len(clip_embeddings) != len(all_frames):
            logger.error(f"Mismatch: {len(clip_embeddings)} embeddings vs {len(all_frames)} frames")
            return []

        n_frames = len(clip_embeddings)
        if n_frames == 0:
            return []

        # Normalize embeddings
        norms = np.linalg.norm(clip_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = clip_embeddings / norms

        # Greedy selection: pick most diverse frames
        selected_indices = []

        # Always select first frame
        if n_frames > 0:
            selected_indices.append(0)

        while len(selected_indices) < min(target_k, n_frames):
            # Compute similarity to already selected frames
            max_sims = np.zeros(n_frames)

            for i in range(n_frames):
                if i in selected_indices:
                    max_sims[i] = 1.0  # Already selected
                    continue

                # Compute max similarity to any selected frame
                sims = normalized[i] @ normalized[selected_indices].T
                max_sims[i] = np.max(sims)

            # Select frame with minimum max-similarity (most diverse)
            candidates = np.where(max_sims < self.threshold)[0]

            if len(candidates) == 0:
                # No more diverse frames, pick the least similar one
                next_idx = np.argmin(max_sims)
            else:
                # Pick the most diverse among candidates
                next_idx = candidates[np.argmin(max_sims[candidates])]

            selected_indices.append(int(next_idx))

        # Return selected frames
        selected_frames = [all_frames[i] for i in selected_indices]
        logger.info(f"CLIP dedup: selected {len(selected_frames)}/{n_frames} frames")

        return selected_frames