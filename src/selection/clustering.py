"""
Temporal clustering for frame selection.
"""

import logging
from typing import List, Optional, Dict
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameCandidate:
    """Container for frame candidate with metadata."""
    timestamp: float
    frame: np.ndarray
    embedding: Optional[np.ndarray] = None
    scene_id: Optional[int] = None
    importance_score: float = 1.0
    cluster_id: Optional[int] = None
    is_representative: bool = False


class TemporalClusterer:
    """
    Cluster frames within scenes and select representatives.
    """
    
    def __init__(
        self,
        max_frames_per_scene: int = 3,
        min_temporal_gap_s: float = 0.5,
        clustering_method: str = "kmeans"
    ):
        self.max_frames_per_scene = max_frames_per_scene
        self.min_temporal_gap_s = min_temporal_gap_s
        self.clustering_method = clustering_method
    
    def assign_scenes(
        self,
        frames: List[tuple],
        scene_boundaries: List[tuple]
    ) -> List[FrameCandidate]:
        """
        Assign frames to scenes based on timestamps.
        
        Args:
            frames: List of (timestamp, frame) tuples
            scene_boundaries: List of (start, end) tuples for each scene
            
        Returns:
            List of FrameCandidate with scene_id assigned
        """
        candidates = []
        
        for ts, frame in frames:
            # Find which scene this frame belongs to
            scene_id = None
            for i, (start, end) in enumerate(scene_boundaries):
                if start <= ts < end:
                    scene_id = i
                    break
            
            # If no scene found, assign to nearest
            if scene_id is None:
                min_dist = float('inf')
                for i, (start, end) in enumerate(scene_boundaries):
                    dist = min(abs(ts - start), abs(ts - end))
                    if dist < min_dist:
                        min_dist = dist
                        scene_id = i
            
            candidates.append(FrameCandidate(
                timestamp=ts,
                frame=frame,
                scene_id=scene_id
            ))
        
        return candidates
    
    def cluster_and_select(
        self,
        candidates: List[FrameCandidate],
        embeddings: Optional[np.ndarray] = None
    ) -> List[FrameCandidate]:
        """
        Cluster frames within each scene and select representatives.
        
        Args:
            candidates: List of FrameCandidate with scene_id assigned
            embeddings: Optional CLIP embeddings for clustering
            
        Returns:
            List of selected representative FrameCandidate
        """
        if not candidates:
            return []
        
        # Attach embeddings
        if embeddings is not None:
            for i, cand in enumerate(candidates):
                if i < len(embeddings):
                    cand.embedding = embeddings[i]
        
        # Group by scene
        scene_frames: Dict[int, List[FrameCandidate]] = {}
        for cand in candidates:
            scene_id = cand.scene_id or 0
            if scene_id not in scene_frames:
                scene_frames[scene_id] = []
            scene_frames[scene_id].append(cand)
        
        # Select representatives from each scene
        selected = []
        
        for scene_id in sorted(scene_frames.keys()):
            scene_cands = scene_frames[scene_id]
            
            if len(scene_cands) <= self.max_frames_per_scene:
                # Keep all frames in small scenes
                for cand in scene_cands:
                    cand.is_representative = True
                selected.extend(scene_cands)
            else:
                # Cluster and select
                reps = self._select_representatives(scene_cands)
                selected.extend(reps)
        
        # Enforce minimum temporal gap
        selected = self._enforce_temporal_gap(selected)
        
        logger.info(f"Selected {len(selected)} representatives from {len(candidates)} candidates")
        
        return selected
    
    def _select_representatives(
        self,
        scene_frames: List[FrameCandidate]
    ) -> List[FrameCandidate]:
        """Select representative frames from a scene using clustering."""
        
        # Check if we have embeddings
        has_embeddings = all(f.embedding is not None for f in scene_frames)
        
        if has_embeddings and self.clustering_method == "kmeans":
            return self._kmeans_selection(scene_frames)
        else:
            return self._uniform_selection(scene_frames)
    
    def _kmeans_selection(
        self,
        scene_frames: List[FrameCandidate]
    ) -> List[FrameCandidate]:
        """Use K-means clustering to select representatives."""
        from sklearn.cluster import KMeans
        
        n_clusters = min(self.max_frames_per_scene, len(scene_frames))
        
        # Stack embeddings
        embeddings = np.array([f.embedding for f in scene_frames])
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Assign cluster IDs
        for i, cand in enumerate(scene_frames):
            cand.cluster_id = int(labels[i])
        
        # Select frame closest to each centroid
        selected = []
        for cluster_id in range(n_clusters):
            cluster_frames = [f for f in scene_frames if f.cluster_id == cluster_id]
            if not cluster_frames:
                continue
            
            centroid = kmeans.cluster_centers_[cluster_id]
            
            # Find closest frame
            min_dist = float('inf')
            best_frame = None
            
            for frame in cluster_frames:
                dist = np.linalg.norm(frame.embedding - centroid)
                if dist < min_dist:
                    min_dist = dist
                    best_frame = frame
            
            if best_frame:
                best_frame.is_representative = True
                selected.append(best_frame)
        
        # Sort by timestamp
        selected.sort(key=lambda x: x.timestamp)
        
        return selected
    
    def _uniform_selection(
        self,
        scene_frames: List[FrameCandidate]
    ) -> List[FrameCandidate]:
        """Uniformly select frames across the scene."""
        n_select = min(self.max_frames_per_scene, len(scene_frames))
        
        # Sort by timestamp
        sorted_frames = sorted(scene_frames, key=lambda x: x.timestamp)
        
        # Select uniformly spaced indices
        if n_select == 1:
            indices = [len(sorted_frames) // 2]
        else:
            indices = np.linspace(0, len(sorted_frames) - 1, n_select, dtype=int)
        
        selected = []
        for idx in indices:
            frame = sorted_frames[idx]
            frame.is_representative = True
            selected.append(frame)
        
        return selected
    
    def _enforce_temporal_gap(
        self,
        frames: List[FrameCandidate]
    ) -> List[FrameCandidate]:
        """Remove frames that are too close temporally."""
        if not frames:
            return []
        
        # Sort by timestamp
        sorted_frames = sorted(frames, key=lambda x: x.timestamp)
        
        # Keep first frame
        kept = [sorted_frames[0]]
        
        for frame in sorted_frames[1:]:
            if frame.timestamp - kept[-1].timestamp >= self.min_temporal_gap_s:
                kept.append(frame)
        
        return kept