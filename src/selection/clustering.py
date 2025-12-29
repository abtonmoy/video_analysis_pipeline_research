#src\selection\clustering.py
"""
Temporal clustering for frame selection with adaptive density-based allocation.
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
    Cluster frames within scenes and select representatives using density-based allocation.
    
    Instead of fixed frames per scene, allocates frames proportionally to scene duration.
    This ensures long scenes get adequate coverage while short scenes aren't oversampled.
    """
    
    def __init__(
        self,
        target_frame_density: float = 0.25,  # ~1 frame every 4 seconds
        min_frames_per_scene: int = 2,
        max_frames_per_scene: int = 10,
        min_temporal_gap_s: float = 0.5,
        clustering_method: str = "kmeans",
        adaptive_density: bool = True
    ):
        self.target_density = target_frame_density
        self.min_frames_per_scene = min_frames_per_scene
        self.max_frames_per_scene = max_frames_per_scene
        self.min_temporal_gap_s = min_temporal_gap_s
        self.clustering_method = clustering_method
        self.adaptive_density = adaptive_density
    
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
        Cluster frames within each scene and select representatives using density-based allocation.
        
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
        
        # ALWAYS include first and last frame
        first_frame = min(candidates, key=lambda c: c.timestamp)
        last_frame = max(candidates, key=lambda c: c.timestamp)
        
        first_frame.is_representative = True
        last_frame.is_representative = True
        
        must_include_timestamps = {first_frame.timestamp, last_frame.timestamp}
        
        # Group by scene
        scene_frames: Dict[int, List[FrameCandidate]] = {}
        for cand in candidates:
            scene_id = cand.scene_id or 0
            if scene_id not in scene_frames:
                scene_frames[scene_id] = []
            scene_frames[scene_id].append(cand)
        
        # Select representatives from each scene with density-based allocation
        selected = []
        
        for scene_id in sorted(scene_frames.keys()):
            scene_cands = scene_frames[scene_id]
            
            # Calculate scene duration
            scene_start = min(c.timestamp for c in scene_cands)
            scene_end = max(c.timestamp for c in scene_cands)
            scene_duration = scene_end - scene_start
            
            # Adaptive density based on scene characteristics
            if self.adaptive_density:
                density = self._calculate_adaptive_density(scene_cands, self.target_density)
            else:
                density = self.target_density
            
            # Calculate target frame count based on density
            target_count = int(scene_duration * density)
            
            # Apply constraints
            n_frames = max(
                self.min_frames_per_scene,
                min(target_count, self.max_frames_per_scene, len(scene_cands))
            )
            
            logger.debug(f"Scene {scene_id}: duration={scene_duration:.1f}s, "
                        f"density={density:.2f}, target={target_count}, "
                        f"final={n_frames} frames")
            
            # Check how many must-include frames are in this scene
            scene_must_include = [c for c in scene_cands if c.timestamp in must_include_timestamps]
            remaining_slots = n_frames - len(scene_must_include)
            
            if len(scene_cands) <= n_frames:
                # Keep all frames in small scenes
                for cand in scene_cands:
                    cand.is_representative = True
                selected.extend(scene_cands)
            else:
                # Add must-include frames
                selected.extend(scene_must_include)
                
                # Cluster and select from remaining frames
                remaining_cands = [c for c in scene_cands if c.timestamp not in must_include_timestamps]
                
                if remaining_slots > 0 and remaining_cands:
                    reps = self._select_representatives(remaining_cands, remaining_slots)
                    selected.extend(reps)
        
        # Remove duplicates (in case first/last are in same scene)
        unique_selected = []
        seen_timestamps = set()
        
        for cand in selected:
            if cand.timestamp not in seen_timestamps:
                unique_selected.append(cand)
                seen_timestamps.add(cand.timestamp)
        
        # Sort by timestamp
        unique_selected.sort(key=lambda x: x.timestamp)
        
        # Enforce minimum temporal gap (but preserve first and last)
        unique_selected = self._enforce_temporal_gap(unique_selected)
        
        logger.info(f"Selected {len(unique_selected)} representatives from {len(candidates)} candidates")
        
        return unique_selected
    
    def _calculate_adaptive_density(self, scene_cands: List[FrameCandidate], base_density: float) -> float:
        """
        Adjust density based on scene complexity.
        
        High variance scenes (lots of changes) get more frames.
        Low variance scenes (static) get fewer frames.
        """
        if len(scene_cands) < 2:
            return base_density
        
        # Compute frame variance (how much changes within scene)
        variance = self._compute_frame_variance([c.frame for c in scene_cands])
        
        # Adjust density based on variance
        if variance > 0.15:  # High complexity
            adjusted_density = base_density * 1.3
            logger.debug(f"High variance ({variance:.3f}), increasing density to {adjusted_density:.3f}")
        elif variance < 0.05:  # Low complexity
            adjusted_density = base_density * 0.7
            logger.debug(f"Low variance ({variance:.3f}), decreasing density to {adjusted_density:.3f}")
        else:
            adjusted_density = base_density
        
        return adjusted_density
    
    def _compute_frame_variance(self, frames: List[np.ndarray]) -> float:
        """Compute variance across frames (pixel-based)."""
        if len(frames) < 2:
            return 0.0
        
        # Sample frames if too many (for efficiency)
        if len(frames) > 10:
            step = len(frames) // 10
            frames = frames[::step]
        
        variances = []
        for i in range(len(frames) - 1):
            # Convert to grayscale for efficiency
            gray1 = np.mean(frames[i], axis=2) if frames[i].ndim == 3 else frames[i]
            gray2 = np.mean(frames[i+1], axis=2) if frames[i+1].ndim == 3 else frames[i+1]
            
            diff = np.abs(gray1.astype(float) - gray2.astype(float))
            variances.append(np.mean(diff) / 255.0)
        
        return np.mean(variances)
    
    def _select_representatives(
        self,
        scene_frames: List[FrameCandidate],
        n_clusters: int
    ) -> List[FrameCandidate]:
        """Select representative frames from a scene using clustering."""
        
        # Check if we have embeddings
        has_embeddings = all(f.embedding is not None for f in scene_frames)
        
        if has_embeddings and self.clustering_method == "kmeans":
            return self._kmeans_selection(scene_frames, n_clusters)
        else:
            return self._uniform_selection(scene_frames, n_clusters)
    
    def _kmeans_selection(
        self,
        scene_frames: List[FrameCandidate],
        n_clusters: int
    ) -> List[FrameCandidate]:
        """Use K-means clustering to select representatives."""
        from sklearn.cluster import KMeans
        
        n_clusters = min(n_clusters, len(scene_frames))
        
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
            best_frame = min(
                cluster_frames,
                key=lambda f: np.linalg.norm(f.embedding - centroid)
            )
            
            best_frame.is_representative = True
            selected.append(best_frame)
        
        # Sort by timestamp
        selected.sort(key=lambda x: x.timestamp)
        
        return selected
    
    def _uniform_selection(
        self,
        scene_frames: List[FrameCandidate],
        n_select: int
    ) -> List[FrameCandidate]:
        """Uniformly select frames across the scene."""
        n_select = min(n_select, len(scene_frames))
        
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
        """
        Remove frames that are too close temporally.
        ALWAYS preserves first and last frame.
        """
        if not frames:
            return []
        
        # Sort by timestamp
        sorted_frames = sorted(frames, key=lambda x: x.timestamp)
        
        # ALWAYS keep first and last frame
        if len(sorted_frames) <= 2:
            return sorted_frames
        
        first_frame = sorted_frames[0]
        last_frame = sorted_frames[-1]
        middle_frames = sorted_frames[1:-1]
        
        # Keep first frame
        kept = [first_frame]
        
        # Filter middle frames
        for frame in middle_frames:
            if frame.timestamp - kept[-1].timestamp >= self.min_temporal_gap_s:
                kept.append(frame)
        
        # ALWAYS keep last frame (even if gap is small)
        if last_frame.timestamp != kept[-1].timestamp:  # Avoid duplicate
            kept.append(last_frame)
        
        return kept