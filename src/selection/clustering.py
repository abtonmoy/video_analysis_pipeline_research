# src/selection/clustering.py
"""
Temporal clustering and NMS-based frame selection with adaptive density-based allocation.

Supports multiple selection methods:
- nms: Non-Maximum Suppression using importance scores (recommended for ads)
- kmeans: K-means clustering on embeddings (good for semantic diversity)
- uniform: Uniform temporal sampling (fallback)

NMS is preferred for ad extraction because it directly uses importance scores
to prioritize frames near key moments (CTAs, brand reveals, audio events).
"""

import logging
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class SelectionMethod(Enum):
    """Available frame selection methods."""
    NMS = "nms"
    KMEANS = "kmeans"
    UNIFORM = "uniform"
    HYBRID = "hybrid"  # NMS with semantic diversity constraint


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
    suppression_reason: Optional[str] = None  # For debugging NMS decisions


class NMSSelector:
    """
    Non-Maximum Suppression based frame selection.
    
    Unlike K-means which ignores importance scores, NMS directly uses them
    to prioritize high-value frames while suppressing redundant neighbors.
    
    Suppression criteria:
    1. Temporal proximity: Frames too close in time
    2. Semantic similarity: Frames with similar CLIP embeddings
    3. Scene-aware: Can optionally enforce per-scene limits
    """
    
    def __init__(
        self,
        temporal_threshold_s: float = 0.5,
        semantic_threshold: float = 0.88,
        use_semantic_suppression: bool = True,
        importance_weight: float = 1.0,
        diversity_bonus: float = 0.1
    ):
        """
        Args:
            temporal_threshold_s: Minimum time gap between selected frames
            semantic_threshold: Cosine similarity above which frames are suppressed
            use_semantic_suppression: Whether to use embedding similarity for suppression
            importance_weight: Weight for importance score in selection
            diversity_bonus: Bonus for frames that are semantically different from selected
        """
        self.temporal_threshold_s = temporal_threshold_s
        self.semantic_threshold = semantic_threshold
        self.use_semantic_suppression = use_semantic_suppression
        self.importance_weight = importance_weight
        self.diversity_bonus = diversity_bonus
    
    def select(
        self,
        candidates: List[FrameCandidate],
        max_frames: int,
        force_include_timestamps: Optional[set] = None
    ) -> List[FrameCandidate]:
        """
        Select frames using NMS with importance-based ordering.
        
        Args:
            candidates: List of FrameCandidate with importance_score set
            max_frames: Maximum number of frames to select
            force_include_timestamps: Set of timestamps that must be included
            
        Returns:
            List of selected FrameCandidate, sorted by timestamp
        """
        if not candidates:
            return []
        
        if max_frames <= 0:
            return []
        
        force_include = force_include_timestamps or set()
        
        # Separate forced frames from candidates
        forced_frames = [c for c in candidates if c.timestamp in force_include]
        regular_candidates = [c for c in candidates if c.timestamp not in force_include]
        
        # Mark forced frames as selected
        for frame in forced_frames:
            frame.is_representative = True
        
        # Start with forced frames
        selected = list(forced_frames)
        remaining_slots = max_frames - len(selected)
        
        if remaining_slots <= 0 or not regular_candidates:
            return sorted(selected, key=lambda c: c.timestamp)
        
        # Compute effective scores (importance + diversity bonus)
        effective_scores = self._compute_effective_scores(regular_candidates, selected)
        
        # Sort by effective score (highest first)
        scored_candidates = list(zip(regular_candidates, effective_scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # NMS selection loop
        for cand, score in scored_candidates:
            if len(selected) >= max_frames:
                break
            
            suppressed, reason = self._is_suppressed(cand, selected)
            
            if suppressed:
                cand.suppression_reason = reason
                logger.debug(f"Frame at {cand.timestamp:.2f}s suppressed: {reason}")
                continue
            
            cand.is_representative = True
            selected.append(cand)
            
            # Recompute effective scores for remaining candidates (optional, for diversity)
            # This is expensive but improves diversity - can be disabled for speed
        
        # Sort by timestamp for output
        selected.sort(key=lambda c: c.timestamp)
        
        logger.debug(f"NMS selected {len(selected)} frames from {len(candidates)} candidates")
        
        return selected
    
    def _compute_effective_scores(
        self,
        candidates: List[FrameCandidate],
        already_selected: List[FrameCandidate]
    ) -> List[float]:
        """
        Compute effective selection scores incorporating diversity bonus.
        
        Frames that are semantically different from already-selected frames
        get a small bonus to encourage diversity.
        """
        scores = []
        
        for cand in candidates:
            base_score = cand.importance_score * self.importance_weight
            
            # Add diversity bonus based on distance from selected frames
            if already_selected and cand.embedding is not None and self.diversity_bonus > 0:
                min_similarity = 1.0
                for sel in already_selected:
                    if sel.embedding is not None:
                        sim = np.dot(cand.embedding, sel.embedding)
                        min_similarity = min(min_similarity, sim)
                
                # Higher bonus for frames that are different from selected
                diversity = 1.0 - min_similarity
                base_score += diversity * self.diversity_bonus
            
            scores.append(base_score)
        
        return scores
    
    def _is_suppressed(
        self,
        candidate: FrameCandidate,
        selected: List[FrameCandidate]
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if candidate should be suppressed by any selected frame.
        
        Returns:
            Tuple of (is_suppressed, reason)
        """
        for sel in selected:
            # Temporal suppression
            time_diff = abs(candidate.timestamp - sel.timestamp)
            if time_diff < self.temporal_threshold_s:
                return True, f"temporal (dt={time_diff:.2f}s < {self.temporal_threshold_s}s)"
            
            # Semantic suppression
            if (self.use_semantic_suppression and 
                candidate.embedding is not None and 
                sel.embedding is not None):
                
                similarity = np.dot(candidate.embedding, sel.embedding)
                if similarity > self.semantic_threshold:
                    return True, f"semantic (sim={similarity:.3f} > {self.semantic_threshold})"
        
        return False, None


class TemporalClusterer:
    """
    Cluster frames within scenes and select representatives.
    
    Supports multiple selection methods:
    - nms: Non-Maximum Suppression (importance-aware, recommended)
    - kmeans: K-means clustering (semantic diversity)
    - uniform: Uniform temporal sampling (fallback)
    - hybrid: NMS with K-means diversity constraints
    
    Instead of fixed frames per scene, allocates frames proportionally to scene duration.
    This ensures long scenes get adequate coverage while short scenes aren't oversampled.
    """
    
    def __init__(
        self,
        target_frame_density: float = 0.25,  # ~1 frame every 4 seconds
        min_frames_per_scene: int = 2,
        max_frames_per_scene: int = 10,
        min_temporal_gap_s: float = 0.5,
        clustering_method: str = "nms",  # Changed default to nms
        adaptive_density: bool = True,
        # NMS-specific parameters
        semantic_threshold: float = 0.88,
        use_semantic_suppression: bool = True,
        diversity_bonus: float = 0.1
    ):
        self.target_density = target_frame_density
        self.min_frames_per_scene = min_frames_per_scene
        self.max_frames_per_scene = max_frames_per_scene
        self.min_temporal_gap_s = min_temporal_gap_s
        self.clustering_method = clustering_method
        self.adaptive_density = adaptive_density
        
        # NMS selector
        self.nms_selector = NMSSelector(
            temporal_threshold_s=min_temporal_gap_s,
            semantic_threshold=semantic_threshold,
            use_semantic_suppression=use_semantic_suppression,
            diversity_bonus=diversity_bonus
        )
    
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
            embeddings: Optional CLIP embeddings for clustering/NMS
            
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
        
        # Identify first and last frames (ALWAYS include these)
        first_frame = min(candidates, key=lambda c: c.timestamp)
        last_frame = max(candidates, key=lambda c: c.timestamp)
        
        # Boost importance of first/last frames to ensure they survive NMS
        first_frame.importance_score *= 2.0
        last_frame.importance_score *= 1.8
        
        must_include_timestamps = {first_frame.timestamp, last_frame.timestamp}
        
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
            
            # Calculate scene duration
            scene_start = min(c.timestamp for c in scene_cands)
            scene_end = max(c.timestamp for c in scene_cands)
            scene_duration = max(scene_end - scene_start, 0.1)  # Avoid division by zero
            
            # Adaptive density based on scene characteristics
            if self.adaptive_density:
                density = self._calculate_adaptive_density(scene_cands, self.target_density)
            else:
                density = self.target_density
            
            # Calculate target frame count based on density
            target_count = max(1, int(scene_duration * density))
            
            # Apply constraints
            n_frames = max(
                self.min_frames_per_scene,
                min(target_count, self.max_frames_per_scene, len(scene_cands))
            )
            
            logger.debug(f"Scene {scene_id}: duration={scene_duration:.1f}s, "
                        f"density={density:.2f}, target={target_count}, "
                        f"final={n_frames} frames")
            
            # Determine must-include frames for this scene
            scene_must_include = {c.timestamp for c in scene_cands 
                                  if c.timestamp in must_include_timestamps}
            
            if len(scene_cands) <= n_frames:
                # Keep all frames in small scenes
                for cand in scene_cands:
                    cand.is_representative = True
                selected.extend(scene_cands)
            else:
                # Select using configured method
                scene_selected = self._select_representatives(
                    scene_cands, 
                    n_frames,
                    force_include=scene_must_include
                )
                selected.extend(scene_selected)
        
        # Remove duplicates (in case first/last are in same scene)
        unique_selected = []
        seen_timestamps = set()
        
        for cand in selected:
            if cand.timestamp not in seen_timestamps:
                unique_selected.append(cand)
                seen_timestamps.add(cand.timestamp)
        
        # Sort by timestamp
        unique_selected.sort(key=lambda x: x.timestamp)
        
        # Final verification: ensure first and last are included
        unique_selected = self._ensure_endpoints(unique_selected, first_frame, last_frame)
        
        logger.info(f"Selected {len(unique_selected)} representatives from {len(candidates)} "
                   f"candidates using {self.clustering_method}")
        
        return unique_selected
    
    def _select_representatives(
        self,
        scene_frames: List[FrameCandidate],
        n_frames: int,
        force_include: Optional[set] = None
    ) -> List[FrameCandidate]:
        """
        Select representative frames from a scene using configured method.
        
        Args:
            scene_frames: Candidates from a single scene
            n_frames: Target number of frames to select
            force_include: Timestamps that must be included
            
        Returns:
            List of selected FrameCandidate
        """
        has_embeddings = all(f.embedding is not None for f in scene_frames)
        method = self.clustering_method.lower()
        
        if method == "nms" or method == SelectionMethod.NMS.value:
            return self._nms_selection(scene_frames, n_frames, force_include)
        
        elif method == "hybrid" or method == SelectionMethod.HYBRID.value:
            return self._hybrid_selection(scene_frames, n_frames, force_include)
        
        elif method == "kmeans" or method == SelectionMethod.KMEANS.value:
            if has_embeddings:
                return self._kmeans_selection(scene_frames, n_frames, force_include)
            else:
                logger.warning("K-means requested but no embeddings available, falling back to NMS")
                return self._nms_selection(scene_frames, n_frames, force_include)
        
        else:  # uniform or unknown
            return self._uniform_selection(scene_frames, n_frames, force_include)
    
    def _nms_selection(
        self,
        scene_frames: List[FrameCandidate],
        n_frames: int,
        force_include: Optional[set] = None
    ) -> List[FrameCandidate]:
        """
        Select frames using Non-Maximum Suppression.
        
        Prioritizes frames by importance score while suppressing
        temporally and semantically similar frames.
        """
        return self.nms_selector.select(
            scene_frames,
            max_frames=n_frames,
            force_include_timestamps=force_include
        )
    
    def _hybrid_selection(
        self,
        scene_frames: List[FrameCandidate],
        n_frames: int,
        force_include: Optional[set] = None
    ) -> List[FrameCandidate]:
        """
        Hybrid selection: NMS for importance, K-means for diversity guarantee.
        
        1. Use NMS to get top importance frames
        2. Check semantic coverage using K-means clusters
        3. If any cluster is unrepresented, swap in its best frame
        """
        # First pass: NMS selection
        nms_selected = self._nms_selection(scene_frames, n_frames, force_include)
        
        # If no embeddings, just return NMS result
        if not all(f.embedding is not None for f in scene_frames):
            return nms_selected
        
        # Check cluster coverage
        try:
            from sklearn.cluster import KMeans
            
            n_clusters = min(n_frames, len(scene_frames))
            if n_clusters < 2:
                return nms_selected
            
            embeddings = np.array([f.embedding for f in scene_frames])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            # Assign cluster IDs
            for i, cand in enumerate(scene_frames):
                cand.cluster_id = int(labels[i])
            
            # Find which clusters are represented
            selected_clusters = {f.cluster_id for f in nms_selected if f.cluster_id is not None}
            all_clusters = set(range(n_clusters))
            missing_clusters = all_clusters - selected_clusters
            
            if not missing_clusters:
                return nms_selected
            
            # For each missing cluster, find best frame and consider swapping
            selected_list = list(nms_selected)
            force_timestamps = force_include or set()
            
            for cluster_id in missing_clusters:
                cluster_frames = [f for f in scene_frames 
                                 if f.cluster_id == cluster_id and f.timestamp not in force_timestamps]
                
                if not cluster_frames:
                    continue
                
                # Find best frame in cluster by importance
                best_in_cluster = max(cluster_frames, key=lambda f: f.importance_score)
                
                # Find lowest-importance selected frame (not forced)
                swappable = [f for f in selected_list 
                            if f.timestamp not in force_timestamps]
                
                if not swappable:
                    continue
                
                worst_selected = min(swappable, key=lambda f: f.importance_score)
                
                # Swap if cluster frame has reasonable importance
                # (at least 70% of the worst selected frame's importance)
                if best_in_cluster.importance_score >= worst_selected.importance_score * 0.7:
                    worst_selected.is_representative = False
                    selected_list.remove(worst_selected)
                    best_in_cluster.is_representative = True
                    selected_list.append(best_in_cluster)
                    logger.debug(f"Hybrid swap: cluster {cluster_id} frame at "
                               f"{best_in_cluster.timestamp:.2f}s for {worst_selected.timestamp:.2f}s")
            
            selected_list.sort(key=lambda x: x.timestamp)
            return selected_list
            
        except ImportError:
            logger.warning("sklearn not available for hybrid selection, using NMS only")
            return nms_selected
    
    def _kmeans_selection(
        self,
        scene_frames: List[FrameCandidate],
        n_clusters: int,
        force_include: Optional[set] = None
    ) -> List[FrameCandidate]:
        """
        Use K-means clustering to select representatives.
        
        Now improved to prefer higher-importance frames within each cluster.
        """
        from sklearn.cluster import KMeans
        
        force_include = force_include or set()
        n_clusters = min(n_clusters, len(scene_frames))
        
        # Handle forced frames
        forced_frames = [f for f in scene_frames if f.timestamp in force_include]
        regular_frames = [f for f in scene_frames if f.timestamp not in force_include]
        
        for f in forced_frames:
            f.is_representative = True
        
        remaining_slots = n_clusters - len(forced_frames)
        
        if remaining_slots <= 0 or not regular_frames:
            return forced_frames
        
        # Stack embeddings
        embeddings = np.array([f.embedding for f in regular_frames])
        
        # Cluster
        actual_clusters = min(remaining_slots, len(regular_frames))
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Assign cluster IDs
        for i, cand in enumerate(regular_frames):
            cand.cluster_id = int(labels[i])
        
        # Select best frame from each cluster (by importance, not just centroid distance)
        selected = list(forced_frames)
        
        for cluster_id in range(actual_clusters):
            cluster_frames = [f for f in regular_frames if f.cluster_id == cluster_id]
            if not cluster_frames:
                continue
            
            centroid = kmeans.cluster_centers_[cluster_id]
            
            # Score: importance * (1 - normalized_distance_to_centroid)
            # This balances importance with representativeness
            max_dist = max(np.linalg.norm(f.embedding - centroid) for f in cluster_frames)
            if max_dist == 0:
                max_dist = 1.0
            
            def combined_score(f):
                dist = np.linalg.norm(f.embedding - centroid)
                centroid_score = 1 - (dist / max_dist)
                return f.importance_score * 0.6 + centroid_score * 0.4
            
            best_frame = max(cluster_frames, key=combined_score)
            best_frame.is_representative = True
            selected.append(best_frame)
        
        # Sort by timestamp
        selected.sort(key=lambda x: x.timestamp)
        
        return selected
    
    def _uniform_selection(
        self,
        scene_frames: List[FrameCandidate],
        n_select: int,
        force_include: Optional[set] = None
    ) -> List[FrameCandidate]:
        """
        Uniformly select frames across the scene.
        
        Now improved to prefer higher-importance frames at each position.
        """
        force_include = force_include or set()
        
        # Handle forced frames
        forced_frames = [f for f in scene_frames if f.timestamp in force_include]
        regular_frames = [f for f in scene_frames if f.timestamp not in force_include]
        
        for f in forced_frames:
            f.is_representative = True
        
        remaining_slots = n_select - len(forced_frames)
        
        if remaining_slots <= 0 or not regular_frames:
            return sorted(forced_frames, key=lambda x: x.timestamp)
        
        n_select = min(remaining_slots, len(regular_frames))
        
        # Sort by timestamp
        sorted_frames = sorted(regular_frames, key=lambda x: x.timestamp)
        
        # Divide into segments and pick best from each
        segment_size = len(sorted_frames) / n_select
        
        selected = list(forced_frames)
        
        for i in range(n_select):
            start_idx = int(i * segment_size)
            end_idx = int((i + 1) * segment_size)
            segment = sorted_frames[start_idx:end_idx]
            
            if segment:
                # Pick highest importance from segment
                best = max(segment, key=lambda f: f.importance_score)
                best.is_representative = True
                selected.append(best)
        
        selected.sort(key=lambda x: x.timestamp)
        return selected
    
    def _calculate_adaptive_density(
        self, 
        scene_cands: List[FrameCandidate], 
        base_density: float
    ) -> float:
        """
        Adjust density based on scene complexity and importance distribution.
        
        High variance scenes (lots of changes) get more frames.
        Scenes with high-importance frames also get more coverage.
        """
        if len(scene_cands) < 2:
            return base_density
        
        # Compute frame variance (how much changes within scene)
        variance = self._compute_frame_variance([c.frame for c in scene_cands])
        
        # Compute importance variance (scenes with high-importance frames deserve more coverage)
        importance_scores = [c.importance_score for c in scene_cands]
        max_importance = max(importance_scores)
        avg_importance = np.mean(importance_scores)
        
        # Base adjustment from visual variance
        if variance > 0.15:  # High complexity
            adjusted_density = base_density * 1.3
        elif variance < 0.05:  # Low complexity
            adjusted_density = base_density * 0.7
        else:
            adjusted_density = base_density
        
        # Boost for high-importance scenes
        if max_importance > 1.5:  # Contains important frames
            adjusted_density *= 1.2
        
        logger.debug(f"Adaptive density: variance={variance:.3f}, "
                    f"max_importance={max_importance:.2f}, "
                    f"density={adjusted_density:.3f}")
        
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
            
            # Handle potential shape mismatch
            if gray1.shape != gray2.shape:
                continue
            
            diff = np.abs(gray1.astype(float) - gray2.astype(float))
            variances.append(np.mean(diff) / 255.0)
        
        return np.mean(variances) if variances else 0.0
    
    def _ensure_endpoints(
        self,
        selected: List[FrameCandidate],
        first_frame: FrameCandidate,
        last_frame: FrameCandidate
    ) -> List[FrameCandidate]:
        """
        Ensure first and last frames are in the selection.
        
        This is a safety net in case they were suppressed.
        """
        timestamps = {f.timestamp for f in selected}
        
        result = list(selected)
        
        if first_frame.timestamp not in timestamps:
            first_frame.is_representative = True
            result.insert(0, first_frame)
            logger.debug(f"Re-added first frame at {first_frame.timestamp:.2f}s")
        
        if last_frame.timestamp not in timestamps:
            last_frame.is_representative = True
            result.append(last_frame)
            logger.debug(f"Re-added last frame at {last_frame.timestamp:.2f}s")
        
        return result


# Backward compatibility alias
def create_temporal_clusterer(
    target_frame_density: float = 0.25,
    min_frames_per_scene: int = 2,
    max_frames_per_scene: int = 10,
    min_temporal_gap_s: float = 0.5,
    clustering_method: str = "nms",
    adaptive_density: bool = True,
    **kwargs
) -> TemporalClusterer:
    """
    Factory function to create TemporalClusterer with configuration.
    
    Args:
        target_frame_density: Frames per second target
        min_frames_per_scene: Minimum frames to select per scene
        max_frames_per_scene: Maximum frames to select per scene
        min_temporal_gap_s: Minimum time between selected frames
        clustering_method: "nms", "kmeans", "uniform", or "hybrid"
        adaptive_density: Whether to adjust density based on scene complexity
        **kwargs: Additional arguments passed to TemporalClusterer
        
    Returns:
        Configured TemporalClusterer instance
    """
    return TemporalClusterer(
        target_frame_density=target_frame_density,
        min_frames_per_scene=min_frames_per_scene,
        max_frames_per_scene=max_frames_per_scene,
        min_temporal_gap_s=min_temporal_gap_s,
        clustering_method=clustering_method,
        adaptive_density=adaptive_density,
        **kwargs
    )