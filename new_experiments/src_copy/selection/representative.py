# src/selection/representative.py
"""
Representative frame selection with importance scoring and temporal-aware NMS.

Features:
- Importance scoring (position, scene, audio, visual)
- Visual feature detection (text, faces) integrated into scoring
- Smart global frame budget with diminishing returns
- Temporal-aware NMS threshold configuration
"""

import logging
from typing import List, Optional, Dict, Tuple
import numpy as np

from .clustering import FrameCandidate, TemporalClusterer, SelectionMethod

logger = logging.getLogger(__name__)


class ImportanceScorer:
    """
    Score frame importance based on various signals.
    
    Importance scores are used by NMS to prioritize frames near key moments:
    - Video opening/closing (brand reveal, CTA)
    - Scene boundaries (transition points)
    - Audio events (speech, music changes)
    - Key phrase occurrences
    """
    
    def __init__(
        self,
        position_weight: float = 1.0,
        scene_weight: float = 1.0,
        audio_weight: float = 1.0,
        key_phrase_boost: float = 1.5
    ):
        """
        Args:
            position_weight: Weight for video position scoring
            scene_weight: Weight for scene position scoring
            audio_weight: Weight for audio event scoring
            key_phrase_boost: Multiplier for frames near key phrases
        """
        self.position_weight = position_weight
        self.scene_weight = scene_weight
        self.audio_weight = audio_weight
        self.key_phrase_boost = key_phrase_boost
    
    def score_by_position(
        self,
        timestamp: float,
        duration: float
    ) -> float:
        """
        Score based on position in video.
        Opening and closing frames are more important for ad extraction.
        
        Returns:
            Score multiplier (1.0 = neutral, >1.0 = important)
        """
        if duration <= 0:
            return 1.0
        
        position = timestamp / duration
        
        # Boost opening (first 10%) - brand introduction
        if position < 0.1:
            return 1.5 * self.position_weight
        
        # Boost closing (last 10%) - CTA typically appears here
        if position > 0.9:
            return 1.4 * self.position_weight
        
        # Slight boost for middle (core message)
        if 0.4 < position < 0.6:
            return 1.1 * self.position_weight
        
        return 1.0
    
    def score_by_audio_events(
        self,
        timestamp: float,
        audio_events: Dict,
        proximity_threshold_s: float = 0.5
    ) -> float:
        """
        Score based on proximity to audio events.
        
        Audio events that increase importance:
        - Energy peaks (attention-grabbing moments)
        - After silence (attention reset points)
        - Speech segments (important content)
        - Key phrases (promotional keywords)
        
        Returns:
            Score multiplier (1.0 = neutral, >1.0 = important)
        """
        score = 1.0
        
        # Check proximity to energy peaks
        energy_peaks = audio_events.get("energy_peaks", [])
        for peak_ts in energy_peaks:
            if abs(timestamp - peak_ts) < proximity_threshold_s:
                score *= 1.3 * self.audio_weight
                break
        
        # Check if after silence (attention reset point)
        silence_segments = audio_events.get("silence_segments", [])
        for start, end in silence_segments:
            if end <= timestamp < end + proximity_threshold_s:
                score *= 1.4 * self.audio_weight
                break
        
        # Check proximity to speech segments
        speech_segments = audio_events.get("speech_segments", [])
        for seg_start, seg_end in speech_segments:
            # Boost frames at speech start (important intro)
            if abs(timestamp - seg_start) < proximity_threshold_s:
                score *= 1.2 * self.audio_weight
                break
            # Boost frames at speech end (important conclusion)
            if abs(timestamp - seg_end) < proximity_threshold_s:
                score *= 1.15 * self.audio_weight
                break
        
        # Check proximity to key phrases (strongest signal)
        key_phrases = audio_events.get("key_phrases", [])
        for phrase_info in key_phrases:
            phrase_ts = phrase_info.get("timestamp", phrase_info.get("start", 0))
            if abs(timestamp - phrase_ts) < proximity_threshold_s:
                score *= self.key_phrase_boost
                logger.debug(f"Frame at {timestamp:.2f}s boosted for key phrase: "
                           f"{phrase_info.get('text', 'unknown')}")
                break
        
        return score
    
    def score_by_scene_position(
        self,
        timestamp: float,
        scene_start: float,
        scene_end: float
    ) -> float:
        """
        Score based on position within scene.
        First and last frames of scenes capture transitions.
        
        Returns:
            Score multiplier (1.0 = neutral, >1.0 = important)
        """
        scene_duration = scene_end - scene_start
        if scene_duration <= 0:
            return 1.0
        
        position_in_scene = (timestamp - scene_start) / scene_duration
        
        # Boost scene start (new content introduction)
        if position_in_scene < 0.15:
            return 1.4 * self.scene_weight
        
        # Boost scene end (transition point)
        if position_in_scene > 0.85:
            return 1.2 * self.scene_weight
        
        return 1.0
    
    def score_by_visual_features(
        self,
        frame: FrameCandidate,
        visual_features: Optional[Dict] = None
    ) -> float:
        """
        Score based on visual features of the frame.
        
        Features that increase importance:
        - Text presence (overlays, CTAs)
        - Face detection (testimonials, presenters)
        - Logo detection (brand moments)
        - High contrast/saturation (attention-grabbing)
        
        Returns:
            Score multiplier (1.0 = neutral, >1.0 = important)
        """
        if not visual_features:
            return 1.0
        
        score = 1.0
        
        # Text presence (detected via OCR or classifier)
        if visual_features.get("has_text", False):
            score *= 1.3
        
        # Face presence
        if visual_features.get("has_face", False):
            score *= 1.2
        
        # Logo presence
        if visual_features.get("has_logo", False):
            score *= 1.4
        
        return score
    
    def compute_importance(
        self,
        frame: FrameCandidate,
        video_duration: float,
        scene_boundaries: Optional[List[Tuple[float, float]]] = None,
        audio_events: Optional[Dict] = None,
        visual_features: Optional[Dict] = None
    ) -> float:
        """
        Compute overall importance score for a frame.
        
        Combines multiple signals multiplicatively so high-importance
        frames score significantly higher than average frames.
        
        Args:
            frame: FrameCandidate to score
            video_duration: Total video duration in seconds
            scene_boundaries: List of (start, end) tuples for scenes
            audio_events: Dict with audio event information
            visual_features: Dict with visual feature detections
            
        Returns:
            Importance score (baseline ~1.0, high importance >2.0)
        """
        score = 1.0
        
        # Position in video
        score *= self.score_by_position(frame.timestamp, video_duration)
        
        # Position in scene
        if scene_boundaries and frame.scene_id is not None:
            if 0 <= frame.scene_id < len(scene_boundaries):
                start, end = scene_boundaries[frame.scene_id]
                score *= self.score_by_scene_position(frame.timestamp, start, end)
        
        # Audio events
        if audio_events:
            score *= self.score_by_audio_events(frame.timestamp, audio_events)
        
        # Visual features
        if visual_features:
            score *= self.score_by_visual_features(frame, visual_features)
        
        return score


class FrameSelector:
    """
    Main frame selection class that combines clustering/NMS and importance scoring.
    
    ENHANCED: Now supports temporal-aware NMS thresholds that adapt based on
    the time distance between frames, preventing over-suppression of important
    repeated content like brand logos and CTAs at video start/end.
    """
    
    def __init__(
        self,
        target_frame_density: float = 0.25,
        min_frames_per_scene: int = 2,
        max_frames_per_scene: int = 10,
        min_temporal_gap_s: float = 0.5,
        clustering_method: str = "nms",
        adaptive_density: bool = True,
        use_importance_scoring: bool = True,
        # NMS-specific options
        semantic_threshold: float = 0.88,
        use_semantic_suppression: bool = True,
        diversity_bonus: float = 0.1,
        # Temporal-aware threshold options
        use_temporal_aware_threshold: bool = True,
        temporal_threshold_scaling: float = 0.3,
        temporal_decay_rate: float = 5.0,
        # Importance scorer options
        position_weight: float = 1.0,
        scene_weight: float = 1.0,
        audio_weight: float = 1.0,
        key_phrase_boost: float = 1.5,
        # Smart frame budget
        global_max_frames: int = 25,
        # Visual feature detection
        use_visual_features: bool = True,
        # Novel Hamiltonian Information Budget
        use_hib_budget: bool = True,
    ):
        """
        Args:
            target_frame_density: Target frames per second
            min_frames_per_scene: Minimum frames to keep per scene
            max_frames_per_scene: Maximum frames to keep per scene
            min_temporal_gap_s: Minimum time between selected frames
            clustering_method: "nms" (recommended), "kmeans", "uniform", or "hybrid"
            adaptive_density: Adjust density based on scene complexity
            use_importance_scoring: Whether to compute importance scores
            semantic_threshold: Base cosine similarity threshold for NMS suppression
            use_semantic_suppression: Use embedding similarity in NMS
            diversity_bonus: Bonus for semantically diverse frames in NMS
            use_temporal_aware_threshold: Enable temporal-aware threshold adaptation
            temporal_threshold_scaling: How much to relax threshold (0.0-1.0)
            temporal_decay_rate: Time constant for exponential decay (seconds)
            position_weight: Weight for video position in importance
            scene_weight: Weight for scene position in importance
            audio_weight: Weight for audio events in importance
            key_phrase_boost: Multiplier for frames near key phrases
            global_max_frames: Maximum total frames across all scenes (smart budget)
            use_visual_features: Whether to detect text/faces for scoring
        """
        self.clusterer = TemporalClusterer(
            target_frame_density=target_frame_density,
            min_frames_per_scene=min_frames_per_scene,
            max_frames_per_scene=max_frames_per_scene,
            min_temporal_gap_s=min_temporal_gap_s,
            clustering_method=clustering_method,
            adaptive_density=adaptive_density,
            semantic_threshold=semantic_threshold,
            use_semantic_suppression=use_semantic_suppression,
            diversity_bonus=diversity_bonus,
            use_temporal_aware_threshold=use_temporal_aware_threshold,
            temporal_threshold_scaling=temporal_threshold_scaling,
            temporal_decay_rate=temporal_decay_rate,
        )

        self.scorer = ImportanceScorer(
            position_weight=position_weight,
            scene_weight=scene_weight,
            audio_weight=audio_weight,
            key_phrase_boost=key_phrase_boost,
        ) if use_importance_scoring else None

        self.use_importance_scoring = use_importance_scoring
        self.global_max_frames = global_max_frames
        self.use_visual_features = use_visual_features
        self.use_hib_budget = use_hib_budget
        self._visual_detector = None

    def _get_visual_detector(self):
        """Lazy-load visual feature detector."""
        if self._visual_detector is None:
            try:
                from ..detection.visual_features import VisualFeatureDetector
                self._visual_detector = VisualFeatureDetector()
            except Exception as e:
                logger.warning(f"Visual feature detector unavailable: {e}")
                self.use_visual_features = False
        return self._visual_detector
    
    def _compute_frame_budget(
        self, 
        video_duration: float, 
        candidates: Optional[List[FrameCandidate]] = None,
        scene_boundaries: Optional[List[Tuple[float, float]]] = None,
        density: float = 0.25
    ) -> int:
        """
        Compute smart frame budget based on Hamiltonian Information Budget (HIB).

        Uses Information Entropy (Semantic Velocity + Attention Yield) to 
        dynamically allocate the token budget.

        Args:
            video_duration: Video duration in seconds
            candidates: Optional list of FrameCandidates with embeddings and importance scores
            scene_boundaries: Optional list of scene boundaries
            density: Base frames per second

        Returns:
            Maximum total frames to select
        """
        num_scenes = len(scene_boundaries) if scene_boundaries else 1
        
        # 1. Base temporal requirement (The "Coverage" floor)
        # We need a minimum amount of frames just to prove time passed, scaling with scenes.
        base_budget = max(5, num_scenes + 1)
        
        # 2. Potential Energy (E_p) = Attention Yield
        # Average importance score across all candidates
        if candidates:
            scores = [c.importance_score for c in candidates]
            e_p = sum(scores) / len(scores) if scores else 1.0
        else:
            e_p = 1.0

        # 3. Kinetic Energy (E_k) = Semantic Velocity & ISD
        # Variance/Derivative of embeddings over time
        e_k = 1.0
        isd = 1  # Intrinsic Semantic Dimensionality
        if candidates:
            embeddings_list = [c.embedding for c in candidates if c.embedding is not None]
            if len(embeddings_list) > 1:
                stacked_emb = np.vstack(embeddings_list)
                # Mean L2 distance between consecutive temporal frames
                diffs = np.linalg.norm(np.diff(stacked_emb, axis=0), axis=1)
                mean_diff = float(np.mean(diffs)) if len(diffs) > 0 else 0.5
                
                # For normalized CLIP embeddings, diffs are typically between 0.1 and 1.0
                # We scale this so average kinetic energy is roughly 1.0
                e_k = max(0.2, mean_diff * 2.5)
                
                # Calculate Intrinsic Semantic Dimensionality (ISD) via SVD
                if len(embeddings_list) > 2:
                    # Center the embeddings for PCA
                    centered_emb = stacked_emb - np.mean(stacked_emb, axis=0)
                    try:
                        # Full matrices=False for efficiency
                        _, s, _ = np.linalg.svd(centered_emb, full_matrices=False)
                        # Calculate explained variance ratio
                        var_explained = (s ** 2) / (np.sum(s ** 2) + 1e-9)
                        cum_var = np.cumsum(var_explained)
                        # ISD is the number of components needed to explain 90% of variance
                        isd = int(np.argmax(cum_var >= 0.90)) + 1
                    except np.linalg.LinAlgError:
                        logger.warning("SVD failed to converge, falling back to heuristic ISD.")
                        isd = max(1, len(embeddings_list) // 5)

        # 4. Total Information Density Multiplier
        # H(t) = a*E_k + b*E_p (represented multiplicatively for budget scaling)
        info_multiplier = (e_k * e_p)
        
        # 5. Dynamic Calculation
        raw_adaptive = base_budget + (video_duration * density * info_multiplier)
        
        # Switch between HIB and the legacy static budget
        if not self.use_hib_budget:
            raw_legacy = video_duration * density
            # Retain global_max_frames as a hard ceiling for legacy mode only
            budget = min(max(5, int(raw_legacy)), self.global_max_frames)
            logger.info(f"Using legacy static budget: {budget}")
            return budget
            
        # 6. Apply ISD bounding (No absolute hard-coded ceiling)
        # We ensure a minimum of base_budget (usually 5) to guarantee basic coverage
        dynamic_max = int(max(5, base_budget) + (isd * 1.5))
        budget = min(dynamic_max, max(base_budget, int(raw_adaptive)))
        
        logger.info(
            f"ISD Adaptive Budget | Scenes: {num_scenes} | "
            f"ISD: {isd} -> Max Cap: {dynamic_max} | "
            f"E_k (Velocity): {e_k:.2f} | E_p (Yield): {e_p:.2f} | "
            f"Raw Adaptive: {int(raw_adaptive)} -> Final Budget: {budget}"
        )
        
        return budget

    def select(
        self,
        frames: List[Tuple[float, np.ndarray]],
        embeddings: Optional[np.ndarray],
        scene_boundaries: List[Tuple[float, float]],
        video_duration: float,
        audio_events: Optional[Dict] = None,
        visual_features: Optional[Dict] = None,
    ) -> List[FrameCandidate]:
        """
        Select representative frames from candidates.

        Now with:
        - Automatic visual feature detection (text, faces)
        - Smart global frame budget with diminishing returns

        Args:
            frames: List of (timestamp, frame) tuples
            embeddings: Optional CLIP embeddings for semantic suppression
            scene_boundaries: List of (start, end) tuples defining scenes
            video_duration: Total video duration in seconds
            audio_events: Optional dict with audio event information
            visual_features: Optional per-frame visual feature dict

        Returns:
            List of selected FrameCandidate, sorted by timestamp
        """
        if not frames:
            return []

        # Assign scenes to frames
        candidates = self.clusterer.assign_scenes(frames, scene_boundaries)

        # Auto-detect visual features if enabled and not provided
        if self.use_visual_features and visual_features is None:
            detector = self._get_visual_detector()
            if detector:
                logger.info("Detecting visual features (text, faces) for importance scoring...")
                vf_results = detector.detect_batch(frames)
                # Create per-candidate visual features lookup
                visual_features_by_ts = vf_results
            else:
                visual_features_by_ts = None
        else:
            visual_features_by_ts = visual_features

        # Compute importance scores
        if self.scorer:
            for cand in candidates:
                cand_vf = None
                if visual_features_by_ts and isinstance(visual_features_by_ts, dict):
                    cand_vf = visual_features_by_ts.get(cand.timestamp)

                cand.importance_score = self.scorer.compute_importance(
                    cand,
                    video_duration,
                    scene_boundaries,
                    audio_events,
                    cand_vf,
                )

            # Log importance distribution
            scores = [c.importance_score for c in candidates]
            logger.debug(
                f"Importance scores: min={min(scores):.2f}, "
                f"max={max(scores):.2f}, mean={np.mean(scores):.2f}"
            )

        # Cluster/NMS and select
        selected = self.clusterer.cluster_and_select(candidates, embeddings)

        # Apply global frame budget — keep top-scoring frames
        budget = self._compute_frame_budget(
            video_duration=video_duration,
            candidates=candidates,
            scene_boundaries=scene_boundaries,
            density=self.clusterer.target_density
        )
        if len(selected) > budget:
            logger.info(
                f"Applying global frame budget: {len(selected)} -> {budget} frames"
            )
            selected.sort(key=lambda c: c.importance_score, reverse=True)
            selected = selected[:budget]
            selected.sort(key=lambda c: c.timestamp)

        # Log selection stats
        if selected:
            selected_scores = [c.importance_score for c in selected]
            logger.info(
                f"Selected {len(selected)} frames: "
                f"importance range [{min(selected_scores):.2f}, {max(selected_scores):.2f}]"
            )

        return selected
    
    def get_selection_stats(
        self,
        candidates: List[FrameCandidate],
        selected: List[FrameCandidate]
    ) -> Dict:
        """
        Compute statistics about the selection process.
        
        Useful for debugging and evaluation.
        """
        if not candidates or not selected:
            return {}
        
        candidate_scores = [c.importance_score for c in candidates]
        selected_scores = [c.importance_score for c in selected]
        
        return {
            "total_candidates": len(candidates),
            "selected_count": len(selected),
            "reduction_rate": 1 - (len(selected) / len(candidates)),
            "candidate_importance": {
                "min": min(candidate_scores),
                "max": max(candidate_scores),
                "mean": np.mean(candidate_scores),
                "std": np.std(candidate_scores)
            },
            "selected_importance": {
                "min": min(selected_scores),
                "max": max(selected_scores),
                "mean": np.mean(selected_scores),
                "std": np.std(selected_scores)
            },
            "importance_lift": np.mean(selected_scores) / np.mean(candidate_scores)
        }



def create_selector(config: Dict) -> FrameSelector:
    """
    Create FrameSelector from configuration dictionary.
    
    ENHANCED: Now reads temporal-aware threshold configuration from:
    ```yaml
    selection:
      nms:
        temporal_aware:
          enabled: true
          scaling: 0.3
          decay_rate: 5.0
    ```
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured FrameSelector instance
    """
    selection_config = config.get("selection", {})
    nms_config = selection_config.get("nms", {})
    importance_config = selection_config.get("importance", {})
    
    # NEW: Read temporal-aware configuration
    temporal_aware_config = nms_config.get("temporal_aware", {})
    
    # Determine clustering method
    method = selection_config.get("method", selection_config.get("clustering_method", "nms"))
    
    return FrameSelector(
        target_frame_density=selection_config.get("target_frame_density", 0.25),
        min_frames_per_scene=selection_config.get("min_frames_per_scene", 2),
        max_frames_per_scene=selection_config.get("max_frames_per_scene", 10),
        min_temporal_gap_s=selection_config.get("min_temporal_gap_s", 0.5),
        clustering_method=method,
        adaptive_density=selection_config.get("adaptive_density", True),
        use_importance_scoring=importance_config.get("enabled", True),
        # NMS options
        semantic_threshold=nms_config.get("semantic_threshold", 0.88),
        use_semantic_suppression=nms_config.get("use_semantic_suppression", True),
        diversity_bonus=nms_config.get("diversity_bonus", 0.1),
        # Temporal-aware options
        use_temporal_aware_threshold=temporal_aware_config.get("enabled", True),
        temporal_threshold_scaling=temporal_aware_config.get("scaling", 0.3),
        temporal_decay_rate=temporal_aware_config.get("decay_rate", 5.0),
        # Importance options
        position_weight=importance_config.get("position_weight", 1.0),
        scene_weight=importance_config.get("scene_weight", 1.0),
        audio_weight=importance_config.get("audio_weight", 1.0),
        key_phrase_boost=importance_config.get("key_phrase_boost", 1.5),
        # Smart frame budget
        global_max_frames=selection_config.get("global_max_frames", 25),
        use_hib_budget=selection_config.get("use_hib_budget", True),
        # Visual features
        use_visual_features=selection_config.get("use_visual_features", True),
    )