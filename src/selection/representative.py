# src/selection/representative.py
"""
Representative frame selection with importance scoring.

Integrates with clustering.py to provide importance-aware frame selection.
The key improvement is that NMS-based selection now directly uses importance
scores, unlike K-means which ignored them.
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
    
    The key improvement over the original K-means approach is that NMS directly
    uses importance scores to select frames, ensuring high-value moments
    (CTAs, brand reveals, key phrases) are prioritized.
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
        # Importance scorer options
        position_weight: float = 1.0,
        scene_weight: float = 1.0,
        audio_weight: float = 1.0,
        key_phrase_boost: float = 1.5
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
            semantic_threshold: Cosine similarity threshold for NMS suppression
            use_semantic_suppression: Use embedding similarity in NMS
            diversity_bonus: Bonus for semantically diverse frames in NMS
            position_weight: Weight for video position in importance
            scene_weight: Weight for scene position in importance
            audio_weight: Weight for audio events in importance
            key_phrase_boost: Multiplier for frames near key phrases
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
            diversity_bonus=diversity_bonus
        )
        
        self.scorer = ImportanceScorer(
            position_weight=position_weight,
            scene_weight=scene_weight,
            audio_weight=audio_weight,
            key_phrase_boost=key_phrase_boost
        ) if use_importance_scoring else None
        
        self.use_importance_scoring = use_importance_scoring
    
    def select(
        self,
        frames: List[Tuple[float, np.ndarray]],
        embeddings: Optional[np.ndarray],
        scene_boundaries: List[Tuple[float, float]],
        video_duration: float,
        audio_events: Optional[Dict] = None,
        visual_features: Optional[Dict] = None
    ) -> List[FrameCandidate]:
        """
        Select representative frames from candidates.
        
        Args:
            frames: List of (timestamp, frame) tuples
            embeddings: Optional CLIP embeddings for semantic suppression
            scene_boundaries: List of (start, end) tuples defining scenes
            video_duration: Total video duration in seconds
            audio_events: Optional dict with audio event information:
                - energy_peaks: List of timestamps
                - silence_segments: List of (start, end) tuples
                - speech_segments: List of (start, end) tuples
                - key_phrases: List of dicts with 'timestamp' and 'text'
            visual_features: Optional dict with visual feature info
            
        Returns:
            List of selected FrameCandidate, sorted by timestamp
        """
        if not frames:
            return []
        
        # Assign scenes to frames
        candidates = self.clusterer.assign_scenes(frames, scene_boundaries)
        
        # Compute importance scores
        if self.scorer:
            for cand in candidates:
                cand.importance_score = self.scorer.compute_importance(
                    cand,
                    video_duration,
                    scene_boundaries,
                    audio_events,
                    visual_features
                )
            
            # Log importance distribution
            scores = [c.importance_score for c in candidates]
            logger.debug(f"Importance scores: min={min(scores):.2f}, "
                        f"max={max(scores):.2f}, mean={np.mean(scores):.2f}")
        
        # Cluster/NMS and select
        selected = self.clusterer.cluster_and_select(candidates, embeddings)
        
        # Log selection stats
        if selected:
            selected_scores = [c.importance_score for c in selected]
            logger.info(f"Selected {len(selected)} frames: "
                       f"importance range [{min(selected_scores):.2f}, {max(selected_scores):.2f}]")
        
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
    
    Expected config structure:
    ```yaml
    selection:
      method: "nms"  # or "kmeans", "uniform", "hybrid"
      target_frame_density: 0.25
      min_frames_per_scene: 2
      max_frames_per_scene: 10
      min_temporal_gap_s: 0.5
      adaptive_density: true
      
      # NMS-specific (optional)
      nms:
        semantic_threshold: 0.88
        use_semantic_suppression: true
        diversity_bonus: 0.1
      
      # Importance scoring (optional)
      importance:
        enabled: true
        position_weight: 1.0
        scene_weight: 1.0
        audio_weight: 1.0
        key_phrase_boost: 1.5
    ```
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured FrameSelector instance
    """
    selection_config = config.get("selection", {})
    nms_config = selection_config.get("nms", {})
    importance_config = selection_config.get("importance", {})
    
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
        # Importance options
        position_weight=importance_config.get("position_weight", 1.0),
        scene_weight=importance_config.get("scene_weight", 1.0),
        audio_weight=importance_config.get("audio_weight", 1.0),
        key_phrase_boost=importance_config.get("key_phrase_boost", 1.5)
    )