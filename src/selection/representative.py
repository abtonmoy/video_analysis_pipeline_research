"""
Representative frame selection with importance scoring.
"""

import logging
from typing import List, Optional, Dict, Tuple
import numpy as np

from .clustering import FrameCandidate, TemporalClusterer

logger = logging.getLogger(__name__)


class ImportanceScorer:
    """
    Score frame importance based on various signals.
    """
    
    def __init__(self):
        pass
    
    def score_by_position(
        self,
        timestamp: float,
        duration: float
    ) -> float:
        """
        Score based on position in video.
        Opening and closing frames are more important.
        """
        position = timestamp / duration if duration > 0 else 0
        
        # Boost opening (first 10%)
        if position < 0.1:
            return 1.5
        
        # Boost closing (last 10%)
        if position > 0.9:
            return 1.3
        
        return 1.0
    
    def score_by_audio_events(
        self,
        timestamp: float,
        audio_events: Dict,
        proximity_threshold_s: float = 0.5
    ) -> float:
        """
        Score based on proximity to audio events.
        """
        score = 1.0
        
        # Check proximity to energy peaks
        for peak_ts in audio_events.get("energy_peaks", []):
            if abs(timestamp - peak_ts) < proximity_threshold_s:
                score *= 1.3
                break
        
        # Check if after silence (attention reset point)
        for start, end in audio_events.get("silence_segments", []):
            if end <= timestamp < end + proximity_threshold_s:
                score *= 1.4
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
        First and last frames of scenes are important.
        """
        scene_duration = scene_end - scene_start
        if scene_duration <= 0:
            return 1.0
        
        position_in_scene = (timestamp - scene_start) / scene_duration
        
        # Boost scene start
        if position_in_scene < 0.15:
            return 1.4
        
        # Boost scene end
        if position_in_scene > 0.85:
            return 1.2
        
        return 1.0
    
    def compute_importance(
        self,
        frame: FrameCandidate,
        video_duration: float,
        scene_boundaries: Optional[List[Tuple[float, float]]] = None,
        audio_events: Optional[Dict] = None
    ) -> float:
        """
        Compute overall importance score for a frame.
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
        
        return score


class FrameSelector:
    """
    Main frame selection class that combines clustering and importance scoring.
    """
    
    def __init__(
        self,
        max_frames_per_scene: int = 3,
        min_temporal_gap_s: float = 0.5,
        clustering_method: str = "kmeans",
        use_importance_scoring: bool = True
    ):
        self.clusterer = TemporalClusterer(
            max_frames_per_scene=max_frames_per_scene,
            min_temporal_gap_s=min_temporal_gap_s,
            clustering_method=clustering_method
        )
        self.scorer = ImportanceScorer() if use_importance_scoring else None
    
    def select(
        self,
        frames: List[Tuple[float, np.ndarray]],
        embeddings: Optional[np.ndarray],
        scene_boundaries: List[Tuple[float, float]],
        video_duration: float,
        audio_events: Optional[Dict] = None
    ) -> List[FrameCandidate]:
        """
        Select representative frames from candidates.
        
        Args:
            frames: List of (timestamp, frame) tuples
            embeddings: Optional CLIP embeddings
            scene_boundaries: List of (start, end) tuples
            video_duration: Total video duration in seconds
            audio_events: Optional audio event dict
            
        Returns:
            List of selected FrameCandidate
        """
        # Assign scenes
        candidates = self.clusterer.assign_scenes(frames, scene_boundaries)
        
        # Score importance
        if self.scorer:
            for cand in candidates:
                cand.importance_score = self.scorer.compute_importance(
                    cand,
                    video_duration,
                    scene_boundaries,
                    audio_events
                )
        
        # Cluster and select
        selected = self.clusterer.cluster_and_select(candidates, embeddings)
        
        return selected


def create_selector(config: Dict) -> FrameSelector:
    """Create FrameSelector from config dict."""
    selection_config = config.get("selection", {})
    
    return FrameSelector(
        max_frames_per_scene=selection_config.get("max_frames_per_scene", 3),
        min_temporal_gap_s=selection_config.get("min_temporal_gap_s", 0.5),
        clustering_method=selection_config.get("method", "clustering"),
        use_importance_scoring=True
    )