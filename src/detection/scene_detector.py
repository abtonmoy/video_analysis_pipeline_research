
import logging
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
import numpy as np
import cv2
from .change_detector import ChangeDetector, AdaptiveChangeDetector, FrameDifferenceDetector

logger = logging.getLogger(__name__)

class SceneDetector:
    """
    Wrapper for scene detection using PySceneDetect.
    """
    
    def __init__(
        self,
        method: str = "content",
        threshold: float = 27.0,
        min_scene_length_s: float = 0.5
    ):
        self.method = method
        self.threshold = threshold
        self.min_scene_length_s = min_scene_length_s
    
    def detect_scenes(self, video_path: str) -> List[Tuple[float, float]]:
        """
        Detect scene boundaries in video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        try:
            from scenedetect import detect, ContentDetector, ThresholdDetector
            from scenedetect.scene_manager import save_images
        except ImportError:
            logger.warning("scenedetect not installed, using fallback")
            return self._fallback_detection(video_path)
        
        # Select detector
        if self.method == "content":
            detector = ContentDetector(threshold=self.threshold)
        elif self.method == "threshold":
            detector = ThresholdDetector(threshold=self.threshold)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Detect scenes
        try:
            scene_list = detect(video_path, detector)
            
            # Convert to (start, end) tuples in seconds
            scenes = []
            for scene in scene_list:
                start = scene[0].get_seconds()
                end = scene[1].get_seconds()
                
                # Filter by minimum length
                if end - start >= self.min_scene_length_s:
                    scenes.append((start, end))
            
            logger.info(f"Detected {len(scenes)} scenes in video")
            return scenes
            
        except Exception as e:
            logger.warning(f"Scene detection failed: {e}, using fallback")
            return self._fallback_detection(video_path)
    
    def _fallback_detection(self, video_path: str) -> List[Tuple[float, float]]:
        """
        Fallback scene detection using simple frame difference.
        """
        from ..utils import get_video_metadata, VideoFrameIterator
        
        metadata = get_video_metadata(video_path)
        detector = FrameDifferenceDetector()
        
        scene_boundaries = [0.0]
        previous_frame = None
        
        with VideoFrameIterator(video_path, interval_ms=200) as frame_iter:
            for timestamp, frame in frame_iter:
                if previous_frame is not None:
                    change = detector.compute_change(previous_frame, frame)
                    if change > 0.3:  # Hard threshold for fallback
                        if timestamp - scene_boundaries[-1] >= self.min_scene_length_s:
                            scene_boundaries.append(timestamp)
                
                previous_frame = frame
        
        scene_boundaries.append(metadata.duration)
        
        # Convert to (start, end) tuples
        scenes = []
        for i in range(len(scene_boundaries) - 1):
            scenes.append((scene_boundaries[i], scene_boundaries[i + 1]))
        
        return scenes


class CandidateFrameExtractor:
    """
    Extract candidate frames for further processing based on change detection.
    """
    
    def __init__(
        self,
        change_detector: ChangeDetector,
        threshold: float = 0.15,
        min_interval_ms: float = 100,
        sample_interval_ms: float = 50
    ):
        self.change_detector = change_detector
        self.threshold = threshold
        self.min_interval_ms = min_interval_ms
        self.sample_interval_ms = sample_interval_ms
    
    def extract_candidates(
        self,
        video_path: str,
        max_resolution: int = 720
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Extract candidate frames where significant change occurs.
        
        Args:
            video_path: Path to video file
            max_resolution: Maximum frame height
            
        Returns:
            List of (timestamp, frame) tuples
        """
        from ..utils.video_utils import VideoFrameIterator
        
        candidates = []
        previous_frame = None
        last_candidate_ms = -self.min_interval_ms
        
        with VideoFrameIterator(
            video_path,
            interval_ms=self.sample_interval_ms,
            max_resolution=max_resolution
        ) as frame_iter:
            
            for timestamp, frame in frame_iter:
                current_ms = timestamp * 1000
                
                # Always include first frame
                if previous_frame is None:
                    candidates.append((timestamp, frame.copy()))
                    last_candidate_ms = current_ms
                    previous_frame = frame
                    continue
                
                # Check if enough time has passed
                if current_ms - last_candidate_ms < self.min_interval_ms:
                    previous_frame = frame
                    continue
                
                # Check for significant change
                change = self.change_detector.compute_change(previous_frame, frame)
                
                if change > self.threshold:
                    candidates.append((timestamp, frame.copy()))
                    last_candidate_ms = current_ms
                
                previous_frame = frame
        
        logger.info(f"Extracted {len(candidates)} candidate frames")
        return candidates
