# src\utils\metrics.py
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
import cv2
import numpy as np
from PIL import Image

from .video_utils import VideoMetadata

logger = logging.getLogger(__name__)
@dataclass
class FrameInfo:
    """Information about a single extracted frame."""
    timestamp: float
    frame: Optional[np.ndarray] = None
    phash: Optional[str] = None
    clip_embedding: Optional[np.ndarray] = None
    scene_id: Optional[int] = None
    importance_score: float = 1.0
    
    # Metadata
    is_scene_start: bool = False
    is_scene_end: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding large arrays)."""
        return {
            "timestamp": self.timestamp,
            "scene_id": self.scene_id,
            "importance_score": self.importance_score,
            "is_scene_start": self.is_scene_start,
            "is_scene_end": self.is_scene_end,
            "has_frame": self.frame is not None,
            "has_embedding": self.clip_embedding is not None
        }


@dataclass
class SceneInfo:
    """Information about a detected scene."""
    scene_id: int
    start_time: float
    end_time: float
    frame_count: int = 0
    representative_frames: List[FrameInfo] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass 
class PipelineResult:
    """Result container for pipeline processing."""
    video_path: str
    metadata: VideoMetadata
    scenes: List[SceneInfo]
    selected_frames: List[FrameInfo]
    extraction_result: Optional[Dict] = None
    
    # Metrics
    total_frames_sampled: int = 0
    frames_after_phash: int = 0
    frames_after_ssim: int = 0
    frames_after_clip: int = 0
    final_frame_count: int = 0
    processing_time_s: float = 0.0
    
    @property
    def reduction_rate(self) -> float:
        """Calculate frame reduction rate."""
        if self.total_frames_sampled == 0:
            return 0.0
        return 1 - (self.final_frame_count / self.total_frames_sampled)
    
    def get_metrics(self) -> Dict:
        """Get summary metrics."""
        return {
            "video_duration_s": self.metadata.duration,
            "total_frames_sampled": self.total_frames_sampled,
            "frames_after_phash": self.frames_after_phash,
            "frames_after_ssim": self.frames_after_ssim,
            "frames_after_clip": self.frames_after_clip,
            "final_frame_count": self.final_frame_count,
            "reduction_rate": self.reduction_rate,
            "num_scenes": len(self.scenes),
            "processing_time_s": self.processing_time_s
        }

