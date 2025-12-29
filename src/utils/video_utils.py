# src\utils\video_utils.py
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

@dataclass
class VideoMetadata:
    """Container for video metadata."""
    path: str
    duration: float  # seconds
    fps: float
    frame_count: int
    width: int
    height: int
    codec: str = ""
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0


def get_video_metadata(video_path: str) -> VideoMetadata:
    """
    Extract metadata from video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        VideoMetadata object
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
    
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return VideoMetadata(
        path=video_path,
        duration=duration,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        codec=codec_str
    )


def extract_frame_at_time(video_path: str, timestamp: float) -> np.ndarray:
    """
    Extract a single frame at specified timestamp.
    
    Args:
        video_path: Path to video file
        timestamp: Time in seconds
        
    Returns:
        Frame as numpy array (BGR)
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Cannot extract frame at {timestamp}s from {video_path}")
    
    return frame


def extract_frames_at_times(
    video_path: str, 
    timestamps: List[float],
    max_resolution: Optional[int] = None
) -> List[Tuple[float, np.ndarray]]:
    """
    Extract multiple frames at specified timestamps.
    
    Args:
        video_path: Path to video file
        timestamps: List of times in seconds
        max_resolution: Optional max height to resize to
        
    Returns:
        List of (timestamp, frame) tuples
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frames = []
    sorted_timestamps = sorted(timestamps)
    
    for ts in sorted_timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        
        if ret:
            if max_resolution and frame.shape[0] > max_resolution:
                scale = max_resolution / frame.shape[0]
                new_width = int(frame.shape[1] * scale)
                frame = cv2.resize(frame, (new_width, max_resolution))
            
            frames.append((ts, frame))
        else:
            logger.warning(f"Failed to extract frame at {ts}s")
    
    cap.release()
    return frames


class VideoFrameIterator:
    """Iterator for extracting frames from video at regular intervals."""
    
    def __init__(
        self, 
        video_path: str, 
        interval_ms: float = 100,
        max_resolution: Optional[int] = None
    ):
        self.video_path = video_path
        self.interval_ms = interval_ms
        self.max_resolution = max_resolution
        self.cap = None
        self.metadata = get_video_metadata(video_path)
        
    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
    
    def __iter__(self):
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("VideoFrameIterator must be used as context manager")
        
        current_ms = 0
        duration_ms = self.metadata.duration * 1000
        
        while current_ms < duration_ms:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, current_ms)
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            if self.max_resolution and frame.shape[0] > self.max_resolution:
                scale = self.max_resolution / frame.shape[0]
                new_width = int(frame.shape[1] * scale)
                frame = cv2.resize(frame, (new_width, self.max_resolution))
            
            yield current_ms / 1000, frame
            current_ms += self.interval_ms

