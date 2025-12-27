import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from ..utils.video_utils import VideoMetadata, get_video_metadata, VideoFrameIterator

logger = logging.getLogger(__name__)


class VideoLoader:
    """
    Handles video loading, validation, and preprocessing.
    """
    
    SUPPORTED_FORMATS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
    
    def __init__(
        self,
        max_resolution: int = 720,
        extract_audio: bool = True
    ):
        self.max_resolution = max_resolution
        self.extract_audio = extract_audio
    
    def load(self, video_path: str) -> Tuple[VideoMetadata, Optional[str]]:
        """
        Load video and optionally extract audio.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (VideoMetadata, audio_path or None)
        """
        video_path = str(Path(video_path).resolve())
        
        # Validate file
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        ext = Path(video_path).suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {ext}. Supported: {self.SUPPORTED_FORMATS}")
        
        # Get metadata
        metadata = get_video_metadata(video_path)
        logger.info(f"Loaded video: {metadata.duration:.1f}s, {metadata.width}x{metadata.height}, {metadata.fps:.1f}fps")
        
        # Extract audio if requested
        audio_path = None
        if self.extract_audio:
            audio_path = self._extract_audio(video_path)
        
        return metadata, audio_path
    
    def _extract_audio(self, video_path: str) -> Optional[str]:
        """
        Extract audio track from video using ffmpeg.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file, or None if failed
        """
        # Create outputs/audio directory if it doesn't exist
        audio_dir = Path('outputs/audio')
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output path with same name as video but .wav extension
        video_name = Path(video_path).stem
        output_path = audio_dir / f"{video_name}.wav"
        
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',
                '-ar', '16000',  # 16kHz for speech processing
                '-ac', '1',  # Mono
                str(output_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and output_path.exists():
                logger.info(f"Extracted audio to: {output_path}")
                return str(output_path)
            else:
                logger.warning(f"Audio extraction failed: {result.stderr}")
                return None
                
        except FileNotFoundError:
            logger.warning("ffmpeg not found, skipping audio extraction")
            return None
        except Exception as e:
            logger.warning(f"Audio extraction error: {e}")
            return None
    
    def get_frame_iterator(
        self,
        video_path: str,
        interval_ms: float = 100
    ) -> VideoFrameIterator:
        """
        Get an iterator for extracting frames at regular intervals.
        
        Args:
            video_path: Path to video file
            interval_ms: Interval between frames in milliseconds
            
        Returns:
            VideoFrameIterator context manager
        """
        return VideoFrameIterator(
            video_path,
            interval_ms=interval_ms,
            max_resolution=self.max_resolution
        )