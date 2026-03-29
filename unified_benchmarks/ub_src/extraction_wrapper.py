#!/usr/bin/env python3
"""
VLM extraction wrapper for unified benchmarks.

Wraps the existing AdExtractor to provide consistent extraction interface.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import logging

# Add parent directory to path to import from src
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Remove current src from path to avoid shadowing
unified_src = Path(__file__).parent
if str(unified_src) in sys.path:
    sys.path.remove(str(unified_src))

logger = logging.getLogger(__name__)

# Lazy imports to avoid loading heavy modules at startup
_clip_deduplicator = None
_ad_extractor_bare = None
_ad_extractor_full = None


def get_clip_deduplicator():
    """Get or create CLIP deduplicator instance."""
    global _clip_deduplicator
    if _clip_deduplicator is None:
        try:
            from src.deduplication.clip_embed import CLIPDeduplicator
            _clip_deduplicator = CLIPDeduplicator(
                model_name="ViT-B-32",
                pretrained="openai",
                threshold=0.90,
                device="auto"
            )
        except Exception as e:
            logger.error(f"Failed to initialize CLIP: {e}")
            raise
    return _clip_deduplicator


def get_ad_extractor_bare(config: Optional[Dict] = None):
    """Get or create bare AdExtractor (minimal settings)."""
    global _ad_extractor_bare
    if _ad_extractor_bare is None:
        try:
            from src.extraction.llm_client import AdExtractor

            # Handle both flat and nested config
            if config:
                if "extraction" in config:
                    ext_config = config["extraction"]
                else:
                    ext_config = config
            else:
                ext_config = {}

            _ad_extractor_bare = AdExtractor(
                provider=ext_config.get("provider", "gemini"),
                model=ext_config.get("model", "gemini-2.5-flash"),
                max_tokens=ext_config.get("max_tokens", 4000),
                temperature=0.0,
                schema_mode="fixed",  # Skip type detection
                temporal_context=False,
                include_timestamps=False,
                include_time_deltas=False,
                include_position_labels=False,
                include_narrative_instructions=False,
            )
        except Exception as e:
            logger.error(f"Failed to initialize bare extractor: {e}")
            raise
    return _ad_extractor_bare


def get_ad_extractor_full(config: Optional[Dict] = None):
    """Get or create full AdExtractor (complete settings)."""
    global _ad_extractor_full
    if _ad_extractor_full is None:
        try:
            from src.extraction.llm_client import create_extractor

            if config:
                # Ensure config has 'extraction' key
                if "extraction" not in config:
                    config = {"extraction": config}
                _ad_extractor_full = create_extractor(config)
            else:
                # Default config
                _ad_extractor_full = create_extractor({
                    "extraction": {
                        "provider": "gemini",
                        "model": "gemini-2.5-flash",
                        "max_tokens": 4000,
                        "temperature": 0.0,
                    }
                })
        except Exception as e:
            logger.error(f"Failed to initialize full extractor: {e}")
            raise
    return _ad_extractor_full


class ExtractionWrapper:
    """Wrapper for VLM extraction with bare and full modes."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._bare = None
        self._full = None

    @property
    def bare(self):
        """Get bare extractor (lazy initialization)."""
        if self._bare is None:
            self._bare = get_ad_extractor_bare(self.config)
        return self._bare

    @property
    def full(self):
        """Get full extractor (lazy initialization)."""
        if self._full is None:
            self._full = get_ad_extractor_full(self.config)
        return self._full

    def extract_bare(
        self,
        frames: List[Tuple[float, np.ndarray]],
        duration: float
    ) -> Dict[str, Any]:
        """
        Extract with minimal settings (1 LLM call).

        Args:
            frames: List of (timestamp, frame) tuples
            duration: Video duration in seconds

        Returns:
            Extraction result dict
        """
        try:
            result = self.bare.extract(frames, duration, audio_context=None)
            return result if result else {}
        except Exception as e:
            logger.error(f"Bare extraction failed: {e}")
            return {"error": str(e)}

    def extract_full(
        self,
        frames: List[Tuple[float, np.ndarray]],
        duration: float,
        audio_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Extract with full settings (2 LLM calls for adaptive schema).

        Args:
            frames: List of (timestamp, frame) tuples
            duration: Video duration in seconds
            audio_context: Optional audio context dict

        Returns:
            Extraction result dict
        """
        try:
            result = self.full.extract(frames, duration, audio_context=audio_context)
            return result if result else {}
        except Exception as e:
            logger.error(f"Full extraction failed: {e}")
            return {"error": str(e)}


def decode_all_frames(
    video_path: str,
    interval_ms: int = 100,
    max_resolution: int = 720
) -> Tuple[List[Tuple[float, np.ndarray]], float, int]:
    """
    Decode all frames from video at specified interval.

    Args:
        video_path: Path to video file
        interval_ms: Sampling interval in milliseconds
        max_resolution: Maximum resolution (resize if larger)

    Returns:
        Tuple of (frames, duration, total_frames)
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # Calculate interval in frames
    interval_frames = int(fps * interval_ms / 1000)
    if interval_frames < 1:
        interval_frames = 1

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval_frames == 0:
            # Resize if needed
            h, w = frame.shape[:2]
            if max(h, w) > max_resolution:
                scale = max_resolution / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))

            timestamp = frame_idx / fps
            frames.append((timestamp, frame))

        frame_idx += 1

    cap.release()
    return frames, duration, total_frames


def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get video metadata."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration": duration,
    }
