"""
PySceneDetect baseline — select one representative frame per detected scene.

Uses PySceneDetect's ContentDetector to find scene boundaries, then picks
the frame closest to the midpoint of each scene. This is a common
industry approach for keyframe extraction.
"""

import logging
from typing import Any, List, Tuple

import cv2
import numpy as np

from benchmarks.base import BaselineMethod

logger = logging.getLogger(__name__)


class PySceneDetectBaseline(BaselineMethod):
    """Select one frame per scene detected by PySceneDetect."""

    def __init__(self, threshold: float = 27.0, min_scene_len_s: float = 0.5):
        self.threshold = threshold
        self.min_scene_len_s = min_scene_len_s

    @property
    def name(self) -> str:
        return "pyscenedetect"

    def select_frames(
        self, video_path: str, **kwargs: Any
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Detect scenes via PySceneDetect, pick midpoint frame of each scene.
        """
        try:
            from scenedetect import detect, ContentDetector
        except ImportError:
            logger.error(
                "scenedetect not installed. Run: pip install scenedetect[opencv]"
            )
            return self._fallback(video_path, **kwargs)

        # Detect scenes
        try:
            scene_list = detect(
                video_path,
                ContentDetector(threshold=self.threshold),
            )
        except Exception as e:
            logger.warning(f"PySceneDetect failed: {e}, using fallback")
            return self._fallback(video_path, **kwargs)

        if not scene_list:
            logger.warning("No scenes detected, treating entire video as one scene")
            return self._fallback(video_path, **kwargs)

        # Filter by minimum scene length
        scenes = []
        for start_tc, end_tc in scene_list:
            start_s = start_tc.get_seconds()
            end_s = end_tc.get_seconds()
            if end_s - start_s >= self.min_scene_len_s:
                scenes.append((start_s, end_s))

        if not scenes:
            scenes = [
                (s.get_seconds(), e.get_seconds()) for s, e in scene_list
            ]

        # For each scene, seek to the midpoint and grab the frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        max_res = kwargs.get("max_resolution", 720)

        selected: List[Tuple[float, np.ndarray]] = []
        for start_s, end_s in scenes:
            mid_s = (start_s + end_s) / 2.0
            mid_frame_idx = int(mid_s * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Resize if needed
            h, w = frame.shape[:2]
            if max(h, w) > max_res:
                scale = max_res / max(h, w)
                frame = cv2.resize(
                    frame,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            selected.append((mid_s, frame))

        cap.release()

        # Always include first and last frame if not already covered
        if selected and selected[0][0] > 1.0:
            cap2 = cv2.VideoCapture(video_path)
            ret, frame = cap2.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                if max(h, w) > max_res:
                    scale = max_res / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                selected.insert(0, (0.0, frame))
            cap2.release()

        logger.info(
            f"PySceneDetect: {len(scene_list)} raw scenes → "
            f"{len(scenes)} filtered → {len(selected)} frames"
        )
        return sorted(selected, key=lambda x: x[0])

    def _fallback(
        self, video_path: str, **kwargs
    ) -> List[Tuple[float, np.ndarray]]:
        """If PySceneDetect fails, fall back to uniform 1 FPS."""
        from benchmarks.methods.uniform import UniformSampling

        logger.warning("Falling back to Uniform 1 FPS")
        return UniformSampling().select_frames(video_path, **kwargs)