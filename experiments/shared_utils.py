"""
Shared utilities for all experiment scripts.

Handles:
- Pipeline results loading (both list and dict formats, auto-create if missing)
- Video file discovery
- Video metadata extraction
- Frame decoding
- Self-contained extraction comparison (no benchmarks.metrics dependency)
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from project root
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)


# ============================================================================
# Pipeline results
# ============================================================================

def load_pipeline_results(path: str) -> Dict[str, Dict]:
    """
    Load pipeline results, converting to a dict keyed by video filename.
    Creates an empty file if it doesn't exist.

    Handles:
      - {"results": [{"video_name": "...", ...}]}   (main.py / run_benchmarks output)
      - {"video_name.mp4": {...}, ...}                (dict-keyed)
    """
    p = Path(path)

    if not p.exists():
        logger.warning(f"Pipeline results not found at {path} — creating empty file")
        p.parent.mkdir(parents=True, exist_ok=True)
        empty = {"metadata": {"created_at": datetime.now().isoformat()}, "results": []}
        with open(p, "w") as f:
            json.dump(empty, f, indent=2)
        return {}

    with open(p) as f:
        data = json.load(f)

    # Already a dict keyed by video name?
    if "results" not in data and isinstance(data, dict):
        first_key = next(iter(data), "")
        if "." in first_key:
            return data

    # List-based format
    indexed = {}
    results_list = data.get("results", [])
    if isinstance(results_list, list):
        for r in results_list:
            vname = r.get("video_name") or Path(r.get("video_path", "")).name
            if vname:
                indexed[vname] = r

    logger.info(f"Loaded pipeline results for {len(indexed)} videos")
    return indexed


# ============================================================================
# Video discovery & info
# ============================================================================

def find_video_files(video_dir: str) -> List[str]:
    """Find all video files in a directory."""
    vdir = Path(video_dir)
    if not vdir.exists():
        logger.error(f"Video directory does not exist: {video_dir}")
        return []

    files = []
    for ext in VIDEO_EXTENSIONS:
        files.extend(vdir.glob(f"*{ext}"))
        files.extend(vdir.glob(f"*{ext.upper()}"))

    return sorted(set(str(p) for p in files))


def get_video_info(video_path: str) -> Tuple[int, float, float]:
    """Get (total_frames, fps, duration) from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total / fps if fps > 0 else 0.0
    cap.release()
    return total, fps, duration


def decode_frames(
    video_path: str,
    interval_ms: float = 100.0,
    max_resolution: int = 720,
) -> List[Tuple[float, np.ndarray]]:
    """Decode video frames at a fixed time interval."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps > 0 else 0.0

    frames = []
    interval_s = interval_ms / 1000.0
    t = 0.0

    while t < duration:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            t += interval_s
            continue

        h, w = frame.shape[:2]
        if max(h, w) > max_resolution:
            scale = max_resolution / max(h, w)
            frame = cv2.resize(
                frame, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        frames.append((t, frame))
        t += interval_s

    cap.release()
    return frames


# ============================================================================
# Extraction comparison (self-contained)
# ============================================================================

def compare_extractions(
    baseline_result: Dict[str, Any],
    reference_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare a baseline extraction against the pipeline reference."""
    if not baseline_result or "error" in baseline_result:
        return {
            "topic_match": None,
            "brand_match": None,
            "cta_detected": None,
            "promo_detected": None,
            "error": (baseline_result or {}).get("error", "unknown"),
        }

    def _sg(d, keys):
        for k in keys:
            if not isinstance(d, dict):
                return None
            d = d.get(k)
        return d

    if not reference_result:
        return {
            "topic_match": None,
            "brand_match": None,
            "cta_detected": bool(_sg(baseline_result, ["call_to_action", "cta_present"])),
            "promo_detected": bool(_sg(baseline_result, ["promotion", "promo_present"])),
        }

    # Topic
    b_t = _sg(baseline_result, ["topic", "topic_id"])
    r_t = _sg(reference_result, ["topic", "topic_id"])
    topic_match = None
    if b_t is not None and r_t is not None:
        try:
            topic_match = int(b_t) == int(r_t)
        except (ValueError, TypeError):
            pass

    # Brand
    b_b = _sg(baseline_result, ["brand", "brand_name_text"])
    r_b = _sg(reference_result, ["brand", "brand_name_text"])
    brand_match = None
    if b_b and r_b:
        bl, rl = str(b_b).lower().strip(), str(r_b).lower().strip()
        brand_match = bl == rl or bl in rl or rl in bl

    cta = _sg(baseline_result, ["call_to_action", "cta_present"])
    promo = _sg(baseline_result, ["promotion", "promo_present"])

    return {
        "topic_match": topic_match,
        "brand_match": brand_match,
        "cta_detected": bool(cta) if cta is not None else None,
        "promo_detected": bool(promo) if promo is not None else None,
    }