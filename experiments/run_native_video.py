#!/usr/bin/env python3
"""
Gemini Native Video vs. Frame-Based Comparison
================================================
Fills %%PLACEHOLDER in Section 6.2:
"Gemini's native video input mode achieves XX% topic accuracy at $YY/video..."

Usage:
    python -m experiments.run_native_video \
        --video_dir data/ads/ads/videos \
        --pipeline_results native_results/analysis.json \
        --output_dir native_results/native_video \
        --max_videos 5

Requires: GEMINI_API_KEY or GOOGLE_API_KEY env var
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Add project root to sys.path so `from src.…` works ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Load .env from project root ──
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
GEMINI_VIDEO_COST_PER_SECOND = 0.001
GEMINI_FLASH_COST_PER_1K_INPUT = 0.00015
GEMINI_FLASH_COST_PER_1K_OUTPUT = 0.0006
TOKENS_PER_FRAME = 765


# ====================================================================
# Self-contained helpers (NO external imports from benchmarks/)
# ====================================================================

def load_pipeline_results(path: str) -> Dict[str, Dict]:
    """Load pipeline results → dict keyed by filename. Creates file if missing."""
    p = Path(path)
    if not p.exists():
        logger.warning(f"Pipeline results not found → creating empty file at {path}")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump({"metadata": {"created_at": datetime.now().isoformat()}, "results": []}, f, indent=2)
        return {}

    with open(p) as f:
        data = json.load(f)

    # Dict-keyed format?
    if "results" not in data and isinstance(data, dict):
        first_key = next(iter(data), "")
        if "." in first_key:
            return data

    # List-based format: {"results": [{...}, ...]}
    indexed = {}
    for r in data.get("results", []):
        vname = r.get("video_name") or Path(r.get("video_path", "")).name
        if vname:
            indexed[vname] = r
    logger.info(f"Loaded pipeline results for {len(indexed)} videos")
    return indexed


def find_video_files(video_dir: str) -> List[str]:
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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total / fps if fps > 0 else 0.0
    cap.release()
    return total, fps, duration


def decode_frames(
    video_path: str, interval_ms: float = 100.0, max_resolution: int = 720,
) -> List[Tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps > 0 else 0.0
    frames, t, step = [], 0.0, interval_ms / 1000.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            if max(h, w) > max_resolution:
                s = max_resolution / max(h, w)
                frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            frames.append((t, frame))
        t += step
    cap.release()
    return frames


def compare_extractions(baseline: Dict, reference: Dict) -> Dict[str, Any]:
    def _sg(d, keys):
        for k in keys:
            if not isinstance(d, dict):
                return None
            d = d.get(k)
        return d

    if not baseline or "error" in baseline:
        return {"topic_match": None, "brand_match": None, "cta_detected": None,
                "promo_detected": None, "error": (baseline or {}).get("error", "unknown")}

    if not reference:
        return {"topic_match": None, "brand_match": None,
                "cta_detected": bool(_sg(baseline, ["call_to_action", "cta_present"])),
                "promo_detected": bool(_sg(baseline, ["promotion", "promo_present"]))}

    b_t, r_t = _sg(baseline, ["topic", "topic_id"]), _sg(reference, ["topic", "topic_id"])
    topic_match = None
    if b_t is not None and r_t is not None:
        try:
            topic_match = int(b_t) == int(r_t)
        except (ValueError, TypeError):
            pass

    b_b, r_b = _sg(baseline, ["brand", "brand_name_text"]), _sg(reference, ["brand", "brand_name_text"])
    brand_match = None
    if b_b and r_b:
        bl, rl = str(b_b).lower().strip(), str(r_b).lower().strip()
        brand_match = bl == rl or bl in rl or rl in bl

    cta = _sg(baseline, ["call_to_action", "cta_present"])
    promo = _sg(baseline, ["promotion", "promo_present"])
    return {"topic_match": topic_match, "brand_match": brand_match,
            "cta_detected": bool(cta) if cta is not None else None,
            "promo_detected": bool(promo) if promo is not None else None}


# ====================================================================
# Extraction wrappers
# ====================================================================

def run_frame_based_extraction(frames, duration):
    from src.extraction.llm_client import AdExtractor
    ext = AdExtractor(provider="gemini", model="gemini-2.5-flash",
                      max_tokens=4000, temperature=0.0, schema_mode="fixed",
                      temporal_context=True, include_timestamps=True)
    t0 = time.time()
    result = ext.extract(frames, duration, audio_context=None)
    return result, time.time() - t0


def run_native_video_extraction(video_path, duration):
    from src.extraction.llm_client import GeminiVideoClient
    prompt = """You are analyzing a video advertisement. Extract structured info.

Return ONLY valid JSON:
{
  "brand": {"brand_name_text": "string or null"},
  "product": {"product_category": "string"},
  "topic": {"topic_id": integer, "topic_name": "string"},
  "sentiment": {"overall_tone": "string"},
  "call_to_action": {"cta_present": boolean, "cta_type": "string or null"},
  "promotion": {"promo_present": boolean},
  "engagement_metrics": {"effectiveness_score": 1-5}
}

Topic IDs: 1=Food, 2=Beverages, 3=Health, 4=Beauty, 5=Fashion,
6=Home, 7=Tech, 8=Auto, 9=Finance, 10=Travel, 11=Entertainment,
12=Education, 13=Social/PSA, 14=Politics, 15=Sports, 16=Pets,
17=Family, 18=Luxury, 19=Cleaning, 20=Real Estate"""

    client = GeminiVideoClient(model="gemini-2.5-flash", max_tokens=4000, temperature=0.0)
    t0 = time.time()
    try:
        raw = client.extract_from_video(video_path, prompt)
        latency = time.time() - t0
        text = raw.strip()
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
        bm = re.search(r"\{.*\}", text, re.DOTALL)
        if bm:
            text = bm.group(0)
        return json.loads(text), latency
    except Exception as e:
        return {"error": str(e)}, time.time() - t0


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Gemini Native Video vs Frame-Based")
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--pipeline_results", required=True,
                        help="Path to pipeline results JSON (auto-created if missing)")
    parser.add_argument("--output_dir", default="results/native_video")
    parser.add_argument("--max_videos", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--throttle_s", type=float, default=4.0)
    args = parser.parse_args()

    # ── Check API key ──
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("Set GEMINI_API_KEY or GOOGLE_API_KEY env var first.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load / create pipeline results ──
    pipeline_results = load_pipeline_results(args.pipeline_results)

    # ── Find videos ──
    video_files = find_video_files(args.video_dir)
    if not video_files:
        logger.error(f"No video files found in {args.video_dir}")
        sys.exit(1)
    logger.info(f"Found {len(video_files)} videos in {args.video_dir}")

    # Prefer videos that have pipeline references (for accuracy comparison)
    if pipeline_results:
        with_ref = [v for v in video_files
                    if pipeline_results.get(Path(v).name, {}).get("extraction")]
        logger.info(f"  {len(with_ref)} have pipeline references")
        candidates = with_ref or video_files
    else:
        logger.info("  No pipeline references — running without accuracy comparison")
        candidates = video_files

    random.seed(args.seed)
    sample = random.sample(candidates, min(args.max_videos, len(candidates)))
    logger.info(f"Selected {len(sample)} videos\n")

    # ── Run comparison ──
    rows: List[Dict] = []
    for i, vpath in enumerate(sample):
        vname = Path(vpath).name
        logger.info(f"[{i+1}/{len(sample)}] {vname}")

        ref = pipeline_results.get(vname, {})
        ref_ext = ref.get("extraction", {})
        try:
            total_frames, fps, duration = get_video_info(vpath)
        except Exception as e:
            logger.error(f"  Cannot read video: {e}")
            continue

        # ── Frame-based ──
        pk = ref.get("pipeline_stats", {}).get("final_frame_count") or min(25, max(5, int(duration)))
        all_f = decode_frames(vpath, interval_ms=100, max_resolution=720)
        if not all_f:
            logger.warning(f"  No frames decoded, skipping")
            continue
        step = max(1, len(all_f) // pk)
        sel = all_f[::step][:pk]
        nf = len(sel)
        logger.info(f"  Frame-based: {nf} frames, duration={duration:.1f}s")

        f_err = None
        try:
            f_res, f_lat = run_frame_based_extraction(sel, duration)
        except Exception as e:
            f_res, f_lat, f_err = {"error": str(e)}, 0.0, str(e)
            logger.error(f"    Frame extraction failed: {e}")

        f_cmp = compare_extractions(f_res, ref_ext)
        f_cost = (nf * TOKENS_PER_FRAME / 1000) * GEMINI_FLASH_COST_PER_1K_INPUT \
                 + (1000 / 1000) * GEMINI_FLASH_COST_PER_1K_OUTPUT

        time.sleep(args.throttle_s)

        # ── Native video ──
        logger.info(f"  Native video: uploading {duration:.1f}s")
        n_err = None
        try:
            n_res, n_lat = run_native_video_extraction(vpath, duration)
        except Exception as e:
            n_res, n_lat, n_err = {"error": str(e)}, 0.0, str(e)
            logger.error(f"    Native extraction failed: {e}")

        n_cmp = compare_extractions(n_res, ref_ext)
        n_cost = duration * GEMINI_VIDEO_COST_PER_SECOND \
                 + (1000 / 1000) * GEMINI_FLASH_COST_PER_1K_OUTPUT

        time.sleep(args.throttle_s)

        rows.append({
            "video": vname, "duration": round(duration, 2), "num_frames_used": nf,
            "frame_latency_s": round(f_lat, 2), "frame_cost_usd": round(f_cost, 5),
            "frame_topic_match": f_cmp.get("topic_match"),
            "frame_brand_match": f_cmp.get("brand_match"),
            "frame_cta": f_cmp.get("cta_detected"), "frame_error": f_err,
            "native_latency_s": round(n_lat, 2), "native_cost_usd": round(n_cost, 5),
            "native_topic_match": n_cmp.get("topic_match"),
            "native_brand_match": n_cmp.get("brand_match"),
            "native_cta": n_cmp.get("cta_detected"), "native_error": n_err,
        })

        # Incremental save
        pd.DataFrame(rows).to_csv(output_dir / "native_video_results.csv", index=False)
        logger.info(f"  → Frame={'OK' if not f_err else 'FAIL'}, "
                    f"Native={'OK' if not n_err else 'FAIL'}\n")

    # ── Summary ──
    if not rows:
        print("\nNo results collected.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "native_video_results.csv", index=False)

    print("\n" + "=" * 60)
    print("Gemini Native Video vs. Frame-Based")
    print("=" * 60)

    for label, pfx in [("Frame-Based", "frame_"), ("Native Video", "native_")]:
        v = df[df[f"{pfx}error"].isna()]
        if v.empty:
            print(f"\n{label}: all failed")
            continue
        tv = v[f"{pfx}topic_match"].dropna()
        print(f"\n{label} (n={len(v)}):")
        print(f"  Topic Accuracy:  {tv.mean()*100:.1f}%" if len(tv) else "  Topic Accuracy:  N/A")
        print(f"  Mean Latency:    {v[f'{pfx}latency_s'].mean():.1f}s")
        print(f"  Mean Cost/Video: ${v[f'{pfx}cost_usd'].mean():.4f}")

    both = df[df["frame_error"].isna() & df["native_error"].isna()]
    bt = both.dropna(subset=["frame_topic_match", "native_topic_match"])
    if len(bt):
        agree = (bt["frame_topic_match"] == bt["native_topic_match"]).mean()
        print(f"\nAgreement (n={len(bt)}): {agree*100:.1f}%")

    summary = {"generated_at": datetime.now().isoformat(), "n_videos": len(df)}
    for lbl, pfx in [("frame_based", "frame_"), ("native_video", "native_")]:
        v = df[df[f"{pfx}error"].isna()]
        tv = v[f"{pfx}topic_match"].dropna()
        summary[lbl] = {
            "n_valid": len(v), "n_errors": len(df) - len(v),
            "topic_accuracy": round(tv.mean()*100, 1) if len(tv) else None,
            "mean_cost": round(v[f"{pfx}cost_usd"].mean(), 5) if len(v) else None,
            "mean_latency": round(v[f"{pfx}latency_s"].mean(), 2) if len(v) else None,
        }
    with open(output_dir / "native_video_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()