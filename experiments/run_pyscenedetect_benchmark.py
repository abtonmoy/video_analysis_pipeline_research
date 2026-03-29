#!/usr/bin/env python3
"""
PySceneDetect Baseline Benchmark
==================================
Fills %%PLACEHOLDER for PySceneDetect row in Table 2.

Usage:
    python -m experiments.run_pyscenedetect_benchmark \
        --video_dir data/ads/ads/videos \
        --pipeline_results results/analysis.json \
        --output_dir results/pyscenedetect \
        --max_videos 50

Requirements: pip install scenedetect[opencv]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_utils import load_pipeline_results, find_video_files, get_video_info, compare_extractions

# Import the PySceneDetect baseline method
from benchmarks.methods.pyscenedetect import PySceneDetectBaseline


def main():
    parser = argparse.ArgumentParser(description="PySceneDetect Baseline Benchmark")
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--pipeline_results", required=True)
    parser.add_argument("--output_dir", default="results/pyscenedetect")
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--run_extraction", action="store_true")
    parser.add_argument("--threshold", type=float, default=27.0)
    parser.add_argument("--throttle_s", type=float, default=3.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_results = load_pipeline_results(args.pipeline_results)
    video_files = find_video_files(args.video_dir)

    if not video_files:
        logger.error(f"No videos in {args.video_dir}")
        sys.exit(1)

    if args.max_videos:
        video_files = video_files[:args.max_videos]

    logger.info(f"Running PySceneDetect on {len(video_files)} videos")

    method = PySceneDetectBaseline(threshold=args.threshold)

    # Optional CLIP for info density
    clip_dedup = None
    try:
        from src.deduplication.clip_embed import CLIPDeduplicator
        clip_dedup = CLIPDeduplicator(model_name="ViT-B-32", device="auto")
    except Exception as e:
        logger.warning(f"CLIP not available: {e}")

    # Optional extractor
    extractor = None
    if args.run_extraction:
        try:
            from benchmarks.extraction_wrapper import ExtractionWrapper
            extractor = ExtractionWrapper({
                "extraction": {"provider": "gemini", "model": "gemini-2.0-flash-exp",
                               "max_tokens": 4000, "temperature": 0.0}
            })
        except Exception as e:
            logger.warning(f"Extraction not available: {e}")

    rows = []
    for i, vpath in enumerate(video_files):
        vname = Path(vpath).name
        logger.info(f"[{i+1}/{len(video_files)}] {vname}")

        try:
            total_frames, fps, duration = get_video_info(vpath)
            frames, latency = method.run_timed(vpath, max_resolution=720)
            if not frames:
                continue

            count = len(frames)
            compression = total_frames / count if count > 0 else float("inf")
            vlm_cost = (count * 765 / 1000) * 0.015

            info_density = 0.0
            if clip_dedup and count >= 2:
                try:
                    from benchmarks.metrics import compute_info_density
                    info_density = compute_info_density(frames, clip_dedup)
                except Exception:
                    pass

            row = {"video": vname, "duration": round(duration, 2),
                   "total_frames": total_frames, "selected_count": count,
                   "compression_ratio": round(compression, 2),
                   "latency_s": round(latency, 3), "vlm_cost_usd": round(vlm_cost, 4),
                   "info_density": round(info_density, 5)}

            if extractor:
                ref = pipeline_results.get(vname, {}).get("extraction", {})
                try:
                    result = extractor.extract_bare(frames, duration)
                    cmp = compare_extractions(result, ref)
                    row["topic_match"] = cmp.get("topic_match")
                    row["brand_match"] = cmp.get("brand_match")
                    time.sleep(args.throttle_s)
                except Exception as e:
                    logger.error(f"  Extraction failed: {e}")

            rows.append(row)
        except Exception as e:
            logger.error(f"  Failed: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "pyscenedetect_results.csv", index=False)

    print("\n" + "=" * 70)
    print("PySceneDetect Baseline (Table 2 row)")
    print("=" * 70)
    print(f"Videos: {len(df)}")
    print(f"Mean frames: {df['selected_count'].mean():.1f}")
    print(f"Mean cost:   ${df['vlm_cost_usd'].mean():.3f}")
    print(f"Mean compression: {df['compression_ratio'].mean():.1f}x")

    if "topic_match" in df.columns:
        valid = df["topic_match"].dropna()
        if len(valid) > 0:
            print(f"Topic accuracy: {valid.mean()*100:.1f}% (n={len(valid)})")

    with open(output_dir / "pyscenedetect_summary.json", "w") as f:
        json.dump({"n": len(df), "mean_frames": round(df["selected_count"].mean(), 1),
                    "mean_cost": round(df["vlm_cost_usd"].mean(), 4)}, f, indent=2)


if __name__ == "__main__":
    main()