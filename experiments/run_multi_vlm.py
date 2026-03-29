#!/usr/bin/env python3
"""
Multi-VLM Provider Comparison
===============================
Fills %%PLACEHOLDER in Section 6.2:
"To assess provider generality, we ran our frames through Claude on 50 videos..."

Usage:
    python -m experiments.run_multi_vlm \
        --video_dir data/ads/ads/videos \
        --pipeline_results results/analysis.json \
        --output_dir results/multi_vlm \
        --providers gemini anthropic openai \
        --max_videos 50

Requires API keys as env vars: GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_utils import load_pipeline_results, find_video_files, get_video_info, decode_frames, compare_extractions

PROVIDER_CONFIGS = {
    "gemini": {"provider": "gemini", "model": "gemini-2.0-flash-exp",
               "env_key": "GEMINI_API_KEY", "alt_env": "GOOGLE_API_KEY",
               "cost_1k_in": 0.00015, "cost_1k_out": 0.0006},
    "anthropic": {"provider": "anthropic", "model": "claude-sonnet-4-20250514",
                  "env_key": "ANTHROPIC_API_KEY", "alt_env": "",
                  "cost_1k_in": 0.003, "cost_1k_out": 0.015},
    "openai": {"provider": "openai", "model": "gpt-4o",
               "env_key": "OPENAI_API_KEY", "alt_env": "",
               "cost_1k_in": 0.0025, "cost_1k_out": 0.01},
}


def check_api_keys(providers):
    available = []
    for p in providers:
        cfg = PROVIDER_CONFIGS.get(p)
        if not cfg:
            continue
        key = os.environ.get(cfg["env_key"]) or os.environ.get(cfg.get("alt_env", ""), "")
        if key:
            available.append(p)
            logger.info(f"  {p}: API key found")
        else:
            logger.warning(f"  {p}: No API key ({cfg['env_key']}), skipping")
    return available


def create_extractor(provider_name):
    from src.extraction.llm_client import AdExtractor
    cfg = PROVIDER_CONFIGS[provider_name]
    return AdExtractor(
        provider=cfg["provider"], model=cfg["model"], max_tokens=4000,
        temperature=0.0, schema_mode="fixed", temporal_context=True,
        include_timestamps=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Multi-VLM Provider Comparison")
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--pipeline_results", required=True)
    parser.add_argument("--output_dir", default="results/multi_vlm")
    parser.add_argument("--providers", nargs="+", default=["gemini", "anthropic", "openai"],
                        choices=list(PROVIDER_CONFIGS.keys()))
    parser.add_argument("--max_videos", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--throttle_s", type=float, default=3.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    available = check_api_keys(args.providers)
    if not available:
        logger.error("No providers with valid API keys.")
        sys.exit(1)

    pipeline_results = load_pipeline_results(args.pipeline_results)
    video_files = find_video_files(args.video_dir)
    if not video_files:
        logger.error(f"No videos found in {args.video_dir}")
        sys.exit(1)

    # Prefer videos with pipeline results
    candidates = [v for v in video_files
                  if pipeline_results.get(Path(v).name, {}).get("extraction")] or video_files

    random.seed(args.seed)
    sample = random.sample(candidates, min(args.max_videos, len(candidates)))
    logger.info(f"Selected {len(sample)} videos, providers: {available}")

    extractors = {}
    for p in available:
        try:
            extractors[p] = create_extractor(p)
        except Exception as e:
            logger.error(f"Failed to init {p}: {e}")

    rows = []
    for i, vpath in enumerate(sample):
        vname = Path(vpath).name
        logger.info(f"\n[{i+1}/{len(sample)}] {vname}")

        ref_extraction = pipeline_results.get(vname, {}).get("extraction", {})
        try:
            _, _, duration = get_video_info(vpath)
        except Exception as e:
            logger.error(f"  Cannot read: {e}")
            continue

        pipeline_k = pipeline_results.get(vname, {}).get("pipeline_stats", {}).get("final_frame_count")
        if not pipeline_k:
            pipeline_k = min(25, max(5, int(duration)))

        all_frames = decode_frames(vpath, interval_ms=100, max_resolution=720)
        if not all_frames:
            continue
        step = max(1, len(all_frames) // pipeline_k)
        frames = all_frames[::step][:pipeline_k]

        for pname, extractor in extractors.items():
            logger.info(f"  {pname}...")
            cfg = PROVIDER_CONFIGS[pname]
            est_cost = (len(frames) * 765 / 1000) * cfg["cost_1k_in"] + (1000/1000) * cfg["cost_1k_out"]

            t0 = time.time()
            try:
                result = extractor.extract(frames, duration, audio_context=None)
                error = None
            except Exception as e:
                result = {"error": str(e)}
                error = str(e)
            latency = time.time() - t0

            cmp = compare_extractions(result, ref_extraction)
            rows.append({
                "video": vname, "provider": pname, "num_frames": len(frames),
                "latency_s": round(latency, 2), "est_cost": round(est_cost, 5),
                "topic_match": cmp.get("topic_match"), "brand_match": cmp.get("brand_match"),
                "cta_detected": cmp.get("cta_detected"), "error": error,
            })
            time.sleep(args.throttle_s)

        # Incremental save
        pd.DataFrame(rows).to_csv(output_dir / "multi_vlm_results.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "multi_vlm_results.csv", index=False)

    print("\n" + "=" * 70)
    print("Multi-VLM Provider Comparison")
    print("=" * 70)
    summary = {}
    for p in available:
        sub = df[df["provider"] == p]
        valid = sub[sub["error"].isna()]
        n = len(valid)
        if n == 0:
            print(f"\n{p}: All failed")
            continue
        tv = valid["topic_match"].dropna()
        topic = tv.mean() * 100 if len(tv) > 0 else None
        print(f"\n{p} (n={n}):")
        print(f"  Topic Accuracy:  {topic:.1f}%" if topic else "  Topic Accuracy:  N/A")
        print(f"  Mean Latency:    {valid['latency_s'].mean():.1f}s")
        print(f"  Mean Cost:       ${valid['est_cost'].mean():.4f}")
        summary[p] = {"n": n, "topic_acc": round(topic, 1) if topic else None,
                       "mean_latency": round(valid["latency_s"].mean(), 2),
                       "mean_cost": round(valid["est_cost"].mean(), 5)}

    with open(output_dir / "multi_vlm_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()