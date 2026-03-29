#!/usr/bin/env python3
"""
Budget Formula Ablation Experiment
===================================
Fills %%PLACEHOLDER in Section 5.5 of the paper.

Compares 5 budget allocation strategies:
  1. Full Cascade   — ISD cap + Semantic Energy scaling (Eq. 6)
  2. ISD-only       — ISD cap, no energy scaling
  3. Energy-only    — Semantic Energy scaling, no ISD cap
  4. Linear-duration — budget = duration_s * 0.5
  5. Fixed-25       — always 25 frames

Also runs ISD multiplier sensitivity: {0.5, 1.0, 1.5, 2.0, 3.0}

Usage:
    python -m experiments.run_budget_ablation \
        --video_dir data/ads/ads/videos \
        --pipeline_results results/analysis.json \
        --output_dir results/budget_ablation \
        --max_videos 100
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Add experiments dir to path for shared_utils
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_utils import (
    load_pipeline_results, find_video_files, get_video_info, decode_frames,
)


# ============================================================================
# Budget Strategy Implementations
# ============================================================================

def compute_isd(embeddings: np.ndarray, tau: float = 0.90) -> int:
    """Intrinsic Semantic Dimensionality via SVD (Eq. 4)."""
    if embeddings.shape[0] < 3:
        return max(1, embeddings.shape[0] // 2)
    centered = embeddings - embeddings.mean(axis=0)
    try:
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return max(1, embeddings.shape[0] // 5)
    var_explained = (s ** 2) / (np.sum(s ** 2) + 1e-9)
    cum_var = np.cumsum(var_explained)
    return int(np.argmax(cum_var >= tau)) + 1


def compute_semantic_velocity(embeddings: np.ndarray) -> float:
    """Mean L2 distance between consecutive frame embeddings."""
    if embeddings.shape[0] < 2:
        return 0.5
    diffs = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
    return float(np.mean(diffs))


def compute_attention_yield(
    frames: List[Tuple[float, np.ndarray]], duration: float,
) -> float:
    """Simplified Attention Yield (positional weight only for speed)."""
    if not frames or duration <= 0:
        return 1.0
    scores = []
    for ts, _ in frames:
        pos = ts / duration
        pw = 1.5 if pos < 0.1 else (1.4 if pos > 0.9 else 1.0)
        scores.append(0.4 * 0.5 + 0.35 * 0.3 + 0.25 * pw)
    return float(np.mean(scores))


def budget_full_cascade(num_scenes, k_star, semantic_energy, duration,
                        density=0.25, isd_multiplier=1.5, **_):
    base = max(5, num_scenes + 1)
    isd_cap = base + int(isd_multiplier * k_star)
    energy_scaled = max(base, int(base + duration * density * semantic_energy))
    return min(isd_cap, energy_scaled)


def budget_isd_only(num_scenes, k_star, isd_multiplier=1.5, **_):
    base = max(5, num_scenes + 1)
    return base + int(isd_multiplier * k_star)


def budget_energy_only(num_scenes, semantic_energy, duration, density=0.25, **_):
    base = max(5, num_scenes + 1)
    return max(base, int(base + duration * density * semantic_energy))


def budget_linear_duration(duration, rate=0.5, **_):
    return max(5, int(duration * rate))


def budget_fixed(fixed_k=25, **_):
    return fixed_k


BUDGET_STRATEGIES = {
    "full_cascade": budget_full_cascade,
    "isd_only": budget_isd_only,
    "energy_only": budget_energy_only,
    "linear_duration": budget_linear_duration,
    "fixed_25": budget_fixed,
}

ISD_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 3.0]


def detect_scenes_count(video_path: str) -> int:
    """Quick scene count via PySceneDetect or fallback."""
    try:
        from scenedetect import detect, ContentDetector
        scenes = detect(video_path, ContentDetector(threshold=27.0))
        return max(1, len(scenes))
    except Exception:
        cap = cv2.VideoCapture(video_path)
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1, cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        return max(1, int(dur / 10))


def main():
    parser = argparse.ArgumentParser(description="Budget Formula Ablation")
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--pipeline_results", required=True)
    parser.add_argument("--output_dir", default="results/budget_ablation")
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--sample_interval_ms", type=float, default=100.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_results = load_pipeline_results(args.pipeline_results)
    video_files = find_video_files(args.video_dir)

    if not video_files:
        # If pipeline results exist, try to match from there
        video_dir = Path(args.video_dir)
        for vname in pipeline_results:
            vpath = video_dir / vname
            if vpath.exists():
                video_files.append(str(vpath))

    if not video_files:
        logger.error(f"No video files found in {args.video_dir}")
        sys.exit(1)

    if args.max_videos:
        video_files = video_files[: args.max_videos]

    logger.info(f"Processing {len(video_files)} videos")

    # Load CLIP
    logger.info("Loading CLIP model...")
    try:
        from src.deduplication.clip_embed import CLIPDeduplicator
        clip_dedup = CLIPDeduplicator(model_name="ViT-B-32", device="auto")
    except Exception as e:
        logger.error(f"Failed to load CLIP: {e}")
        sys.exit(1)

    rows = []
    for i, vpath in enumerate(video_files):
        vname = Path(vpath).name
        logger.info(f"[{i+1}/{len(video_files)}] {vname}")

        try:
            total_frames, fps, duration = get_video_info(vpath)
            frames = decode_frames(vpath, interval_ms=args.sample_interval_ms)
            if not frames:
                continue

            embeddings = clip_dedup.compute_signatures_batch([f for _, f in frames])
            num_scenes = detect_scenes_count(vpath)

            k_star = compute_isd(embeddings)
            sem_vel = compute_semantic_velocity(embeddings)
            attn_yield = compute_attention_yield(frames, duration)
            sem_energy = sem_vel * attn_yield

            row = {
                "video": vname, "duration": round(duration, 2),
                "num_scenes": num_scenes, "k_star": k_star,
                "semantic_velocity": round(sem_vel, 4),
                "attention_yield": round(attn_yield, 4),
                "semantic_energy": round(sem_energy, 4),
                "num_candidates": len(frames),
            }

            shared = dict(num_scenes=num_scenes, k_star=k_star,
                          semantic_energy=sem_energy, duration=duration)

            for name, fn in BUDGET_STRATEGIES.items():
                row[f"budget_{name}"] = fn(**shared)

            for mult in ISD_MULTIPLIERS:
                row[f"budget_mult_{mult}"] = budget_full_cascade(**shared, isd_multiplier=mult)

            rows.append(row)

        except Exception as e:
            logger.error(f"  Failed: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "budget_ablation.csv", index=False)
    logger.info(f"Saved {len(rows)} rows to {output_dir / 'budget_ablation.csv'}")

    # Summary
    budget_cols = [c for c in df.columns if c.startswith("budget_") and "mult" not in c]
    print("\n" + "=" * 70)
    print("Budget Formula Ablation Results")
    print("=" * 70)
    print(f"{'Strategy':<20} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>6} {'Max':>6}")
    print("-" * 60)
    for col in budget_cols:
        label = col.replace("budget_", "")
        print(f"{label:<20} {df[col].mean():>8.1f} {df[col].median():>8.1f} "
              f"{df[col].std():>8.1f} {df[col].min():>6} {df[col].max():>6}")

    print(f"\nISD Multiplier Sensitivity:")
    print(f"{'Multiplier':<12} {'Mean':>10} {'Median':>10}")
    print("-" * 35)
    for m in ISD_MULTIPLIERS:
        col = f"budget_mult_{m}"
        if col in df.columns:
            print(f"{m:<12} {df[col].mean():>10.1f} {df[col].median():>10.1f}")

    summary = {col: {"mean": round(df[col].mean(), 1), "median": round(df[col].median(), 1),
                      "std": round(df[col].std(), 1)} for col in budget_cols}
    with open(output_dir / "budget_ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()