#!/usr/bin/env python3
"""
Experiment 3: Content-Type Adaptivity

Demonstrates that budget adapts to video content type:
- Low motion (static/rotating product): Low ISD → Low budget
- Medium motion (testimonial/dialogue): Medium ISD → Medium budget  
- High motion (rapid scene cuts): High ISD → High budget

Usage:
    uv run python -m experiments.03_adaptivity \
        --pipeline_results ../../filtered_results.json \
        --embeddings_dir ../results/embeddings \
        --output_dir ../results
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.isd import compute_isd, compute_cut_frequency
from src.analysis import load_pipeline_results, categorize_by_motion, save_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Content-Type Adaptivity")
    parser.add_argument("--pipeline_results", required=True)
    parser.add_argument("--embeddings_dir", required=True)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    pipeline_data = load_pipeline_results(args.pipeline_results)
    embeddings_dir = Path(args.embeddings_dir)
    
    # Categorize and compute
    logger.info("Categorizing videos by motion...")
    rows = []
    
    for video_name, vdata in pipeline_data.items():
        emb_path = embeddings_dir / f"{video_name}.npy"
        
        if not emb_path.exists():
            continue
        
        try:
            embeddings = np.load(emb_path)
            if len(embeddings) < 3:
                continue
            
            isd = compute_isd(embeddings)
            
            # Get metadata
            metadata = vdata.get("metadata", {})
            scenes = vdata.get("scenes", [])
            extraction = vdata.get("extraction", {})
            duration = metadata.get("duration", 0)
            num_scenes = len(scenes)
            cut_freq = compute_cut_frequency(scenes, duration)
            
            # Categorize
            category = categorize_by_motion(cut_freq)
            ad_type = extraction.get("ad_type", "unknown")
            
            rows.append({
                "video_name": video_name,
                "isd": isd,
                "cut_frequency": cut_freq,
                "category": category,
                "duration": duration,
                "num_scenes": num_scenes,
                "ad_type": ad_type,
            })
            
        except Exception as e:
            logger.error(f"Failed {video_name}: {e}")
    
    if not rows:
        logger.error("No valid data!")
        return
    
    df = pd.DataFrame(rows)
    
    # Compute per-category statistics
    category_stats = {}
    for category in ["low_motion", "medium_motion", "high_motion"]:
        cat_df = df[df["category"] == category]
        if len(cat_df) == 0:
            continue
        
        category_stats[category] = {
            "n_videos": len(cat_df),
            "mean_isd": float(cat_df["isd"].mean()),
            "std_isd": float(cat_df["isd"].std()),
            "median_isd": float(cat_df["isd"].median()),
            "isd_range": [int(cat_df["isd"].min()), int(cat_df["isd"].max())],
            "mean_cut_freq": float(cat_df["cut_frequency"].mean()),
            "mean_duration": float(cat_df["duration"].mean()),
        }
    
    # Compute adaptivity ratio
    if "high_motion" in category_stats and "low_motion" in category_stats:
        high_isd = category_stats["high_motion"]["mean_isd"]
        low_isd = category_stats["low_motion"]["mean_isd"]
        adaptivity_ratio = high_isd / low_isd if low_isd > 0 else 0
    else:
        adaptivity_ratio = 0
    
    # Example videos
    examples = {}
    for category in category_stats.keys():
        cat_df = df[df["category"] == category]
        # Get representative example (closest to median)
        median_isd = cat_df["isd"].median()
        closest_idx = (cat_df["isd"] - median_isd).abs().idxmin()
        example = cat_df.loc[closest_idx]
        examples[category] = {
            "video_name": example["video_name"],
            "isd": int(example["isd"]),
            "cut_frequency": float(example["cut_frequency"]),
            "ad_type": example["ad_type"],
        }
    
    summary = {
        "n_videos": len(df),
        "category_statistics": category_stats,
        "adaptivity_ratio": adaptivity_ratio,
        "interpretation": {
            "low_motion": "Static/rotating products, single-scene testimonials",
            "medium_motion": "Dialogue with moderate scene changes",
            "high_motion": "Rapid cuts, action sequences, dynamic content",
        },
        "examples": examples,
        "key_finding": f"Budget adapts {adaptivity_ratio:.1f}x between low and high motion content",
    }
    
    # Save results
    df.to_csv(output_dir / "03_adaptivity_data.csv", index=False)
    save_results(summary, output_dir, "03_adaptivity_summary")
    
    # Print results
    logger.info("\n" + "="*70)
    logger.info("Content-Type Adaptivity Results")
    logger.info("="*70)
    logger.info(f"{'Category':<15} {'N':>6} {'Mean ISD':>10} {'Std ISD':>10} {'Cut Freq':>10}")
    logger.info("-"*70)
    
    for category in ["low_motion", "medium_motion", "high_motion"]:
        if category in category_stats:
            s = category_stats[category]
            logger.info(f"{category:<15} {s['n_videos']:>6} {s['mean_isd']:>10.1f} {s['std_isd']:>10.1f} {s['mean_cut_freq']:>10.2f}")
    
    logger.info("\n" + "="*70)
    logger.info(f"Adaptivity Ratio: {adaptivity_ratio:.2f}x")
    logger.info("="*70)
    
    logger.info("\nExample Videos:")
    for category, ex in examples.items():
        logger.info(f"  {category}: {ex['video_name']} (ISD={ex['isd']}, type={ex['ad_type']})")


if __name__ == "__main__":
    main()
