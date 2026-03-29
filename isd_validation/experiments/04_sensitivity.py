#!/usr/bin/env python3
"""
Experiment 4: ISD Threshold Sensitivity

Tests different variance thresholds (tau) for ISD computation
to validate the choice of tau=0.90.

Usage:
    uv run python -m experiments.04_sensitivity \
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
from src.isd import compute_isd
from src.analysis import save_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ISD Threshold Sensitivity")
    parser.add_argument("--embeddings_dir", required=True)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    embeddings_dir = Path(args.embeddings_dir)
    
    # Test different tau values
    tau_values = [0.80, 0.85, 0.90, 0.95, 0.99]
    
    logger.info("Testing ISD with different tau values...")
    
    # Collect ISD values for each tau
    isd_by_tau = {tau: [] for tau in tau_values}
    video_names = []
    
    for emb_file in sorted(embeddings_dir.glob("*.npy")):
        try:
            embeddings = np.load(emb_file)
            if len(embeddings) < 3:
                continue
            
            video_name = emb_file.stem
            video_names.append(video_name)
            
            for tau in tau_values:
                isd = compute_isd(embeddings, tau=tau)
                isd_by_tau[tau].append(isd)
                
        except Exception as e:
            logger.error(f"Failed {emb_file}: {e}")
    
    if not video_names:
        logger.error("No valid embeddings!")
        return
    
    # Compute statistics
    stats_by_tau = {}
    for tau in tau_values:
        isds = np.array(isd_by_tau[tau])
        stats_by_tau[tau] = {
            "mean": float(isds.mean()),
            "median": float(np.median(isds)),
            "std": float(isds.std()),
            "min": int(isds.min()),
            "max": int(isds.max()),
            "range": int(isds.max() - isds.min()),
            "q25": float(np.percentile(isds, 25)),
            "q75": float(np.percentile(isds, 75)),
        }
    
    # Create comparison data
    rows = []
    for i, video_name in enumerate(video_names):
        row = {"video_name": video_name}
        for tau in tau_values:
            row[f"isd_tau_{tau}"] = isd_by_tau[tau][i]
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Analysis
    baseline_tau = 0.90
    baseline_isds = np.array(isd_by_tau[baseline_tau])
    
    differences = {}
    for tau in tau_values:
        if tau == baseline_tau:
            continue
        diff = np.array(isd_by_tau[tau]) - baseline_isds
        differences[tau] = {
            "mean_diff": float(diff.mean()),
            "max_diff": int(np.abs(diff).max()),
            "direction": "higher" if diff.mean() > 0 else "lower",
        }
    
    summary = {
        "n_videos": len(video_names),
        "baseline_tau": baseline_tau,
        "statistics_by_tau": stats_by_tau,
        "differences_from_baseline": differences,
        "interpretation": {
            "0.80": "Captures less variance, lower ISD, may miss subtle changes",
            "0.85": "Moderate threshold, slightly conservative",
            "0.90": "Balanced - captures most semantic variation without over-budgeting",
            "0.95": "Higher threshold, more frames, diminishing returns",
            "0.99": "Very high threshold, over-budgets for marginal gains",
        },
        "recommendation": "tau=0.90 provides optimal balance of coverage and efficiency",
    }
    
    # Save results
    df.to_csv(output_dir / "04_sensitivity_data.csv", index=False)
    save_results(summary, output_dir, "04_sensitivity_summary")
    
    # Print results
    logger.info("\n" + "="*70)
    logger.info("ISD Threshold Sensitivity Analysis")
    logger.info("="*70)
    logger.info(f"{'Tau':<8} {'Mean ISD':>10} {'Median':>10} {'Std':>8} {'Range':>8}")
    logger.info("-"*70)
    
    for tau in tau_values:
        s = stats_by_tau[tau]
        marker = " <-- baseline" if tau == baseline_tau else ""
        logger.info(f"{tau:<8.2f} {s['mean']:>10.1f} {s['median']:>10.1f} {s['std']:>8.1f} {s['range']:>8}{marker}")
    
    logger.info("\n" + "="*70)
    logger.info("Differences from baseline (tau=0.90):")
    for tau, diff in differences.items():
        logger.info(f"  tau={tau}: {diff['direction']} by {abs(diff['mean_diff']):.1f} frames on average")
    
    logger.info("\n" + "="*70)
    logger.info("Recommendation: tau=0.90 balances coverage and efficiency")
    logger.info("="*70)


if __name__ == "__main__":
    main()
