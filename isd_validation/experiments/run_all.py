#!/usr/bin/env python3
"""
Run all ISD validation experiments.

Usage:
    uv run python -m experiments.run_all \
        --pipeline_results ../../filtered_results.json \
        --embeddings_dir ../results/embeddings \
        --output_dir ../results
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def run_experiment(script_name: str, args: list):
    """Run a single experiment script."""
    script_path = Path(__file__).parent / script_name
    cmd = [sys.executable, "-m", f"experiments.{script_name.replace('.py', '')}"] + args
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Running {script_name}")
    logger.info(f"{'='*70}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        logger.error(f"{script_name} failed with code {result.returncode}")
        return False
    
    logger.info(f"{script_name} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run all ISD validation experiments")
    parser.add_argument("--pipeline_results", required=True)
    parser.add_argument("--embeddings_dir", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--skip", nargs="+", default=[], help="Experiments to skip")
    args = parser.parse_args()
    
    base_args = [
        "--pipeline_results", args.pipeline_results,
        "--embeddings_dir", args.embeddings_dir,
        "--output_dir", args.output_dir,
    ]
    
    experiments = [
        ("01_correlation.py", base_args),
        ("02_ablation.py", base_args),
        ("03_adaptivity.py", base_args),
        ("04_sensitivity.py", ["--embeddings_dir", args.embeddings_dir, "--output_dir", args.output_dir]),
    ]
    
    success_count = 0
    
    for script_name, script_args in experiments:
        if script_name.replace(".py", "") in args.skip:
            logger.info(f"Skipping {script_name}")
            continue
        
        if run_experiment(script_name, script_args):
            success_count += 1
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Completed {success_count}/{len(experiments)} experiments")
    logger.info(f"{'='*70}")
    
    # Generate figures
    logger.info("\nGenerating figures...")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.visualization import generate_all_figures
        
        results_dir = Path(args.output_dir)
        figures_dir = Path(args.output_dir).parent / "figures"
        generate_all_figures(results_dir, figures_dir)
        
    except Exception as e:
        logger.error(f"Figure generation failed: {e}")
    
    logger.info("\nAll done!")


if __name__ == "__main__":
    main()
