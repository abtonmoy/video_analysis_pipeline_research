#!/usr/bin/env python3
"""
CLI entry point for unified benchmarks.

Usage:
    python scripts/run_benchmark.py \
        --video_dir /path/to/videos \
        --pipeline_results /path/to/results.json \
        --output_dir ./results \
        --methods uniform_1fps random histogram

    python scripts/run_benchmark.py \
        --video_dir /path/to/videos \
        --pipeline_results /path/to/results.json \
        --selection_only

    python scripts/run_benchmark.py \
        --video_dir /path/to/videos \
        --pipeline_results /path/to/results.json \
        --generate_figures
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Load .env from parent directory
from dotenv import load_dotenv
parent_dir = Path(__file__).parent.parent.parent
env_path = parent_dir / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ub_src.runner import BenchmarkRunner
from ub_src.visualization import generate_all_figures
from ub_src.aggregation import aggregate_by_method, save_summary

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML or JSON."""
    import yaml

    with open(config_path) as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)


def get_video_paths(video_dir: str, max_videos: int = None) -> list:
    """Get list of video files from directory."""
    video_dir = Path(video_dir)
    extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']

    videos = []
    for ext in extensions:
        videos.extend(video_dir.glob(f'*{ext}'))

    videos = sorted(videos)

    if max_videos:
        videos = videos[:max_videos]

    return [str(v) for v in videos]


def main():
    parser = argparse.ArgumentParser(
        description='Run unified benchmarks for AdaFrame',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all methods on all videos
  python scripts/run_benchmark.py --video_dir ./videos --pipeline_results ./results.json

  # Run specific methods
  python scripts/run_benchmark.py --video_dir ./videos --pipeline_results ./results.json \\
      --methods uniform_1fps random histogram

  # Frame selection only (no VLM calls)
  python scripts/run_benchmark.py --video_dir ./videos --pipeline_results ./results.json \\
      --selection_only

  # Generate figures from existing results
  python scripts/run_benchmark.py --generate_figures --output_dir ./results
        """
    )

    parser.add_argument('--video_dir', type=str,
                        help='Directory containing video files')
    parser.add_argument('--pipeline_results', type=str,
                        help='Path to AdaFrame pipeline results JSON')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--config', type=str, default='./config.yaml',
                        help='Configuration file path')

    parser.add_argument('--methods', nargs='+',
                        help='Methods to run (default: all)')
    parser.add_argument('--skip_gpu', action='store_true',
                        help='Skip GPU-dependent methods')

    parser.add_argument('--selection_only', action='store_true',
                        help='Only compute frame selection metrics (no VLM)')
    parser.add_argument('--bare_only', action='store_true',
                        help='Only run bare extraction')
    parser.add_argument('--full_only', action='store_true',
                        help='Only run full extraction')

    parser.add_argument('--max_videos', type=int,
                        help='Maximum number of videos to process')
    parser.add_argument('--generate_figures', action='store_true',
                        help='Generate figures from existing results')

    parser.add_argument('--retry_queue_dir', type=str,
                        default='./retry_queue',
                        help='Directory for retry queue')
    parser.add_argument('--failed_log', type=str,
                        default='./failed.log',
                        help='Path to failed log file')

    args = parser.parse_args()

    # Generate figures only
    if args.generate_figures:
        logger.info("Generating figures from existing results...")
        from ub_src.visualization import generate_all_figures_from_combined
        generate_all_figures_from_combined(args.output_dir)
        logger.info("Done!")
        return

    # Validate required args
    if not args.video_dir or not args.pipeline_results:
        parser.error("--video_dir and --pipeline_results are required")

    # Load config
    logger.info(f"Loading config from {args.config}")
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        config = {
            "provider": "gemini",
            "model": "gemini-2.0-flash-exp",
            "max_tokens": 4000,
            "temperature": 0.0,
            "throttle_seconds": 4.0,
            "sample_interval_ms": 100,
            "max_resolution": 720,
        }

    # Get video paths
    logger.info(f"Scanning {args.video_dir} for videos...")
    video_paths = get_video_paths(args.video_dir, args.max_videos)
    logger.info(f"Found {len(video_paths)} videos")

    if not video_paths:
        logger.error("No videos found!")
        sys.exit(1)

    # Create runner
    runner = BenchmarkRunner(
        config=config,
        pipeline_results_path=args.pipeline_results,
        output_dir=args.output_dir,
        methods=args.methods,
        skip_gpu=args.skip_gpu,
        selection_only=args.selection_only,
        bare_only=args.bare_only,
        full_only=args.full_only,
    )

    # Run benchmark
    logger.info("Starting benchmark...")
    summary = runner.run(
        video_paths=video_paths,
        retry_queue_dir=args.retry_queue_dir,
        failed_log_path=args.failed_log,
    )

    # Generate figures
    logger.info("Generating figures...")
    from ub_src.visualization import generate_all_figures_from_combined
    generate_all_figures_from_combined(args.output_dir)

    logger.info("Benchmark complete!")
    logger.info(f"Results saved to {args.output_dir}")

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Videos processed: {summary['n_videos']}")
    print(f"Methods: {summary['n_methods']}")
    print("\nResults by method:")
    for method, stats in summary.get("methods", {}).items():
        frames = stats["frames"]["mean"]
        cost = stats["cost_usd"]["mean"]
        topic_acc = stats["accuracy"]["topic_accuracy"]
        print(f"  {method:20s}: {frames:5.1f} frames, ${cost:.4f}, {topic_acc:.1f}% topic acc")
    print("="*60)


if __name__ == "__main__":
    main()
