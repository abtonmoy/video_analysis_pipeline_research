# main.py
"""
Main entry point for batch video advertisement analysis.

Usage:
    python main.py --input data/ads --output outputs/results.json
    python main.py -i data/ads -o outputs/results.json --workers 4
    python main.py -i data/ads --skip-extraction  # No LLM calls
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import sys

from dotenv import load_dotenv
load_dotenv()  # Load API keys from .env

from src.pipeline import AdVideoPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch process video advertisements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos in data/ads directory
  python main.py --input data/ads
  
  # Process with 4 parallel workers
  python main.py -i data/ads --workers 4
  
  # Skip LLM extraction (faster, no API costs)
  python main.py -i data/ads --skip-extraction
  
  # Custom output file
  python main.py -i data/ads -o results/my_results.json
  
  # Use custom config
  python main.py -i data/ads --config config/fast.yaml
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input directory containing video files'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='outputs/results.json',
        help='Output JSON file path (default: outputs/results.json)'
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config/default.yaml',
        help='Path to configuration file (default: config/default.yaml)'
    )
    
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)'
    )
    
    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip LLM extraction (only run pipeline stages 1-6)'
    )
    
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.mp4', '.mov', '.avi', '.mkv', '.webm'],
        help='Video file extensions to process (default: .mp4 .mov .avi .mkv .webm)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def find_videos(input_dir: str, extensions: List[str]) -> List[str]:
    """
    Find all video files in directory.
    
    Args:
        input_dir: Input directory path
        extensions: List of video file extensions
        
    Returns:
        List of video file paths
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    video_files = []
    
    for ext in extensions:
        # Case-insensitive search
        video_files.extend(input_path.glob(f"*{ext}"))
        video_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    # Remove duplicates and sort
    video_files = sorted(set(video_files))
    
    return [str(p) for p in video_files]


def result_to_dict(result) -> Dict[str, Any]:
    """
    Convert PipelineResult to dictionary for JSON serialization.
    
    Args:
        result: PipelineResult object
        
    Returns:
        Dictionary representation
    """
    if result is None:
        return {"error": "Processing failed", "status": "failed"}
    
    return {
        "status": "success",
        "video_path": result.video_path,
        "metadata": {
            "duration": result.metadata.duration,
            "fps": result.metadata.fps,
            "width": result.metadata.width,
            "height": result.metadata.height
        },
        "scenes": [
            {
                "scene_id": scene.scene_id,
                "start_time": scene.start_time,
                "end_time": scene.end_time
            }
            for scene in result.scenes
        ],
        "selected_frames": [
            {
                "timestamp": frame.timestamp,
                "scene_id": frame.scene_id,
                "importance_score": frame.importance_score
            }
            for frame in result.selected_frames
        ],
        "pipeline_stats": {
            "total_frames_sampled": result.total_frames_sampled,
            "frames_after_phash": result.frames_after_phash,
            "frames_after_ssim": result.frames_after_ssim,
            "frames_after_clip": result.frames_after_clip,
            "final_frame_count": result.final_frame_count,
            "reduction_rate": result.reduction_rate,
            "processing_time_s": result.processing_time_s
        },
        "extraction": result.extraction_result if result.extraction_result else None
    }


def save_results(results: List[Dict], output_path: str):
    """
    Save results to JSON file.
    
    Args:
        results: List of result dictionaries
        output_path: Output file path
    """
    output_file = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare output data
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_videos": len(results),
            "successful": sum(1 for r in results if r.get("status") == "success"),
            "failed": sum(1 for r in results if r.get("status") == "failed")
        },
        "results": results
    }
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {output_file}")


def print_summary(results: List[Dict]):
    """
    Print summary statistics.
    
    Args:
        results: List of result dictionaries
    """
    print("\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "failed"]
    
    print(f"\nTotal videos: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\n" + "-"*80)
        print("SUCCESSFUL VIDEOS")
        print("-"*80)
        
        total_reduction = 0
        total_time = 0
        
        for result in successful:
            video_name = Path(result['video_path']).name
            stats = result['pipeline_stats']
            
            print(f"\n{video_name}:")
            print(f"  Duration: {result['metadata']['duration']:.1f}s")
            print(f"  Scenes: {len(result['scenes'])}")
            print(f"  Frames: {stats['total_frames_sampled']} → {stats['final_frame_count']}")
            print(f"  Reduction: {stats['reduction_rate']:.1%}")
            print(f"  Processing time: {stats['processing_time_s']:.1f}s")
            
            if result['extraction']:
                extraction = result['extraction']
                print(f"  Brand: {extraction.get('brand', {}).get('name', 'N/A')}")
                print(f"  Ad type: {extraction.get('_metadata', {}).get('ad_type', 'N/A')}")
            
            total_reduction += stats['reduction_rate']
            total_time += stats['processing_time_s']
        
        print("\n" + "-"*80)
        print("AGGREGATE STATISTICS")
        print("-"*80)
        print(f"Average reduction rate: {total_reduction / len(successful):.1%}")
        print(f"Total processing time: {total_time:.1f}s")
        print(f"Average time per video: {total_time / len(successful):.1f}s")
    
    if failed:
        print("\n" + "-"*80)
        print("FAILED VIDEOS")
        print("-"*80)
        for result in failed:
            video_name = Path(result.get('video_path', 'unknown')).name
            print(f"  - {video_name}")
    
    print("\n" + "="*80)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Find videos
    logger.info(f"Searching for videos in: {args.input}")
    video_paths = find_videos(args.input, args.extensions)
    
    if not video_paths:
        logger.error(f"No video files found in: {args.input}")
        logger.error(f"Searched for extensions: {args.extensions}")
        sys.exit(1)
    
    logger.info(f"Found {len(video_paths)} video(s)")
    for path in video_paths:
        logger.info(f"  - {Path(path).name}")
    
    # Initialize pipeline
    logger.info(f"Initializing pipeline with config: {args.config}")
    try:
        pipeline = AdVideoPipeline(config_path=args.config)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Process videos
    logger.info(f"Processing {len(video_paths)} video(s) with {args.workers} worker(s)")
    logger.info(f"LLM extraction: {'DISABLED' if args.skip_extraction else 'ENABLED'}")
    
    try:
        results = pipeline.process_batch(
            video_paths=video_paths,
            max_workers=args.workers,
            skip_extraction=args.skip_extraction
        )
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Convert results to dictionaries
    logger.info("Converting results to JSON format")
    result_dicts = [result_to_dict(r) for r in results]
    
    # Save results
    logger.info(f"Saving results to: {args.output}")
    try:
        save_results(result_dicts, args.output)
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        sys.exit(1)
    
    # Print summary
    print_summary(result_dicts)
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()