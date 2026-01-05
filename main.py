# main.py
"""
Batch video advertisement analysis with incremental saving and resume capability.

Usage:
    python main.py
    python main.py --skip-extraction  # No LLM calls
    python main.py --reset  # Start fresh, ignore previous progress
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
import sys

from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

from src.pipeline import AdVideoPipeline

# Configuration
INPUT_DIR = "data/hussain_videos"
OUTPUT_DIR = "results"
RESULTS_FILE = "results/processing_results.json"
PROGRESS_FILE = "results/progress.json"
CONFIG_PATH = "config/default.yaml"
VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv', '.webm']

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch process video advertisements with resume capability',
    )
    
    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip LLM extraction (only run pipeline stages 1-6)'
    )
    
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset progress and start from scratch'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def ensure_output_dir():
    """Ensure output directory exists."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def find_videos(input_dir: str, extensions: List[str]) -> List[str]:
    """
    Find all video files in directory.
    
    Args:
        input_dir: Input directory path
        extensions: List of video file extensions
        
    Returns:
        List of video file paths (sorted)
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    video_files = []
    
    for ext in extensions:
        video_files.extend(input_path.glob(f"*{ext}"))
        video_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    video_files = sorted(set(video_files))
    
    return [str(p) for p in video_files]


def load_progress() -> Dict[str, Any]:
    """
    Load progress from progress file.
    
    Returns:
        Progress dictionary with processed videos set
    """
    progress_path = Path(PROGRESS_FILE)
    
    if progress_path.exists():
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    "processed_videos": set(data.get("processed_videos", [])),
                    "failed_videos": set(data.get("failed_videos", [])),
                    "started_at": data.get("started_at"),
                    "last_updated": data.get("last_updated")
                }
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load progress file: {e}")
    
    return {
        "processed_videos": set(),
        "failed_videos": set(),
        "started_at": datetime.now().isoformat(),
        "last_updated": None
    }


def save_progress(progress: Dict[str, Any]):
    """
    Save progress to progress file.
    
    Args:
        progress: Progress dictionary
    """
    progress_path = Path(PROGRESS_FILE)
    
    data = {
        "processed_videos": list(progress["processed_videos"]),
        "failed_videos": list(progress["failed_videos"]),
        "started_at": progress["started_at"],
        "last_updated": datetime.now().isoformat()
    }
    
    with open(progress_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def load_results() -> Dict[str, Any]:
    """
    Load existing results from results file.
    
    Returns:
        Results dictionary
    """
    results_path = Path(RESULTS_FILE)
    
    if results_path.exists():
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load results file: {e}")
    
    return {
        "metadata": {
            "started_at": datetime.now().isoformat(),
            "input_directory": INPUT_DIR,
            "total_videos": 0,
            "successful": 0,
            "failed": 0
        },
        "results": []
    }


def save_results(results_data: Dict[str, Any]):
    """
    Save results to results file.
    
    Args:
        results_data: Results dictionary
    """
    results_path = Path(RESULTS_FILE)
    
    # Update metadata
    results_data["metadata"]["last_updated"] = datetime.now().isoformat()
    results_data["metadata"]["successful"] = sum(
        1 for r in results_data["results"] if r.get("status") == "success"
    )
    results_data["metadata"]["failed"] = sum(
        1 for r in results_data["results"] if r.get("status") == "failed"
    )
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)


def result_to_dict(result, video_path: str) -> Dict[str, Any]:
    """
    Convert PipelineResult to dictionary for JSON serialization.
    
    Args:
        result: PipelineResult object
        video_path: Original video path
        
    Returns:
        Dictionary representation
    """
    if result is None:
        return {
            "status": "failed",
            "video_path": video_path,
            "video_name": Path(video_path).name,
            "error": "Processing failed",
            "processed_at": datetime.now().isoformat()
        }
    
    return {
        "status": "success",
        "video_path": result.video_path,
        "video_name": Path(result.video_path).name,
        "processed_at": datetime.now().isoformat(),
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


def print_summary(results_data: Dict[str, Any]):
    """Print summary statistics."""
    results = results_data.get("results", [])
    
    print("\n" + "=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "failed"]
    
    print(f"\nTotal videos processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\n" + "-" * 80)
        print("SUCCESSFUL VIDEOS")
        print("-" * 80)
        
        total_reduction = 0
        total_time = 0
        
        for result in successful:
            video_name = result.get('video_name', 'unknown')
            stats = result.get('pipeline_stats', {})
            
            print(f"\n{video_name}:")
            print(f"  Duration: {result.get('metadata', {}).get('duration', 0):.1f}s")
            print(f"  Scenes: {len(result.get('scenes', []))}")
            print(f"  Frames: {stats.get('total_frames_sampled', 0)} → {stats.get('final_frame_count', 0)}")
            print(f"  Reduction: {stats.get('reduction_rate', 0):.1%}")
            print(f"  Processing time: {stats.get('processing_time_s', 0):.1f}s")
            
            if result.get('extraction'):
                extraction = result['extraction']
                print(f"  Brand: {extraction.get('brand', {}).get('name', 'N/A')}")
                print(f"  Ad type: {extraction.get('_metadata', {}).get('ad_type', 'N/A')}")
            
            total_reduction += stats.get('reduction_rate', 0)
            total_time += stats.get('processing_time_s', 0)
        
        print("\n" + "-" * 80)
        print("AGGREGATE STATISTICS")
        print("-" * 80)
        print(f"Average reduction rate: {total_reduction / len(successful):.1%}")
        print(f"Total processing time: {total_time:.1f}s")
        print(f"Average time per video: {total_time / len(successful):.1f}s")
    
    if failed:
        print("\n" + "-" * 80)
        print("FAILED VIDEOS")
        print("-" * 80)
        for result in failed:
            video_name = result.get('video_name', 'unknown')
            error = result.get('error', 'Unknown error')
            print(f"  - {video_name}: {error}")
    
    print("\n" + "=" * 80)


def process_videos(
    video_paths: List[str],
    pipeline: AdVideoPipeline,
    progress: Dict[str, Any],
    results_data: Dict[str, Any],
    skip_extraction: bool = False
):
    """
    Process videos with progress bar and incremental saving.
    
    Args:
        video_paths: List of video paths to process
        pipeline: AdVideoPipeline instance
        progress: Progress tracking dictionary
        results_data: Results dictionary
        skip_extraction: Whether to skip LLM extraction
    """
    # Filter out already processed videos
    pending_videos = [
        vp for vp in video_paths 
        if vp not in progress["processed_videos"] and vp not in progress["failed_videos"]
    ]
    
    already_done = len(video_paths) - len(pending_videos)
    
    if already_done > 0:
        logger.info(f"Resuming: {already_done} videos already processed, {len(pending_videos)} remaining")
    
    if not pending_videos:
        logger.info("All videos have been processed!")
        return
    
    # Update total count
    results_data["metadata"]["total_videos"] = len(video_paths)
    
    # Process with progress bar
    with tqdm(
        total=len(video_paths),
        initial=already_done,
        desc="Processing videos",
        unit="video",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    ) as pbar:
        
        for video_path in pending_videos:
            video_name = Path(video_path).name
            pbar.set_postfix_str(f"Current: {video_name[:30]}...")
            
            try:
                logger.info(f"Processing: {video_name}")
                result = pipeline.process(video_path, skip_extraction=skip_extraction)
                
                # Convert to dict and add to results
                result_dict = result_to_dict(result, video_path)
                results_data["results"].append(result_dict)
                
                # Update progress
                progress["processed_videos"].add(video_path)
                
                logger.info(f"Completed: {video_name} - "
                           f"{result.final_frame_count} frames, "
                           f"{result.reduction_rate:.1%} reduction")
                
            except KeyboardInterrupt:
                logger.warning("\nProcessing interrupted by user")
                save_progress(progress)
                save_results(results_data)
                logger.info("Progress saved. Run again to resume.")
                raise
                
            except Exception as e:
                logger.error(f"Failed to process {video_name}: {e}")
                
                # Record failure
                result_dict = {
                    "status": "failed",
                    "video_path": video_path,
                    "video_name": video_name,
                    "error": str(e),
                    "processed_at": datetime.now().isoformat()
                }
                results_data["results"].append(result_dict)
                progress["failed_videos"].add(video_path)
            
            # Save progress and results after each video
            save_progress(progress)
            save_results(results_data)
            
            pbar.update(1)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure output directory exists
    ensure_output_dir()
    
    # Handle reset flag
    if args.reset:
        logger.info("Resetting progress...")
        progress_path = Path(PROGRESS_FILE)
        results_path = Path(RESULTS_FILE)
        
        if progress_path.exists():
            progress_path.unlink()
        if results_path.exists():
            results_path.unlink()
        
        logger.info("Progress reset complete")
    
    # Find videos
    logger.info(f"Searching for videos in: {INPUT_DIR}")
    
    try:
        video_paths = find_videos(INPUT_DIR, VIDEO_EXTENSIONS)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    
    if not video_paths:
        logger.error(f"No video files found in: {INPUT_DIR}")
        logger.error(f"Searched for extensions: {VIDEO_EXTENSIONS}")
        sys.exit(1)
    
    logger.info(f"Found {len(video_paths)} video(s)")
    for path in video_paths:
        logger.info(f"  - {Path(path).name}")
    
    # Load progress and results
    progress = load_progress()
    results_data = load_results()
    
    # Initialize pipeline
    logger.info(f"Initializing pipeline with config: {CONFIG_PATH}")
    try:
        pipeline = AdVideoPipeline(config_path=CONFIG_PATH)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Process videos
    logger.info(f"LLM extraction: {'DISABLED' if args.skip_extraction else 'ENABLED'}")
    
    try:
        process_videos(
            video_paths=video_paths,
            pipeline=pipeline,
            progress=progress,
            results_data=results_data,
            skip_extraction=args.skip_extraction
        )
    except KeyboardInterrupt:
        print("\n")
        logger.info("Processing stopped. Run again to resume from where you left off.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save what we have
        save_progress(progress)
        save_results(results_data)
        sys.exit(1)
    
    # Final save
    save_progress(progress)
    save_results(results_data)
    
    # Print summary
    print_summary(results_data)
    
    # Print output file locations
    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"Progress saved to: {PROGRESS_FILE}")
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()