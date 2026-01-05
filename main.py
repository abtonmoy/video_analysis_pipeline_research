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

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.logging import RichHandler
from rich.theme import Theme
from rich import box
from rich.text import Text

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

# Setup Rich console with custom theme
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "highlight": "magenta",
})
console = Console(theme=custom_theme)

# Setup logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(console=console, rich_tracebacks=True, show_path=False),
        logging.FileHandler('results/processing.log')
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
    """Load progress from progress file."""
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
            console.print(f"[warning]Could not load progress file: {e}[/warning]")
    
    return {
        "processed_videos": set(),
        "failed_videos": set(),
        "started_at": datetime.now().isoformat(),
        "last_updated": None
    }


def save_progress(progress: Dict[str, Any]):
    """Save progress to progress file."""
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
    """Load existing results from results file."""
    results_path = Path(RESULTS_FILE)
    
    if results_path.exists():
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            console.print(f"[warning]Could not load results file: {e}[/warning]")
    
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
    """Save results to results file."""
    results_path = Path(RESULTS_FILE)
    
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
    """Convert PipelineResult to dictionary for JSON serialization."""
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


def print_header():
    """Print application header."""
    header = Text()
    header.append("Video Advertisement Batch Processor", style="bold cyan")
    
    console.print(Panel(
        header,
        subtitle="[dim]Incremental Processing with Resume Support[/dim]",
        box=box.DOUBLE,
        padding=(1, 2)
    ))


def print_config_info(video_count: int, skip_extraction: bool):
    """Print configuration information."""
    table = Table(show_header=False, box=box.ROUNDED, padding=(0, 2))
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Input Directory", INPUT_DIR)
    table.add_row("Videos Found", str(video_count))
    table.add_row("Output File", RESULTS_FILE)
    table.add_row("LLM Extraction", "[red]Disabled[/red]" if skip_extraction else "[green]Enabled[/green]")
    
    console.print(Panel(table, title="[bold]Configuration[/bold]", box=box.ROUNDED))


def print_video_list(video_paths: List[str]):
    """Print list of videos to process."""
    table = Table(title="Videos to Process", box=box.SIMPLE_HEAD)
    table.add_column("#", style="dim", width=4)
    table.add_column("Filename", style="cyan")
    table.add_column("Size", justify="right", style="green")
    
    for i, path in enumerate(video_paths, 1):
        p = Path(path)
        size_mb = p.stat().st_size / (1024 * 1024)
        table.add_row(str(i), p.name, f"{size_mb:.1f} MB")
    
    console.print(table)
    console.print()


def print_summary(results_data: Dict[str, Any]):
    """Print beautiful summary statistics."""
    results = results_data.get("results", [])
    
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "failed"]
    
    # Summary Panel
    summary_text = Text()
    summary_text.append("Successful: ", style="bold")
    summary_text.append(f"{len(successful)}", style="bold green")
    summary_text.append("  |  ", style="dim")
    summary_text.append("Failed: ", style="bold")
    summary_text.append(f"{len(failed)}", style="bold red")
    summary_text.append("  |  ", style="dim")
    summary_text.append("Total: ", style="bold")
    summary_text.append(f"{len(results)}", style="bold cyan")
    
    console.print(Panel(summary_text, title="[bold]Processing Summary[/bold]", box=box.DOUBLE))
    
    # Successful videos table
    if successful:
        table = Table(
            title="Successfully Processed Videos",
            box=box.ROUNDED,
            show_lines=True,
            title_style="bold green"
        )
        table.add_column("Video", style="cyan", max_width=30)
        table.add_column("Duration", justify="right")
        table.add_column("Scenes", justify="center")
        table.add_column("Frames", justify="center")
        table.add_column("Reduction", justify="right", style="green")
        table.add_column("Time", justify="right", style="yellow")
        table.add_column("Brand", max_width=15)
        
        total_reduction = 0
        total_time = 0
        
        for result in successful:
            video_name = result.get('video_name', 'unknown')
            stats = result.get('pipeline_stats', {})
            metadata = result.get('metadata', {})
            extraction = result.get('extraction', {})
            
            # Get brand name
            brand = "N/A"
            if extraction:
                brand = extraction.get('brand', {}).get('name', 'N/A')
            
            # Frame reduction display
            frames_before = stats.get('total_frames_sampled', 0)
            frames_after = stats.get('final_frame_count', 0)
            frames_display = f"{frames_before} -> {frames_after}"
            
            table.add_row(
                video_name[:30],
                f"{metadata.get('duration', 0):.1f}s",
                str(len(result.get('scenes', []))),
                frames_display,
                f"{stats.get('reduction_rate', 0):.1%}",
                f"{stats.get('processing_time_s', 0):.1f}s",
                brand[:15] if brand else "N/A"
            )
            
            total_reduction += stats.get('reduction_rate', 0)
            total_time += stats.get('processing_time_s', 0)
        
        console.print(table)
        
        # Aggregate stats
        agg_table = Table(show_header=False, box=box.SIMPLE)
        agg_table.add_column("Metric", style="bold")
        agg_table.add_column("Value", style="cyan")
        
        avg_reduction = total_reduction / len(successful) if successful else 0
        avg_time = total_time / len(successful) if successful else 0
        
        agg_table.add_row("Average Reduction Rate", f"{avg_reduction:.1%}")
        agg_table.add_row("Total Processing Time", f"{total_time:.1f}s")
        agg_table.add_row("Average Time per Video", f"{avg_time:.1f}s")
        
        console.print(Panel(agg_table, title="[bold]Aggregate Statistics[/bold]", box=box.ROUNDED))
    
    # Failed videos table
    if failed:
        table = Table(
            title="Failed Videos",
            box=box.ROUNDED,
            title_style="bold red"
        )
        table.add_column("Video", style="cyan")
        table.add_column("Error", style="red")
        
        for result in failed:
            video_name = result.get('video_name', 'unknown')
            error = result.get('error', 'Unknown error')
            table.add_row(video_name, error[:50])
        
        console.print(table)
    
    # Output files info
    console.print()
    files_panel = Panel(
        f"Results:  [cyan]{RESULTS_FILE}[/cyan]\n"
        f"Progress: [cyan]{PROGRESS_FILE}[/cyan]\n"
        f"Log:      [cyan]results/processing.log[/cyan]",
        title="[bold]Output Files[/bold]",
        box=box.ROUNDED
    )
    console.print(files_panel)


def process_videos(
    video_paths: List[str],
    pipeline: AdVideoPipeline,
    progress: Dict[str, Any],
    results_data: Dict[str, Any],
    skip_extraction: bool = False
):
    """Process videos with rich progress bar and incremental saving."""
    
    # Filter out already processed videos
    pending_videos = [
        vp for vp in video_paths 
        if vp not in progress["processed_videos"] and vp not in progress["failed_videos"]
    ]
    
    already_done = len(video_paths) - len(pending_videos)
    
    if already_done > 0:
        console.print(Panel(
            f"[cyan]Resuming:[/cyan] {already_done} videos already processed, "
            f"[yellow]{len(pending_videos)}[/yellow] remaining",
            box=box.ROUNDED
        ))
    
    if not pending_videos:
        console.print("[success]All videos have been processed![/success]")
        return
    
    # Update total count
    results_data["metadata"]["total_videos"] = len(video_paths)
    
    # Process with rich progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TextColumn("[dim]|[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]|[/dim]"),
        TimeRemainingColumn(),
        console=console,
        expand=False
    ) as progress_bar:
        
        overall_task = progress_bar.add_task(
            "[cyan]Overall Progress",
            total=len(video_paths),
            completed=already_done
        )
        
        current_task = progress_bar.add_task(
            "[yellow]Current Video",
            total=100,
            visible=True
        )
        
        for video_path in pending_videos:
            video_name = Path(video_path).name
            progress_bar.update(current_task, description=f"[yellow]{video_name[:40]}", completed=0)
            
            try:
                logger.info(f"Processing: {video_name}")
                progress_bar.update(current_task, completed=10)
                
                result = pipeline.process(video_path, skip_extraction=skip_extraction)
                progress_bar.update(current_task, completed=90)
                
                # Convert to dict and add to results
                result_dict = result_to_dict(result, video_path)
                results_data["results"].append(result_dict)
                
                # Update progress
                progress["processed_videos"].add(video_path)
                
                progress_bar.update(current_task, completed=100)
                
                console.print(
                    f"  [success][+][/success] {video_name} - "
                    f"[cyan]{result.final_frame_count}[/cyan] frames, "
                    f"[green]{result.reduction_rate:.1%}[/green] reduction, "
                    f"[yellow]{result.processing_time_s:.1f}s[/yellow]"
                )
                
            except KeyboardInterrupt:
                console.print("\n[warning]Processing interrupted by user[/warning]")
                save_progress(progress)
                save_results(results_data)
                console.print("[info]Progress saved. Run again to resume.[/info]")
                raise
                
            except Exception as e:
                logger.error(f"Failed to process {video_name}: {e}")
                
                result_dict = {
                    "status": "failed",
                    "video_path": video_path,
                    "video_name": video_name,
                    "error": str(e),
                    "processed_at": datetime.now().isoformat()
                }
                results_data["results"].append(result_dict)
                progress["failed_videos"].add(video_path)
                
                console.print(f"  [error][x][/error] {video_name} - [red]{str(e)[:50]}[/red]")
            
            # Save progress and results after each video
            save_progress(progress)
            save_results(results_data)
            
            progress_bar.update(overall_task, advance=1)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure output directory exists
    ensure_output_dir()
    
    # Print header
    print_header()
    
    # Handle reset flag
    if args.reset:
        console.print("[warning]Resetting progress...[/warning]")
        progress_path = Path(PROGRESS_FILE)
        results_path = Path(RESULTS_FILE)
        
        if progress_path.exists():
            progress_path.unlink()
        if results_path.exists():
            results_path.unlink()
        
        console.print("[success]Progress reset complete[/success]\n")
    
    # Find videos
    console.print(f"[info]Searching for videos in: {INPUT_DIR}[/info]")
    
    try:
        video_paths = find_videos(INPUT_DIR, VIDEO_EXTENSIONS)
    except ValueError as e:
        console.print(f"[error]Error: {e}[/error]")
        sys.exit(1)
    
    if not video_paths:
        console.print(f"[error]No video files found in: {INPUT_DIR}[/error]")
        console.print(f"[dim]Searched for extensions: {VIDEO_EXTENSIONS}[/dim]")
        sys.exit(1)
    
    # Print config and video list
    print_config_info(len(video_paths), args.skip_extraction)
    print_video_list(video_paths)
    
    # Load progress and results
    progress = load_progress()
    results_data = load_results()
    
    # Initialize pipeline
    console.print("[info]Initializing pipeline...[/info]")
    try:
        pipeline = AdVideoPipeline(config_path=CONFIG_PATH)
        console.print("[success]Pipeline initialized[/success]\n")
    except Exception as e:
        console.print(f"[error]Failed to initialize pipeline: {e}[/error]")
        sys.exit(1)
    
    # Process videos
    console.print(Panel("[bold cyan]Starting Video Processing[/bold cyan]", box=box.DOUBLE))
    
    try:
        process_videos(
            video_paths=video_paths,
            pipeline=pipeline,
            progress=progress,
            results_data=results_data,
            skip_extraction=args.skip_extraction
        )
    except KeyboardInterrupt:
        console.print("\n[info]Processing stopped. Run again to resume.[/info]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[error]Batch processing failed: {e}[/error]")
        console.print_exception()
        
        save_progress(progress)
        save_results(results_data)
        sys.exit(1)
    
    # Final save
    save_progress(progress)
    save_results(results_data)
    
    # Print summary
    console.print()
    print_summary(results_data)
    
    console.print("\n[success]Processing complete![/success]")


if __name__ == "__main__":
    main()