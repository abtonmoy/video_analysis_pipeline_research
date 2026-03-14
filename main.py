# main.py
"""
Batch video advertisement analysis with parallel processing, incremental saving, and resume capability.

Usage:
    python main.py
    python main.py --skip-extraction  # No LLM calls
    python main.py --reset  # Start fresh, ignore previous progress
    python main.py --workers 8  # Use 8 parallel workers
    python main.py --output results/my_custom_results.json  # Save results to a custom file
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
import sys
import os
import warnings
import time

# Suppress warnings early
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.logging import RichHandler
from rich.theme import Theme
from rich import box
from rich.text import Text
from rich.live import Live
from rich.layout import Layout

from dotenv import load_dotenv
load_dotenv()

# Configuration
INPUT_DIR = "data/hussain_videos"
OUTPUT_DIR = "main_results"
RESULTS_FILE = "main_results/processing_results.json"
PROGRESS_FILE = "main_results/progress.json"
CONFIG_PATH = "config/default.yaml"
VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
DEFAULT_WORKERS = 4

# Setup Rich console with custom theme
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "highlight": "magenta",
    "dim": "dim white",
})
console = Console(theme=custom_theme)


def setup_logging(verbose: bool = False):
    """Setup logging."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    level = logging.DEBUG if verbose else logging.WARNING
    
    # Suppress noisy loggers
    for logger_name in ["whisper", "torch", "PIL", "urllib3", "open_clip", "lpips", 
                        "matplotlib", "numba", "filelock", "transformers", "multiprocessing"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    log_file = Path(OUTPUT_DIR) / 'processing.log'
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    console_handler = RichHandler(console=console, rich_tracebacks=True, show_path=False)
    console_handler.setLevel(level)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[file_handler, console_handler]
    )
    
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch process video advertisements with parallel processing',
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
        '--workers',
        type=int,
        default=DEFAULT_WORKERS,
        help=f'Number of parallel workers (default: {DEFAULT_WORKERS})'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--batch',
        type=str,
        help='Directory containing videos to process in batch'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to the output results JSON file (directories will be created automatically)'
    )
    
    return parser.parse_args()


def ensure_output_dir():
    """Ensure output directory exists."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def find_videos(input_dir: str, extensions: List[str]) -> List[str]:
    """Find all video files in directory."""
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


def print_header():
    """Print application header."""
    header = Text()
    header.append("Video Advertisement Batch Processor", style="bold cyan")
    header.append(" | ", style="dim")
    header.append("Parallel Mode", style="bold yellow")
    
    console.print(Panel(
        header,
        subtitle="[dim]Incremental Processing with Resume Support[/dim]",
        box=box.DOUBLE,
        padding=(1, 2)
    ))


def print_config_info(video_count: int, pending_count: int, skip_extraction: bool, workers: int):
    """Print configuration information."""
    table = Table(show_header=False, box=box.ROUNDED, padding=(0, 2))
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Input Directory", INPUT_DIR)
    table.add_row("Total Videos", str(video_count))
    table.add_row("Pending Videos", f"[yellow]{pending_count}[/yellow]")
    table.add_row("Output File", RESULTS_FILE)
    table.add_row("Parallel Workers", f"[yellow]{workers}[/yellow]")
    table.add_row("LLM Extraction", "[red]Disabled[/red]" if skip_extraction else "[green]Enabled[/green]")
    
    console.print(Panel(table, title="[bold]Configuration[/bold]", box=box.ROUNDED))


def print_video_list(video_paths: List[str], max_display: int = 15):
    """Print list of videos to process."""
    if not video_paths:
        return
        
    table = Table(title=f"Videos to Process ({len(video_paths)} total)", box=box.SIMPLE_HEAD)
    table.add_column("#", style="dim", width=4)
    table.add_column("Filename", style="cyan")
    table.add_column("Size", justify="right", style="green")
    
    display_paths = video_paths[:max_display]
    
    for i, path in enumerate(display_paths, 1):
        p = Path(path)
        size_mb = p.stat().st_size / (1024 * 1024)
        table.add_row(str(i), p.name, f"{size_mb:.1f} MB")
    
    if len(video_paths) > max_display:
        table.add_row("...", f"[dim]and {len(video_paths) - max_display} more[/dim]", "")
    
    console.print(table)
    console.print()


def print_summary(results_data: Dict[str, Any]):
    """Print summary statistics."""
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
    
    # Successful videos table (show last 20)
    if successful:
        display_results = successful[-20:] if len(successful) > 20 else successful
        
        table = Table(
            title=f"Successfully Processed (last {len(display_results)} of {len(successful)})",
            box=box.ROUNDED,
            show_lines=False,
            title_style="bold green"
        )
        table.add_column("Video", style="cyan", max_width=35)
        table.add_column("Duration", justify="right")
        table.add_column("Frames", justify="center")
        table.add_column("Reduction", justify="right", style="green")
        table.add_column("Time", justify="right", style="yellow")
        
        total_reduction = 0
        total_time = 0
        
        for result in successful:
            total_reduction += result.get('pipeline_stats', {}).get('reduction_rate', 0)
            total_time += result.get('pipeline_stats', {}).get('processing_time_s', 0)
        
        for result in display_results:
            video_name = result.get('video_name', 'unknown')
            stats = result.get('pipeline_stats', {})
            metadata = result.get('metadata', {})
            
            frames_before = stats.get('total_frames_sampled', 0)
            frames_after = stats.get('final_frame_count', 0)
            frames_display = f"{frames_before}->{frames_after}"
            
            table.add_row(
                video_name[:35],
                f"{metadata.get('duration', 0):.1f}s",
                frames_display,
                f"{stats.get('reduction_rate', 0):.1%}",
                f"{stats.get('processing_time_s', 0):.1f}s",
            )
        
        console.print(table)
        
        # Aggregate stats
        agg_table = Table(show_header=False, box=box.SIMPLE)
        agg_table.add_column("Metric", style="bold")
        agg_table.add_column("Value", style="cyan")
        
        avg_reduction = total_reduction / len(successful) if successful else 0
        avg_time = total_time / len(successful) if successful else 0
        
        agg_table.add_row("Average Reduction Rate", f"{avg_reduction:.1%}")
        agg_table.add_row("Total Processing Time", f"{total_time:.1f}s ({total_time/60:.1f} min)")
        agg_table.add_row("Average Time per Video", f"{avg_time:.1f}s")
        
        console.print(Panel(agg_table, title="[bold]Aggregate Statistics[/bold]", box=box.ROUNDED))
    
    # Failed videos table
    if failed:
        table = Table(
            title=f"Failed Videos ({len(failed)})",
            box=box.ROUNDED,
            title_style="bold red"
        )
        table.add_column("Video", style="cyan")
        table.add_column("Error", style="red", max_width=50)
        
        for result in failed[:20]:  # Show first 20 failures
            video_name = result.get('video_name', 'unknown')
            error = result.get('error', 'Unknown error')
            table.add_row(video_name, error[:50])
        
        if len(failed) > 20:
            table.add_row(f"... and {len(failed) - 20} more", "")
        
        console.print(table)
    
    # Output files info
    console.print()
    files_panel = Panel(
        f"Results:  [cyan]{RESULTS_FILE}[/cyan]\n"
        f"Progress: [cyan]{PROGRESS_FILE}[/cyan]\n"
        f"Log:      [cyan]{Path(OUTPUT_DIR) / 'processing.log'}[/cyan]",
        title="[bold]Output Files[/bold]",
        box=box.ROUNDED
    )
    console.print(files_panel)


def process_videos_parallel(
    video_paths: List[str],
    progress_data: Dict[str, Any],
    results_data: Dict[str, Any],
    num_workers: int = 4,
    skip_extraction: bool = False
):
    """Process videos in parallel with progress display."""
    
    # Import parallel pipeline here to avoid multiprocessing issues
    from src.parallel_pipeline import ParallelPipeline, VideoResult
    
    # Filter out already processed videos
    pending_videos = [
        vp for vp in video_paths 
        if vp not in progress_data["processed_videos"] and vp not in progress_data["failed_videos"]
    ]
    
    already_done = len(progress_data["processed_videos"])
    already_failed = len(progress_data["failed_videos"])
    
    if already_done + already_failed > 0:
        console.print(Panel(
            f"[cyan]Resuming:[/cyan] [green]{already_done}[/green] done, "
            f"[red]{already_failed}[/red] failed, "
            f"[yellow]{len(pending_videos)}[/yellow] remaining",
            box=box.ROUNDED
        ))
    
    if not pending_videos:
        console.print("[success]All videos have been processed![/success]")
        return
    
    # Update metadata
    results_data["metadata"]["total_videos"] = len(video_paths)
    
    # Stats tracking
    stats = {
        "successful": already_done,
        "failed": already_failed,
        "total": len(video_paths)
    }
    
    console.print(f"\n[info]Initializing {num_workers} worker processes (loading models)...[/info]")
    console.print("[dim]This may take a minute on first run...[/dim]\n")
    
    # Create progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("[dim]|[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]|[/dim]"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=2,
        transient=False
    ) as progress:
        
        task = progress.add_task(
            "[cyan]Processing videos",
            total=len(video_paths),
            completed=already_done + already_failed
        )
        
        def on_video_complete(result: VideoResult):
            """Callback when a video completes."""
            nonlocal stats
            
            # Update progress tracking
            if result.success:
                progress_data["processed_videos"].add(result.video_path)
                stats["successful"] += 1
            else:
                progress_data["failed_videos"].add(result.video_path)
                stats["failed"] += 1
            
            # Add result
            if result.result_dict:
                results_data["results"].append(result.result_dict)
            
            # Save incrementally
            save_progress(progress_data)
            save_results(results_data)
            
            # Update progress bar
            progress.update(task, advance=1)
            
            # Print result
            if result.success:
                result_dict = result.result_dict or {}
                pipeline_stats = result_dict.get('pipeline_stats', {})
                console.print(
                    f"  [success][+][/success] {result.video_name} - "
                    f"[cyan]{pipeline_stats.get('final_frame_count', '?')}[/cyan] frames, "
                    f"[green]{pipeline_stats.get('reduction_rate', 0):.1%}[/green] reduction, "
                    f"[yellow]{result.processing_time:.1f}s[/yellow]"
                )
            else:
                console.print(
                    f"  [error][x][/error] {result.video_name} - "
                    f"[red]{result.error[:60] if result.error else 'Unknown error'}[/red]"
                )
        
        try:
            # Create and run parallel pipeline
            with ParallelPipeline(
                config_path=CONFIG_PATH,
                num_workers=num_workers,
                suppress_worker_logs=True
            ) as pipeline:
                
                console.print(f"[success]Workers initialized! Starting processing...[/success]\n")
                
                # Process videos
                pipeline.process_batch(
                    video_paths=pending_videos,
                    skip_extraction=skip_extraction,
                    callback=on_video_complete
                )
                
        except KeyboardInterrupt:
            console.print("\n[warning]Processing interrupted by user[/warning]")
            save_progress(progress_data)
            save_results(results_data)
            console.print("[info]Progress saved. Run again to resume.[/info]")
            raise
    
    console.print()
    console.print(
        f"[info]Batch complete: "
        f"[green]{stats['successful']}[/green] successful, "
        f"[red]{stats['failed']}[/red] failed[/info]"
    )


def main():
    """Main entry point."""
    global INPUT_DIR, OUTPUT_DIR, RESULTS_FILE, PROGRESS_FILE
    
    # Required for Windows multiprocessing
    if sys.platform == 'win32':
        import multiprocessing
        multiprocessing.freeze_support()
    
    args = parse_args()
    
    if args.batch:
        INPUT_DIR = args.batch
        
    if args.output:
        RESULTS_FILE = args.output
        output_path = Path(RESULTS_FILE)
        OUTPUT_DIR = str(output_path.parent)
        PROGRESS_FILE = str(output_path.parent / "progress.json")
        
    # Ensure output directory exists (this handles creating the folder if it doesn't exist)
    ensure_output_dir()
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose)
    
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
    
    # Load progress and results
    progress_data = load_progress()
    results_data = load_results()
    
    # Calculate pending
    pending_videos = [
        vp for vp in video_paths 
        if vp not in progress_data["processed_videos"] and vp not in progress_data["failed_videos"]
    ]
    
    # Print config and video list
    print_config_info(len(video_paths), len(pending_videos), args.skip_extraction, args.workers)
    print_video_list(pending_videos)
    
    if not pending_videos:
        console.print("[success]All videos have been processed![/success]\n")
        print_summary(results_data)
        return
    
    # Process videos
    console.print(Panel(
        f"[bold cyan]Starting Parallel Processing[/bold cyan]\n"
        f"[dim]{args.workers} workers | {len(pending_videos)} videos[/dim]",
        box=box.DOUBLE
    ))
    
    start_time = time.time()
    
    try:
        process_videos_parallel(
            video_paths=video_paths,
            progress_data=progress_data,
            results_data=results_data,
            num_workers=args.workers,
            skip_extraction=args.skip_extraction
        )
    except KeyboardInterrupt:
        console.print("\n[info]Processing stopped. Run again to resume.[/info]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[error]Batch processing failed: {e}[/error]")
        import traceback
        traceback.print_exc()
        
        save_progress(progress_data)
        save_results(results_data)
        sys.exit(1)
    
    total_time = time.time() - start_time
    
    # Final save
    save_progress(progress_data)
    save_results(results_data)
    
    # Print summary
    console.print()
    print_summary(results_data)
    
    console.print(f"\n[success]Processing complete in {total_time/60:.1f} minutes![/success]")


if __name__ == "__main__":
    main()