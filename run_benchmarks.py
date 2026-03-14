# run_benchmarks.py
"""
Batch benchmark runner with parallel processing, incremental saving, and resume capability.

Similar to main.py but designed for running benchmarks across multiple videos.

Usage:
    python run_benchmarks.py --video_dir data/ads --pipeline_results results/analysis.json
    python run_benchmarks.py --video_dir data/ads --pipeline_results results/analysis.json --workers 8
    python run_benchmarks.py --video_dir data/ads --pipeline_results results/analysis.json --selection_only
    python run_benchmarks.py --video_dir data/ads --pipeline_results results/analysis.json --reset
    python run_benchmarks.py --video_dir data/ads --pipeline_results results/analysis.json --methods uniform_1fps histogram
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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.logging import RichHandler
from rich.theme import Theme
from rich import box
from rich.text import Text

from dotenv import load_dotenv
load_dotenv()

# Configuration
DEFAULT_VIDEO_DIR = "data/ads"
DEFAULT_PIPELINE_RESULTS = "test_results/processing_results.json"
OUTPUT_DIR = "test_results"
RESULTS_FILE = "test_results/benchmark_results.json"
PROGRESS_FILE = "test_results/benchmark_progress.json"
CONFIG_PATH = "config/benchmark.yaml"
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

    log_file = Path(OUTPUT_DIR) / 'benchmark.log'
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    console_handler = RichHandler(console=console, rich_tracebacks=True, show_path=False)
    console_handler.setLevel(level)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        datefmt="[%(time)s]",
        handlers=[file_handler, console_handler]
    )

    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch benchmark frame-selection methods with parallel processing',
    )

    parser.add_argument(
        '--video_dir',
        type=str,
        default=DEFAULT_VIDEO_DIR,
        help=f'Directory containing videos to benchmark (default: {DEFAULT_VIDEO_DIR})'
    )

    parser.add_argument(
        '--pipeline_results',
        type=str,
        default=DEFAULT_PIPELINE_RESULTS,
        help=f'Path to existing pipeline results JSON (default: {DEFAULT_PIPELINE_RESULTS})'
    )

    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip LLM extraction (only compute frame selection metrics)'
    )

    parser.add_argument(
        '--selection-only',
        action='store_true',
        help='Only compute frame selection metrics, no LLM calls (same as --skip-extraction)'
    )

    parser.add_argument(
        '--bare-only',
        action='store_true',
        help='Only run bare extraction (1 LLM call per baseline)'
    )

    parser.add_argument(
        '--full-only',
        action='store_true',
        help='Only run full extraction (2 LLM calls per baseline)'
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
        '--methods',
        type=str,
        nargs='*',
        default=None,
        help='Specific baselines to run (e.g., uniform_1fps histogram clip_only)'
    )

    parser.add_argument(
        '--skip-gpu',
        action='store_true',
        help='Skip GPU-dependent methods (clip_only, kmeans)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Path to the output results JSON file (directories will be created automatically)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=CONFIG_PATH,
        help=f'Benchmark config file (default: {CONFIG_PATH})'
    )

    return parser.parse_args()


def ensure_output_dir():
    """Ensure output directory exists."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def ensure_pipeline_results(path: str):
    """Create an empty pipeline results file if it doesn't exist."""
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({"results": []}, f, indent=2)
        console.print(f"[info]Created empty pipeline results file: [cyan]{path}[/cyan][/info]")


def find_videos(video_dir: str, extensions: List[str]) -> List[str]:
    """Find all video files in directory."""
    video_path = Path(video_dir)

    if not video_path.exists():
        raise ValueError(f"Video directory does not exist: {video_dir}")

    if not video_path.is_dir():
        raise ValueError(f"Video path is not a directory: {video_dir}")

    video_files = []

    for ext in extensions:
        video_files.extend(video_path.glob(f"*{ext}"))
        video_files.extend(video_path.glob(f"*{ext.upper()}"))

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
    """Save progress to progress file, creating directories if needed."""
    progress_path = Path(PROGRESS_FILE)

    # Ensure parent directory exists
    progress_path.parent.mkdir(parents=True, exist_ok=True)

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
            "video_directory": DEFAULT_VIDEO_DIR,
            "pipeline_results": DEFAULT_PIPELINE_RESULTS,
            "total_videos": 0,
            "successful": 0,
            "failed": 0,
            "total_llm_calls": 0
        },
        "per_video": {}
    }


def save_results(results_data: Dict[str, Any]):
    """Save results to results file, creating directories if needed."""
    results_path = Path(RESULTS_FILE)

    # Ensure parent directory exists
    results_path.parent.mkdir(parents=True, exist_ok=True)

    results_data["metadata"]["last_updated"] = datetime.now().isoformat()
    results_data["metadata"]["successful"] = sum(
        1 for vname, r in results_data["per_video"].items()
        if "error" not in r
    )
    results_data["metadata"]["failed"] = sum(
        1 for vname, r in results_data["per_video"].items()
        if "error" in r
    )

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)


def print_header():
    """Print application header."""
    header = Text()
    header.append("Benchmark Batch Runner", style="bold cyan")
    header.append(" | ", style="dim")
    header.append("Parallel Mode", style="bold yellow")

    console.print(Panel(
        header,
        subtitle="[dim]Incremental Processing with Resume Support[/dim]",
        box=box.DOUBLE,
        padding=(1, 2)
    ))


def print_config_info(video_count: int, pending_count: int, args, workers: int):
    """Print configuration information."""
    table = Table(show_header=False, box=box.ROUNDED, padding=(0, 2))
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Video Directory", args.video_dir)
    table.add_row("Pipeline Results", args.pipeline_results)
    table.add_row("Total Videos", str(video_count))
    table.add_row("Pending Videos", f"[yellow]{pending_count}[/yellow]")
    table.add_row("Output File", RESULTS_FILE)
    table.add_row("Parallel Workers", f"[yellow]{workers}[/yellow]")

    # Extraction mode
    if args.selection_only or args.skip_extraction:
        mode = "[green]Selection Only (no LLM)[/green]"
    elif args.bare_only:
        mode = "[yellow]Bare Only (1 LLM call)[/yellow]"
    elif args.full_only:
        mode = "[red]Full Only (2 LLM calls)[/red]"
    else:
        mode = "[magenta]Full (bare + full)[/magenta]"

    table.add_row("Extraction Mode", mode)

    # Methods
    methods_str = ", ".join(args.methods) if args.methods else "All baselines"
    table.add_row("Methods", methods_str)

    # GPU skip
    if args.skip_gpu:
        table.add_row("GPU Methods", "[red]Skipped[/red]")

    console.print(Panel(table, title="[bold]Configuration[/bold]", box=box.ROUNDED))


def print_video_list(video_paths: List[str], max_display: int = 15):
    """Print list of videos to process."""
    if not video_paths:
        return

    table = Table(title=f"Videos to Benchmark ({len(video_paths)} total)", box=box.SIMPLE_HEAD)
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
    per_video = results_data.get("per_video", {})

    successful = [(k, v) for k, v in per_video.items() if "error" not in v]
    failed = [(k, v) for k, v in per_video.items() if "error" in v]

    # Summary Panel
    summary_text = Text()
    summary_text.append("Successful: ", style="bold")
    summary_text.append(f"{len(successful)}", style="bold green")
    summary_text.append("  |  ", style="dim")
    summary_text.append("Failed: ", style="bold")
    summary_text.append(f"{len(failed)}", style="bold red")
    summary_text.append("  |  ", style="dim")
    summary_text.append("Total: ", style="bold")
    summary_text.append(f"{len(per_video)}", style="bold cyan")

    console.print(Panel(summary_text, title="[bold]Benchmark Summary[/bold]", box=box.DOUBLE))

    # Successful videos table (show last 20)
    if successful:
        display_results = successful[-20:] if len(successful) > 20 else successful

        table = Table(
            title=f"Successfully Benchmarked (last {len(display_results)} of {len(successful)})",
            box=box.ROUNDED,
            show_lines=False,
            title_style="bold green"
        )
        table.add_column("Video", style="cyan", max_width=35)
        table.add_column("Duration", justify="right")
        table.add_column("Baselines", justify="center")
        table.add_column("LLM Calls", justify="right", style="yellow")

        for vname, result in display_results:
            baselines = result.get("baselines", {})
            n_baselines = len(baselines)
            llm_calls = sum(
                1 for bdata in baselines.values()
                if isinstance(bdata, dict) and ("bare_extraction" in bdata or "full_extraction" in bdata)
            ) * 2  # Approximate

            table.add_row(
                vname[:35],
                f"{result.get('video_metadata', {}).get('duration', 0):.1f}s",
                str(n_baselines),
                str(llm_calls),
            )

        console.print(table)

    # Failed videos table
    if failed:
        table = Table(
            title=f"Failed Videos ({len(failed)})",
            box=box.ROUNDED,
            title_style="bold red"
        )
        table.add_column("Video", style="cyan")
        table.add_column("Error", style="red", max_width=50)

        for vname, result in failed[:20]:
            error = result.get('error', 'Unknown error')
            table.add_row(vname, error[:50])

        if len(failed) > 20:
            table.add_row(f"... and {len(failed) - 20} more", "")

        console.print(table)

    # Output files info
    console.print()
    files_panel = Panel(
        f"Results:  [cyan]{RESULTS_FILE}[/cyan]\n"
        f"Progress: [cyan]{PROGRESS_FILE}[/cyan]\n"
        f"Log:      [cyan]{Path(OUTPUT_DIR) / 'benchmark.log'}[/cyan]",
        title="[bold]Output Files[/bold]",
        box=box.ROUNDED
    )
    console.print(files_panel)


# Module-level worker function for multiprocessing compatibility
def _worker_process_video(
    video_path: str,
    pipeline_results_path: str,
    config: Dict[str, Any],
    selection_only: bool,
    bare_only: bool,
    full_only: bool,
    skip_gpu: bool,
    methods: Optional[List[str]],
    output_dir: str,
) -> tuple:
    """Worker function for parallel processing (must be module-level for pickling)."""
    try:
        # Setup logging in worker process
        log_file = Path(output_dir) / 'benchmark.log'
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(str(log_file))],
            force=True,
        )

        from benchmarks.runner import BenchmarkRunner

        # Create a temporary runner for this video
        runner = BenchmarkRunner(
            config=config,
            pipeline_results_path=pipeline_results_path,
            output_dir=output_dir,
            methods=methods,
            skip_gpu=skip_gpu,
            selection_only=selection_only,
            bare_only=bare_only,
            full_only=full_only,
        )

        # Process single video
        results = runner.run([video_path])

        video_name = Path(video_path).name
        result = results["per_video"].get(video_name, {"error": "No result returned"})
        return (video_name, result, None)
    except Exception as e:
        video_name = Path(video_path).name
        return (video_name, None, str(e))


def process_videos_parallel(
    video_paths: List[str],
    progress_data: Dict[str, Any],
    results_data: Dict[str, Any],
    args,
    num_workers: int = 4,
):
    """Process videos in parallel with progress display."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from src.utils.config import load_config, deep_merge

    # Filter out already processed videos
    pending_videos = [
        vp for vp in video_paths
        if Path(vp).name not in progress_data["processed_videos"]
        and Path(vp).name not in progress_data["failed_videos"]
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
        console.print("[success]All videos have been benchmarked![/success]")
        return

    # Update metadata
    results_data["metadata"]["total_videos"] = len(video_paths)
    results_data["metadata"]["video_directory"] = args.video_dir
    results_data["metadata"]["pipeline_results"] = args.pipeline_results

    # Stats tracking
    stats = {
        "successful": already_done,
        "failed": already_failed,
        "total": len(video_paths)
    }

    console.print(f"\n[info]Initializing {num_workers} worker processes...[/info]")
    console.print("[dim]This may take a minute on first run...[/dim]\n")

    # Load config once (will be passed to workers)
    config = {}
    default_path = Path("config/default.yaml")
    if default_path.exists():
        config = load_config(str(default_path))

    config_path = Path(args.config)
    if config_path.exists():
        bench_config = load_config(str(config_path))
        config = deep_merge(config, bench_config)

        # Manually promote benchmark extraction overrides to top level
        if "benchmark" in bench_config and "extraction" in bench_config["benchmark"]:
            if "extraction" not in config:
                config["extraction"] = {}
            for k, v in bench_config["benchmark"]["extraction"].items():
                config["extraction"][k] = v

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
            "[cyan]Benchmarking videos",
            total=len(video_paths),
            completed=already_done + already_failed
        )

        try:
            # Use ProcessPoolExecutor for parallel processing
            # Note: Each worker will initialize its own models
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks - use module-level worker function
                future_to_video = {
                    executor.submit(
                        _worker_process_video,
                        vp,
                        args.pipeline_results,
                        config,
                        (args.selection_only or args.skip_extraction),
                        args.bare_only,
                        args.full_only,
                        args.skip_gpu,
                        args.methods,
                        OUTPUT_DIR,
                    ): vp
                    for vp in pending_videos
                }

                console.print(f"[success]Workers initialized! Starting benchmarking...[/success]\n")

                # Process completed tasks
                for future in as_completed(future_to_video):
                    video_path = future_to_video[future]
                    video_name = Path(video_path).name

                    try:
                        vname, result, error = future.result()

                        if error:
                            # Failed
                            progress_data["failed_videos"].add(video_name)
                            stats["failed"] += 1
                            results_data["per_video"][video_name] = {"error": error}

                            console.print(
                                f"  [error][x][/error] {video_name} - "
                                f"[red]{error[:60]}[/red]"
                            )
                        else:
                            # Success
                            progress_data["processed_videos"].add(video_name)
                            stats["successful"] += 1
                            results_data["per_video"][video_name] = result

                            baselines = result.get("baselines", {})
                            n_baselines = len(baselines)

                            console.print(
                                f"  [success][+][/success] {video_name} - "
                                f"[cyan]{n_baselines}[/cyan] baselines"
                            )

                        # Save incrementally
                        save_progress(progress_data)
                        save_results(results_data)

                        # Update progress bar
                        progress.update(task, advance=1)

                    except Exception as e:
                        console.print(
                            f"  [error][!] {video_name} - Unexpected error: {e}[/error]"
                        )
                        progress_data["failed_videos"].add(video_name)
                        stats["failed"] += 1
                        results_data["per_video"][video_name] = {"error": str(e)}
                        progress.update(task, advance=1)

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
    global OUTPUT_DIR, RESULTS_FILE, PROGRESS_FILE

    # Required for Windows multiprocessing
    if sys.platform == 'win32':
        import multiprocessing
        multiprocessing.freeze_support()

    args = parse_args()

    # Handle custom output path
    if args.output:
        RESULTS_FILE = args.output
        output_path = Path(RESULTS_FILE)
        OUTPUT_DIR = str(output_path.parent)
        PROGRESS_FILE = str(output_path.parent / "benchmark_progress.json")

        # Create parent directories for custom output path if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        console.print(f"[info]Output directory created/verified: [cyan]{OUTPUT_DIR}[/cyan][/info]")
    else:
        # Ensure default output directory exists
        ensure_output_dir()

    # Ensure pipeline results file exists
    ensure_pipeline_results(args.pipeline_results)

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
    console.print(f"[info]Searching for videos in: {args.video_dir}[/info]")

    try:
        video_paths = find_videos(args.video_dir, VIDEO_EXTENSIONS)
    except ValueError as e:
        console.print(f"[error]Error: {e}[/error]")
        sys.exit(1)

    if not video_paths:
        console.print(f"[error]No video files found in: {args.video_dir}[/error]")
        console.print(f"[dim]Searched for extensions: {VIDEO_EXTENSIONS}[/dim]")
        sys.exit(1)

    # Load progress and results
    progress_data = load_progress()
    results_data = load_results()

    # Calculate pending
    pending_videos = [
        vp for vp in video_paths
        if Path(vp).name not in progress_data["processed_videos"]
        and Path(vp).name not in progress_data["failed_videos"]
    ]

    # Print config and video list
    print_config_info(len(video_paths), len(pending_videos), args, args.workers)
    print_video_list(pending_videos)

    if not pending_videos:
        console.print("[success]All videos have been benchmarked![/success]\n")
        print_summary(results_data)
        return

    # Process videos
    console.print(Panel(
        f"[bold cyan]Starting Parallel Benchmarking[/bold cyan]\n"
        f"[dim]{args.workers} workers | {len(pending_videos)} videos[/dim]",
        box=box.DOUBLE
    ))

    start_time = time.time()

    try:
        process_videos_parallel(
            video_paths=video_paths,
            progress_data=progress_data,
            results_data=results_data,
            args=args,
            num_workers=args.workers,
        )
    except KeyboardInterrupt:
        console.print("\n[info]Processing stopped. Run again to resume.[/info]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[error]Batch benchmarking failed: {e}[/error]")
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

    console.print(f"\n[success]Benchmarking complete in {total_time/60:.1f} minutes![/success]")


if __name__ == "__main__":
    main()