#!/usr/bin/env python3
"""
Main benchmark runner for unified benchmarks.

Orchestrates frame selection, metric computation, VLM extraction,
and result aggregation across multiple baseline methods.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ub_src.methods import ALL_METHODS, BaseMethod
from ub_src.extraction_wrapper import ExtractionWrapper, decode_all_frames, get_video_info
from ub_src.metrics import compute_selection_metrics, compare_extractions, aggregate_metrics
from ub_src.retry_utils import call_vlm_with_retry_queue

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Orchestrates benchmark execution across videos and baseline methods.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        pipeline_results_path: str,
        output_dir: str,
        methods: Optional[List[str]] = None,
        skip_gpu: bool = False,
        selection_only: bool = False,
        bare_only: bool = False,
        full_only: bool = False,
    ):
        """
        Initialize benchmark runner.

        Args:
            config: Configuration dict
            pipeline_results_path: Path to AdaFrame pipeline results JSON
            output_dir: Directory for output files
            methods: List of method names to run (None = all)
            skip_gpu: Skip GPU-dependent methods
            selection_only: Only compute frame selection metrics
            bare_only: Only run bare extraction
            full_only: Only run full extraction
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.selection_only = selection_only
        self.bare_only = bare_only
        self.full_only = full_only
        self.skip_gpu = skip_gpu

        # Throttle between LLM calls
        self.llm_throttle = config.get("throttle_seconds", 4.0)

        # Determine extraction modes
        self.run_bare = not full_only and not selection_only
        self.run_full = not bare_only and not selection_only

        # Load pipeline reference results
        self.pipeline_results = self._load_pipeline_results(pipeline_results_path)

        # Initialize methods
        self.methods = self._init_methods(methods)

        # Shared infrastructure
        self._extraction_wrapper = None

        # Incremental saving setup
        self.jsonl_path = self.output_dir / "results.jsonl"
        self.failed_jsonl_path = self.output_dir / "failed.jsonl"

        logger.info(f"Initialized runner with {len(self.methods)} methods")
        logger.info(f"Extraction modes: bare={self.run_bare}, full={self.run_full}")
        logger.info(f"Incremental saving to: {self.jsonl_path}")

    def _load_pipeline_results(self, path: str) -> Dict[str, Any]:
        """Load AdaFrame pipeline results."""
        with open(path) as f:
            data = json.load(f)

        # Convert to dict by video name for easy lookup
        results = {}
        for result in data.get("results", []):
            video_name = result.get("video_name", "")
            if video_name:
                results[video_name] = result

        logger.info(f"Loaded {len(results)} pipeline results from {path}")
        return results

    def _init_methods(self, method_names: Optional[List[str]]) -> List[BaseMethod]:
        """Initialize baseline methods."""
        methods = []

        if method_names is None:
            method_names = list(ALL_METHODS.keys())

        for name in method_names:
            if name not in ALL_METHODS:
                logger.warning(f"Unknown method: {name}, skipping")
                continue

            cls = ALL_METHODS[name]
            instance = cls()

            if instance.requires_gpu and self.skip_gpu:
                logger.info(f"Skipping GPU method: {name}")
                continue

            methods.append(instance)

        return methods

    @property
    def extraction_wrapper(self) -> ExtractionWrapper:
        """Get extraction wrapper (lazy initialization)."""
        if self._extraction_wrapper is None:
            self._extraction_wrapper = ExtractionWrapper(self.config)
        return self._extraction_wrapper

    def run(
        self,
        video_paths: List[str],
        retry_queue_dir: Optional[str] = None,
        failed_log_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run benchmark on all videos.

        Args:
            video_paths: List of video file paths
            retry_queue_dir: Directory for retry queue
            failed_log_path: Path for failure log

        Returns:
            Summary dict with aggregated results
        """
        all_results = []
        failed_videos = []

        for i, video_path in enumerate(video_paths):
            video_name = Path(video_path).name
            logger.info(f"\n[{i+1}/{len(video_paths)}] Processing {video_name}")

            try:
                result = self._process_video(
                    video_path,
                    retry_queue_dir=retry_queue_dir,
                    failed_log_path=failed_log_path,
                )
                all_results.extend(result)
                # Save incrementally after each video
                self._save_results_incremental(result)
            except Exception as e:
                logger.error(f"Failed to process {video_name}: {e}")
                failed_entry = {"video": video_name, "error": str(e)}
                failed_videos.append(failed_entry)
                # Save failed video incrementally
                self._save_failed_incremental(failed_entry)

        # Aggregate results by method
        summary = self._aggregate_by_method(all_results)

        # Save final results (per-method folders, etc.)
        self._save_results(all_results, summary, failed_videos)

        return summary

    def _process_video(
        self,
        video_path: str,
        retry_queue_dir: Optional[str] = None,
        failed_log_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process a single video with all methods.

        Returns:
            List of result dicts (one per method)
        """
        video_name = Path(video_path).name

        # Get video info
        try:
            info = get_video_info(video_path)
        except Exception as e:
            logger.error(f"Cannot get video info: {e}")
            return []

        duration = info["duration"]
        total_frames = info["total_frames"]

        # Get pipeline reference
        pipeline_ref = self.pipeline_results.get(video_name, {})
        pipeline_extraction = pipeline_ref.get("extraction", {})
        pipeline_k = pipeline_ref.get("pipeline_stats", {}).get("final_frame_count", 25)

        # Pre-decode frames for GPU methods
        all_frames = None
        clip_embeddings = None

        if any(m.requires_gpu for m in self.methods):
            logger.info("  Pre-decoding frames for GPU methods...")
            all_frames, _, _ = decode_all_frames(
                video_path,
                interval_ms=self.config.get("sample_interval_ms", 100),
                max_resolution=self.config.get("max_resolution", 720)
            )

            if all_frames:
                from ub_src.extraction_wrapper import get_clip_deduplicator
                clip = get_clip_deduplicator()
                frame_arrays = [f for _, f in all_frames]
                clip_embeddings = clip.compute_signatures_batch(frame_arrays)
                logger.info(f"  Computed {len(clip_embeddings)} CLIP embeddings")

        # Run each method
        results = []

        for method in self.methods:
            logger.info(f"  Running {method.name}...")

            try:
                result = self._run_method(
                    method=method,
                    video_path=video_path,
                    video_name=video_name,
                    duration=duration,
                    total_frames=total_frames,
                    pipeline_k=pipeline_k,
                    pipeline_extraction=pipeline_extraction,
                    all_frames=all_frames,
                    clip_embeddings=clip_embeddings,
                    retry_queue_dir=retry_queue_dir,
                    failed_log_path=failed_log_path,
                )
                results.append(result)

                # Throttle between methods
                if not self.selection_only:
                    time.sleep(self.llm_throttle)

            except Exception as e:
                logger.error(f"    {method.name} failed: {e}")
                results.append({
                    "video_name": video_name,
                    "method": method.name,
                    "status": "error",
                    "error": str(e),
                })

        return results

    def _run_method(
        self,
        method: BaseMethod,
        video_path: str,
        video_name: str,
        duration: float,
        total_frames: int,
        pipeline_k: int,
        pipeline_extraction: Dict,
        all_frames: Optional[List] = None,
        clip_embeddings: Optional[np.ndarray] = None,
        retry_queue_dir: Optional[str] = None,
        failed_log_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a single method on a video."""

        # Frame selection
        shared_kwargs = {
            "target_k": pipeline_k,
            "all_frames": all_frames,
            "clip_embeddings": clip_embeddings,
            "duration": duration,
        }

        frames, latency = method.run_timed(video_path, **shared_kwargs)

        # Compute selection metrics
        selection = compute_selection_metrics(
            frames, total_frames, latency,
            clip_embeddings=clip_embeddings[:len(frames)] if clip_embeddings is not None and len(frames) > 0 else None
        )

        result = {
            "status": "success",
            "video_path": video_path,
            "video_name": video_name,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "method": method.name,
            "metadata": {
                "duration": duration,
                "fps": total_frames / duration if duration > 0 else 0,
                "total_frames": total_frames,
            },
            "pipeline_stats": selection,
        }

        # Extraction
        if not self.selection_only and frames:
            extraction_result = self._run_extraction(
                video_name=video_name,
                frames=frames,
                duration=duration,
                pipeline_extraction=pipeline_extraction,
                retry_queue_dir=retry_queue_dir,
                failed_log_path=failed_log_path,
            )
            result["extraction"] = extraction_result.get("extraction", {})
            result["comparison"] = extraction_result.get("comparison", {})

        return result

    def _run_extraction(
        self,
        video_name: str,
        frames: List[Tuple[float, np.ndarray]],
        duration: float,
        pipeline_extraction: Dict,
        retry_queue_dir: Optional[str] = None,
        failed_log_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run VLM extraction with retry handling."""

        result = {"extraction": {}, "comparison": {}}

        # Bare extraction
        if self.run_bare:
            def extract_bare():
                return self.extraction_wrapper.extract_bare(frames, duration)

            if retry_queue_dir and failed_log_path:
                bare_result = call_vlm_with_retry_queue(
                    video_id=f"{video_name}_bare",
                    extract_fn=extract_bare,
                    provider=self.config.get("provider", "gemini"),
                    model=self.config.get("model", "gemini-2.0-flash-exp"),
                    retry_queue_dir=retry_queue_dir,
                    failed_log_path=failed_log_path,
                    extra_metadata={"mode": "bare", "n_frames": len(frames)},
                )
            else:
                try:
                    bare_result = extract_bare()
                except Exception as e:
                    bare_result = {"error": str(e)}

            if bare_result:
                result["extraction"] = bare_result
                result["comparison"] = compare_extractions(bare_result, pipeline_extraction)

        # Full extraction
        if self.run_full:
            time.sleep(self.llm_throttle)  # Throttle between calls

            def extract_full():
                return self.extraction_wrapper.extract_full(frames, duration, audio_context=None)

            if retry_queue_dir and failed_log_path:
                full_result = call_vlm_with_retry_queue(
                    video_id=f"{video_name}_full",
                    extract_fn=extract_full,
                    provider=self.config.get("provider", "gemini"),
                    model=self.config.get("model", "gemini-2.0-flash-exp"),
                    retry_queue_dir=retry_queue_dir,
                    failed_log_path=failed_log_path,
                    extra_metadata={"mode": "full", "n_frames": len(frames)},
                )
            else:
                try:
                    full_result = extract_full()
                except Exception as e:
                    full_result = {"error": str(e)}

            if full_result:
                result["extraction"] = full_result
                result["comparison"] = compare_extractions(full_result, pipeline_extraction)

        return result

    def _aggregate_by_method(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results by method."""
        by_method = {}

        for result in results:
            method = result.get("method", "unknown")
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(result)

        summary = {
            "n_videos": len(set(r.get("video_name") for r in results)),
            "n_methods": len(by_method),
            "methods": {},
        }

        for method, method_results in by_method.items():
            summary["methods"][method] = aggregate_metrics(method_results)

        return summary

    def _save_results_incremental(self, results: List[Dict]):
        """Save results incrementally to JSONL file."""
        try:
            with open(self.jsonl_path, 'a') as f:
                for result in results:
                    f.write(json.dumps(result, default=str) + '\n')
        except Exception as e:
            logger.error(f"Failed to save incremental results: {e}")

    def _save_failed_incremental(self, failed_entry: Dict):
        """Save failed video incrementally to JSONL file."""
        try:
            with open(self.failed_jsonl_path, 'a') as f:
                f.write(json.dumps(failed_entry, default=str) + '\n')
        except Exception as e:
            logger.error(f"Failed to save incremental failed entry: {e}")

    def _save_results(
        self,
        all_results: List[Dict],
        summary: Dict,
        failed_videos: List[Dict],
    ):
        """Save results to per-method folders."""

        # Group results by method
        by_method = {}
        for result in all_results:
            method = result.get("method", "unknown")
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(result)

        # Save per-method results
        for method, method_results in by_method.items():
            method_dir = self.output_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)

            # Method-specific summary
            method_summary = summary.get("methods", {}).get(method, {})
            method_summary["method"] = method
            method_summary["n_videos"] = len(method_results)
            method_summary["n_success"] = len([r for r in method_results if r.get("status") == "success"])
            method_summary["n_failed"] = len([r for r in method_results if r.get("status") == "error"])

            # Save per-method results.json
            results_file = method_dir / "results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    "metadata": {
                        "method": method,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "n_videos": len(method_results),
                        "config": self.config,
                    },
                    "results": method_results,
                }, f, indent=2, default=str)

            # Save per-method summary.json
            summary_file = method_dir / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(method_summary, f, indent=2, default=str)

            # Save per-method CSV
            import csv
            csv_file = method_dir / "results.csv"
            if method_results:
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=method_results[0].keys())
                    writer.writeheader()
                    for row in method_results:
                        writer.writerow(row)

            logger.info(f"Saved {method} results to {method_dir}")

        # Save combined results (all methods together)
        combined_dir = self.output_dir / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)

        combined_results_file = combined_dir / "all_results.json"
        with open(combined_results_file, 'w') as f:
            json.dump({
                "metadata": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "n_videos": summary["n_videos"],
                    "n_methods": summary["n_methods"],
                    "config": self.config,
                },
                "results": all_results,
                "failed": failed_videos,
            }, f, indent=2, default=str)

        combined_summary_file = combined_dir / "all_summary.json"
        with open(combined_summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        import csv
        combined_csv_file = combined_dir / "all_results.csv"
        if all_results:
            with open(combined_csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                for row in all_results:
                    writer.writerow(row)

        logger.info(f"Saved combined results to {combined_dir}")
        logger.info(f"Incremental results (JSONL): {self.jsonl_path}")
        if self.failed_jsonl_path.exists():
            logger.info(f"Failed videos log: {self.failed_jsonl_path}")
