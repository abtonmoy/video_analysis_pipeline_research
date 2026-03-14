"""
BenchmarkRunner — main orchestrator for the benchmarking suite.

Iterates over videos, runs each enabled baseline, optionally feeds
selected frames through bare and/or full LLM extraction, and writes
structured results (JSON + CSV).
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from benchmarks.base import BaselineMethod, decode_frames_at_interval, get_video_info
from benchmarks.extraction_wrapper import ExtractionWrapper
from benchmarks.metrics import compare_extractions, compute_selection_metrics
# from benchmarks.methods import ALL_METHODS
from benchmarks import ALL_METHODS
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Orchestrates benchmark execution across videos and baselines.

    Shared resources (CLIP embeddings, audio context, decoded frames)
    are computed once per video and injected into baselines via kwargs.
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
        Args:
            config: Merged config dict (benchmark.yaml + defaults)
            pipeline_results_path: Path to your pipeline's results JSON
            output_dir: Where to write benchmark outputs
            methods: List of method names to run (None = all from config)
            skip_gpu: Skip GPU-dependent methods (clip_only, kmeans)
            selection_only: Only compute frame-selection metrics, no LLM
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

        # Determine which extraction modes to run
        self.run_bare = not full_only and not selection_only
        self.run_full = not bare_only and not selection_only

        # Load pipeline reference results
        self.pipeline_results = self._load_pipeline_results(pipeline_results_path)

        # Initialize baseline methods
        bench_config = config.get("benchmark", {})
        enabled = methods or bench_config.get("methods", list(ALL_METHODS.keys()))
        thresholds = bench_config.get("thresholds", {})

        self.methods: List[BaselineMethod] = []
        for name in enabled:
            if name not in ALL_METHODS:
                logger.warning(f"Unknown method '{name}', skipping")
                continue
            cls = ALL_METHODS[name]
            # Pass threshold overrides to constructor where applicable
            instance = self._create_method(cls, thresholds)
            if instance.requires_gpu and skip_gpu:
                logger.info(f"Skipping GPU method: {name}")
                continue
            self.methods.append(instance)

        logger.info(f"Benchmark methods: {[m.name for m in self.methods]}")

        # Shared infrastructure (lazy init)
        self._clip_dedup = None
        self._extraction_wrapper = None
        self._audio_extractor = None
        self._video_loader = None

    # ------------------------------------------------------------------
    # Lazy-loaded shared resources
    # ------------------------------------------------------------------

    @property
    def clip_dedup(self):
        if self._clip_dedup is None:
            from src.deduplication.clip_embed import CLIPDeduplicator
            clip_cfg = self.config.get("benchmark", {}).get("clip", {})
            self._clip_dedup = CLIPDeduplicator(
                model_name=clip_cfg.get("model", "ViT-B-32"),
                pretrained=clip_cfg.get("pretrained", "openai"),
                device=clip_cfg.get("device", "auto"),
                batch_size=clip_cfg.get("batch_size", 32),
            )
        return self._clip_dedup

    @property
    def extraction_wrapper(self):
        if self._extraction_wrapper is None:
            self._extraction_wrapper = ExtractionWrapper(self.config)
        return self._extraction_wrapper

    @property
    def audio_extractor(self):
        if self._audio_extractor is None:
            from src.ingestion.audio_extractor import AudioExtractor
            self._audio_extractor = AudioExtractor()
        return self._audio_extractor

    @property
    def video_loader(self):
        if self._video_loader is None:
            from src.ingestion.video_loader import VideoLoader
            ing = self.config.get("ingestion", {})
            self._video_loader = VideoLoader(
                max_resolution=ing.get("max_resolution", 720),
                extract_audio=ing.get("extract_audio", True),
            )
        return self._video_loader

    # ------------------------------------------------------------------
    # Pipeline results loading
    # ------------------------------------------------------------------

    def _load_pipeline_results(self, path: str) -> Dict[str, Dict]:
        """Load existing pipeline results, indexed by video filename."""
        p = Path(path)
        if not p.exists():
            logger.warning(f"Pipeline results not found: {path}")
            return {}

        with open(p) as f:
            data = json.load(f)

        indexed = {}
        results_list = data.get("results", [])
        for r in results_list:
            vname = r.get("video_name") or Path(r.get("video_path", "")).name
            if vname:
                indexed[vname] = r

        logger.info(f"Loaded pipeline results for {len(indexed)} videos")
        return indexed

    # ------------------------------------------------------------------
    # Method factory
    # ------------------------------------------------------------------

    @staticmethod
    def _create_method(cls, thresholds: Dict) -> BaselineMethod:
        """Instantiate a baseline method, passing relevant thresholds."""
        name = cls.__name__

        if name == "HistogramDedup":
            return cls(threshold=thresholds.get("histogram_correlation", 0.95))
        elif name == "ORBDedup":
            return cls(
                match_threshold=thresholds.get("orb_good_matches", 40),
                distance_threshold=thresholds.get("orb_match_distance", 50),
            )
        elif name == "OpticalFlowPeaks":
            return cls(percentile=thresholds.get("optical_flow_percentile", 85.0))
        elif name == "CLIPOnlyDedup":
            return cls(threshold=thresholds.get("clip_cosine", 0.92))
        elif name == "KMeansClustering":
            return cls(seconds_per_cluster=thresholds.get("kmeans_seconds_per_cluster", 3.0))
        else:
            return cls()

    # ------------------------------------------------------------------
    # Shared pre-computation per video
    # ------------------------------------------------------------------

    def _precompute_shared(
        self, video_path: str
    ) -> Tuple[
        Optional[List[Tuple[float, np.ndarray]]],
        Optional[np.ndarray],
        Optional[Dict],
        int,
    ]:
        """
        Pre-compute shared resources for a video:
        - all_frames + clip_embeddings (if GPU methods enabled)
        - audio_context (if full extraction enabled)
        - total_frame_count

        Returns: (all_frames, clip_embeddings, audio_context, total_frames)
        """
        bench_cfg = self.config.get("benchmark", {})
        video_cfg = bench_cfg.get("video", {})
        max_res = video_cfg.get("max_resolution", 720)
        interval_ms = video_cfg.get("sample_interval_ms", 100)

        total_frames, fps, duration = get_video_info(video_path)

        # Pre-decode frames if any method needs them
        all_frames = None
        clip_embeddings = None
        needs_all_frames = any(
            m.requires_gpu or m.name == "random" for m in self.methods
        )

        if needs_all_frames:
            logger.info("Pre-decoding frames for shared use...")
            all_frames = decode_frames_at_interval(
                video_path, interval_ms=interval_ms, max_resolution=max_res
            )
            logger.info(f"Decoded {len(all_frames)} frames at {interval_ms}ms interval")

        # CLIP embeddings for GPU methods + info density
        needs_clip = any(m.requires_gpu for m in self.methods) or not self.selection_only
        if needs_clip and all_frames and not self.skip_gpu:
            logger.info("Computing shared CLIP embeddings...")
            frame_arrays = [f for _, f in all_frames]
            clip_embeddings = self.clip_dedup.compute_signatures_batch(frame_arrays)
            logger.info(f"CLIP embeddings: {clip_embeddings.shape}")

        # Audio context (only if full extraction is enabled)
        audio_context = None
        if self.run_full:
            try:
                metadata, audio_path = self.video_loader.load(video_path)
                if audio_path:
                    audio_cfg = self.config.get("audio_analysis", {})
                    trans_cfg = audio_cfg.get("transcription", {})
                    audio_context = self.audio_extractor.extract_full_context(
                        audio_path,
                        transcribe=trans_cfg.get("enabled", True),
                        model_size=trans_cfg.get("model", "base"),
                    )
                    logger.info(f"Audio context extracted: mood={audio_context.get('mood')}")
            except Exception as e:
                logger.warning(f"Audio extraction failed: {e}")

        return all_frames, clip_embeddings, audio_context, total_frames

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self, video_paths: List[str]) -> Dict[str, Any]:
        """
        Run all enabled baselines on all videos.

        Returns the full results dict (also written to disk).
        """
        all_results: Dict[str, Any] = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "baselines_run": [m.name for m in self.methods],
                "extraction_modes": self._extraction_modes(),
                "videos_processed": 0,
                "total_llm_calls": 0,
            },
            "per_video": {},
        }

        csv_rows: List[Dict] = []
        total_llm_calls = 0

        for vpath in video_paths:
            vname = Path(vpath).name
            logger.info(f"\n{'='*60}\nBenchmarking: {vname}\n{'='*60}")

            try:
                video_result, rows, n_calls = self._process_video(vpath)
                all_results["per_video"][vname] = video_result
                csv_rows.extend(rows)
                total_llm_calls += n_calls
            except Exception as e:
                logger.error(f"Failed to benchmark {vname}: {e}", exc_info=True)
                all_results["per_video"][vname] = {"error": str(e)}

        all_results["metadata"]["videos_processed"] = len(
            [v for v in all_results["per_video"].values() if "error" not in v]
        )
        all_results["metadata"]["total_llm_calls"] = total_llm_calls

        # Write outputs
        self._write_json(all_results)
        self._write_csv(csv_rows)

        return all_results

    def _process_video(
        self, video_path: str
    ) -> Tuple[Dict[str, Any], List[Dict], int]:
        """
        Process a single video through all baselines.

        Returns: (video_result_dict, csv_rows, llm_call_count)
        """
        vname = Path(video_path).name
        total_frames, fps, duration = get_video_info(video_path)

        # Load pipeline reference
        pipeline_ref = self.pipeline_results.get(vname, {})
        pipeline_k = (
            pipeline_ref.get("pipeline_stats", {}).get("final_frame_count")
            or max(1, int(duration))
        )
        pipeline_extraction = pipeline_ref.get("extraction", {})

        # Pre-compute shared resources
        all_frames, clip_embeddings, audio_context, _ = self._precompute_shared(
            video_path
        )

        video_result: Dict[str, Any] = {
            "video_metadata": {
                "duration": round(duration, 2),
                "fps": round(fps, 1),
                "total_frames": total_frames,
            },
            "audio_context_available": audio_context is not None,
            "pipeline_reference": {
                "final_frame_count": pipeline_k,
                "reduction_rate": pipeline_ref.get("pipeline_stats", {}).get(
                    "reduction_rate"
                ),
            },
            "baselines": {},
        }

        csv_rows: List[Dict] = []
        llm_calls = 0

        # Build shared kwargs for baselines
        shared_kwargs = {
            "target_k": pipeline_k,
            "all_frames": all_frames,
            "clip_embeddings": clip_embeddings,
            "max_resolution": self.config.get("benchmark", {})
            .get("video", {})
            .get("max_resolution", 720),
        }

        for method in self.methods:
            logger.info(f"  Running: {method.name}")

            try:
                result = self._run_single_baseline(
                    method=method,
                    video_path=video_path,
                    duration=duration,
                    total_frames=total_frames,
                    audio_context=audio_context,
                    pipeline_extraction=pipeline_extraction,
                    shared_kwargs=shared_kwargs,
                )

                video_result["baselines"][method.name] = result["data"]
                csv_rows.append(
                    {"video": vname, "method": method.name, **result["csv_row"]}
                )
                llm_calls += result["llm_calls"]

            except Exception as e:
                logger.error(f"  {method.name} failed: {e}", exc_info=True)
                video_result["baselines"][method.name] = {"error": str(e)}
                csv_rows.append(
                    {"video": vname, "method": method.name, "error": str(e)}
                )

        return video_result, csv_rows, llm_calls

    def _run_single_baseline(
        self,
        method: BaselineMethod,
        video_path: str,
        duration: float,
        total_frames: int,
        audio_context: Optional[Dict],
        pipeline_extraction: Dict,
        shared_kwargs: Dict,
    ) -> Dict[str, Any]:
        """
        Run one baseline on one video: select frames → metrics → extraction.

        Returns dict with 'data' (for JSON), 'csv_row', and 'llm_calls'.
        """
        # 1. Frame selection (timed)
        frames, latency = method.run_timed(video_path, **shared_kwargs)

        # 2. Selection metrics
        clip_for_density = self.clip_dedup if not self.skip_gpu else None
        selection = compute_selection_metrics(
            frames, total_frames, latency, clip_for_density
        )

        data: Dict[str, Any] = {"selection": selection}
        csv_row: Dict[str, Any] = {**selection}
        calls = 0

        # 3. Bare extraction
        if self.run_bare and frames:
            logger.info(f"    Bare extraction ({len(frames)} frames)...")
            bare_result = self.extraction_wrapper.extract_bare(frames, duration)
            bare_cmp = compare_extractions(bare_result, pipeline_extraction)
            data["bare_extraction"] = bare_result
            data["bare_vs_pipeline"] = bare_cmp
            csv_row.update({f"bare_{k}": v for k, v in bare_cmp.items()})
            calls += 1

        # 4. Full extraction
        if self.run_full and frames:
            logger.info(f"    Full extraction ({len(frames)} frames)...")
            full_result = self.extraction_wrapper.extract_full(
                frames, duration, audio_context
            )
            full_cmp = compare_extractions(full_result, pipeline_extraction)
            data["full_extraction"] = full_result
            data["full_vs_pipeline"] = full_cmp
            csv_row.update({f"full_{k}": v for k, v in full_cmp.items()})
            calls += 2  # adaptive schema = type detection + extraction

        return {"data": data, "csv_row": csv_row, "llm_calls": calls}

    # ------------------------------------------------------------------
    # Output writing
    # ------------------------------------------------------------------

    def _write_json(self, results: Dict):
        path = self.output_dir / "benchmark_results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"JSON results written to {path}")

    def _write_csv(self, rows: List[Dict]):
        if not rows:
            return
        path = self.output_dir / "benchmark_results.csv"
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        logger.info(f"CSV results written to {path}")
        logger.info(f"\n{df.to_string(index=False)}")

    def _extraction_modes(self) -> List[str]:
        modes = []
        if self.run_bare:
            modes.append("bare")
        if self.run_full:
            modes.append("full")
        if self.selection_only:
            modes.append("selection_only")
        return modes