# src/parallel_pipeline.py
"""
Parallel batch processing pipeline for video advertisement analysis.

Uses multiprocessing with pre-loaded models per worker to achieve true parallelism.
Each worker process initializes its own pipeline instance once, then processes
multiple videos without reloading models.

IMPORTANT: Models are warmed up during initialization using dummy data to ensure
all lazy-loaded components (including LPIPS, CLIP, etc.) are fully loaded before
processing begins.
"""

import os
import sys
import time
import logging
import multiprocessing as mp
from multiprocessing import Pool, get_context
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime
import warnings
import traceback

# Suppress warnings before imports
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class VideoResult:
    """Result from processing a single video."""
    video_path: str
    video_name: str
    success: bool
    result_dict: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float = 0.0


# Global worker state (one per process)
_worker_pipeline = None
_worker_id = None
_worker_initialized = False


def _warmup_models(pipeline):
    """
    Warm up all models by running them with dummy data.
    
    This forces all lazy-loaded sub-components to initialize,
    including LPIPS, CLIP embeddings, etc.
    """
    import numpy as np
    from PIL import Image
    
    logging.info("Warming up models with dummy data...")
    
    # Create dummy frames (small 64x64 images to be fast)
    dummy_frames = []
    for i in range(5):
        # Create slightly different images so deduplication has something to compare
        img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        dummy_frames.append((float(i), img))  # (timestamp, frame)
    
    try:
        # Warm up the deduplicator - this triggers LPIPS, CLIP, hash functions
        # The deduplicate method expects list of (timestamp, frame) tuples
        deduped, embeddings, stats = pipeline.deduplicator.deduplicate(dummy_frames)
        logging.info(f"Deduplicator warmed up: processed {len(dummy_frames)} -> {len(deduped)} frames")
    except Exception as e:
        logging.warning(f"Deduplicator warmup failed (non-critical): {e}")
    
    try:
        # Warm up the selector with dummy data
        from src.selection.clustering import FrameCandidate
        dummy_candidates = [
            FrameCandidate(
                timestamp=float(i),
                frame=dummy_frames[i][1],
                scene_id=0,
                importance_score=0.5
            )
            for i in range(min(3, len(dummy_frames)))
        ]
        
        # Create dummy embeddings
        dummy_embeddings = [np.random.rand(512).astype(np.float32) for _ in dummy_candidates]
        
        selected = pipeline.selector.select(
            frames=dummy_candidates,
            embeddings=dummy_embeddings,
            scene_boundaries=[(0.0, 5.0)],
            video_duration=5.0,
            audio_events=None
        )
        logging.info(f"Selector warmed up: selected {len(selected)} frames")
    except Exception as e:
        logging.warning(f"Selector warmup failed (non-critical): {e}")
    
    logging.info("Model warmup complete!")


def _init_worker(config_path: str, worker_id_queue, suppress_logs: bool = True):
    """
    Initialize worker process with its own pipeline instance.
    Called once per worker when the pool starts.
    
    This function:
    1. Creates a pipeline instance
    2. Loads all lazy-loaded model properties
    3. Warms up models with dummy data to force LPIPS/CLIP initialization
    
    Args:
        config_path: Path to the pipeline configuration file
        worker_id_queue: Queue to get unique worker IDs
        suppress_logs: Whether to suppress verbose logs in workers
    """
    global _worker_pipeline, _worker_id, _worker_initialized
    
    # Get unique worker ID from queue
    try:
        _worker_id = worker_id_queue.get_nowait()
    except:
        _worker_id = os.getpid()
    
    print(f"[Worker {_worker_id}] Initializing pipeline and loading models...")
    
    # Suppress noisy logs in workers AFTER we print our status
    if suppress_logs:
        # Keep our own logs visible but suppress library noise
        logging.getLogger().setLevel(logging.WARNING)
        for logger_name in ["whisper", "torch", "PIL", "urllib3", "open_clip", 
                          "lpips", "matplotlib", "numba", "filelock", "transformers",
                          "pyscenedetect", "cv2"]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    try:
        # Import here to avoid issues with multiprocessing
        from src.pipeline import AdVideoPipeline
        
        print(f"[Worker {_worker_id}] Creating pipeline instance...")
        
        # Create pipeline instance for this worker
        _worker_pipeline = AdVideoPipeline(config_path=config_path)
        
        # Step 1: Access all lazy-loaded properties to trigger basic initialization
        print(f"[Worker {_worker_id}] Loading video loader...")
        _ = _worker_pipeline.loader
        
        print(f"[Worker {_worker_id}] Loading audio extractor...")
        _ = _worker_pipeline.audio_extractor
        
        print(f"[Worker {_worker_id}] Loading scene detector...")
        _ = _worker_pipeline.scene_detector
        
        print(f"[Worker {_worker_id}] Loading deduplicator...")
        _ = _worker_pipeline.deduplicator
        
        print(f"[Worker {_worker_id}] Loading selector...")
        _ = _worker_pipeline.selector
        
        print(f"[Worker {_worker_id}] Loading extractor...")
        _ = _worker_pipeline.extractor
        
        # Step 2: Warm up models with dummy data to force LPIPS, CLIP, etc. to fully load
        print(f"[Worker {_worker_id}] Warming up models (this loads LPIPS, CLIP, etc.)...")
        _warmup_models(_worker_pipeline)
        
        _worker_initialized = True
        
        print(f"[Worker {_worker_id}] ✓ Fully initialized and ready!")
        
    except Exception as e:
        print(f"[Worker {_worker_id}] ✗ Failed to initialize: {e}")
        traceback.print_exc()
        _worker_initialized = False
        raise


def _process_video_worker(args: tuple) -> VideoResult:
    """
    Process a single video using the worker's pre-loaded pipeline.
    This function runs in a worker process.
    
    Args:
        args: Tuple of (video_path, skip_extraction)
        
    Returns:
        VideoResult with processing outcome
    """
    video_path, skip_extraction = args
    video_name = Path(video_path).name
    start_time = time.time()
    
    global _worker_pipeline, _worker_initialized, _worker_id
    
    if not _worker_initialized or _worker_pipeline is None:
        return VideoResult(
            video_path=video_path,
            video_name=video_name,
            success=False,
            error="Worker pipeline not initialized"
        )
    
    try:
        result = _worker_pipeline.process(video_path, skip_extraction=skip_extraction)
        processing_time = time.time() - start_time
        
        # Convert to serializable dict
        result_dict = _result_to_dict(result, video_path)
        
        return VideoResult(
            video_path=video_path,
            video_name=video_name,
            success=True,
            result_dict=result_dict,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        
        return VideoResult(
            video_path=video_path,
            video_name=video_name,
            success=False,
            error=error_msg,
            result_dict={
                "status": "failed",
                "video_path": video_path,
                "video_name": video_name,
                "error": error_msg,
                "processed_at": datetime.now().isoformat()
            },
            processing_time=processing_time
        )


def _result_to_dict(result, video_path: str) -> Dict[str, Any]:
    """Convert PipelineResult to serializable dictionary."""
    if result is None:
        return {
            "status": "failed",
            "video_path": video_path,
            "video_name": Path(video_path).name,
            "error": "Processing returned None",
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


class ParallelPipeline:
    """
    Parallel video processing pipeline using multiprocessing.
    
    Each worker process has its own pipeline instance with pre-loaded models.
    Videos are distributed across workers for true parallel processing.
    
    Initialization flow:
    1. Pool is created with N workers
    2. Each worker runs _init_worker() ONCE:
       - Creates pipeline instance
       - Loads all lazy properties (loader, detector, deduplicator, etc.)
       - Warms up models with dummy data (forces LPIPS, CLIP to fully initialize)
    3. Workers are now ready to process videos without any model loading
    """
    
    def __init__(
        self,
        config_path: str = "config/default.yaml",
        num_workers: int = 4,
        suppress_worker_logs: bool = True
    ):
        """
        Initialize parallel pipeline.
        
        Args:
            config_path: Path to pipeline configuration file
            num_workers: Number of worker processes (each loads models once)
            suppress_worker_logs: If True, suppress library logs in worker processes
        """
        self.config_path = config_path
        self.num_workers = num_workers
        self.suppress_worker_logs = suppress_worker_logs
        self._pool = None
        self._manager = None
        self._worker_id_queue = None
    
    def _create_pool(self):
        """Create worker pool with initialized pipelines."""
        if self._pool is not None:
            return
        
        # Use spawn method to avoid fork issues with CUDA/models
        ctx = get_context('spawn')
        
        # Create a manager for sharing the worker ID queue
        self._manager = ctx.Manager()
        self._worker_id_queue = self._manager.Queue()
        
        # Populate queue with worker IDs
        for i in range(self.num_workers):
            self._worker_id_queue.put(i)
        
        print(f"\n{'='*60}")
        print(f"Creating pool with {self.num_workers} workers...")
        print(f"Each worker will load models once during initialization.")
        print(f"{'='*60}\n")
        
        # Create pool with initializer
        # Each worker will call _init_worker ONCE when it starts
        self._pool = ctx.Pool(
            processes=self.num_workers,
            initializer=_init_worker,
            initargs=(self.config_path, self._worker_id_queue, self.suppress_worker_logs)
        )
        
        print(f"\n{'='*60}")
        print(f"All {self.num_workers} workers initialized and ready!")
        print(f"{'='*60}\n")
    
    def process_batch(
        self,
        video_paths: List[str],
        skip_extraction: bool = False,
        callback: Optional[Callable[[VideoResult], None]] = None
    ) -> List[VideoResult]:
        """
        Process multiple videos in parallel.
        
        Args:
            video_paths: List of video file paths
            skip_extraction: If True, skip LLM extraction
            callback: Optional callback function called after each video completes
            
        Returns:
            List of VideoResult objects
        """
        if not video_paths:
            return []
        
        self._create_pool()
        
        # Prepare arguments for each video
        args_list = [(vp, skip_extraction) for vp in video_paths]
        
        results = []
        
        # Use imap_unordered for better responsiveness
        for result in self._pool.imap_unordered(_process_video_worker, args_list):
            results.append(result)
            if callback:
                callback(result)
        
        return results
    
    def process_batch_with_progress(
        self,
        video_paths: List[str],
        skip_extraction: bool = False,
        on_complete: Optional[Callable[[VideoResult], None]] = None,
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> List[VideoResult]:
        """
        Process videos with progress callbacks.
        
        Args:
            video_paths: List of video file paths
            skip_extraction: If True, skip LLM extraction
            on_complete: Called when a video completes (receives VideoResult)
            on_progress: Called to update progress (receives completed_count, total_count)
            
        Returns:
            List of VideoResult objects
        """
        if not video_paths:
            return []
        
        self._create_pool()
        
        args_list = [(vp, skip_extraction) for vp in video_paths]
        total = len(args_list)
        completed = 0
        results = []
        
        for result in self._pool.imap_unordered(_process_video_worker, args_list):
            results.append(result)
            completed += 1
            
            if on_complete:
                on_complete(result)
            
            if on_progress:
                on_progress(completed, total)
        
        return results
    
    def shutdown(self):
        """Shutdown worker pool and release resources."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures proper cleanup."""
        self.shutdown()
        return False


def process_videos_parallel(
    video_paths: List[str],
    config_path: str = "config/default.yaml",
    num_workers: int = 4,
    skip_extraction: bool = False,
    on_complete: Optional[Callable[[VideoResult], None]] = None
) -> List[VideoResult]:
    """
    Convenience function for parallel video processing.
    
    Args:
        video_paths: List of video file paths
        config_path: Path to pipeline configuration
        num_workers: Number of parallel workers (default: 4)
        skip_extraction: If True, skip LLM extraction
        on_complete: Optional callback for each completed video
        
    Returns:
        List of VideoResult objects
    """
    with ParallelPipeline(config_path, num_workers) as pipeline:
        return pipeline.process_batch(
            video_paths,
            skip_extraction=skip_extraction,
            callback=on_complete
        )