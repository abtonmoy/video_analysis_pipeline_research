"""
Main pipeline orchestrator for video advertisement analysis.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from src.utils.config import load_config, deep_merge, get_device
from src.utils.logging import setup_logging
from src.utils.metrics import PipelineResult, FrameInfo, SceneInfo
from src.utils.video_utils import get_video_metadata

from src.ingestion.video_loader import VideoLoader
from src.ingestion.audio_extractor import AudioExtractor
from src.detection.change_detector import get_change_detector
from src.detection.scene_detector import CandidateFrameExtractor, SceneDetector
    
from src.deduplication.hierarchical import create_deduplicator
from src.selection.representative import create_selector 
from src.selection.clustering import FrameCandidate
from src.extraction import create_extractor

logger = logging.getLogger(__name__)


class AdVideoPipeline:
    """
    Main pipeline for adaptive video advertisement analysis.
    
    Pipeline stages:
    1. Video ingestion & metadata extraction
    2. Lightweight change detection for candidate extraction
    3. Scene boundary detection
    4. Hierarchical deduplication (pHash → SSIM → CLIP)
    5. Temporal clustering & representative selection
    6. LLM extraction with temporal context
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None,
        overrides: Optional[Dict] = None
    ):
        """
        Initialize pipeline.
        
        Args:
            config_path: Path to YAML config file
            config: Config dictionary (alternative to config_path)
            overrides: Optional overrides to apply to config
        """
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_config(config_path, overrides)
        else:
            # Use defaults
            self.config = self._get_default_config()
        
        if overrides and config is not None:
            self.config = deep_merge(self.config, overrides)
        
        # Setup logging
        log_config = self.config.get("logging", {})
        setup_logging(
            level=log_config.get("level", "INFO"),
            log_file=log_config.get("log_file")
        )
        
        # Initialize components (lazy loading)
        self._loader = None
        self._audio_extractor = None
        self._scene_detector = None
        self._deduplicator = None
        self._selector = None
        self._extractor = None
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "ingestion": {"max_resolution": 720, "extract_audio": True},
            "change_detection": {"method": "histogram", "threshold": 0.15, "min_interval_ms": 100},
            "scene_detection": {"method": "content", "threshold": 27.0, "min_scene_length_s": 0.5},
            "deduplication": {
                "phash": {"enabled": True, "threshold": 8},
                "ssim": {"enabled": True, "threshold": 0.92},
                "clip": {"enabled": True, "model": "ViT-B-32", "threshold": 0.90, "device": "auto"}
            },
            "selection": {"max_frames_per_scene": 3, "min_temporal_gap_s": 0.5},
            "extraction": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "schema": {"mode": "adaptive"},
                "temporal_context": {"enabled": True}
            }
        }
    
    @property
    def loader(self) -> VideoLoader:
        if self._loader is None:
            ingestion_config = self.config.get("ingestion", {})
            self._loader = VideoLoader(
                max_resolution=ingestion_config.get("max_resolution", 720),
                extract_audio=ingestion_config.get("extract_audio", True)
            )
        return self._loader
    
    @property
    def audio_extractor(self) -> AudioExtractor:
        if self._audio_extractor is None:
            self._audio_extractor = AudioExtractor()
        return self._audio_extractor
    
    @property
    def scene_detector(self) -> SceneDetector:
        if self._scene_detector is None:
            scene_config = self.config.get("scene_detection", {})
            self._scene_detector = SceneDetector(
                method=scene_config.get("method", "content"),
                threshold=scene_config.get("threshold", 27.0),
                min_scene_length_s=scene_config.get("min_scene_length_s", 0.5)
            )
        return self._scene_detector
    
    @property
    def deduplicator(self):
        if self._deduplicator is None:
            self._deduplicator = create_deduplicator(self.config)
        return self._deduplicator
    
    @property
    def selector(self):
        if self._selector is None:
            self._selector = create_selector(self.config)
        return self._selector
    
    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = create_extractor(self.config)
        return self._extractor
    
    def process(
        self,
        video_path: str,
        skip_extraction: bool = True
    ) -> PipelineResult:
        """
        Process a single video through the pipeline.
        
        Args:
            video_path: Path to video file
            skip_extraction: If True, skip LLM extraction step
            
        Returns:
            PipelineResult with all outputs and metrics
        """
        start_time = time.time()
        video_path = str(Path(video_path).resolve())
        
        logger.info(f"Processing video: {video_path}")
        
        # Stage 1: Load video and extract metadata
        logger.info("Stage 1: Loading video...")
        metadata, audio_path = self.loader.load(video_path)
        
        # Stage 2: Detect scenes
        logger.info("Stage 2: Detecting scenes...")
        scene_boundaries = self.scene_detector.detect_scenes(video_path)
        
        if not scene_boundaries:
            # Fallback: treat entire video as one scene
            scene_boundaries = [(0.0, metadata.duration)]
        
        # Stage 3: Extract candidate frames
        logger.info("Stage 3: Extracting candidate frames...")
        change_config = self.config.get("change_detection", {})
        change_detector = get_change_detector(change_config.get("method", "histogram"))
        
        candidate_extractor = CandidateFrameExtractor(
            change_detector=change_detector,
            threshold=change_config.get("threshold", 0.15),
            min_interval_ms=change_config.get("min_interval_ms", 100)
        )
        
        candidates = candidate_extractor.extract_candidates(
            video_path,
            max_resolution=self.config.get("ingestion", {}).get("max_resolution", 720)
        )
        
        total_frames_sampled = len(candidates)
        logger.info(f"Extracted {total_frames_sampled} candidate frames")
        
        # Stage 4: Hierarchical deduplication
        logger.info("Stage 4: Hierarchical deduplication...")
        deduped_frames, embeddings, dedup_stats = self.deduplicator.deduplicate(candidates)
        
        # Stage 5: Extract audio events (optional)
        audio_events = None
        if audio_path:
            try:
                logger.info("Extracting audio events...")
                audio_events = self.audio_extractor.get_audio_events(audio_path)
            except Exception as e:
                logger.warning(f"Audio event extraction failed: {e}")
        
        # Stage 6: Select representatives
        logger.info("Stage 5: Selecting representative frames...")
        selected_candidates = self.selector.select(
            frames=deduped_frames,
            embeddings=embeddings,
            scene_boundaries=scene_boundaries,
            video_duration=metadata.duration,
            audio_events=audio_events
        )
        
        # Convert to frame list
        selected_frames = [(c.timestamp, c.frame) for c in selected_candidates]
        
        # Stage 7: LLM Extraction
        extraction_result = None
        if not skip_extraction and selected_frames:
            logger.info("Stage 6: LLM extraction...")
            try:
                extraction_result = self.extractor.extract(
                    frames=selected_frames,
                    video_duration=metadata.duration
                )
            except Exception as e:
                logger.error(f"Extraction failed: {e}")
                extraction_result = {"error": str(e)}
        
        # Build result
        processing_time = time.time() - start_time
        
        # Convert scenes
        scenes = [
            SceneInfo(
                scene_id=i,
                start_time=start,
                end_time=end
            )
            for i, (start, end) in enumerate(scene_boundaries)
        ]
        
        # Convert selected frames to FrameInfo
        frame_infos = [
            FrameInfo(
                timestamp=c.timestamp,
                scene_id=c.scene_id,
                importance_score=c.importance_score
            )
            for c in selected_candidates
        ]
        
        result = PipelineResult(
            video_path=video_path,
            metadata=metadata,
            scenes=scenes,
            selected_frames=frame_infos,
            extraction_result=extraction_result,
            total_frames_sampled=total_frames_sampled,
            frames_after_phash=dedup_stats.get("after_phash", total_frames_sampled),
            frames_after_ssim=dedup_stats.get("after_ssim", dedup_stats.get("after_phash", total_frames_sampled)),
            frames_after_clip=dedup_stats.get("after_clip", dedup_stats.get("after_ssim", total_frames_sampled)),
            final_frame_count=len(selected_frames),
            processing_time_s=processing_time
        )
        
        logger.info(f"Pipeline complete: {result.final_frame_count} frames selected "
                   f"({result.reduction_rate:.1%} reduction) in {processing_time:.1f}s")
        
        return result
    
    def process_batch(
        self,
        video_paths: List[str],
        max_workers: int = 4,
        skip_extraction: bool = False
    ) -> List[PipelineResult]:
        """
        Process multiple videos in parallel.
        
        Args:
            video_paths: List of video file paths
            max_workers: Number of parallel workers
            skip_extraction: If True, skip LLM extraction
            
        Returns:
            List of PipelineResult objects
        """
        logger.info(f"Processing batch of {len(video_paths)} videos with {max_workers} workers")
        
        results = []
        
        if max_workers == 1:
            # Sequential processing
            for video_path in video_paths:
                try:
                    result = self.process(video_path, skip_extraction=skip_extraction)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process {video_path}: {e}")
                    results.append(None)
        else:
            # Parallel processing - process sequentially to avoid multiprocessing issues
            # In production, would use proper multiprocessing with serializable config
            for video_path in video_paths:
                try:
                    result = self.process(video_path, skip_extraction=skip_extraction)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process {video_path}: {e}")
                    results.append(None)
        
        successful = sum(1 for r in results if r is not None)
        logger.info(f"Batch complete: {successful}/{len(video_paths)} videos processed successfully")
        
        return results
    
    def get_metrics(self, result: PipelineResult) -> Dict[str, Any]:
        """Get summary metrics from a pipeline result."""
        return result.get_metrics()


# ============================================================================
# Convenience functions
# ============================================================================

def process_video(
    video_path: str,
    config_path: Optional[str] = None,
    **kwargs
) -> PipelineResult:
    """
    Convenience function to process a single video.
    
    Args:
        video_path: Path to video file
        config_path: Optional path to config file
        **kwargs: Override configuration options
        
    Returns:
        PipelineResult
    """
    pipeline = AdVideoPipeline(config_path=config_path, overrides=kwargs)
    return pipeline.process(video_path)


def process_directory(
    directory: str,
    config_path: Optional[str] = None,
    max_workers: int = 4,
    extensions: List[str] = None,
    **kwargs
) -> List[PipelineResult]:
    """
    Process all videos in a directory.
    
    Args:
        directory: Directory containing videos
        config_path: Optional path to config file
        max_workers: Number of parallel workers
        extensions: List of video extensions to process
        **kwargs: Override configuration options
        
    Returns:
        List of PipelineResult objects
    """
    if extensions is None:
        extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
    
    directory = Path(directory)
    video_paths = []
    
    for ext in extensions:
        video_paths.extend(directory.glob(f"*{ext}"))
        video_paths.extend(directory.glob(f"*{ext.upper()}"))
    
    video_paths = [str(p) for p in sorted(set(video_paths))]
    
    if not video_paths:
        logger.warning(f"No videos found in {directory}")
        return []
    
    pipeline = AdVideoPipeline(config_path=config_path, overrides=kwargs)
    return pipeline.process_batch(video_paths, max_workers=max_workers)
