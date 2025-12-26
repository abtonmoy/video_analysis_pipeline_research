import logging
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import cv2

from base import BaseDeduplicator
from phash import PHashDeduplicator
from ssim import SSIMDeduplicator
from clip_embed import CLIPDeduplicator

logger = logging.getLogger(__name__)
# ============================================================================
# Hierarchical Deduplicator
# ============================================================================

class HierarchicalDeduplicator:
    """
    Hierarchical deduplication using cheap-to-expensive methods.
    
    Pipeline: pHash -> SSIM -> CLIP
    Each stage filters frames before passing to the more expensive next stage.
    """
    
    def __init__(
        self,
        phash_enabled: bool = True,
        phash_threshold: int = 8,
        ssim_enabled: bool = True,
        ssim_threshold: float = 0.92,
        clip_enabled: bool = True,
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        clip_threshold: float = 0.90,
        clip_device: str = "auto",
        clip_batch_size: int = 32
    ):
        # Initialize deduplicators
        self.phash_enabled = phash_enabled
        self.ssim_enabled = ssim_enabled
        self.clip_enabled = clip_enabled
        
        if phash_enabled:
            self.phash = PHashDeduplicator(threshold=phash_threshold)
        
        if ssim_enabled:
            self.ssim = SSIMDeduplicator(threshold=ssim_threshold)
        
        if clip_enabled:
            self.clip = CLIPDeduplicator(
                model_name=clip_model,
                pretrained=clip_pretrained,
                threshold=clip_threshold,
                device=clip_device,
                batch_size=clip_batch_size
            )
    
    def deduplicate(
        self,
        frames: List[Tuple[float, np.ndarray]]
    ) -> Tuple[List[Tuple[float, np.ndarray]], Optional[np.ndarray], Dict[str, int]]:
        """
        Apply hierarchical deduplication.
        
        Args:
            frames: List of (timestamp, frame) tuples
            
        Returns:
            Tuple of:
                - Deduplicated frames
                - CLIP embeddings (if CLIP enabled, else None)
                - Stats dict with frame counts at each stage
        """
        stats = {"input": len(frames)}
        current_frames = frames
        embeddings = None
        
        # Stage 1: pHash (fastest)
        if self.phash_enabled and len(current_frames) > 1:
            current_frames = self.phash.deduplicate(current_frames)
            stats["after_phash"] = len(current_frames)
        
        # Stage 2: SSIM (medium)
        if self.ssim_enabled and len(current_frames) > 1:
            current_frames = self.ssim.deduplicate(current_frames)
            stats["after_ssim"] = len(current_frames)
        
        # Stage 3: CLIP (slowest but semantic)
        if self.clip_enabled and len(current_frames) > 1:
            current_frames, embeddings = self.clip.deduplicate(current_frames)
            stats["after_clip"] = len(current_frames)
        
        stats["output"] = len(current_frames)
        
        logger.info(f"Hierarchical dedup: {stats['input']} -> {stats['output']} frames")
        
        return current_frames, embeddings, stats


# ============================================================================
# Convenience function
# ============================================================================

def create_deduplicator(config: Dict[str, Any]) -> HierarchicalDeduplicator:
    """
    Create HierarchicalDeduplicator from config dict.
    
    Args:
        config: Configuration dictionary with deduplication settings
        
    Returns:
        Configured HierarchicalDeduplicator
    """
    dedup_config = config.get("deduplication", {})
    
    return HierarchicalDeduplicator(
        phash_enabled=dedup_config.get("phash", {}).get("enabled", True),
        phash_threshold=dedup_config.get("phash", {}).get("threshold", 8),
        ssim_enabled=dedup_config.get("ssim", {}).get("enabled", True),
        ssim_threshold=dedup_config.get("ssim", {}).get("threshold", 0.92),
        clip_enabled=dedup_config.get("clip", {}).get("enabled", True),
        clip_model=dedup_config.get("clip", {}).get("model", "ViT-B-32"),
        clip_pretrained=dedup_config.get("clip", {}).get("pretrained", "openai"),
        clip_threshold=dedup_config.get("clip", {}).get("threshold", 0.90),
        clip_device=dedup_config.get("clip", {}).get("device", "auto"),
        clip_batch_size=dedup_config.get("clip", {}).get("batch_size", 32)
    )
