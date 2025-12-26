"""
Hierarchical deduplication module using pHash, SSIM, and CLIP.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


# ============================================================================
# Base Deduplicator
# ============================================================================

class BaseDeduplicator(ABC):
    """Abstract base class for frame deduplication."""
    
    @abstractmethod
    def are_similar(self, frame1: Any, frame2: Any) -> bool:
        """Check if two frames are similar."""
        pass
    
    @abstractmethod
    def compute_signature(self, frame: np.ndarray) -> Any:
        """Compute a signature/hash for a frame."""
        pass
    
    def deduplicate(
        self,
        frames: List[Tuple[float, np.ndarray]]
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Remove duplicate frames from list.
        
        Args:
            frames: List of (timestamp, frame) tuples
            
        Returns:
            Deduplicated list of (timestamp, frame) tuples
        """
        if not frames:
            return []
        
        # Compute signatures
        signatures = [(ts, frame, self.compute_signature(frame)) for ts, frame in frames]
        
        # Keep first frame always
        kept = [signatures[0]]
        
        for ts, frame, sig in signatures[1:]:
            # Compare with all kept frames
            is_duplicate = False
            for _, _, kept_sig in kept:
                if self.are_similar(sig, kept_sig):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept.append((ts, frame, sig))
        
        logger.info(f"{self.__class__.__name__}: {len(frames)} -> {len(kept)} frames")
        return [(ts, frame) for ts, frame, _ in kept]
