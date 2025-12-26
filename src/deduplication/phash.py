import logging
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import cv2

from base import BaseDeduplicator

logger = logging.getLogger(__name__)

# ============================================================================
# pHash Deduplicator
# ============================================================================

class PHashDeduplicator(BaseDeduplicator):
    """
    Perceptual hash based deduplication.
    Very fast, catches near-identical frames.
    """
    
    def __init__(self, threshold: int = 8):
        """
        Args:
            threshold: Maximum Hamming distance to consider similar (default: 8)
        """
        self.threshold = threshold
        self._imagehash = None
    
    def _get_imagehash(self):
        """Lazy load imagehash."""
        if self._imagehash is None:
            import imagehash
            self._imagehash = imagehash
        return self._imagehash
    
    def compute_signature(self, frame: np.ndarray) -> 'imagehash.ImageHash':
        """Compute perceptual hash of frame."""
        imagehash = self._get_imagehash()
        
        # Convert to PIL
        if isinstance(frame, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            pil_image = frame
        
        return imagehash.phash(pil_image)
    
    def are_similar(self, hash1, hash2) -> bool:
        """Check if two hashes are similar within threshold."""
        return (hash1 - hash2) < self.threshold

