# src\deduplication\dhash.py
"""
Difference Hash (dHash) based deduplication.
Excellent for detecting brightness changes by comparing adjacent pixels.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import cv2

from .base import BaseDeduplicator

logger = logging.getLogger(__name__)


class DHashDeduplicator(BaseDeduplicator):
    """
    Difference hash (dHash) based deduplication.
    
    dHash works by computing the gradient (difference) between adjacent pixels,
    making it excellent for detecting brightness and contrast changes.
    Very fast and robust to scaling.
    """
    
    def __init__(self, threshold: int = 8, hash_size: int = 8):
        """
        Args:
            threshold: Maximum Hamming distance to consider similar (default: 8)
            hash_size: Size of the hash (default: 8 produces 64-bit hash)
        """
        self.threshold = threshold
        self.hash_size = hash_size
        self._imagehash = None
    
    def _get_imagehash(self):
        """Lazy load imagehash."""
        if self._imagehash is None:
            import imagehash
            self._imagehash = imagehash
        return self._imagehash
    
    def compute_signature(self, frame: np.ndarray) -> 'imagehash.ImageHash':
        """
        Compute difference hash of frame.
        
        dHash algorithm:
        1. Reduce to grayscale
        2. Resize to (hash_size+1, hash_size)
        3. Compute horizontal gradient (compare adjacent pixels)
        4. Convert to binary hash
        """
        imagehash = self._get_imagehash()
        
        # Convert to PIL
        if isinstance(frame, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            pil_image = frame
        
        return imagehash.dhash(pil_image, hash_size=self.hash_size)
    
    def are_similar(self, hash1, hash2) -> bool:
        """Check if two hashes are similar within threshold."""
        return (hash1 - hash2) < self.threshold
    
    def get_hamming_distance(self, hash1, hash2) -> int:
        """Get the Hamming distance between two hashes."""
        return hash1 - hash2