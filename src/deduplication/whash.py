# src\deduplication\whash.py
"""
Wavelet Hash (wHash) based deduplication.
Robust to noise and small perturbations using wavelet decomposition.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import cv2

from .base import BaseDeduplicator

logger = logging.getLogger(__name__)


class WHashDeduplicator(BaseDeduplicator):
    """
    Wavelet hash (wHash) based deduplication.
    
    wHash uses discrete wavelet transform (Haar wavelet) to decompose
    the image into frequency components, making it robust to noise
    and small perturbations while preserving structural information.
    """
    
    def __init__(self, threshold: int = 8, hash_size: int = 8, mode: str = 'haar'):
        """
        Args:
            threshold: Maximum Hamming distance to consider similar (default: 8)
            hash_size: Size of the hash (default: 8 produces 64-bit hash)
            mode: Wavelet mode - 'haar' (default) or 'db4'
        """
        self.threshold = threshold
        self.hash_size = hash_size
        self.mode = mode
        self._imagehash = None
    
    def _get_imagehash(self):
        """Lazy load imagehash."""
        if self._imagehash is None:
            import imagehash
            self._imagehash = imagehash
        return self._imagehash
    
    def compute_signature(self, frame: np.ndarray) -> 'imagehash.ImageHash':
        """
        Compute wavelet hash of frame.
        
        wHash algorithm:
        1. Convert to grayscale
        2. Resize to power of 2 size
        3. Apply Haar wavelet transform
        4. Extract low-frequency coefficients
        5. Convert to binary hash based on median
        """
        imagehash = self._get_imagehash()
        
        # Convert to PIL
        if isinstance(frame, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            pil_image = frame
        
        return imagehash.whash(pil_image, hash_size=self.hash_size, mode=self.mode)
    
    def are_similar(self, hash1, hash2) -> bool:
        """Check if two hashes are similar within threshold."""
        return (hash1 - hash2) < self.threshold
    
    def get_hamming_distance(self, hash1, hash2) -> int:
        """Get the Hamming distance between two hashes."""
        return hash1 - hash2