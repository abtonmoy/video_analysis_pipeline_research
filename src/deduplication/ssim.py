import logging
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import cv2

from base import BaseDeduplicator
from phash import PHashDeduplicator

logger = logging.getLogger(__name__)
# ============================================================================
# SSIM Deduplicator
# ============================================================================

class SSIMDeduplicator(BaseDeduplicator):
    """
    Structural Similarity Index based deduplication.
    Medium speed, catches frames with same structure but different details.
    """
    
    def __init__(self, threshold: float = 0.92):
        """
        Args:
            threshold: Minimum SSIM score to consider similar (default: 0.92)
        """
        self.threshold = threshold
    
    def compute_signature(self, frame: np.ndarray) -> np.ndarray:
        """Store grayscale frame as signature."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize to standard size for comparison
        return cv2.resize(gray, (256, 256))
    
    def are_similar(self, gray1: np.ndarray, gray2: np.ndarray) -> bool:
        """Compute SSIM and check threshold."""
        from skimage.metrics import structural_similarity as ssim
        
        score = ssim(gray1, gray2)
        return score > self.threshold
