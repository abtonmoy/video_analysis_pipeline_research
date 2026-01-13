# src\detection\change_detector.py
import logging
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
import numpy as np
import cv2

logger = logging.getLogger(__name__)


# ============================================================================
# Change Detection
# ============================================================================

class ChangeDetector(ABC):
    """Abstract base class for frame change detection."""
    
    @abstractmethod
    def compute_change(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute change score between two frames.
        
        Args:
            frame1: First frame (BGR)
            frame2: Second frame (BGR)
            
        Returns:
            Change score (higher = more change)
        """
        pass
    
    def is_significant_change(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        threshold: float
    ) -> bool:
        """Check if change exceeds threshold."""
        return self.compute_change(frame1, frame2) > threshold


class FrameDifferenceDetector(ChangeDetector):
    """Detect changes using pixel-wise frame difference."""
    
    def compute_change(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute normalized L1 difference between frames."""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Resize if needed
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        
        diff = np.abs(gray1.astype(float) - gray2.astype(float))
        return np.mean(diff) / 255.0


class HistogramDetector(ChangeDetector):
    """Detect changes using color histogram comparison."""
    
    def __init__(self, bins: int = 16):
        self.bins = bins
    
    def compute_change(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute chi-square distance between color histograms."""
        hist1 = self._compute_histogram(frame1)
        hist2 = self._compute_histogram(frame2)
        
        # Chi-square distance
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    
    def _compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Compute 3D color histogram."""
        hist = cv2.calcHist(
            [frame], [0, 1, 2], None,
            [self.bins, self.bins, self.bins],
            [0, 256, 0, 256, 0, 256]
        )
        cv2.normalize(hist, hist)
        return hist.flatten()


class EdgeChangeDetector(ChangeDetector):
    """Detect changes using edge comparison."""
    
    def compute_change(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute edge change ratio between frames."""
        edges1 = self._compute_edges(frame1)
        edges2 = self._compute_edges(frame2)
        
        # Resize if needed
        if edges1.shape != edges2.shape:
            edges2 = cv2.resize(edges2, (edges1.shape[1], edges1.shape[0]))
        
        # XOR to find changed edges
        changed = cv2.bitwise_xor(edges1, edges2)
        total = max(cv2.countNonZero(edges1), cv2.countNonZero(edges2), 1)
        
        return cv2.countNonZero(changed) / total
    
    def _compute_edges(self, frame: np.ndarray) -> np.ndarray:
        """Compute Canny edges."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 100, 200)


def get_change_detector(method: str = "histogram") -> ChangeDetector:
    """
    Factory function to get change detector by name.
    
    Args:
        method: "frame_diff", "histogram", or "edge"
        
    Returns:
        ChangeDetector instance
    """
    detectors = {
        "frame_diff": FrameDifferenceDetector,
        "histogram": HistogramDetector,
        "edge": EdgeChangeDetector
    }
    
    if method not in detectors:
        raise ValueError(f"Unknown method: {method}. Available: {list(detectors.keys())}")
    
    return detectors[method]()


class AdaptiveChangeDetector(ChangeDetector):
    """
    Adaptive change detection that adjusts threshold based on video statistics.
    """
    
    def __init__(
        self,
        method: str = "histogram",
        base_threshold: float = 0.15,
        adaptation_window: int = 30
    ):
        self.detector = get_change_detector(method)
        self.base_threshold = base_threshold
        self.adaptation_window = adaptation_window
        self.change_history = []

    def compute_change(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute change score between two frames using the underlying detector.
        
        Args:
            frame1: First frame (BGR)
            frame2: Second frame (BGR)
            
        Returns:
            Change score (higher = more change)
        """
        return self.detector.compute_change(frame1, frame2)
    
    def get_adaptive_threshold(self) -> float:
        """Compute adaptive threshold based on recent changes."""
        if len(self.change_history) < self.adaptation_window:
            return self.base_threshold
        
        recent = self.change_history[-self.adaptation_window:]
        mean_change = np.mean(recent)
        std_change = np.std(recent)
        
        # Threshold at mean + 1 std, but bounded
        adaptive = mean_change + std_change
        return max(self.base_threshold * 0.5, min(adaptive, self.base_threshold * 2))
    
    def process_frame(
        self,
        current_frame: np.ndarray,
        previous_frame: np.ndarray
    ) -> Tuple[float, bool]:
        """
        Process frame and determine if it represents significant change.
        
        Returns:
            Tuple of (change_score, is_significant)
        """
        change = self.detector.compute_change(previous_frame, current_frame)
        self.change_history.append(change)
        
        threshold = self.get_adaptive_threshold()
        is_significant = change > threshold
        
        return change, is_significant

