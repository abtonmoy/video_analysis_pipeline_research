# src\deduplication\lpips.py
"""
LPIPS (Learned Perceptual Image Patch Similarity) based deduplication.

LPIPS uses deep features from pretrained networks (AlexNet, VGG, SqueezeNet)
to compute perceptual similarity. It's more aligned with human perception
than traditional metrics like SSIM or MSE.

Key advantages over SSIM:
- Better correlation with human perceptual judgments
- More robust to minor geometric/photometric changes
- Captures semantic similarity, not just structural similarity
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import cv2

from .base import BaseDeduplicator

logger = logging.getLogger(__name__)


class LPIPSDeduplicator(BaseDeduplicator):
    """
    LPIPS-based deduplication using learned perceptual similarity.
    
    LPIPS (Learned Perceptual Image Patch Similarity) computes distance
    in deep feature space, providing perceptual similarity scores that
    correlate well with human judgments.
    
    Lower LPIPS distance = more similar images.
    """
    
    def __init__(
        self,
        threshold: float = 0.15,
        net: str = "alex",
        device: str = "auto",
        spatial: bool = False
    ):
        """
        Args:
            threshold: Maximum LPIPS distance to consider similar (default: 0.15)
                      Lower = stricter matching. Typical values:
                      - 0.1: Very strict, only near-identical
                      - 0.15: Moderate (recommended)
                      - 0.2: Lenient, catches more variations
                      - 0.3: Very lenient
            net: Network backbone - "alex" (fastest), "vgg" (most accurate), "squeeze"
            device: "auto", "cuda", or "cpu"
            spatial: If True, return spatial map instead of scalar
        """
        self.threshold = threshold
        self.net = net
        self.spatial = spatial
        
        # Set device
        import torch
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self._model = None
        self._transform = None
    
    def _load_model(self):
        """Lazy load LPIPS model."""
        if self._model is not None:
            return
        
        try:
            import lpips
            import torch
            
            self._model = lpips.LPIPS(net=self.net, spatial=self.spatial)
            self._model = self._model.to(self.device)
            self._model.eval()
            
            logger.info(f"Loaded LPIPS model: {self.net} on {self.device}")
            
        except ImportError:
            raise ImportError(
                "lpips not installed. Run: pip install lpips\n"
                "Note: lpips requires torch to be installed first."
            )
    
    def _preprocess_frame(self, frame: np.ndarray) -> 'torch.Tensor':
        """
        Preprocess frame for LPIPS.
        
        LPIPS expects images in [-1, 1] range with shape (N, C, H, W).
        
        Args:
            frame: BGR numpy array (H, W, C)
            
        Returns:
            Preprocessed tensor
        """
        import torch
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to standard size for efficiency (LPIPS works on any size but smaller is faster)
        rgb = cv2.resize(rgb, (224, 224))
        
        # Convert to float and normalize to [-1, 1]
        tensor = torch.from_numpy(rgb).float()
        tensor = tensor / 127.5 - 1.0  # [0, 255] -> [-1, 1]
        
        # Rearrange to (C, H, W) and add batch dimension
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def compute_signature(self, frame: np.ndarray) -> np.ndarray:
        """
        Store preprocessed frame as signature.
        
        Unlike hash-based methods, LPIPS compares images directly,
        so we store the preprocessed frame for later comparison.
        """
        # Store resized RGB frame for comparison
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        return resized
    
    def compute_distance(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute LPIPS distance between two frames.
        
        Args:
            frame1: First frame (RGB numpy array from compute_signature)
            frame2: Second frame (RGB numpy array from compute_signature)
            
        Returns:
            LPIPS distance (lower = more similar)
        """
        self._load_model()
        
        import torch
        
        # Convert to tensors
        def to_tensor(rgb):
            tensor = torch.from_numpy(rgb).float()
            tensor = tensor / 127.5 - 1.0
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            return tensor.to(self.device)
        
        t1 = to_tensor(frame1)
        t2 = to_tensor(frame2)
        
        with torch.no_grad():
            distance = self._model(t1, t2)
        
        return distance.item()
    
    def are_similar(self, sig1: np.ndarray, sig2: np.ndarray) -> bool:
        """
        Check if two frames are perceptually similar.
        
        Args:
            sig1: First signature (preprocessed frame)
            sig2: Second signature (preprocessed frame)
            
        Returns:
            True if LPIPS distance < threshold
        """
        distance = self.compute_distance(sig1, sig2)
        return distance < self.threshold
    
    def compute_distances_batch(
        self,
        reference: np.ndarray,
        candidates: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute LPIPS distances from reference to all candidates efficiently.
        
        Args:
            reference: Reference frame signature
            candidates: List of candidate frame signatures
            
        Returns:
            Array of distances
        """
        self._load_model()
        
        import torch
        
        def to_tensor(rgb):
            tensor = torch.from_numpy(rgb).float()
            tensor = tensor / 127.5 - 1.0
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            return tensor.to(self.device)
        
        ref_tensor = to_tensor(reference)
        
        distances = []
        
        # Process in batches to avoid OOM
        batch_size = 16
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            
            # Stack batch
            batch_tensors = torch.cat([to_tensor(c) for c in batch], dim=0)
            
            # Expand reference to match batch size
            ref_expanded = ref_tensor.expand(len(batch), -1, -1, -1)
            
            with torch.no_grad():
                batch_distances = self._model(ref_expanded, batch_tensors)
            
            distances.extend(batch_distances.cpu().numpy().flatten().tolist())
        
        return np.array(distances)
    
    def deduplicate(
        self,
        frames: List[Tuple[float, np.ndarray]]
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Remove perceptually duplicate frames.
        
        Uses efficient batched comparison when possible.
        
        Args:
            frames: List of (timestamp, frame) tuples
            
        Returns:
            Deduplicated list of (timestamp, frame) tuples
        """
        if not frames:
            return []
        
        if len(frames) == 1:
            return frames
        
        self._load_model()
        
        # Compute signatures (preprocessed frames)
        signatures = [(ts, frame, self.compute_signature(frame)) for ts, frame in frames]
        
        # Keep first frame always
        kept = [signatures[0]]
        
        for ts, frame, sig in signatures[1:]:
            is_duplicate = False
            
            # Compare with all kept frames
            for _, _, kept_sig in kept:
                distance = self.compute_distance(sig, kept_sig)
                if distance < self.threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept.append((ts, frame, sig))
        
        logger.info(f"LPIPSDeduplicator: {len(frames)} -> {len(kept)} frames "
                   f"(threshold={self.threshold}, net={self.net})")
        
        return [(ts, frame) for ts, frame, _ in kept]
    
    def get_perceptual_distance(
        self, 
        frame1: np.ndarray, 
        frame2: np.ndarray
    ) -> Dict[str, Any]:
        """
        Get detailed perceptual distance information between two frames.
        
        Useful for debugging and analysis.
        
        Args:
            frame1: First frame (BGR numpy array)
            frame2: Second frame (BGR numpy array)
            
        Returns:
            Dict with distance, similarity assessment, and metadata
        """
        sig1 = self.compute_signature(frame1)
        sig2 = self.compute_signature(frame2)
        
        distance = self.compute_distance(sig1, sig2)
        
        return {
            'lpips_distance': distance,
            'threshold': self.threshold,
            'is_similar': distance < self.threshold,
            'similarity_percent': max(0, (1 - distance) * 100),
            'network': self.net,
            'device': self.device
        }