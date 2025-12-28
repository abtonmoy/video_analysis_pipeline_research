import logging
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import cv2

from .base import BaseDeduplicator
from .phash import PHashDeduplicator
from .ssim import SSIMDeduplicator

logger = logging.getLogger(__name__)

# ============================================================================
# CLIP Deduplicator
# ============================================================================

class CLIPDeduplicator(BaseDeduplicator):
    """
    CLIP embedding based deduplication.
    Slower but catches semantically similar frames.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        threshold: float = 0.90,
        device: str = "auto",
        batch_size: int = 32
    ):
        """
        Args:
            model_name: CLIP model name
            pretrained: Pretrained weights
            threshold: Minimum cosine similarity to consider similar
            device: "auto", "cuda", or "cpu"
            batch_size: Batch size for embedding computation
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.threshold = threshold
        self.batch_size = batch_size
        
        # Set device
        import torch
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self._model = None
        self._preprocess = None
        self._tokenizer = None
    
    def _load_model(self):
        """Lazy load CLIP model."""
        if self._model is not None:
            return
        
        try:
            import open_clip
            
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device
            )
            self._model.eval()
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            
            logger.info(f"Loaded CLIP model: {self.model_name} on {self.device}")
            
        except ImportError:
            raise ImportError("open_clip not installed. Run: pip install open-clip-torch")
    
    def compute_signature(self, frame: np.ndarray) -> np.ndarray:
        """Compute CLIP embedding for frame."""
        self._load_model()
        
        import torch
        
        # Convert to PIL
        if isinstance(frame, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            pil_image = frame
        
        # Preprocess and embed
        image_input = self._preprocess(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self._model.encode_image(image_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy().flatten()
    
    def compute_signatures_batch(
        self,
        frames: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute CLIP embeddings for multiple frames efficiently.
        
        Args:
            frames: List of frames (BGR numpy arrays)
            
        Returns:
            Array of shape (num_frames, embedding_dim)
        """
        self._load_model()
        
        import torch
        
        embeddings = []
        
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i + self.batch_size]
            
            # Convert and preprocess batch
            pil_images = [
                Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                for f in batch
            ]
            
            batch_tensor = torch.stack([
                self._preprocess(img) for img in pil_images
            ]).to(self.device)
            
            # Batch inference
            with torch.no_grad():
                batch_embeddings = self._model.encode_image(batch_tensor)
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
            
            embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def are_similar(self, emb1: np.ndarray, emb2: np.ndarray) -> bool:
        """Check cosine similarity between embeddings."""
        similarity = np.dot(emb1, emb2)
        return similarity > self.threshold
    
    def deduplicate(
        self,
        frames: List[Tuple[float, np.ndarray]]
    ) -> Tuple[List[Tuple[float, np.ndarray]], np.ndarray]:
        """
        Deduplicate frames and return embeddings for later use.
        
        Returns:
            Tuple of (deduplicated frames, embeddings for kept frames)
        """
        if not frames:
            return [], np.array([])
        
        # Batch compute embeddings
        frame_arrays = [f for _, f in frames]
        embeddings = self.compute_signatures_batch(frame_arrays)
        
        # Keep first frame always
        kept_indices = [0]
        
        for i in range(1, len(frames)):
            is_duplicate = False
            
            for kept_idx in kept_indices:
                similarity = np.dot(embeddings[i], embeddings[kept_idx])
                if similarity > self.threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept_indices.append(i)
        
        kept_frames = [(frames[i][0], frames[i][1]) for i in kept_indices]
        kept_embeddings = embeddings[kept_indices]
        
        logger.info(f"CLIPDeduplicator: {len(frames)} -> {len(kept_frames)} frames")
        
        return kept_frames, kept_embeddings