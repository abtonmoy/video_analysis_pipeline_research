# src\deduplication\hierarchical.py
"""
Hierarchical deduplication using cheap-to-expensive methods.
Features a voting system across multiple hash algorithms for robust detection.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import cv2

from .base import BaseDeduplicator
from .phash import PHashDeduplicator
from .dhash import DHashDeduplicator
from .whash import WHashDeduplicator
from .ssim import SSIMDeduplicator
from .clip_embed import CLIPDeduplicator

logger = logging.getLogger(__name__)


# ============================================================================
# Hash Voting Deduplicator
# ============================================================================

class HashVotingDeduplicator(BaseDeduplicator):
    """
    Multi-hash voting deduplicator combining pHash, dHash, and wHash.
    
    Each hash type excels at different scenarios:
    - pHash (Frequency domain): Good for scaling/compression artifacts
    - dHash (Gradient domain): Excellent for brightness/contrast changes  
    - wHash (Wavelet domain): Robust to noise and small perturbations
    
    Frames are considered similar if at least `min_votes` hash algorithms
    agree they are similar.
    """
    
    def __init__(
        self,
        phash_threshold: int = 8,
        dhash_threshold: int = 8,
        whash_threshold: int = 8,
        min_votes: int = 2,
        hash_size: int = 8
    ):
        """
        Args:
            phash_threshold: Max Hamming distance for pHash similarity
            dhash_threshold: Max Hamming distance for dHash similarity
            whash_threshold: Max Hamming distance for wHash similarity
            min_votes: Minimum number of hash algorithms that must agree (1-3)
            hash_size: Size of hash for dHash and wHash (default: 8)
        """
        self.min_votes = min(max(min_votes, 1), 3)  # Clamp to 1-3
        
        self.phash = PHashDeduplicator(threshold=phash_threshold)
        self.dhash = DHashDeduplicator(threshold=dhash_threshold, hash_size=hash_size)
        self.whash = WHashDeduplicator(threshold=whash_threshold, hash_size=hash_size)
        
        self.phash_threshold = phash_threshold
        self.dhash_threshold = dhash_threshold
        self.whash_threshold = whash_threshold
    
    def compute_signature(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Compute all three hash signatures for a frame.
        
        Returns:
            Dict with 'phash', 'dhash', 'whash' keys
        """
        return {
            'phash': self.phash.compute_signature(frame),
            'dhash': self.dhash.compute_signature(frame),
            'whash': self.whash.compute_signature(frame)
        }
    
    def are_similar(self, sig1: Dict[str, Any], sig2: Dict[str, Any]) -> bool:
        """
        Check if two frames are similar using voting.
        
        Returns True if at least `min_votes` hash algorithms agree.
        """
        votes = 0
        
        # Vote 1: pHash
        if self.phash.are_similar(sig1['phash'], sig2['phash']):
            votes += 1
        
        # Vote 2: dHash
        if self.dhash.are_similar(sig1['dhash'], sig2['dhash']):
            votes += 1
        
        # Vote 3: wHash
        if self.whash.are_similar(sig1['whash'], sig2['whash']):
            votes += 1
        
        return votes >= self.min_votes
    
    def get_vote_details(
        self, 
        sig1: Dict[str, Any], 
        sig2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get detailed voting information between two signatures.
        
        Returns:
            Dict with distances, individual votes, and total votes
        """
        phash_dist = sig1['phash'] - sig2['phash']
        dhash_dist = sig1['dhash'] - sig2['dhash']
        whash_dist = sig1['whash'] - sig2['whash']
        
        phash_vote = phash_dist < self.phash_threshold
        dhash_vote = dhash_dist < self.dhash_threshold
        whash_vote = whash_dist < self.whash_threshold
        
        return {
            'phash_distance': phash_dist,
            'dhash_distance': dhash_dist,
            'whash_distance': whash_dist,
            'phash_vote': phash_vote,
            'dhash_vote': dhash_vote,
            'whash_vote': whash_vote,
            'total_votes': sum([phash_vote, dhash_vote, whash_vote]),
            'is_similar': sum([phash_vote, dhash_vote, whash_vote]) >= self.min_votes
        }


# ============================================================================
# Hierarchical Deduplicator
# ============================================================================

class HierarchicalDeduplicator:
    """
    Hierarchical deduplication using cheap-to-expensive methods.
    
    Pipeline: Hash Voting (pHash + dHash + wHash) -> SSIM -> CLIP
    
    The first stage uses a voting system across three complementary hash algorithms:
    - pHash: Frequency domain - good for scaling/compression
    - dHash: Gradient domain - excellent for brightness changes
    - wHash: Wavelet domain - robust to noise
    
    Each subsequent stage filters frames before passing to the more expensive next stage.
    """
    
    def __init__(
        self,
        # Hash voting parameters
        hash_voting_enabled: bool = True,
        phash_threshold: int = 8,
        dhash_threshold: int = 8,
        whash_threshold: int = 8,
        min_hash_votes: int = 2,
        hash_size: int = 8,
        # SSIM parameters
        ssim_enabled: bool = False,
        ssim_threshold: float = 0.92,
        # CLIP parameters
        clip_enabled: bool = True,
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        clip_threshold: float = 0.90,
        clip_device: str = "auto",
        clip_batch_size: int = 32
    ):
        """
        Args:
            hash_voting_enabled: Enable multi-hash voting (pHash + dHash + wHash)
            phash_threshold: Max Hamming distance for pHash
            dhash_threshold: Max Hamming distance for dHash
            whash_threshold: Max Hamming distance for wHash
            min_hash_votes: Minimum votes needed to consider similar (1-3)
            hash_size: Size of hash for dHash and wHash
            ssim_enabled: Enable SSIM stage
            ssim_threshold: SSIM similarity threshold
            clip_enabled: Enable CLIP semantic stage
            clip_model: CLIP model name
            clip_pretrained: CLIP pretrained weights
            clip_threshold: CLIP cosine similarity threshold
            clip_device: Device for CLIP ('auto', 'cuda', 'cpu')
            clip_batch_size: Batch size for CLIP inference
        """
        self.hash_voting_enabled = hash_voting_enabled
        self.ssim_enabled = ssim_enabled
        self.clip_enabled = clip_enabled
        
        # Initialize hash voting deduplicator
        if hash_voting_enabled:
            self.hash_voter = HashVotingDeduplicator(
                phash_threshold=phash_threshold,
                dhash_threshold=dhash_threshold,
                whash_threshold=whash_threshold,
                min_votes=min_hash_votes,
                hash_size=hash_size
            )
        
        # Initialize SSIM deduplicator
        if ssim_enabled:
            self.ssim = SSIMDeduplicator(threshold=ssim_threshold)
        
        # Initialize CLIP deduplicator
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
        
        # Stage 1: Hash Voting (fastest - combines pHash, dHash, wHash)
        if self.hash_voting_enabled and len(current_frames) > 1:
            current_frames = self.hash_voter.deduplicate(current_frames)
            stats["after_hash_voting"] = len(current_frames)
            logger.debug(
                f"Hash voting stage: {stats['input']} -> {stats['after_hash_voting']} frames "
                f"(min_votes={self.hash_voter.min_votes})"
            )
        
        # Stage 2: SSIM (medium speed)
        if self.ssim_enabled and len(current_frames) > 1:
            current_frames = self.ssim.deduplicate(current_frames)
            stats["after_ssim"] = len(current_frames)
        
        # Stage 3: CLIP (slowest but semantic understanding)
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
        
    Expected config format:
        {
            "deduplication": {
                "hash_voting": {
                    "enabled": True,
                    "phash_threshold": 8,
                    "dhash_threshold": 8,
                    "whash_threshold": 8,
                    "min_votes": 2,
                    "hash_size": 8
                },
                "ssim": {
                    "enabled": False,
                    "threshold": 0.92
                },
                "clip": {
                    "enabled": True,
                    "model": "ViT-B/32",
                    "threshold": 0.90,
                    "device": "cpu",
                    "batch_size": 32
                }
            }
        }
        
    Returns:
        Configured HierarchicalDeduplicator
    """
    dedup_config = config.get("deduplication", {})
    hash_config = dedup_config.get("hash_voting", {})
    
    # Normalize CLIP model name (handle both "ViT-B/32" and "ViT-B-32" formats)
    clip_model = dedup_config.get("clip", {}).get("model", "ViT-B-32")
    clip_model = clip_model.replace("/", "-")  # Normalize to open_clip format
    
    return HierarchicalDeduplicator(
        # Hash voting parameters
        hash_voting_enabled=hash_config.get("enabled", True),
        phash_threshold=hash_config.get("phash_threshold", 8),
        dhash_threshold=hash_config.get("dhash_threshold", 8),
        whash_threshold=hash_config.get("whash_threshold", 8),
        min_hash_votes=hash_config.get("min_votes", 2),
        hash_size=hash_config.get("hash_size", 8),
        # SSIM parameters
        ssim_enabled=dedup_config.get("ssim", {}).get("enabled", False),
        ssim_threshold=dedup_config.get("ssim", {}).get("threshold", 0.92),
        # CLIP parameters
        clip_enabled=dedup_config.get("clip", {}).get("enabled", True),
        clip_model=clip_model,
        clip_pretrained=dedup_config.get("clip", {}).get("pretrained", "openai"),
        clip_threshold=dedup_config.get("clip", {}).get("threshold", 0.90),
        clip_device=dedup_config.get("clip", {}).get("device", "auto"),
        clip_batch_size=dedup_config.get("clip", {}).get("batch_size", 32)
    )