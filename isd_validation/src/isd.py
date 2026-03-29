"""
Core ISD computation and utilities.
"""

import numpy as np
from typing import Optional


def compute_isd(embeddings: np.ndarray, tau: float = 0.90) -> int:
    """
    Compute Intrinsic Semantic Dimensionality via SVD.
    
    ISD is the number of principal components needed to explain `tau` fraction
    of variance in the embedding matrix.
    
    Args:
        embeddings: Array of shape (n_frames, n_dims)
        tau: Variance threshold (default 0.90 for 90%)
    
    Returns:
        ISD value (integer >= 1)
    
    Reference:
        Equation 4 in paper: k* = min{k : sum(sigma_i^2)/sum(sigma_j^2) >= tau}
    """
    if embeddings.shape[0] < 3:
        return max(1, embeddings.shape[0] // 2)
    
    # Center the embeddings
    centered = embeddings - embeddings.mean(axis=0)
    
    try:
        # Compute SVD
        _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
        
        # Compute explained variance
        var_explained = (singular_values ** 2) / (np.sum(singular_values ** 2) + 1e-9)
        cum_var = np.cumsum(var_explained)
        
        # Find number of components for tau variance
        isd = int(np.argmax(cum_var >= tau)) + 1
        return isd
        
    except np.linalg.LinAlgError:
        # Fallback if SVD fails
        return max(1, embeddings.shape[0] // 5)


def compute_semantic_velocity(embeddings: np.ndarray) -> float:
    """
    Compute mean L2 distance between consecutive frame embeddings.
    
    This measures how quickly visual content changes over time.
    Higher values indicate more dynamic content.
    
    Args:
        embeddings: Array of shape (n_frames, n_dims)
    
    Returns:
        Mean L2 distance (float)
    
    Reference:
        Equation 2 in paper: v_bar = mean(||phi(f_{t+1}) - phi(f_t)||_2)
    """
    if embeddings.shape[0] < 2:
        return 0.5
    
    diffs = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
    return float(np.mean(diffs))


def compute_visual_diversity(embeddings: np.ndarray) -> float:
    """
    Compute mean pairwise cosine distance between all frames.
    
    This measures overall visual diversity independent of temporal order.
    
    Args:
        embeddings: Array of shape (n_frames, n_dims)
    
    Returns:
        Mean pairwise distance (float)
    """
    if embeddings.shape[0] < 2:
        return 0.0
    
    # Normalize embeddings
    normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    
    # Compute pairwise distances
    similarities = normalized @ normalized.T
    distances = 1 - similarities
    
    # Mean of upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(distances), k=1).astype(bool)
    return float(np.mean(distances[mask]))


def compute_cut_frequency(scene_boundaries: list, duration: float) -> float:
    """
    Compute scene changes per second.
    
    Args:
        scene_boundaries: List of (start, end) tuples
        duration: Video duration in seconds
    
    Returns:
        Cut frequency (scenes per second)
    """
    if duration <= 0:
        return 0.0
    return len(scene_boundaries) / duration
