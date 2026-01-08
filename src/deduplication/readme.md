# Frame Deduplication Module

A comprehensive Python module for detecting and removing duplicate or near-duplicate frames from video sequences. The module implements multiple deduplication strategies ranging from fast hash-based methods to sophisticated deep learning approaches.

## Overview

This module provides a hierarchical deduplication pipeline that progressively applies cheap-to-expensive similarity detection methods:

1. **Hash Voting** (fastest) - Combines multiple perceptual hashing algorithms
2. **SSIM/LPIPS** (medium) - Structural or perceptual similarity metrics
3. **CLIP** (slowest) - Semantic understanding via vision-language models

## Features

- Multiple deduplication algorithms with different speed/accuracy tradeoffs
- Hierarchical pipeline for efficient processing
- Voting system across complementary hash algorithms
- Batch processing support for deep learning methods
- Flexible configuration and threshold tuning
- Detailed statistics and debugging information

## Installation

### Core Requirements

```bash
pip install numpy opencv-python pillow
```

### Hash-Based Methods

```bash
pip install imagehash
```

### Structural Similarity

```bash
pip install scikit-image
```

### Perceptual Similarity (LPIPS)

```bash
pip install lpips torch torchvision
```

### Semantic Similarity (CLIP)

```bash
pip install open-clip-torch torch torchvision
```

## Quick Start

### Basic Usage

```python
from deduplication.hierarchical import HierarchicalDeduplicator

# Create deduplicator with default settings
dedup = HierarchicalDeduplicator()

# Deduplicate frames
frames = [(0.0, frame1), (1.0, frame2), (2.0, frame3)]  # List of (timestamp, frame) tuples
deduplicated_frames, embeddings, stats = dedup.deduplicate(frames)

print(f"Reduced from {stats['input']} to {stats['output']} frames")
```

### Configuration-Based Setup

```python
from deduplication.hierarchical import create_deduplicator

config = {
    "deduplication": {
        "hash_voting": {
            "enabled": True,
            "phash_threshold": 8,
            "dhash_threshold": 8,
            "whash_threshold": 8,
            "min_votes": 2,
            "hash_size": 8
        },
        "clip": {
            "enabled": True,
            "model": "ViT-B-32",
            "threshold": 0.90,
            "device": "auto",
            "batch_size": 32
        }
    }
}

dedup = create_deduplicator(config)
deduplicated_frames, embeddings, stats = dedup.deduplicate(frames)
```

## Deduplication Methods

### Hash-Based Methods (Fastest)

#### pHash (Perceptual Hash)

- **Speed**: Very fast
- **Use case**: Near-identical frames, robust to scaling/compression
- **Algorithm**: DCT-based frequency domain analysis
- **Threshold**: Hamming distance (typical: 8)

```python
from deduplication.phash import PHashDeduplicator

dedup = PHashDeduplicator(threshold=8)
result = dedup.deduplicate(frames)
```

#### dHash (Difference Hash)

- **Speed**: Very fast
- **Use case**: Excellent for brightness/contrast changes
- **Algorithm**: Gradient-based (compares adjacent pixels)
- **Threshold**: Hamming distance (typical: 8)

```python
from deduplication.dhash import DHashDeduplicator

dedup = DHashDeduplicator(threshold=8, hash_size=8)
result = dedup.deduplicate(frames)
```

#### wHash (Wavelet Hash)

- **Speed**: Very fast
- **Use case**: Robust to noise and small perturbations
- **Algorithm**: Haar wavelet decomposition
- **Threshold**: Hamming distance (typical: 8)

```python
from deduplication.whash import WHashDeduplicator

dedup = WHashDeduplicator(threshold=8, hash_size=8, mode='haar')
result = dedup.deduplicate(frames)
```

### Hash Voting (Recommended for Speed)

Combines pHash, dHash, and wHash with a voting mechanism. Frames are considered similar if at least `min_votes` algorithms agree.

```python
from deduplication.hierarchical import HashVotingDeduplicator

dedup = HashVotingDeduplicator(
    phash_threshold=8,
    dhash_threshold=8,
    whash_threshold=8,
    min_votes=2,  # At least 2 out of 3 must agree
    hash_size=8
)

result = dedup.deduplicate(frames)
```

**Why voting works well:**

- pHash: Good for scaling/compression artifacts
- dHash: Excellent for brightness/contrast changes
- wHash: Robust to noise and perturbations
- Combined: Catches duplicates across different degradation types

### Structural Similarity (SSIM)

- **Speed**: Medium
- **Use case**: Identical frames with same structure
- **Algorithm**: Structural similarity index
- **Threshold**: Similarity score (typical: 0.92)

```python
from deduplication.ssim import SSIMDeduplicator

dedup = SSIMDeduplicator(threshold=0.92)
result = dedup.deduplicate(frames)
```

### Perceptual Similarity (LPIPS)

- **Speed**: Medium-slow (GPU recommended)
- **Use case**: Better human perception correlation than SSIM
- **Algorithm**: Learned features from AlexNet/VGG/SqueezeNet
- **Threshold**: Perceptual distance (typical: 0.15, lower = stricter)

```python
from deduplication.lpips import LPIPSDeduplicator

dedup = LPIPSDeduplicator(
    threshold=0.15,
    net="alex",  # "alex", "vgg", or "squeeze"
    device="auto"
)

result = dedup.deduplicate(frames)
```

**LPIPS threshold guide:**

- 0.1: Very strict, only near-identical
- 0.15: Moderate (recommended)
- 0.2: Lenient
- 0.3: Very lenient

### Semantic Similarity (CLIP)

- **Speed**: Slowest (GPU strongly recommended)
- **Use case**: Semantic deduplication (e.g., same scene from different angles)
- **Algorithm**: Vision-language model embeddings
- **Threshold**: Cosine similarity (typical: 0.90)

```python
from deduplication.clip_embed import CLIPDeduplicator

dedup = CLIPDeduplicator(
    model_name="ViT-B-32",
    pretrained="openai",
    threshold=0.90,
    device="auto",
    batch_size=32
)

deduplicated_frames, embeddings = dedup.deduplicate(frames)
```

## Hierarchical Pipeline

The recommended approach for most use cases combines multiple methods in a hierarchical pipeline:

```python
from deduplication.hierarchical import HierarchicalDeduplicator

dedup = HierarchicalDeduplicator(
    # Stage 1: Hash voting (fastest)
    hash_voting_enabled=True,
    phash_threshold=8,
    dhash_threshold=8,
    whash_threshold=8,
    min_hash_votes=2,

    # Stage 2a: SSIM (optional, faster)
    ssim_enabled=False,
    ssim_threshold=0.92,

    # Stage 2b: LPIPS (optional, slower but better)
    lpips_enabled=False,
    lpips_threshold=0.15,
    lpips_net="alex",

    # Stage 3: CLIP (semantic understanding)
    clip_enabled=True,
    clip_model="ViT-B-32",
    clip_threshold=0.90,
    clip_device="auto",
    clip_batch_size=32
)

frames, embeddings, stats = dedup.deduplicate(frames)
```

**Pipeline stages:**

1. **Hash Voting**: Removes obvious duplicates quickly (90%+ of work)
2. **SSIM/LPIPS**: Catches structural/perceptual duplicates (optional refinement)
3. **CLIP**: Semantic deduplication for remaining frames

## Performance Considerations

### Speed Comparison (1000 frames)

| Method      | Speed | GPU Required | Memory |
| ----------- | ----- | ------------ | ------ |
| pHash       | ~1s   | No           | Low    |
| dHash       | ~1s   | No           | Low    |
| wHash       | ~1s   | No           | Low    |
| Hash Voting | ~2s   | No           | Low    |
| SSIM        | ~10s  | No           | Medium |
| LPIPS       | ~30s  | Recommended  | High   |
| CLIP        | ~60s  | Recommended  | High   |

### Recommended Configurations

**Fast (CPU-friendly):**

```python
HierarchicalDeduplicator(
    hash_voting_enabled=True,
    min_hash_votes=2,
    ssim_enabled=False,
    lpips_enabled=False,
    clip_enabled=False
)
```

**Balanced (moderate GPU):**

```python
HierarchicalDeduplicator(
    hash_voting_enabled=True,
    min_hash_votes=2,
    clip_enabled=True,
    clip_device="cuda"
)
```

**High Quality (strong GPU):**

```python
HierarchicalDeduplicator(
    hash_voting_enabled=True,
    min_hash_votes=2,
    lpips_enabled=True,
    lpips_net="alex",
    clip_enabled=True,
    clip_device="cuda"
)
```

## Advanced Usage

### Get Detailed Similarity Information

```python
from deduplication.hierarchical import HashVotingDeduplicator

dedup = HashVotingDeduplicator(min_votes=2)
sig1 = dedup.compute_signature(frame1)
sig2 = dedup.compute_signature(frame2)

details = dedup.get_vote_details(sig1, sig2)
print(f"pHash distance: {details['phash_distance']}")
print(f"dHash distance: {details['dhash_distance']}")
print(f"wHash distance: {details['whash_distance']}")
print(f"Total votes: {details['total_votes']}")
print(f"Similar: {details['is_similar']}")
```

### LPIPS Distance Analysis

```python
from deduplication.lpips import LPIPSDeduplicator

dedup = LPIPSDeduplicator(threshold=0.15)
info = dedup.get_perceptual_distance(frame1, frame2)
print(f"LPIPS distance: {info['lpips_distance']:.4f}")
print(f"Similarity: {info['similarity_percent']:.1f}%")
```

### Custom Deduplication Logic

```python
from deduplication.base import BaseDeduplicator

class CustomDeduplicator(BaseDeduplicator):
    def compute_signature(self, frame):
        # Your custom signature computation
        return your_signature

    def are_similar(self, sig1, sig2):
        # Your custom similarity check
        return your_similarity_check(sig1, sig2)
```

## Threshold Tuning Guide

### Hash Methods (Hamming Distance)

- **0-5**: Very strict, only nearly identical frames
- **6-8**: Moderate (recommended)
- **9-12**: Lenient, catches more variations
- **13+**: Very lenient, may cause false positives

### SSIM (Structural Similarity)

- **0.95+**: Very strict
- **0.90-0.94**: Moderate (recommended)
- **0.85-0.89**: Lenient
- **< 0.85**: Very lenient

### LPIPS (Perceptual Distance)

- **< 0.1**: Very strict
- **0.1-0.15**: Moderate (recommended)
- **0.15-0.25**: Lenient
- **> 0.25**: Very lenient

### CLIP (Cosine Similarity)

- **0.95+**: Very strict, only semantically identical
- **0.90-0.94**: Moderate (recommended)
- **0.85-0.89**: Lenient, similar scenes
- **< 0.85**: Very lenient, may group unrelated content

## Architecture

```
deduplication/
├── base.py              # Abstract base class
├── phash.py             # Perceptual hash
├── dhash.py             # Difference hash
├── whash.py             # Wavelet hash
├── ssim.py              # Structural similarity
├── lpips.py             # Learned perceptual similarity
├── clip_embed.py        # CLIP embeddings
└── hierarchical.py      # Multi-stage pipeline
```

## Contributing

Contributions welcome! Areas for improvement:

- Additional hash algorithms (aHash, colorHash)
- Video-specific temporal coherence checks
- Scene detection integration
- Parallel processing optimizations
- Additional deep learning backbones
