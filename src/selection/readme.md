# Frame Selection System

A sophisticated frame selection system for video analysis that intelligently identifies and extracts the most important frames from video content. Originally designed for advertisement analysis, this system prioritizes key moments like calls-to-action (CTAs), brand reveals, and important audio events.

## Overview

This system solves a fundamental challenge in video processing: **how do you select the most representative and important frames from thousands of candidates?**

Rather than simply sampling frames uniformly or using naive clustering, this system:

- **Scores frame importance** based on position, audio events, scene boundaries, and visual features
- **Uses Non-Maximum Suppression (NMS)** to select high-importance frames while avoiding redundancy
- **Adapts to scene complexity** by allocating more frames to complex or important scenes
- **Ensures temporal and semantic diversity** while respecting importance scores

## Core Components

### 1. Importance Scoring (`representative.py`)

The `ImportanceScorer` evaluates each frame based on multiple signals:

**Position-Based Scoring:**

- Opening frames (first 10%) - typically contain brand introductions
- Closing frames (last 10%) - often feature CTAs
- Middle frames (40-60%) - core messaging

**Audio Event Scoring:**

- Energy peaks (attention-grabbing moments)
- Post-silence frames (attention reset points)
- Speech segment boundaries (important transitions)
- Key phrase proximity (promotional keywords) - **strongest signal**

**Scene-Based Scoring:**

- Scene openings (new content introduction)
- Scene endings (transition points)

**Visual Feature Scoring** (optional):

- Text overlays (CTAs, captions)
- Face detection (testimonials, presenters)
- Logo detection (brand moments)

All signals are combined **multiplicatively**, meaning frames near multiple important events score significantly higher (e.g., a frame at video opening + near key phrase + after silence could score 2.5-3.0x baseline).

### 2. Frame Selection Methods (`clustering.py`)

The system supports four selection strategies:

#### **NMS (Non-Maximum Suppression)** - RECOMMENDED

The default and most effective method for importance-aware selection.

**How it works:**

1. Sort frames by importance score (highest first)
2. Iteratively select frames, suppressing nearby candidates that are:
   - Too close temporally (< 0.5s by default)
   - Too similar semantically (cosine similarity > 0.88)
3. Add diversity bonus to encourage semantic variety

**Why NMS is better than K-means:**

- Directly uses importance scores (K-means ignores them)
- Guarantees temporal spacing
- Respects forced frames (first/last)
- More interpretable selection decisions

#### **Hybrid**

Combines NMS importance-awareness with K-means diversity guarantees.

**How it works:**

1. Use NMS to select top importance frames
2. Check if all semantic clusters are represented
3. Swap low-importance frames for cluster representatives if needed

**Best for:** Videos where semantic diversity is critical (e.g., product showcases)

#### **K-means**

Clusters frames by CLIP embeddings, selects representatives from each cluster.

**Improvements over naive K-means:**

- Selects highest-importance frame near each cluster centroid (60% importance + 40% proximity)
- Respects forced frames

**Best for:** When semantic coverage matters more than importance

#### **Uniform**

Divides timeline into segments, picks highest-importance frame from each.

**Best for:** Fallback when embeddings unavailable

### 3. Adaptive Density Allocation

Instead of fixed frames-per-scene, the system allocates frames proportionally:

```python
frames_for_scene = scene_duration * density_factor
```

**Density adjustments:**

- High visual complexity (variance > 0.15): +30% density
- Low visual complexity (variance < 0.05): -30% density
- High-importance scenes (max score > 1.5): +20% density

This ensures long scenes get adequate coverage while short scenes aren't oversampled.

### 4. Scene-Aware Processing

The `TemporalClusterer` processes each scene independently:

1. Assign frames to scenes based on boundaries
2. Calculate adaptive density for each scene
3. Select representatives using chosen method
4. Always include first and last frames of video

## Usage

### Basic Example

```python
from src.selection.representative import FrameSelector

# Initialize selector
selector = FrameSelector(
    target_frame_density=0.25,        # ~1 frame every 4 seconds
    min_frames_per_scene=2,           # At least 2 frames per scene
    max_frames_per_scene=10,          # At most 10 frames per scene
    min_temporal_gap_s=0.5,           # 0.5s minimum between frames
    clustering_method="nms",          # Use NMS (recommended)
    adaptive_density=True,            # Adjust for scene complexity
)

# Select frames
selected = selector.select(
    frames=[(timestamp, frame_array), ...],     # Your frame candidates
    embeddings=clip_embeddings,                  # CLIP embeddings (optional)
    scene_boundaries=[(start, end), ...],       # Scene boundaries
    video_duration=30.0,                        # Total duration
    audio_events={                              # Audio analysis results
        "energy_peaks": [2.1, 15.3, 28.7],
        "silence_segments": [(5.0, 6.5), ...],
        "speech_segments": [(0.0, 4.5), ...],
        "key_phrases": [
            {"timestamp": 28.5, "text": "buy now"}
        ]
    },
    visual_features={                           # Visual analysis (optional)
        "has_text": True,
        "has_face": False,
        "has_logo": True
    }
)

# Access selected frames
for frame_candidate in selected:
    print(f"Selected frame at {frame_candidate.timestamp:.2f}s")
    print(f"  Importance: {frame_candidate.importance_score:.2f}")
    print(f"  Scene: {frame_candidate.scene_id}")
```

### Configuration File

```yaml
selection:
  method: "nms" # "nms", "kmeans", "uniform", or "hybrid"
  target_frame_density: 0.25 # Frames per second target
  min_frames_per_scene: 2
  max_frames_per_scene: 10
  min_temporal_gap_s: 0.5
  adaptive_density: true

  # NMS-specific settings
  nms:
    semantic_threshold: 0.88 # Similarity threshold for suppression
    use_semantic_suppression: true # Use CLIP embeddings
    diversity_bonus: 0.1 # Bonus for diverse frames

  # Importance scoring weights
  importance:
    enabled: true
    position_weight: 1.0 # Video position importance
    scene_weight: 1.0 # Scene position importance
    audio_weight: 1.0 # Audio event importance
    key_phrase_boost: 1.5 # Multiplier for key phrases
```

Load with:

```python
from src.selection.representative import create_selector

selector = create_selector(config)
```

## Data Structures

### FrameCandidate

Container for frame metadata:

```python
@dataclass
class FrameCandidate:
    timestamp: float                      # Frame time in seconds
    frame: np.ndarray                     # Frame pixels
    embedding: Optional[np.ndarray]       # CLIP embedding
    scene_id: Optional[int]               # Scene assignment
    importance_score: float               # Computed importance (1.0 = baseline)
    cluster_id: Optional[int]             # K-means cluster (if used)
    is_representative: bool               # Selected for output?
    suppression_reason: Optional[str]     # Why suppressed (debugging)
```

### Audio Events Dictionary

Expected structure:

```python
audio_events = {
    "energy_peaks": [2.1, 5.3, ...],                    # Timestamps of energy peaks
    "silence_segments": [(5.0, 6.5), ...],              # (start, end) tuples
    "speech_segments": [(0.0, 4.5), (10.2, 15.8), ...], # (start, end) tuples
    "key_phrases": [
        {"timestamp": 28.5, "text": "buy now"},
        {"timestamp": 15.2, "text": "limited time"},
        ...
    ]
}
```

## Algorithm Details

### NMS Selection Process

```
1. Separate forced frames (first/last) from candidates
2. For each candidate:
   - Compute effective score = importance * weight + diversity_bonus
3. Sort candidates by effective score (descending)
4. Iterate through sorted candidates:
   - Check if suppressed by already-selected frames
   - If temporal gap < threshold: SUPPRESS
   - If semantic similarity > threshold: SUPPRESS
   - Otherwise: SELECT
5. Return selected frames sorted by timestamp
```

### Suppression Criteria

A candidate is suppressed if ANY selected frame satisfies:

```python
# Temporal suppression
abs(candidate.timestamp - selected.timestamp) < temporal_threshold_s

# Semantic suppression (if embeddings available)
cosine_similarity(candidate.embedding, selected.embedding) > semantic_threshold
```

### Adaptive Density Calculation

```python
base_density = 0.25  # ~1 frame per 4 seconds

# Adjust for visual complexity
if frame_variance > 0.15:
    density *= 1.3  # More frames for complex scenes
elif frame_variance < 0.05:
    density *= 0.7  # Fewer frames for static scenes

# Adjust for importance
if max_importance_in_scene > 1.5:
    density *= 1.2  # More frames for important scenes
```

## Performance Characteristics

**Computational Complexity:**

- NMS selection: O(n²) worst case, O(n log n) typical (early termination)
- K-means: O(n _ k _ iterations)
- Uniform: O(n)

**Memory:**

- Stores all candidates in memory
- CLIP embeddings: 512 floats per frame
- Typical: ~100-500 candidates for 30s video

**Typical Results:**

- Input: 500-1000 candidate frames (1-2 fps sampling)
- Output: 8-15 selected frames
- Reduction: 95-98%
- Importance lift: 1.3-1.8x (selected frames 30-80% more important than average)

## Debugging

### Get Selection Statistics

```python
stats = selector.get_selection_stats(candidates, selected)
print(stats)
```

Output:

```python
{
    "total_candidates": 500,
    "selected_count": 12,
    "reduction_rate": 0.976,
    "candidate_importance": {
        "min": 0.85,
        "max": 3.12,
        "mean": 1.15,
        "std": 0.42
    },
    "selected_importance": {
        "min": 1.21,
        "max": 3.12,
        "mean": 1.89,
        "std": 0.58
    },
    "importance_lift": 1.64
}
```

### Inspect Suppression Reasons

```python
for candidate in candidates:
    if not candidate.is_representative and candidate.suppression_reason:
        print(f"{candidate.timestamp:.2f}s: {candidate.suppression_reason}")
```

Output:

```
3.45s: temporal (dt=0.32s < 0.5s)
7.82s: semantic (sim=0.912 > 0.88)
12.10s: temporal (dt=0.41s < 0.5s)
```

## Tuning Guide

### For Advertisement Extraction

```python
selector = FrameSelector(
    clustering_method="nms",
    target_frame_density=0.2,        # Sparse sampling
    semantic_threshold=0.90,          # Allow similar frames (brand repetition)
    key_phrase_boost=2.0,            # Strong emphasis on CTAs
    position_weight=1.5,             # Prioritize opening/closing
)
```

### For Content Summarization

```python
selector = FrameSelector(
    clustering_method="hybrid",      # Balance importance + diversity
    target_frame_density=0.3,        # Denser sampling
    semantic_threshold=0.85,         # Suppress more similar frames
    adaptive_density=True,           # Adapt to scene complexity
)
```

### For Maximum Diversity

```python
selector = FrameSelector(
    clustering_method="kmeans",      # Pure semantic clustering
    target_frame_density=0.25,
    min_frames_per_scene=3,          # Ensure scene coverage
)
```

## Dependencies

- `numpy` - Array operations
- `sklearn` - K-means clustering (optional, only for kmeans/hybrid methods)
- CLIP embeddings (optional but recommended for semantic suppression)

## Common Pitfalls

1. **Missing embeddings**: NMS semantic suppression requires CLIP embeddings. Without them, only temporal suppression is used.

2. **Too aggressive suppression**: If `semantic_threshold` is too low (e.g., 0.7), too many frames are suppressed. Start with 0.88.

3. **Insufficient scene boundaries**: Poor scene detection leads to suboptimal frame allocation. Ensure scene boundaries are accurate.

4. **Ignoring audio events**: The system works without audio, but importance scores are far less effective. Audio events (especially key phrases) are crucial for ad extraction.

5. **Not forcing first/last frames**: The system automatically includes first and last frames, but if you override this, you may miss opening/closing content.
