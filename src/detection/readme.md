# Frame Detection Module

A comprehensive Python module for detecting scene changes and extracting significant frames from video sequences. The module provides multiple change detection algorithms and scene boundary detection with both PySceneDetect integration and fallback methods.

## Overview

This module provides two main capabilities:

1. **Change Detection** - Detect when significant changes occur between consecutive frames
2. **Scene Detection** - Identify scene boundaries and extract representative frames

The module supports multiple detection algorithms with different speed/accuracy tradeoffs and includes adaptive thresholding for dynamic content.

## Features

- Multiple change detection algorithms (frame difference, histogram, edge-based)
- Adaptive threshold adjustment based on video statistics
- Scene boundary detection with PySceneDetect integration
- Fallback detection for systems without PySceneDetect
- Candidate frame extraction for further processing
- Configurable sampling intervals and thresholds
- Memory-efficient frame iteration

## Installation

### Core Requirements

```bash
pip install numpy opencv-python
```

### Optional: Scene Detection

```bash
pip install scenedetect[opencv]
```

Note: PySceneDetect is optional. The module includes a fallback scene detector if PySceneDetect is not installed.

## Quick Start

### Basic Change Detection

```python
from detection.change_detector import get_change_detector

# Create a change detector
detector = get_change_detector("histogram")

# Compute change between two frames
change_score = detector.compute_change(frame1, frame2)

# Check if change is significant
is_significant = detector.is_significant_change(frame1, frame2, threshold=0.15)
```

### Scene Detection

```python
from detection.scene_detector import SceneDetector

# Create scene detector
scene_detector = SceneDetector(
    method="content",
    threshold=27.0,
    min_scene_length_s=0.5
)

# Detect scenes
scenes = scene_detector.detect_scenes("video.mp4")

for start, end in scenes:
    print(f"Scene: {start:.2f}s - {end:.2f}s")
```

### Candidate Frame Extraction

```python
from detection.change_detector import get_change_detector
from detection.scene_detector import CandidateFrameExtractor

# Setup change detector
change_detector = get_change_detector("histogram")

# Create extractor
extractor = CandidateFrameExtractor(
    change_detector=change_detector,
    threshold=0.15,
    min_interval_ms=100,
    sample_interval_ms=50
)

# Extract frames where significant changes occur
candidates = extractor.extract_candidates("video.mp4", max_resolution=720)

print(f"Extracted {len(candidates)} candidate frames")
for timestamp, frame in candidates:
    print(f"Candidate at {timestamp:.2f}s")
```

## Change Detection Algorithms

### 1. Frame Difference Detector (Fastest)

Detects changes using pixel-wise frame differences in grayscale.

**Best for:**

- Fast processing requirements
- Simple content changes
- Overall brightness/contrast shifts

**Algorithm:**

- Convert to grayscale
- Compute absolute difference
- Normalize by pixel count

```python
from detection.change_detector import FrameDifferenceDetector

detector = FrameDifferenceDetector()
change = detector.compute_change(frame1, frame2)
# Returns: 0.0 (identical) to 1.0 (completely different)
```

**Typical thresholds:**

- 0.05-0.10: Very sensitive, catches minor changes
- 0.10-0.20: Moderate (recommended)
- 0.20-0.30: Less sensitive, major changes only

### 2. Histogram Detector (Recommended)

Detects changes using color histogram comparison via chi-square distance.

**Best for:**

- Color composition changes
- Scene transitions
- Lighting changes
- Robust to small motions

**Algorithm:**

- Compute 3D color histogram (16x16x16 bins)
- Compare using chi-square distance
- Normalized for frame size

```python
from detection.change_detector import HistogramDetector

detector = HistogramDetector(bins=16)
change = detector.compute_change(frame1, frame2)
# Returns: chi-square distance (0 = identical, higher = more different)
```

**Typical thresholds:**

- 0.05-0.10: Very sensitive
- 0.10-0.20: Moderate (recommended)
- 0.20-0.40: Less sensitive

**Performance:** ~2-3x slower than frame difference, but more robust

### 3. Edge Change Detector (Specialized)

Detects changes by comparing edge structures using Canny edge detection.

**Best for:**

- Structural changes
- Object appearance/disappearance
- Shape-based transitions
- Invariant to color/brightness changes

**Algorithm:**

- Compute Canny edges (100, 200 thresholds)
- XOR edge maps
- Compute changed edge ratio

```python
from detection.change_detector import EdgeChangeDetector

detector = EdgeChangeDetector()
change = detector.compute_change(frame1, frame2)
# Returns: ratio of changed edges (0.0 to 1.0)
```

**Typical thresholds:**

- 0.10-0.20: Very sensitive
- 0.20-0.30: Moderate (recommended)
- 0.30-0.50: Less sensitive

**Performance:** ~5-10x slower than frame difference due to edge detection

## Adaptive Change Detection

The `AdaptiveChangeDetector` automatically adjusts thresholds based on video statistics, making it ideal for content with varying dynamics.

```python
from detection.change_detector import AdaptiveChangeDetector

detector = AdaptiveChangeDetector(
    method="histogram",           # Base detection method
    base_threshold=0.15,          # Starting threshold
    adaptation_window=30          # Frames to consider for adaptation
)

# Process frames sequentially
for current_frame in frames:
    change_score, is_significant = detector.process_frame(
        current_frame,
        previous_frame
    )

    if is_significant:
        print(f"Significant change detected: {change_score:.3f}")

    previous_frame = current_frame

# Get current adaptive threshold
current_threshold = detector.get_adaptive_threshold()
```

**How it works:**

- Maintains sliding window of recent change scores
- Computes mean and standard deviation
- Sets threshold at `mean + 1σ`
- Bounded between `base_threshold * 0.5` and `base_threshold * 2`

**Benefits:**

- Adapts to slow-paced vs action-heavy content
- Reduces false positives in static scenes
- Increases sensitivity in dynamic scenes

## Scene Detection

### Using PySceneDetect (Recommended)

```python
from detection.scene_detector import SceneDetector

# Content-based detection (default)
detector = SceneDetector(
    method="content",
    threshold=27.0,              # Lower = more sensitive
    min_scene_length_s=0.5       # Minimum scene duration
)

scenes = detector.detect_scenes("video.mp4")
```

**Methods:**

1. **Content Detector** (`method="content"`)

   - Detects fast cuts and content changes
   - Uses frame-by-frame comparison
   - Threshold: 15-40 (default: 27)
   - Lower = more sensitive

2. **Threshold Detector** (`method="threshold"`)
   - Detects cuts based on average frame intensity
   - Uses fade in/out detection
   - Threshold: 8-32 (default: 12)

### Fallback Detection

If PySceneDetect is not installed, the module automatically uses a fallback detector:

```python
# Fallback automatically activated if scenedetect not available
detector = SceneDetector(method="content")
scenes = detector.detect_scenes("video.mp4")
```

**Fallback behavior:**

- Uses `FrameDifferenceDetector`
- Samples at 200ms intervals
- Hard threshold of 0.3
- Less accurate but always available

## Candidate Frame Extraction

Extract frames where significant changes occur for downstream processing (e.g., deduplication, analysis).

```python
from detection.change_detector import HistogramDetector
from detection.scene_detector import CandidateFrameExtractor

# Setup
change_detector = HistogramDetector(bins=16)
extractor = CandidateFrameExtractor(
    change_detector=change_detector,
    threshold=0.15,              # Change detection threshold
    min_interval_ms=100,         # Minimum time between candidates
    sample_interval_ms=50        # Frame sampling interval
)

# Extract candidates
candidates = extractor.extract_candidates(
    video_path="video.mp4",
    max_resolution=720           # Downsample for efficiency
)

# Process candidates
for timestamp, frame in candidates:
    # Frame is a numpy array (BGR format)
    process_frame(frame)
```

**Parameters explained:**

- `threshold`: Change score threshold for significance
- `min_interval_ms`: Prevents selecting frames too close together
- `sample_interval_ms`: How often to check for changes (lower = more CPU)
- `max_resolution`: Downsamples frames to this height for efficiency

**Use cases:**

- Pre-filtering for deduplication pipeline
- Keyframe extraction
- Event detection
- Summary generation

## Algorithm Comparison

| Algorithm        | Speed   | Memory | Best Use Case      | Invariances      |
| ---------------- | ------- | ------ | ------------------ | ---------------- |
| Frame Difference | Fastest | Low    | General changes    | None             |
| Histogram        | Fast    | Medium | Scene transitions  | Small motions    |
| Edge             | Slow    | Medium | Structural changes | Color/brightness |
| Adaptive         | Fast\*  | Medium | Variable content   | Adapts to video  |

\*Speed depends on underlying detector

## Performance Tips

### 1. Choose the Right Detector

```python
# Fast processing, general purpose
detector = get_change_detector("frame_diff")

# Balanced speed/quality (recommended)
detector = get_change_detector("histogram")

# Structural changes only
detector = get_change_detector("edge")
```

### 2. Adjust Sampling Intervals

```python
# Fast extraction (may miss quick changes)
extractor = CandidateFrameExtractor(
    change_detector=detector,
    sample_interval_ms=200  # Check every 200ms
)

# Thorough extraction (slower)
extractor = CandidateFrameExtractor(
    change_detector=detector,
    sample_interval_ms=33   # Check every frame (~30 fps)
)
```

### 3. Use Resolution Limits

```python
# Extract at lower resolution for speed
candidates = extractor.extract_candidates(
    "video.mp4",
    max_resolution=480  # 480p max height
)
```

### 4. Combine with Deduplication

```python
from detection.change_detector import HistogramDetector
from detection.scene_detector import CandidateFrameExtractor
from deduplication.hierarchical import HierarchicalDeduplicator

# Stage 1: Extract candidates (fast)
detector = HistogramDetector()
extractor = CandidateFrameExtractor(detector, threshold=0.15)
candidates = extractor.extract_candidates("video.mp4")

# Stage 2: Deduplicate candidates (slower but fewer frames)
dedup = HierarchicalDeduplicator()
final_frames, embeddings, stats = dedup.deduplicate(candidates)
```

## Threshold Tuning Guide

### Frame Difference

- Video lectures/presentations: 0.15-0.25
- Action videos: 0.10-0.15
- Static camera footage: 0.20-0.30

### Histogram

- Smooth transitions: 0.10-0.15
- Fast cuts: 0.15-0.25
- Mixed content: 0.12-0.18 (recommended)

### Edge

- Object tracking: 0.20-0.30
- Scene changes: 0.25-0.40
- Structural analysis: 0.15-0.25

### PySceneDetect Content

- Sensitive (more scenes): 15-23
- Moderate (recommended): 24-30
- Conservative (fewer scenes): 31-40

## Advanced Usage

### Custom Change Detector

```python
from detection.change_detector import ChangeDetector
import cv2
import numpy as np

class OpticalFlowDetector(ChangeDetector):
    """Detect changes using optical flow magnitude."""

    def compute_change(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return np.mean(magnitude)

# Use custom detector
detector = OpticalFlowDetector()
change = detector.compute_change(frame1, frame2)
```

### Scene-Aware Processing

```python
from detection.scene_detector import SceneDetector
from detection.change_detector import HistogramDetector
from detection.scene_detector import CandidateFrameExtractor

# Detect scenes
scene_detector = SceneDetector()
scenes = scene_detector.detect_scenes("video.mp4")

# Extract candidates per scene
detector = HistogramDetector()
extractor = CandidateFrameExtractor(detector)

for scene_idx, (start, end) in enumerate(scenes):
    print(f"Processing scene {scene_idx}: {start:.2f}s - {end:.2f}s")

    # Extract frames only from this scene
    # (requires custom VideoFrameIterator with time bounds)
    scene_candidates = extract_scene_frames(start, end)

    # Process scene-specific candidates
    process_scene(scene_candidates)
```

### Multi-Method Voting

```python
from detection.change_detector import (
    FrameDifferenceDetector,
    HistogramDetector,
    EdgeChangeDetector
)

class VotingChangeDetector:
    """Combine multiple detectors with voting."""

    def __init__(self):
        self.detectors = [
            (FrameDifferenceDetector(), 0.15),
            (HistogramDetector(), 0.15),
            (EdgeChangeDetector(), 0.25)
        ]

    def compute_change(self, frame1, frame2):
        votes = 0
        for detector, threshold in self.detectors:
            if detector.is_significant_change(frame1, frame2, threshold):
                votes += 1
        return votes >= 2  # Majority vote

detector = VotingChangeDetector()
```

## Common Workflows

### 1. Keyframe Extraction

```python
from detection.change_detector import HistogramDetector
from detection.scene_detector import CandidateFrameExtractor

detector = HistogramDetector(bins=16)
extractor = CandidateFrameExtractor(
    detector,
    threshold=0.18,
    min_interval_ms=500  # At least 500ms between keyframes
)

keyframes = extractor.extract_candidates("video.mp4")
```

### 2. Scene Summarization

```python
from detection.scene_detector import SceneDetector
from detection.change_detector import AdaptiveChangeDetector

# Detect scenes
scene_detector = SceneDetector(threshold=27.0)
scenes = scene_detector.detect_scenes("video.mp4")

# Extract representative frame per scene
change_detector = AdaptiveChangeDetector(method="histogram")

summaries = []
for start, end in scenes:
    # Get middle frame as representative
    mid_time = (start + end) / 2
    frame = extract_frame_at_time(mid_time)
    summaries.append((mid_time, frame))
```

### 3. Event Detection

```python
from detection.change_detector import EdgeChangeDetector

detector = EdgeChangeDetector()
events = []

for i in range(len(frames) - 1):
    change = detector.compute_change(frames[i], frames[i+1])

    if change > 0.4:  # High threshold for significant events
        events.append({
            'timestamp': timestamps[i+1],
            'change_score': change,
            'type': 'structural_change'
        })
```

## Architecture

```
detection/
├── change_detector.py       # Change detection algorithms
│   ├── ChangeDetector       # Abstract base class
│   ├── FrameDifferenceDetector
│   ├── HistogramDetector
│   ├── EdgeChangeDetector
│   └── AdaptiveChangeDetector
│
└── scene_detector.py        # Scene and candidate extraction
    ├── SceneDetector        # PySceneDetect wrapper
    └── CandidateFrameExtractor
```

## Dependencies

**Required:**

- numpy
- opencv-python

**Optional:**

- scenedetect[opencv] - For advanced scene detection

## Troubleshooting

### PySceneDetect Import Error

```python
# Module automatically falls back to simple detection
# To use full features, install:
pip install scenedetect[opencv]
```

### Memory Issues with Large Videos

```python
# Reduce resolution
extractor = CandidateFrameExtractor(detector)
candidates = extractor.extract_candidates(
    "large_video.mp4",
    max_resolution=480  # Lower resolution
)

# Increase sampling interval
extractor = CandidateFrameExtractor(
    detector,
    sample_interval_ms=200  # Check less frequently
)
```

### Too Many/Few Candidates

```python
# Too many: Increase threshold or min_interval
extractor = CandidateFrameExtractor(
    detector,
    threshold=0.25,        # Higher = fewer candidates
    min_interval_ms=200    # More spacing
)

# Too few: Decrease threshold or use adaptive
detector = AdaptiveChangeDetector(
    base_threshold=0.10    # Lower = more candidates
)
```

## License

MIT License

## Contributing

Contributions welcome! Areas for improvement:

- Additional change detection algorithms (optical flow, SIFT/SURF)
- GPU-accelerated detection methods
- Temporal smoothing and filtering
- Multi-scale change detection
- Audio-based scene detection integration
