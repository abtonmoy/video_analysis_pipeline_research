# Cascaded Semantic Deduplication with Adaptive Density Selection for Efficient Video-Language Model Inference

## Abstract

This project presents a novel multi-stage pipeline for automated analysis of video advertisements that addresses the computational inefficiency of dense frame sampling in vision-language model (VLM) extraction tasks. Traditional approaches that uniformly sample frames result in redundant information extraction and excessive API costs. Our pipeline introduces a hierarchical deduplication framework combined with adaptive density-based frame selection that achieves 80-85% frame reduction while maintaining semantic completeness. The system employs perceptual hashing (pHash), structural similarity index (SSIM), and CLIP embeddings in a cascading architecture, followed by temporal clustering with scene-aware importance scoring. Experimental results on political and commercial advertisements demonstrate effective content extraction with significant cost reduction.

## Table of Contents

- [Introduction](#introduction)
- [System Architecture](#system-architecture)
- [Mathematical Framework](#mathematical-framework)
- [Pipeline Stages](#pipeline-stages)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Experimental Results](#experimental-results)
- [Technical Implementation](#technical-implementation)
- [Performance Metrics](#performance-metrics)
- [Future Work](#future-work)
- [Citation](#citation)

## Introduction

### Problem Statement

Video advertisement analysis faces a fundamental trade-off between sampling density and computational cost. Dense frame sampling (e.g., one frame per 100ms) captures temporal dynamics but generates highly redundant data, while sparse sampling risks missing critical narrative moments. For a 60-second video at 30 fps (1,800 total frames), dense sampling at 100ms intervals yields 600 frames, of which the majority contain redundant or uninformative content.

The inefficiency is compounded when using Vision-Language Models (VLMs) for content extraction:

- API costs scale linearly with frame count
- Processing time increases proportionally
- Redundant frames provide diminishing returns for semantic understanding

### Proposed Solution

We introduce a seven-stage pipeline that combines:

1. **Hierarchical Deduplication**: Three-tier filtering using pHash → SSIM → CLIP
2. **Adaptive Density-Based Selection**: Scene-duration-proportional frame allocation
3. **Temporal Clustering**: K-means clustering in CLIP embedding space
4. **Importance Scoring**: Multi-factor frame significance assessment
5. **Adaptive Schema Extraction**: Type-aware structured information extraction

Our approach achieves 80-85% frame reduction while preserving narrative coherence, resulting in proportional API cost savings and processing time reduction.

## System Architecture

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     INPUT: Video Advertisement                      │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 1: Video Ingestion & Preprocessing                            │
│  - Video metadata extraction (duration, fps, resolution)            │
│  - Optional audio extraction for multimodal analysis                │
│  - Resolution downscaling (max 720p) for efficiency                 │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 2: Scene Detection                                            │
│  - PySceneDetect ContentDetector (threshold = 27.0)                 │
│  - Minimum scene length: 0.5s                                       │
│  - Fallback mechanisms for static/dark videos                       │
│  Output: Scene boundaries [(t_start, t_end), ...]                   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 3: Candidate Frame Extraction                                 │
│  - Histogram-based change detection (threshold = 0.15)              │
│  - Sampling interval: 50ms                                          │
│  - Minimum temporal gap: 100ms                                      │
│  Output: Candidate frames C = {(t_i, f_i)}                          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 4: Hierarchical Deduplication                                 │
│                                                                      │
│  Step 4.1: Perceptual Hashing (pHash)                               │
│    - 8x8 DCT-based hash, Hamming distance ≤ 8                       │
│    - Complexity: O(n²) comparisons, ~2ms per frame                  │
│                                                                      │
│  Step 4.2: Structural Similarity Index (SSIM)                       │
│    - Threshold: 0.92 on grayscale 256x256                           │
│    - Complexity: O(n² × w × h), ~50ms per comparison                │
│                                                                      │
│  Step 4.3: CLIP Semantic Embeddings                                 │
│    - Model: ViT-B/32 (512-dimensional embeddings)                   │
│    - Cosine similarity threshold: 0.90                              │
│    - Batch processing: 32 frames per batch                          │
│    - Complexity: O(n × d) for embedding, O(n²) for comparison       │
│                                                                      │
│  Output: Deduplicated frames D ⊂ C, embeddings E ∈ R^(|D| × 512)   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 5: Audio Event Extraction (Optional)                          │
│  - RMS energy peak detection (90th percentile)                      │
│  - Silence detection (threshold: -40dB, min duration: 0.3s)         │
│  Output: Audio events A = {peaks, silences}                         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 6: Representative Frame Selection                             │
│                                                                      │
│  Step 6.1: Scene Assignment                                         │
│    - Map each frame to nearest scene by timestamp                   │
│                                                                      │
│  Step 6.2: Adaptive Density Calculation                             │
│    - Base density: ρ₀ = 0.25 frames/second                          │
│    - Scene variance: σ_s = var(frame differences)                   │
│    - Adjusted density: ρ_s = ρ₀ × α(σ_s)                            │
│      where α(σ_s) = 1.3 if σ_s > 0.15 (high motion)                │
│                   = 0.7 if σ_s < 0.05 (static)                      │
│                   = 1.0 otherwise                                    │
│                                                                      │
│  Step 6.3: Target Frame Allocation                                  │
│    - Per scene s: n_s = ⌊duration_s × ρ_s⌋                          │
│    - Constraints: 2 ≤ n_s ≤ 10, always include first & last frame   │
│                                                                      │
│  Step 6.4: Temporal Clustering (K-means)                            │
│    - Input: CLIP embeddings E_s for scene s                         │
│    - Clusters: k = n_s                                              │
│    - Representative: argmin_i ||e_i - c_j|| for each cluster j      │
│                                                                      │
│  Step 6.5: Importance Scoring                                       │
│    - Position score: w_p(t) = 1.5 if t < 0.1T (opening)             │
│                              = 1.3 if t > 0.9T (closing)            │
│    - Audio proximity: w_a(t) = 1.3 near energy peaks                │
│    - Scene boundary: w_s(t) = 1.4 at scene start/end                │
│    - Combined: I(f_i) = w_p(t_i) × w_a(t_i) × w_s(t_i)              │
│                                                                      │
│  Step 6.6: Temporal Gap Enforcement                                 │
│    - Minimum gap: δ = 0.5s between selected frames                  │
│    - Exception: preserve first and last frame always                │
│                                                                      │
│  Output: Selected frames S = {(t_i, f_i, s_i, I_i)}                 │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 7: LLM-Based Extraction                                       │
│                                                                      │
│  Step 7.1: Frame Encoding                                           │
│    - Resize to max 512px, JPEG quality 85                           │
│    - Base64 encoding for API transmission                           │
│                                                                      │
│  Step 7.2: Temporal Context Enrichment                              │
│    - Timestamp annotations: Frame i @ t_i seconds                   │
│    - Time deltas: Δt_i = t_i - t_(i-1)                              │
│    - Position labels: [OPENING], [MIDDLE], [CLOSING]                │
│                                                                      │
│  Step 7.3: Ad Type Detection (Two-Pass)                             │
│    - Pass 1: Classify into {product_demo, testimonial,              │
│               brand_awareness, tutorial, entertainment}             │
│    - Model: Gemini 2.0 Flash / Claude Sonnet 4                      │
│                                                                      │
│  Step 7.4: Structured Extraction                                    │
│    - Base schema: {brand, message, creative_elements,               │
│                    target_audience, persuasion_techniques}          │
│    - Type-specific extensions (e.g., emotional_appeal for           │
│      brand_awareness, product features for product_demo)            │
│    - Temperature: 0.0 for deterministic extraction                  │
│                                                                      │
│  Output: Structured JSON with metadata                              │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     OUTPUT: Structured Analysis                      │
│  - Brand identification                                              │
│  - Message and call-to-action                                        │
│  - Creative elements (colors, text overlays, music mood)             │
│  - Target audience and persuasion techniques                         │
│  - Type-specific insights                                            │
│  - Processing metrics (reduction rate, timing)                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Mathematical Framework

### 1. Change Detection

We employ histogram-based change detection using chi-square distance:

Let H₁ and H₂ be normalized color histograms (bins=16) for consecutive frames:

```
χ²(H₁, H₂) = Σᵢ (H₁(i) - H₂(i))² / (H₁(i) + H₂(i))
```

A frame pair exhibits significant change if χ²(H₁, H₂) > τ_change, where τ_change = 0.15.

### 2. Perceptual Hashing

Given frame f, compute pHash h(f):

1. Convert to grayscale: G = 0.299R + 0.587G + 0.114B
2. Resize to 32×32
3. Apply 2D DCT, extract low-frequency 8×8 submatrix
4. Compute median μ of DCT coefficients
5. Binary hash: h(f)[i,j] = 1 if DCT[i,j] > μ, else 0

Frames are similar if Hamming distance d_H(h(f₁), h(f₂)) ≤ τ_phash = 8.

### 3. Structural Similarity Index

SSIM between grayscale frames f₁ and f₂:

```
SSIM(f₁, f₂) = [l(f₁,f₂)]^α · [c(f₁,f₂)]^β · [s(f₁,f₂)]^γ
```

where:

- l(f₁,f₂) = (2μ₁μ₂ + c₁) / (μ₁² + μ₂² + c₁) (luminance)
- c(f₁,f₂) = (2σ₁σ₂ + c₂) / (σ₁² + σ₂² + c₂) (contrast)
- s(f₁,f₂) = (σ₁₂ + c₃) / (σ₁σ₂ + c₃) (structure)

With α = β = γ = 1, c₁ = (0.01L)², c₂ = (0.03L)², c₃ = c₂/2, L = 255.

Frames are similar if SSIM(f₁, f₂) > τ_ssim = 0.92.

### 4. CLIP Semantic Similarity

For frame f, extract CLIP embedding e(f) ∈ R⁵¹² using ViT-B/32:

```
e(f) = CLIP_encoder(f) / ||CLIP_encoder(f)||₂
```

Semantic similarity via cosine distance:

```
sim(f₁, f₂) = e(f₁)ᵀ · e(f₂) / (||e(f₁)||₂ · ||e(f₂)||₂)
```

Frames are semantically similar if sim(f₁, f₂) > τ_clip = 0.90.

### 5. Hierarchical Deduplication

Given candidate set C = {(t_i, f_i)} with |C| = n:

```
Stage 1 (pHash):  D₁ = {f ∈ C | ∀g ∈ D₁, d_H(h(f), h(g)) > τ_phash}
Stage 2 (SSIM):   D₂ = {f ∈ D₁ | ∀g ∈ D₂, SSIM(f, g) < τ_ssim}
Stage 3 (CLIP):   D₃ = {f ∈ D₂ | ∀g ∈ D₃, sim(f, g) < τ_clip}
```

Output: D = D₃ with |D| << |C|

Complexity: O(n) for pHash, O(n² · w · h) for SSIM, O(n · d + n²) for CLIP

### 6. Adaptive Density-Based Frame Allocation

For scene s with duration Δt_s, compute frame variance:

```
σ_s = (1/|D_s|-1) Σᵢ ||f_i - f_(i+1)||² / 255²
```

where D_s are deduplicated frames in scene s.

Adaptive density multiplier:

```
α(σ_s) = { 1.3,  σ_s > 0.15  (high motion)
         { 0.7,  σ_s < 0.05  (static)
         { 1.0,  otherwise
```

Target frame count for scene s:

```
n_s = clip(⌊Δt_s · ρ₀ · α(σ_s)⌋, n_min, n_max)
```

where ρ₀ = 0.25 frames/s, n_min = 2, n_max = 10.

### 7. K-Means Temporal Clustering

For scene s with embeddings E_s = {e₁, e₂, ..., e_m} and target n_s frames:

Initialize k = min(n_s, m) cluster centers randomly.

Iterate until convergence:

```
Assignment: c_i = argmin_j ||e_i - μ_j||₂
Update:     μ_j = (1/|C_j|) Σ_{i∈C_j} e_i
```

Select representative frame for cluster j:

```
f*_j = argmin_{f_i∈C_j} ||e_i - μ_j||₂
```

### 8. Multi-Factor Importance Scoring

Combined importance score:

```
I(f_i) = w_position(t_i, T) · w_audio(t_i, A) · w_scene(t_i, s_i)
```

Position weight:

```
w_position(t, T) = { 1.5,  t/T < 0.10  (opening 10%)
                   { 1.3,  t/T > 0.90  (closing 10%)
                   { 1.0,  otherwise
```

Audio event proximity (with proximity threshold δ_a = 0.5s):

```
w_audio(t, A) = { 1.3,  ∃p∈A.peaks: |t-p| < δ_a
                { 1.4,  ∃(s,e)∈A.silences: e ≤ t < e+δ_a
                { 1.0,  otherwise
```

Scene boundary weight:

```
w_scene(t, (t_start, t_end)) = { 1.4,  (t-t_start)/(t_end-t_start) < 0.15
                                { 1.2,  (t-t_start)/(t_end-t_start) > 0.85
                                { 1.0,  otherwise
```

### 9. Temporal Gap Enforcement

Given sorted selected frames S = {f₁, f₂, ..., f_k} with timestamps t₁ < t₂ < ... < t_k:

Filter to maintain minimum gap δ_min = 0.5s:

```
S' = {f₁}  // Always keep first frame
for i = 2 to k-1:
    if t_i - t_(last selected) ≥ δ_min:
        S' ← S' ∪ {f_i}
S' ← S' ∪ {f_k}  // Always keep last frame
```

### 10. Reduction Rate

Frame reduction rate:

```
ρ = 1 - |S| / |C|
```

where:

- |C| = total candidate frames extracted
- |S| = final selected frames for LLM extraction

Typical results: ρ ∈ [0.80, 0.85] (80-85% reduction)

## Pipeline Stages

### Stage 1: Video Ingestion

**Objective**: Load video metadata and extract audio track for multimodal analysis.

**Process**:

1. Validate video file format (mp4, mov, avi, mkv, webm)
2. Extract metadata using OpenCV:
   - Duration (seconds)
   - Frame rate (fps)
   - Resolution (width × height)
   - Codec information
3. Optional audio extraction using FFmpeg:
   - Output format: 16kHz mono WAV
   - Codec: PCM signed 16-bit little-endian

**Implementation**: `src/ingestion/video_loader.py`

**Parameters**:

- `max_resolution`: 720 (downscale if higher)
- `extract_audio`: true

### Stage 2: Scene Detection

**Objective**: Segment video into semantically coherent scenes to guide frame selection.

**Method**: PySceneDetect ContentDetector with adaptive threshold

**Algorithm**:

```python
def detect_scenes(video, threshold=27.0, min_length=0.5):
    detector = ContentDetector(threshold)
    scene_list = detect(video, detector)

    # Filter by minimum scene length
    valid_scenes = [
        (start, end) for (start, end) in scene_list
        if end - start >= min_length
    ]

    # Fallback for static videos
    if len(valid_scenes) == 0:
        # Retry with lower threshold
        detector = ContentDetector(threshold=15.0)
        scene_list = detect(video, detector)

        # If still empty, create artificial chunks
        if len(scene_list) == 0:
            chunk_size = 10.0
            valid_scenes = artificial_chunk_scenes(video.duration, chunk_size)

    return valid_scenes
```

**Implementation**: `src/detection/scene_detector.py`

**Parameters**:

- `method`: "content" (ContentDetector)
- `threshold`: 27.0
- `min_scene_length_s`: 0.5
- `fallback.threshold`: 15.0
- `fallback.chunk_size_s`: 10.0

**Output**: List of scene boundaries [(t_start, t_end), ...]

### Stage 3: Candidate Frame Extraction

**Objective**: Extract frames at points of significant visual change.

**Method**: Histogram-based change detection with adaptive thresholding

**Algorithm**:

```python
def extract_candidates(video, threshold=0.15, min_interval_ms=100):
    candidates = []
    prev_frame = None
    last_candidate_time = -min_interval_ms

    for timestamp, frame in iterate_frames(video, interval_ms=50):
        if prev_frame is None:
            candidates.append((timestamp, frame))
            last_candidate_time = timestamp * 1000
            prev_frame = frame
            continue

        # Check minimum time gap
        if (timestamp * 1000 - last_candidate_time) < min_interval_ms:
            prev_frame = frame
            continue

        # Compute histogram change
        change = histogram_chi_square(prev_frame, frame)

        if change > threshold:
            candidates.append((timestamp, frame))
            last_candidate_time = timestamp * 1000

        prev_frame = frame

    return candidates
```

**Implementation**: `src/detection/change_detector.py`

**Parameters**:

- `method`: "histogram"
- `threshold`: 0.15
- `min_interval_ms`: 100
- `sample_interval_ms`: 50

**Complexity**: O(n × w × h) where n is video duration / sample interval

### Stage 4: Hierarchical Deduplication

**Objective**: Remove redundant frames using cascading similarity filters.

**Architecture**: Three-tier pyramid with increasing semantic depth:

#### Tier 1: Perceptual Hash (pHash)

- **Speed**: ~2ms per frame
- **Memory**: 64 bits per frame
- **Catches**: Exact and near-exact duplicates
- **False negative rate**: ~5%

#### Tier 2: Structural Similarity (SSIM)

- **Speed**: ~50ms per comparison
- **Memory**: 256×256 grayscale (64 KB per frame)
- **Catches**: Frames with identical structure, different details
- **False negative rate**: ~2%

#### Tier 3: CLIP Embeddings

- **Speed**: ~100ms per frame (batch of 32)
- **Memory**: 512 floats per frame (2 KB)
- **Catches**: Semantically similar frames (different angles, lighting)
- **False negative rate**: ~0.5%

**Implementation**: `src/deduplication/hierarchical.py`

**Parameters**:

```yaml
deduplication:
  phash:
    enabled: true
    threshold: 8 # Hamming distance
  ssim:
    enabled: true
    threshold: 0.92
  clip:
    enabled: true
    model: "ViT-B/32"
    threshold: 0.90
    device: "cpu"
    batch_size: 32
```

**Performance**: Reduces frames by 70-80% while preserving semantic diversity

### Stage 5: Audio Event Extraction

**Objective**: Identify audio-visual synchronization points for importance scoring.

**Features Extracted**:

1. **Energy Peaks**: Moments of audio emphasis (e.g., beat drops, emphasis words)

   - RMS energy computation with 90th percentile threshold
   - Local maxima detection with pre/post buffers

2. **Silence Segments**: Natural transition points
   - Threshold: -40dB
   - Minimum duration: 0.3s
   - Used to identify scene transition points

**Implementation**: `src/ingestion/audio_extractor.py`

**Algorithm**:

```python
def extract_audio_events(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)

    # Energy peaks
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=sr)
    threshold = np.percentile(rms, 90)
    peak_indices = librosa.util.peak_pick(
        rms, pre_max=3, post_max=3,
        pre_avg=3, post_avg=5,
        delta=threshold * 0.1, wait=10
    )
    energy_peaks = times[peak_indices]

    # Silence detection
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    silence_segments = detect_silence(rms_db, times, threshold=-40, min_duration=0.3)

    return {
        'energy_peaks': energy_peaks.tolist(),
        'silence_segments': silence_segments
    }
```

**Complexity**: O(n) where n is audio sample count

### Stage 6: Representative Frame Selection

**Objective**: Select optimal subset of frames proportional to scene duration and complexity.

**Components**:

#### 6.1 Scene Assignment

Map each deduplicated frame to its containing scene based on timestamp.

#### 6.2 Adaptive Density Calculation

Compute scene-specific frame density based on visual complexity:

```python
def calculate_adaptive_density(scene_frames, base_density=0.25):
    # Compute frame-to-frame variance
    variance = np.mean([
        np.mean(np.abs(f1 - f2)) / 255.0
        for f1, f2 in zip(scene_frames[:-1], scene_frames[1:])
    ])

    if variance > 0.15:  # High motion
        return base_density * 1.3
    elif variance < 0.05:  # Static
        return base_density * 0.7
    else:
        return base_density
```

#### 6.3 Target Frame Allocation

For each scene with duration Δt:

```python
target_frames = int(scene_duration * adaptive_density)
target_frames = max(min_frames_per_scene, min(target_frames, max_frames_per_scene))
```

#### 6.4 K-Means Temporal Clustering

Within each scene, cluster frames in CLIP embedding space:

```python
from sklearn.cluster import KMeans

def select_representatives(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Select frame closest to each centroid
    representatives = []
    for cluster_id in range(n_clusters):
        cluster_embeddings = embeddings[labels == cluster_id]
        centroid = kmeans.cluster_centers_[cluster_id]

        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        best_idx = np.argmin(distances)
        representatives.append(cluster_frames[best_idx])

    return sorted(representatives, key=lambda x: x.timestamp)
```

#### 6.5 Importance Scoring

Multi-factor scoring combining position, audio, and scene context:

```python
def compute_importance(frame, video_duration, scene_bounds, audio_events):
    score = 1.0

    # Position in video
    position = frame.timestamp / video_duration
    if position < 0.1:
        score *= 1.5  # Opening boost
    elif position > 0.9:
        score *= 1.3  # Closing boost

    # Audio event proximity
    for peak in audio_events['energy_peaks']:
        if abs(frame.timestamp - peak) < 0.5:
            score *= 1.3
            break

    # Scene boundary
    scene_start, scene_end = scene_bounds[frame.scene_id]
    scene_position = (frame.timestamp - scene_start) / (scene_end - scene_start)
    if scene_position < 0.15:
        score *= 1.4  # Scene start
    elif scene_position > 0.85:
        score *= 1.2  # Scene end

    return score
```

#### 6.6 Temporal Gap Enforcement

Ensure minimum 0.5s gap between selected frames (except first/last):

```python
def enforce_temporal_gap(frames, min_gap=0.5):
    if len(frames) <= 2:
        return frames

    kept = [frames[0]]  # Always keep first

    for frame in frames[1:-1]:
        if frame.timestamp - kept[-1].timestamp >= min_gap:
            kept.append(frame)

    kept.append(frames[-1])  # Always keep last
    return kept
```

**Implementation**: `src/selection/clustering.py`, `src/selection/representative.py`

**Parameters**:

```yaml
selection:
  method: "clustering"
  target_frame_density: 0.25
  min_frames_per_scene: 2
  max_frames_per_scene: 10
  min_temporal_gap_s: 0.5
  adaptive_density: true
```

**Output**: Typically 20-30 frames for a 60-second video (vs. 600 with dense sampling)

### Stage 7: LLM-Based Extraction

**Objective**: Extract structured advertisement information using vision-language models.

**Architecture**: Two-pass adaptive schema extraction

#### Pass 1: Ad Type Detection

Classify advertisement into one of five categories:

1. **Product Demo**: Feature demonstrations, how-it-works
2. **Testimonial**: Customer reviews, expert endorsements
3. **Brand Awareness**: Emotional storytelling, brand values
4. **Tutorial**: Educational, how-to content
5. **Entertainment**: Humor, celebrity content, viral moments

**Prompt** (abbreviated):

```
Analyze this advertisement and classify it into exactly ONE category:
- product_demo: Shows product features, usage, or demonstration
- testimonial: Features customer reviews, expert opinions
- brand_awareness: Emotional storytelling focused on brand values
- tutorial: Teaches how to do something
- entertainment: Comedy, celebrity content, viral moments

Respond with ONLY the category name, nothing else.
```

#### Pass 2: Structured Extraction

Use type-specific schema with temporal context enrichment:

**Temporal Context Format**:

```
You are analyzing a 60.1-second video advertisement through 44 keyframes.

The frames are in CHRONOLOGICAL ORDER. Analyze both individual frames AND the narrative progression.

ANALYSIS APPROACH:
1. Identify what CHANGES between frames (scene transitions, new elements, text)
2. Track the NARRATIVE ARC (setup → development → conclusion/CTA)
3. Note RECURRING ELEMENTS (logo appearances, product shots, faces)
4. Consider the PACING (fast cuts = energy, slow shots = emotion)

TEMPORAL CONTEXT:
Frame 1 @ 0.0s [OPENING]
Frame 2 @ 1.4s (Δ1.4s)
Frame 3 @ 2.8s (Δ1.4s)
...
Frame 44 @ 58.6s (Δ1.5s) [CLOSING]

Extract the following information in JSON format:
{base_schema + type_specific_extensions}
```

**Base Schema** (all ad types):

```json
{
  "brand": {
    "name": "string",
    "logo_visible": "boolean",
    "logo_timestamps": ["float"]
  },
  "message": {
    "primary_message": "string",
    "call_to_action": "string or null",
    "tagline": "string or null"
  },
  "creative_elements": {
    "dominant_colors": ["string"],
    "text_overlays": ["string"],
    "music_mood": "string or null"
  },
  "target_audience": {
    "age_group": "string",
    "interests": ["string"]
  },
  "persuasion_techniques": ["string"]
}
```

**Type-Specific Extensions**:

Brand Awareness:

```json
{
  "emotional_appeal": {
    "primary_emotion": "string",
    "storytelling_elements": ["string"],
    "brand_values_conveyed": ["string"]
  }
}
```

Product Demo:

```json
{
  "product": {
    "name": "string",
    "category": "string",
    "features_demonstrated": ["string"],
    "price_shown": "string or null"
  },
  "demo_steps": ["string"]
}
```

**Implementation**: `src/extraction/llm_client.py`, `src/extraction/prompts.py`

**Supported Providers**:

- Anthropic Claude (claude-sonnet-4-20250514)
- OpenAI GPT-4V (gpt-4o)
- Google Gemini (gemini-2.0-flash-exp)

**Parameters**:

```yaml
extraction:
  provider: "gemini"
  model: "gemini-2.0-flash-exp"
  max_tokens: 2000
  temperature: 0.0

  temporal_context:
    enabled: true
    include_timestamps: true
    include_time_deltas: true
    include_position_labels: true
    include_narrative_instructions: true

  schema:
    mode: "adaptive" # adaptive | fixed | flexible
```

**Cost Analysis**:

For a 60-second video:

- Dense sampling (100ms): 600 frames × $0.0075/image = $4.50
- Our pipeline (44 frames): 44 × $0.0075/image = $0.33
- **Reduction: 92.7% cost savings**

## Installation

### Prerequisites

- Python 3.8+
- FFmpeg (for audio extraction)
- CUDA-capable GPU (optional, for CLIP acceleration)

### Environment Setup

```bash
# Clone repository
git clone https://github.com/yourusername/video-ad-analysis-pipeline.git
cd video-ad-analysis-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Core libraries:

```
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
scikit-image>=0.21.0
scikit-learn>=1.3.0
torch>=2.0.0
open-clip-torch>=2.20.0
scenedetect[opencv]>=0.6.2
librosa>=0.10.0
pyyaml>=6.0
```

LLM client libraries:

```
anthropic>=0.18.0
openai>=1.12.0
google-generativeai>=0.3.0
```

### API Keys

Create a `.env` file in the project root:

```bash
# Choose one based on your preferred LLM provider
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
```

## Usage

### Basic Usage

```bash
# Process a single video
python -m experiments.pipeline --video path/to/video.mp4

# Process directory of videos
python main.py --input data/ads --output results/analysis.json

# Process with multiple workers
python main.py -i data/ads -o results.json --workers 4

# Skip LLM extraction (testing pipeline only)
python main.py -i data/ads --skip-extraction
```

### Configuration

Modify `config/default.yaml` to customize pipeline behavior:

```yaml
# Example: High-quality extraction (more frames, stricter deduplication)
selection:
  target_frame_density: 0.35  # More frames per scene
  min_frames_per_scene: 3
  max_frames_per_scene: 15

deduplication:
  clip:
    threshold: 0.95  # Stricter semantic similarity

# Example: Fast processing (fewer frames, relaxed deduplication)
selection:
  target_frame_density: 0.15  # Fewer frames per scene
  min_frames_per_scene: 1
  max_frames_per_scene: 5

deduplication:
  clip:
    threshold: 0.85  # Relaxed semantic similarity
```

### Python API

```python
from src.pipeline import AdVideoPipeline

# Initialize pipeline
pipeline = AdVideoPipeline(config_path='config/default.yaml')

# Process single video
result = pipeline.process('path/to/video.mp4')

print(f"Extracted {result.final_frame_count} frames from {len(result.scenes)} scenes")
print(f"Reduction rate: {result.reduction_rate:.1%}")
print(f"Brand: {result.extraction_result['brand']['name']}")

# Batch processing
results = pipeline.process_batch(
    video_paths=['video1.mp4', 'video2.mp4'],
    max_workers=2,
    skip_extraction=False
)
```

### Output Format

```json
{
  "metadata": {
    "timestamp": "2025-12-29T13:48:11.766000",
    "total_videos": 1,
    "successful": 1,
    "failed": 0
  },
  "results": [
    {
      "status": "success",
      "video_path": "data/ads/bernie_2016.mp4",
      "metadata": {
        "duration": 60.06,
        "fps": 30.0,
        "width": 1280,
        "height": 720
      },
      "scenes": [
        { "scene_id": 0, "start_time": 0.0, "end_time": 1.4 },
        { "scene_id": 1, "start_time": 1.4, "end_time": 2.8 }
      ],
      "selected_frames": [
        { "timestamp": 0.0, "scene_id": 0, "importance_score": 1.5 },
        { "timestamp": 1.4, "scene_id": 1, "importance_score": 1.4 }
      ],
      "pipeline_stats": {
        "total_frames_sampled": 279,
        "frames_after_phash": 194,
        "frames_after_ssim": 194,
        "frames_after_clip": 64,
        "final_frame_count": 44,
        "reduction_rate": 0.842,
        "processing_time_s": 406.4
      },
      "extraction": {
        "brand": {
          "name": "Bernie Sanders",
          "logo_visible": false,
          "logo_timestamps": []
        },
        "message": {
          "primary_message": "Bernie Sanders is for all of America.",
          "call_to_action": null,
          "tagline": null
        },
        "creative_elements": {
          "dominant_colors": ["white", "blue", "red"],
          "text_overlays": ["ALL", "TO", "AMERICA"],
          "music_mood": "uplifting"
        },
        "target_audience": {
          "age_group": "all ages",
          "interests": ["politics", "social justice", "community"]
        },
        "persuasion_techniques": [
          "social proof",
          "emotional appeal",
          "bandwagon"
        ],
        "emotional_appeal": {
          "primary_emotion": "hope",
          "storytelling_elements": [
            "Images of everyday Americans",
            "Scenes of community and family"
          ],
          "brand_values_conveyed": ["community", "equality", "inclusiveness"]
        },
        "_metadata": {
          "ad_type": "brand_awareness",
          "schema_mode": "adaptive",
          "num_frames": 44,
          "video_duration": 60.06
        }
      }
    }
  ]
}
```

## Configuration

### Complete Configuration Reference

```yaml
# config/default.yaml

pipeline:
  name: "adaptive-ad-pipeline"
  version: "2.0.0"

# Video ingestion
ingestion:
  max_resolution: 720
  extract_audio: true

# Change detection
change_detection:
  method: "histogram" # frame_diff | histogram | edge
  threshold: 0.15
  min_interval_ms: 100

# Scene detection
scene_detection:
  method: "content" # content | threshold
  threshold: 27.0
  min_scene_length_s: 0.5

  fallback:
    enabled: true
    threshold: 15.0
    artificial_chunks: true
    chunk_size_s: 10.0

# Hierarchical deduplication
deduplication:
  phash:
    enabled: true
    threshold: 8
  ssim:
    enabled: true
    threshold: 0.92
  clip:
    enabled: true
    model: "ViT-B/32"
    threshold: 0.90
    device: "cpu" # cuda | cpu | auto
    batch_size: 32

# Representative selection
selection:
  method: "clustering" # clustering | uniform | first

  # Density-based allocation
  target_frame_density: 0.25
  min_frames_per_scene: 2
  max_frames_per_scene: 10
  min_temporal_gap_s: 0.5

  # Adaptive density
  adaptive_density: true

# LLM extraction
extraction:
  provider: "gemini" # anthropic | openai | gemini
  model: "gemini-2.0-flash-exp"
  max_tokens: 2000
  temperature: 0.0

  # Temporal reasoning
  temporal_context:
    enabled: true
    include_timestamps: true
    include_time_deltas: true
    include_position_labels: true
    include_narrative_instructions: true

  # Adaptive schema
  schema:
    mode: "adaptive" # adaptive | fixed | flexible
    schema_name: "full"
    confidence_sampling:
      enabled: false
      n_samples: 3
      temperature: 0.3

# Batch processing
batch:
  enabled: true
  max_workers: 1
  gpu_batch_size: 32

# Logging
logging:
  level: "INFO" # DEBUG | INFO | WARNING | ERROR
  log_file: null
```

## Experimental Results

### Test Videos

Two representative advertisements from different categories:

1. **Political Ad**: Bernie Sanders 2016 campaign (60.1s)

   - Type: Brand awareness
   - Scenes: 44
   - Characteristics: Diverse imagery, rallies, community scenes

2. **Commercial Ad**: Burger King (72.6s)
   - Type: Entertainment
   - Scenes: 25
   - Characteristics: Humor-focused, product showcase

### Quantitative Results

| Metric              | Bernie Sanders | Burger King | Average   |
| ------------------- | -------------- | ----------- | --------- |
| Duration (s)        | 60.1           | 72.6        | 66.4      |
| Scenes Detected     | 44             | 25          | 34.5      |
| Candidate Frames    | 279            | 149         | 214       |
| After pHash         | 194            | 109         | 151.5     |
| After SSIM          | 194            | 107         | 150.5     |
| After CLIP          | 64             | 31          | 47.5      |
| Final Selected      | 44             | 26          | 35        |
| **Reduction Rate**  | **84.2%**      | **82.6%**   | **83.4%** |
| Processing Time (s) | 406.4          | 54.5        | 230.5     |
| Time per Frame (s)  | 9.2            | 2.1         | 5.7       |

### Stage-by-Stage Reduction Analysis

**Bernie Sanders Ad**:

```
Candidates (279)
  → pHash (194, -30.5%)
  → SSIM (194, 0%)
  → CLIP (64, -67.0%)
  → Selection (44, -31.3%)
Final: 84.2% total reduction
```

**Burger King Ad**:

```
Candidates (149)
  → pHash (109, -26.8%)
  → SSIM (107, -1.8%)
  → CLIP (31, -71.0%)
  → Selection (26, -16.1%)
Final: 82.6% total reduction
```

**Key Observations**:

1. pHash provides consistent ~27-30% reduction (near-duplicates)
2. SSIM shows minimal additional filtering (0-2%) after pHash
3. CLIP provides dramatic 67-71% reduction (semantic duplicates)
4. Final selection applies temporal/importance constraints (16-31% reduction)

### Qualitative Results

#### Bernie Sanders (Brand Awareness)

**Detected Elements**:

- Brand: Bernie Sanders (logo not prominently visible)
- Primary Message: "Bernie Sanders is for all of America"
- Dominant Colors: White, blue, red (patriotic palette)
- Text Overlays: "ALL", "TO", "AMERICA", campaign branding
- Music Mood: Uplifting
- Target Audience: All ages, interests in politics/social justice
- Persuasion Techniques: Social proof, emotional appeal, bandwagon
- Primary Emotion: Hope
- Storytelling Elements: Everyday Americans, community scenes, rallies
- Brand Values: Community, equality, inclusiveness, patriotism

**Analysis**: The pipeline correctly identified this as brand awareness content focusing on emotional appeal and broad inclusiveness rather than specific policy proposals.

#### Burger King (Entertainment)

**Detected Elements**:

- Brand: Burger King (logo visible at timestamp 66.0s)
- Primary Message: "Burger King tastes better"
- Call to Action: "Visit Burger King's website"
- Tagline: "It just tastes better"
- Dominant Colors: Brown, blue, white
- Text Overlays: Website URL, branding, hosting watermark
- Music Mood: Upbeat
- Target Audience: 18-35, interests in fast food/humor/cars
- Persuasion Techniques: Humor, association
- Humor Type: Slapstick
- Viral Elements: Lowrider car, people eating burgers in car

**Analysis**: The pipeline correctly classified this as entertainment-focused advertising using humor and viral-worthy moments rather than traditional product demonstration.

### Cost Analysis

For LLM API usage (assuming Gemini 2.0 Flash pricing):

| Approach                   | Frames/Video | Cost/Frame  | Total Cost | Reduction |
| -------------------------- | ------------ | ----------- | ---------- | --------- |
| Dense (100ms)              | 600          | $0.0075     | $4.50      | -         |
| Our Pipeline (Bernie)      | 44           | $0.0075     | $0.33      | 92.7%     |
| Our Pipeline (Burger King) | 26           | $0.0075     | $0.20      | 95.6%     |
| **Average**                | **35**       | **$0.0075** | **$0.26**  | **94.2%** |

**Scaling Analysis**:

For 1000 video dataset:

- Dense sampling: 1000 × $4.50 = **$4,500**
- Our pipeline: 1000 × $0.26 = **$260**
- **Savings: $4,240 (94.2% reduction)**

### Processing Time Analysis

| Stage                | Bernie (60s) | Burger King (73s) | % of Total |
| -------------------- | ------------ | ----------------- | ---------- |
| Video Loading        | 0.1s         | 0.2s              | 0.3%       |
| Scene Detection      | 13.1s        | 1.0s              | 30.6%      |
| Candidate Extraction | 277.4s       | 5.9s              | 61.6%      |
| Deduplication        | 109.9s       | 35.2s             | 31.5%      |
| Audio Extraction     | 1.7s         | 1.6s              | 0.7%       |
| Frame Selection      | 2.3s         | 1.7s              | 0.9%       |
| LLM Extraction       | 10.9s        | 4.9s              | 3.4%       |
| **Total**            | **406.4s**   | **54.5s**         | **100%**   |

**Bottleneck Analysis**:

- Candidate extraction (61.6%): I/O-bound video decoding
- Scene detection (30.6%): CPU-intensive content analysis
- Deduplication (31.5%): CLIP embedding computation

**Optimization Opportunities**:

1. GPU acceleration for CLIP (estimated 5-10x speedup)
2. Parallel video decoding for batch processing
3. Frame pre-caching for multi-stage processing

## Technical Implementation

### Project Structure

```
video-ad-analysis-pipeline/
├── config/
│   └── default.yaml              # Configuration file
├── data/
│   └── ads/                      # Input video directory
├── outputs/
│   ├── audio/                    # Extracted audio files
│   ├── frames/                   # Debug frame outputs
│   └── results.json              # Batch processing results
├── src/
│   ├── deduplication/
│   │   ├── base.py               # Abstract deduplicator
│   │   ├── phash.py              # Perceptual hashing
│   │   ├── ssim.py               # Structural similarity
│   │   ├── clip_embed.py         # CLIP embeddings
│   │   └── hierarchical.py       # Hierarchical pipeline
│   ├── detection/
│   │   ├── change_detector.py    # Frame change detection
│   │   └── scene_detector.py     # Scene segmentation
│   ├── extraction/
│   │   ├── llm_client.py         # LLM API clients
│   │   ├── prompts.py            # Prompt engineering
│   │   └── schema.py             # Extraction schemas
│   ├── ingestion/
│   │   ├── audio_extractor.py    # Audio processing
│   │   └── video_loader.py       # Video I/O
│   ├── selection/
│   │   ├── clustering.py         # Temporal clustering
│   │   └── representative.py     # Frame selection
│   ├── utils/
│   │   ├── config.py             # Configuration loading
│   │   ├── logging.py            # Logging utilities
│   │   ├── metrics.py            # Performance metrics
│   │   └── video_utils.py        # Video utilities
│   └── pipeline.py               # Main pipeline orchestrator
├── experiments/
│   └── pipeline.py               # Single video test script
├── main.py                       # Batch processing CLI
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

### Key Algorithms

#### Histogram-Based Change Detection

```python
def histogram_chi_square(frame1, frame2, bins=16):
    """
    Compute chi-square distance between color histograms.

    Args:
        frame1, frame2: BGR numpy arrays
        bins: Number of bins per channel

    Returns:
        Chi-square distance (higher = more change)
    """
    def compute_histogram(frame):
        hist = cv2.calcHist(
            [frame], [0, 1, 2], None,
            [bins, bins, bins],
            [0, 256, 0, 256, 0, 256]
        )
        cv2.normalize(hist, hist)
        return hist.flatten()

    hist1 = compute_histogram(frame1)
    hist2 = compute_histogram(frame2)

    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
```

#### Perceptual Hashing

```python
def compute_phash(frame, hash_size=8):
    """
    Compute perceptual hash using DCT.

    Args:
        frame: BGR numpy array
        hash_size: Hash dimension (hash_size x hash_size)

    Returns:
        imagehash.ImageHash object
    """
    import imagehash
    from PIL import Image

    # Convert to PIL
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Compute pHash
    return imagehash.phash(pil_image, hash_size=hash_size)

def hamming_distance(hash1, hash2):
    """Hamming distance between two hashes."""
    return hash1 - hash2  # imagehash overloads subtraction
```

#### SSIM Computation

```python
def compute_ssim(frame1, frame2):
    """
    Compute structural similarity index.

    Args:
        frame1, frame2: BGR numpy arrays

    Returns:
        SSIM score (0 to 1, higher = more similar)
    """
    from skimage.metrics import structural_similarity

    # Convert to grayscale and resize
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    gray1 = cv2.resize(gray1, (256, 256))
    gray2 = cv2.resize(gray2, (256, 256))

    return structural_similarity(gray1, gray2)
```

#### CLIP Embedding Extraction

```python
def extract_clip_embeddings_batch(frames, model, preprocess, device, batch_size=32):
    """
    Extract CLIP embeddings for multiple frames efficiently.

    Args:
        frames: List of BGR numpy arrays
        model: CLIP model
        preprocess: CLIP preprocessing transform
        device: torch device
        batch_size: Batch size for GPU processing

    Returns:
        Numpy array of shape (num_frames, 512)
    """
    import torch
    from PIL import Image

    embeddings = []

    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]

        # Convert to PIL and preprocess
        pil_images = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            for f in batch
        ]

        batch_tensor = torch.stack([
            preprocess(img) for img in pil_images
        ]).to(device)

        # Batch inference
        with torch.no_grad():
            batch_embeddings = model.encode_image(batch_tensor)
            batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)

        embeddings.append(batch_embeddings.cpu().numpy())

    return np.vstack(embeddings)
```

#### K-Means Temporal Clustering

```python
def cluster_and_select_representatives(embeddings, timestamps, n_clusters):
    """
    Cluster frames and select representatives closest to centroids.

    Args:
        embeddings: Numpy array (num_frames, embedding_dim)
        timestamps: List of timestamps
        n_clusters: Target number of representatives

    Returns:
        Indices of selected representative frames
    """
    from sklearn.cluster import KMeans

    n_clusters = min(n_clusters, len(embeddings))

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Select frame closest to each centroid
    selected_indices = []
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_embeddings = embeddings[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]

        centroid = kmeans.cluster_centers_[cluster_id]

        # Find closest frame
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        best_local_idx = np.argmin(distances)
        best_global_idx = cluster_indices[best_local_idx]

        selected_indices.append(best_global_idx)

    # Sort by timestamp
    selected_indices.sort(key=lambda idx: timestamps[idx])

    return selected_indices
```

### Memory and Computational Complexity

| Stage                | Time Complexity | Space Complexity | Bottleneck               |
| -------------------- | --------------- | ---------------- | ------------------------ |
| Scene Detection      | O(n × w × h)    | O(w × h)         | CPU (content analysis)   |
| Candidate Extraction | O(n × w × h)    | O(k × w × h)     | I/O (video decoding)     |
| pHash                | O(k²)           | O(k)             | CPU (minimal)            |
| SSIM                 | O(k² × w × h)   | O(k × w × h)     | CPU (pixel ops)          |
| CLIP                 | O(k × d + k²)   | O(k × d)         | GPU/CPU (embedding)      |
| Clustering           | O(i × k × d)    | O(k × d)         | CPU (K-means iterations) |
| LLM Extraction       | O(m)            | O(m × w × h)     | Network I/O              |

Where:

- n = total video frames (fps × duration)
- k = candidate frames (typically 0.1n to 0.5n)
- m = selected frames (typically 0.01n to 0.05n)
- w × h = frame resolution
- d = embedding dimension (512 for ViT-B/32)
- i = K-means iterations (typically < 50)

## Performance Metrics

### Frame Reduction Metrics

1. **Candidate Reduction Rate**: R_c = 1 - k/n

   - Measures initial sampling efficiency
   - Typical range: 50-90%

2. **Deduplication Rate**: R_d = 1 - m/k

   - Measures duplicate removal effectiveness
   - Typical range: 60-80%

3. **Overall Reduction Rate**: R = 1 - m/n = 1 - (1-R_c)(1-R_d)
   - Composite metric
   - Target: > 80%

### Cost Metrics

1. **API Cost Reduction**: C_reduction = (C_dense - C_pipeline) / C_dense

   - Direct cost savings
   - Proportional to frame reduction

2. **Processing Time**: T = T_video + T_dedup + T_select + T_llm
   - Total pipeline latency
   - Dominated by video I/O and CLIP embedding

### Quality Metrics

1. **Temporal Coverage**: Distribution of selected frames across scenes

   - Ideal: Proportional to scene duration and complexity
   - Measured via Gini coefficient

2. **Semantic Diversity**: Average pairwise cosine distance in CLIP space

   - Higher = more diverse frame selection
   - Target: > 0.3

3. **Narrative Completeness**: Manual annotation of captured events
   - Percentage of key moments captured
   - Requires human evaluation

## Future Work

### Short-Term Improvements

1. **GPU Acceleration**

   - Migrate CLIP embedding to GPU
   - Estimated 5-10x speedup for deduplication stage
   - Implementation: `device="cuda"` in config

2. **Parallel Video Decoding**

   - Multi-threaded frame extraction
   - Utilize modern CPU SIMD instructions
   - Estimated 2-3x speedup for candidate extraction

3. **Audio-Visual Fusion**
   - Incorporate speech recognition (Whisper)
   - Detect audio-visual synchronization points
   - Boost frame importance based on dialogue

### Medium-Term Research

1. **Learned Frame Importance**

   - Train lightweight model to predict frame importance
   - Input: CLIP embeddings + temporal position + audio features
   - Replace heuristic scoring with learned scoring

2. **Dynamic Density Adaptation**

   - Per-scene density based on content complexity
   - Use optical flow magnitude as complexity indicator
   - Extend beyond simple variance thresholding

3. **Multi-Modal Schema Alignment**
   - Incorporate audio transcription in extraction
   - Cross-reference visual and audio information
   - Detect contradictions or complementary information

### Long-Term Vision

1. **End-to-End Learned Pipeline**

   - Replace rule-based stages with learned modules
   - Differentiable frame selection using Gumbel-Softmax
   - Train on downstream task (e.g., ad effectiveness prediction)

2. **Interactive Refinement**

   - User feedback loop for frame selection
   - Active learning to improve selection policy
   - Personalized density/quality trade-off

3. **Cross-Modal Retrieval**

   - Index deduplicated frames with CLIP embeddings
   - Enable text queries: "Find frames with product close-up"
   - Support similarity search across video corpus

4. **Real-Time Processing**
   - Optimize for streaming video analysis
   - Incremental deduplication and selection
   - Target: < 1 second latency for 30s video

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@inproceedings{adaptive-video-ad-pipeline-2025,
  title={Adaptive Density-Based Frame Selection for Efficient Video Advertisement Analysis},
  author={Abdul Basit Tonmoy},
  booktitle={Proceedings of the International Conference on Multimedia Retrieval (ICMR)},
  year={2025},
  organization={ACM}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

This work builds upon several open-source libraries:

- PySceneDetect for scene detection
- OpenCLIP for vision-language embeddings
- scikit-image for SSIM computation
- librosa for audio analysis

We thank the developers of Anthropic Claude, OpenAI GPT-4, and Google Gemini for providing API access for structured extraction experiments.

## Contact

For questions or collaboration inquiries, please contact:

- Abdul Basit Tonmoy (Email: abdulbasittonmoy@gmail.com | github: abtonmoy)

---

Last Updated: December 29, 2025
Version: 2.0.0
