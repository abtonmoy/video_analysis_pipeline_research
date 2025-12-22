# Adaptive Video Advertisement Analysis Pipeline

## Project Overview

This research project develops an **efficient, hierarchical frame extraction pipeline** for analyzing video advertisements using Large Language Models (LLMs). The core innovation is reducing computational cost and API usage while maintaining extraction quality through intelligent, multi-stage frame selection.

### Research Question

> Can we significantly reduce the number of frames sent to vision-language models for advertisement analysis while preserving (or improving) the quality of extracted insights?

### Target Publication Venues

- **Primary:** ICME, MMM, WACV, ACM Multimedia workshops
- **Reach:** ACM MM, ECCV workshops

---

## Problem Statement

### Current Approach (Naive Pipeline)

```
Video → Fixed-interval sampling (e.g., 1 frame/0.3s) → CLIP embedding →
Cosine similarity deduplication → LLM Vision API → Insights
```

**Problems:**

1. **Wasteful extraction:** Many frames extracted only to be discarded
2. **Arbitrary thresholds:** 0.3s interval is content-agnostic
3. **Expensive deduplication:** CLIP embeddings computed for all frames before filtering
4. **No temporal awareness:** Similarity computed pairwise without scene context
5. **High API costs:** More frames = more tokens = higher costs

### Proposed Approach (Hierarchical Pipeline)

```
Video → Lightweight change detection → Adaptive sampling →
Hierarchical deduplication (pHash → Scene → CLIP) →
Temporal clustering → LLM Vision API → Insights
```

**Improvements:**

1. **Smart extraction:** Only extract frames when content changes significantly
2. **Content-adaptive:** Sampling rate responds to video dynamics
3. **Cheap-to-expensive filtering:** Use fast methods first, expensive methods last
4. **Scene-aware:** Respect narrative structure of advertisements
5. **Cost-efficient:** Fewer, more informative frames sent to LLM

---

## Technical Architecture

### Stage 1: Video Ingestion & Metadata Extraction

**Input:** Video file (MP4, MOV, etc.)

**Operations:**

- Extract video metadata (duration, fps, resolution, codec)
- Extract audio track for parallel processing
- Compute video-level statistics (average brightness, motion intensity)

**Output:**

- Video metadata JSON
- Separated audio file
- Initial video statistics

**Tools:** FFmpeg, OpenCV

### Stage 2: Lightweight Change Detection

**Purpose:** Identify candidate moments where frame extraction is worthwhile

**Methods (in order of computational cost):**

| Method                      | Cost      | What It Detects           |
| --------------------------- | --------- | ------------------------- |
| Frame difference (L1/L2)    | Very Low  | Any pixel changes         |
| Histogram difference        | Low       | Color distribution shifts |
| Motion vectors (from codec) | Near Zero | Movement between frames   |
| Edge change ratio           | Low       | Structural changes        |

**Implementation:**

```python
def compute_frame_difference(frame1, frame2):
    """L1 norm of grayscale difference, normalized by frame size."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = np.abs(gray1.astype(float) - gray2.astype(float))
    return np.mean(diff) / 255.0

def compute_histogram_difference(frame1, frame2):
    """Chi-square distance between color histograms."""
    hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
```

**Output:** List of timestamps where significant change detected

**Adaptive Threshold Logic:**

- Fast-paced ads (many scene cuts): Lower threshold, more candidates
- Slow-paced ads (few cuts): Higher threshold, fewer candidates
- Threshold adjusts based on running statistics of the video

### Stage 3: Scene Boundary Detection

**Purpose:** Segment video into coherent scenes/shots

**Methods:**

| Method                            | Description                    | When to Use          |
| --------------------------------- | ------------------------------ | -------------------- |
| PySceneDetect (ContentDetector)   | Detects content changes        | General purpose      |
| PySceneDetect (ThresholdDetector) | Detects fade-to-black          | TV commercials       |
| TransNetV2                        | Neural shot boundary detection | High accuracy needed |

**Implementation:**

```python
from scenedetect import detect, ContentDetector, ThresholdDetector

def detect_scenes(video_path, method='content'):
    """Detect scene boundaries in video."""
    if method == 'content':
        detector = ContentDetector(threshold=27.0)
    elif method == 'threshold':
        detector = ThresholdDetector(threshold=12)

    scene_list = detect(video_path, detector)
    return [(s[0].get_seconds(), s[1].get_seconds()) for s in scene_list]
```

**Output:** List of (start_time, end_time) tuples for each scene

### Stage 4: Hierarchical Frame Deduplication

**Purpose:** Remove redundant frames using progressively expensive methods

**Layer 1: Perceptual Hashing (pHash)**

- **Cost:** ~0.1ms per frame
- **What it catches:** Near-identical frames, minor compression artifacts
- **Threshold:** Hamming distance < 8

```python
import imagehash
from PIL import Image

def compute_phash(frame):
    """Compute perceptual hash of frame."""
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_image)

def phash_similar(hash1, hash2, threshold=8):
    """Check if two frames are similar via pHash."""
    return hash1 - hash2 < threshold
```

**Layer 2: Structural Similarity (SSIM)**

- **Cost:** ~5ms per frame pair
- **What it catches:** Frames with same structure but different details
- **Threshold:** SSIM > 0.92

```python
from skimage.metrics import structural_similarity as ssim

def ssim_similar(frame1, frame2, threshold=0.92):
    """Check structural similarity between frames."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    score = ssim(gray1, gray2)
    return score > threshold
```

**Layer 3: CLIP Embedding Similarity**

- **Cost:** ~50ms per frame (GPU), ~500ms (CPU)
- **What it catches:** Semantically similar frames with visual differences
- **Threshold:** Cosine similarity > 0.90

```python
import torch
import clip

class CLIPEmbedder:
    def __init__(self, device='cuda'):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

    def embed(self, frame):
        """Compute CLIP embedding for frame."""
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image_input)
        return embedding.cpu().numpy().flatten()

    def similarity(self, emb1, emb2):
        """Compute cosine similarity between embeddings."""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
```

**Hierarchical Flow:**

```
All candidate frames
    │
    ▼ pHash filtering (removes ~40-60%)
Frames passing pHash
    │
    ▼ SSIM filtering (removes ~20-30% more)
Frames passing SSIM
    │
    ▼ CLIP filtering (removes ~10-20% more)
Final keyframes
```

### Stage 5: Temporal Clustering & Representative Selection

**Purpose:** Group remaining frames by scene and select best representatives

**Algorithm:**

1. Group frames by scene boundaries (from Stage 3)
2. Within each scene, cluster by CLIP embeddings
3. Select frame closest to cluster centroid as representative
4. Ensure temporal spread (don't select adjacent frames)

```python
from sklearn.cluster import KMeans

def select_representatives(frames, embeddings, scene_boundaries, max_per_scene=3):
    """Select representative frames from each scene."""
    representatives = []

    for start, end in scene_boundaries:
        # Get frames in this scene
        scene_frames = [(f, e) for f, e in zip(frames, embeddings)
                        if start <= f['timestamp'] < end]

        if len(scene_frames) <= max_per_scene:
            representatives.extend([f for f, e in scene_frames])
            continue

        # Cluster and select centroids
        scene_embeddings = np.array([e for f, e in scene_frames])
        n_clusters = min(max_per_scene, len(scene_frames))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(scene_embeddings)

        # Select frame closest to each centroid
        for i in range(n_clusters):
            cluster_frames = [f for (f, e), label in zip(scene_frames, kmeans.labels_)
                             if label == i]
            cluster_embeddings = [e for (f, e), label in zip(scene_frames, kmeans.labels_)
                                  if label == i]
            centroid = kmeans.cluster_centers_[i]
            distances = [np.linalg.norm(e - centroid) for e in cluster_embeddings]
            best_idx = np.argmin(distances)
            representatives.append(cluster_frames[best_idx])

    return representatives
```

### Stage 6: Audio-Visual Alignment (Optional Enhancement)

**Purpose:** Align frame selection with audio events for better context

**Audio Features to Extract:**

- Speech boundaries (using VAD - Voice Activity Detection)
- Music/jingle detection
- Audio energy peaks
- Silence detection (often indicates scene transitions)

```python
import librosa

def extract_audio_events(audio_path):
    """Extract significant audio events."""
    y, sr = librosa.load(audio_path)

    # Energy-based event detection
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=sr)

    # Find peaks in audio energy
    peaks = librosa.util.peak_pick(rms, pre_max=3, post_max=3,
                                    pre_avg=3, post_avg=5, delta=0.1, wait=10)

    return times[peaks].tolist()
```

**Integration:** Boost importance of frames near audio events

### Stage 7: LLM Vision API Integration

**Purpose:** Extract structured insights from selected keyframes

**Supported APIs:**

- Claude Vision (Anthropic)
- GPT-4V (OpenAI)
- Gemini Pro Vision (Google)

**Extraction Schema:**

```json
{
  "brand": {
    "name": "string",
    "logo_visible": "boolean",
    "logo_timestamp": "float"
  },
  "product": {
    "name": "string",
    "category": "string",
    "features_mentioned": ["string"]
  },
  "message": {
    "primary_message": "string",
    "call_to_action": "string",
    "tagline": "string"
  },
  "creative_elements": {
    "dominant_colors": ["string"],
    "text_overlays": ["string"],
    "faces_detected": "integer",
    "emotions_detected": ["string"]
  },
  "persuasion_techniques": ["string"],
  "target_audience": {
    "age_group": "string",
    "gender": "string",
    "interests": ["string"]
  }
}
```

**Prompt Template:**

```
You are analyzing a video advertisement through a series of keyframes.

Frames are provided in chronological order with timestamps.

Extract the following information in JSON format:
{schema}

Keyframes:
{frames_with_timestamps}

Respond ONLY with valid JSON.
```

---

## Evaluation Framework

### Datasets

| Dataset                        | Size             | Annotations                     | Use Case               |
| ------------------------------ | ---------------- | ------------------------------- | ---------------------- |
| **Hussain et al. (CVPR 2017)** | 3,477 video ads  | Topic, sentiment, action-reason | Primary benchmark      |
| **LAMBDA**                     | 2,205 ads        | Memorability scores             | Secondary validation   |
| **Custom sponsored content**   | 42 videos/images | Full extraction ground truth    | LLM quality evaluation |

### Metrics

#### Efficiency Metrics

| Metric               | Formula                              | Target              |
| -------------------- | ------------------------------------ | ------------------- |
| Frame Reduction Rate | 1 - (selected_frames / total_frames) | > 70%               |
| API Cost Reduction   | 1 - (our_tokens / baseline_tokens)   | > 60%               |
| Processing Time      | Total pipeline time in seconds       | < 2x video duration |
| Memory Usage         | Peak RAM usage in MB                 | < 4GB               |

#### Quality Metrics

| Metric                  | Description                                             | How to Compute            |
| ----------------------- | ------------------------------------------------------- | ------------------------- |
| Extraction Accuracy     | Match between extracted and ground truth fields         | Field-by-field comparison |
| Extraction Completeness | Percentage of ground truth fields recovered             | Recall of fields          |
| Semantic Similarity     | Embedding similarity of extracted vs. ground truth text | SBERT similarity          |
| Human Evaluation        | Blind comparison of outputs                             | A/B preference study      |

### Baselines to Compare Against

| Baseline         | Description                                                |
| ---------------- | ---------------------------------------------------------- |
| **Uniform-0.3s** | Extract frame every 0.3 seconds                            |
| **Uniform-1.0s** | Extract frame every 1.0 seconds                            |
| **CLIP-Only**    | Uniform extraction + CLIP deduplication (current approach) |
| **Scene-Only**   | Extract first frame of each scene                          |
| **LMSKE-style**  | TransNetV2 + CLIP + adaptive clustering                    |
| **Random**       | Random frame selection (negative baseline)                 |

### Ablation Studies

| Experiment             | What We Vary                   | What We Measure                     |
| ---------------------- | ------------------------------ | ----------------------------------- |
| pHash threshold        | Hamming distance: 4, 8, 12, 16 | Frames retained, downstream quality |
| CLIP threshold         | Cosine sim: 0.85, 0.90, 0.95   | Frames retained, downstream quality |
| Hierarchical vs. flat  | With/without pHash/SSIM layers | Processing time, quality            |
| Scene detection method | PySceneDetect vs. TransNetV2   | Accuracy, speed tradeoff            |
| Frames per scene       | 1, 2, 3, 5 representatives     | Coverage vs. efficiency             |

---

## Project Structure

```
ad-video-pipeline/
├── README.md
├── requirements.txt
├── setup.py
│
├── configs/
│   ├── default.yaml           # Default pipeline configuration
│   ├── fast.yaml              # Speed-optimized configuration
│   └── quality.yaml           # Quality-optimized configuration
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py            # Main pipeline orchestrator
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── video_loader.py    # Video loading and metadata
│   │   └── audio_extractor.py # Audio track extraction
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── change_detector.py # Lightweight change detection
│   │   ├── scene_detector.py  # Scene boundary detection
│   │   └── audio_events.py    # Audio event detection
│   │
│   ├── deduplication/
│   │   ├── __init__.py
│   │   ├── phash.py           # Perceptual hashing
│   │   ├── ssim.py            # Structural similarity
│   │   ├── clip_embed.py      # CLIP embeddings
│   │   └── hierarchical.py    # Hierarchical dedup orchestrator
│   │
│   ├── selection/
│   │   ├── __init__.py
│   │   ├── clustering.py      # Temporal clustering
│   │   └── representative.py  # Representative frame selection
│   │
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── llm_client.py      # LLM API wrapper
│   │   ├── prompts.py         # Prompt templates
│   │   └── schema.py          # Extraction schema definitions
│   │
│   └── utils/
│       ├── __init__.py
│       ├── video_utils.py     # Video I/O utilities
│       ├── metrics.py         # Evaluation metrics
│       └── visualization.py   # Result visualization
│
├── data/
│   ├── raw/                   # Original video files
│   ├── processed/             # Extracted frames and features
│   ├── annotations/           # Ground truth annotations
│   └── results/               # Extraction results
│
├── experiments/
│   ├── run_baseline.py        # Run baseline methods
│   ├── run_ablation.py        # Run ablation studies
│   ├── evaluate.py            # Compute metrics
│   └── visualize_results.py   # Generate figures
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_pipeline_demo.ipynb
│   ├── 03_results_analysis.ipynb
│   └── 04_paper_figures.ipynb
│
└── tests/
    ├── test_detection.py
    ├── test_deduplication.py
    └── test_extraction.py
```

---

## Configuration

### Default Configuration (configs/default.yaml)

```yaml
# Pipeline configuration
pipeline:
  name: "adaptive-ad-pipeline"
  version: "1.0.0"

# Video ingestion
ingestion:
  max_resolution: 720 # Downscale if larger
  extract_audio: true

# Change detection
change_detection:
  method: "histogram" # Options: frame_diff, histogram, edge
  threshold: 0.15
  min_interval_ms: 100 # Minimum time between candidates

# Scene detection
scene_detection:
  method: "content" # Options: content, threshold, transnet
  threshold: 27.0
  min_scene_length_s: 0.5

# Hierarchical deduplication
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
    device: "cuda" # Options: cuda, cpu

# Representative selection
selection:
  method: "clustering" # Options: clustering, uniform, first
  max_frames_per_scene: 3
  min_temporal_gap_s: 0.5

# LLM extraction
extraction:
  provider: "anthropic" # Options: anthropic, openai, google
  model: "claude-sonnet-4-20250514"
  max_tokens: 2000
  temperature: 0.0
  schema: "full" # Options: full, minimal, custom

# Evaluation
evaluation:
  metrics:
    - frame_reduction_rate
    - api_cost_reduction
    - processing_time
    - extraction_accuracy
  save_intermediate: true
```

---

## API Usage

### Basic Pipeline Usage

```python
from src.pipeline import AdVideoPipeline

# Initialize pipeline
pipeline = AdVideoPipeline(config_path="configs/default.yaml")

# Process single video
result = pipeline.process("path/to/video.mp4")

# Access results
print(f"Frames extracted: {result.num_frames}")
print(f"Frames selected: {result.num_selected}")
print(f"Reduction rate: {result.reduction_rate:.2%}")
print(f"Extracted insights: {result.insights}")

# Process batch
results = pipeline.process_batch("path/to/video/directory/")
```

### Custom Configuration

```python
from src.pipeline import AdVideoPipeline

# Override specific settings
custom_config = {
    "deduplication": {
        "clip": {
            "threshold": 0.85  # More aggressive deduplication
        }
    },
    "selection": {
        "max_frames_per_scene": 2  # Fewer frames
    }
}

pipeline = AdVideoPipeline(
    config_path="configs/default.yaml",
    overrides=custom_config
)
```

### Evaluation

```python
from experiments.evaluate import Evaluator

evaluator = Evaluator(
    ground_truth_path="data/annotations/",
    results_path="data/results/"
)

# Compute all metrics
metrics = evaluator.compute_all()
print(metrics.to_dataframe())

# Compare against baselines
comparison = evaluator.compare_baselines([
    "uniform_0.3s",
    "clip_only",
    "ours"
])
comparison.plot()
```

---

## Dependencies

### Core Dependencies

```
# requirements.txt

# Video processing
opencv-python>=4.8.0
ffmpeg-python>=0.2.0
scenedetect>=0.6.0

# Image processing
Pillow>=10.0.0
imagehash>=4.3.0
scikit-image>=0.21.0

# Deep learning
torch>=2.0.0
clip @ git+https://github.com/openai/CLIP.git
transformers>=4.30.0

# Audio processing
librosa>=0.10.0
soundfile>=0.12.0

# ML utilities
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0

# LLM APIs
anthropic>=0.18.0
openai>=1.0.0
google-generativeai>=0.3.0

# Utilities
pyyaml>=6.0
tqdm>=4.65.0
pandas>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Evaluation
sentence-transformers>=2.2.0  # For semantic similarity
```

### Optional Dependencies

```
# For TransNetV2 scene detection
transnetv2 @ git+https://github.com/soCzech/TransNetV2.git

# For GPU acceleration
cupy-cuda12x>=12.0.0

# For experiment tracking
wandb>=0.15.0
mlflow>=2.5.0
```

---

## Timeline & Milestones

### Phase 1: Foundation (Weeks 1-2)

- [ ] Set up project structure
- [ ] Implement video ingestion module
- [ ] Implement change detection methods
- [ ] Basic scene detection integration

### Phase 2: Core Pipeline (Weeks 3-4)

- [ ] Implement hierarchical deduplication
- [ ] Implement temporal clustering
- [ ] Integrate LLM extraction
- [ ] End-to-end pipeline working

### Phase 3: Evaluation (Weeks 5-6)

- [ ] Download and preprocess Hussain et al. dataset
- [ ] Implement baseline methods
- [ ] Create ground truth annotations for subset
- [ ] Run full evaluation suite

### Phase 4: Analysis & Writing (Weeks 7-8)

- [ ] Ablation studies
- [ ] Generate paper figures
- [ ] Write paper draft
- [ ] Code cleanup and documentation

---

## Known Limitations & Future Work

### Current Limitations

1. **Single-video processing:** No batch optimization yet
2. **English-only:** OCR and ASR assume English content
3. **No temporal reasoning:** LLM sees frames independently
4. **Fixed schema:** Extraction schema not adaptive to ad type

### Future Work

1. **Multi-video learning:** Learn optimal thresholds across dataset
2. **Reinforcement learning:** Learn frame selection policy
3. **Multimodal fusion:** Better audio-visual integration
4. **Streaming processing:** Real-time frame selection
5. **Domain adaptation:** Transfer to other video types

---

## References

### Key Papers

1. Hussain, Z., et al. "Automatic Understanding of Image and Video Advertisements." CVPR 2017.
2. Tan, K., et al. "Large Model based Sequential Keyframe Extraction for Video Summarization." CMLDS 2024.
3. Hu, W., et al. "M-LLM Based Video Frame Selection for Efficient Video Understanding." CVPR 2025.
4. TriPSS: "A Tri-Modal Keyframe Extraction Framework." ACM MM Workshop 2025.
5. EVS: "Efficient Video Sampling: Pruning Temporally Redundant Tokens." arXiv 2025.

### Tools & Libraries

- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)
- [TransNetV2](https://github.com/soCzech/TransNetV2)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [ImageHash](https://github.com/JohannesBuchner/imagehash)

---

## Contact & Contribution

**Author:** Abdul Basit Tonmoy  
**Email:** abdulbasittonmoy@gmail.com  
**GitHub:** github.com/abtonmoy

For questions, issues, or contributions, please open an issue on the GitHub repository.
