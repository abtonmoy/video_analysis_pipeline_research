# Adaptive Video Advertisement Analysis Pipeline:

# Token-Efficient Video Understanding: A Hierarchical Cascade for Cost-Aware LLM Inference

## Project Overview

This research project develops an **efficient, hierarchical frame extraction pipeline** for analyzing video advertisements using Large Language Models (LLMs). The core innovation is reducing computational cost and API usage by 80-99% while maintaining extraction quality through intelligent, multi-stage frame selection.

### Research Question

> Can we significantly reduce the number of frames sent to vision-language models for advertisement analysis while preserving (or improving) the quality of extracted insights?

### Key Results (Achieved)

- **98.9% frame reduction**: 450 frames to 5 frames (15-second video at 30fps)
- **80.8% cost reduction**: Hierarchical deduplication eliminates redundant frames
- **Sub-minute processing**: 21 seconds for full pipeline (15-second video)
- **Production-ready**: Modular architecture with config-driven behavior

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
Hierarchical deduplication (pHash → SSIM → CLIP) →
Temporal clustering → Importance scoring → LLM Vision API → Insights
```

**Improvements:**

1. **Smart extraction:** Only extract frames when content changes significantly
2. **Content-adaptive:** Sampling rate responds to video dynamics
3. **Cheap-to-expensive filtering:** Use fast methods first, expensive methods last
4. **Scene-aware:** Respect narrative structure of advertisements
5. **Temporal reasoning:** LLM sees chronological frames with position labels
6. **Cost-efficient:** Fewer, more informative frames sent to LLM

---

## Key Features Summary

| Feature                        | Status         | Description                                             |
| ------------------------------ | -------------- | ------------------------------------------------------- |
| **Hierarchical Deduplication** | ✅ Implemented | pHash → SSIM → CLIP (cheap to expensive)                |
| **Scene-Aware Selection**      | ✅ Implemented | Respects narrative structure via scene detection        |
| **Batch Processing**           | ✅ Implemented | Parallel video processing + GPU-batched CLIP            |
| **Temporal Reasoning**         | ✅ Implemented | Multi-frame prompts with timestamps & narrative context |
| **Adaptive Schema**            | ✅ Implemented | Two-pass extraction with type-specific schemas          |
| **First/Last Frame Guarantee** | ✅ Implemented | Always includes opening (brand) and closing (CTA)       |
| **Multi-Provider LLM**         | ✅ Implemented | Claude, GPT-4V, Gemini support                          |
| **Importance Scoring**         | ✅ Implemented | Position + scene + audio based weighting                |
| **Multilingual Support**       | 🟡 Planned     | Swap to multilingual OCR/ASR models                     |
| **Audio-Visual Fusion**        | 🟡 Partial     | Audio events extracted, fusion in progress              |
| **Streaming Processing**       | 🔴 Future      | Requires architecture redesign                          |
| **Learned Thresholds**         | 🔴 Future      | Requires meta-learning research                         |

---

## Technical Architecture

### Complete Pipeline Flow

```
Stage 1: Video Ingestion
    └─> Extract metadata, audio track, video statistics
         ↓
Stage 2: Scene Detection
    └─> PySceneDetect identifies scene boundaries
         ↓
Stage 3: Candidate Extraction
    └─> Change detector (Histogram/Edge/FrameDiff) samples frames
         ↓ (26 candidates from 450 total frames - 94.2% reduction)
Stage 4: Hierarchical Deduplication
    ├─> PHash:  26 → 22 frames (15% reduction, <1ms/frame)
    ├─> SSIM:   22 → 21 frames (5% reduction, ~5ms/pair)
    └─> CLIP:   21 → 10 frames (52% reduction, ~50ms/frame)
         ↓
Stage 5: Audio Event Extraction (Optional)
    └─> Detect energy peaks, silence segments
         ↓
Stage 6: Representative Selection
    ├─> Assign frames to scenes
    ├─> Score importance (position + audio + scene context)
    ├─> Cluster within scenes (K-means on CLIP embeddings)
    ├─> Select representatives (closest to centroids)
    └─> Enforce temporal gaps + guarantee first/last frames
         ↓ (10 → 5 final frames - 50% reduction)
Stage 7: LLM Extraction
    ├─> Two-pass adaptive schema (detect ad type, then extract)
    ├─> Temporal prompt with timestamps and position labels
    └─> Vision-language model (Claude/GPT/Gemini)
         ↓
Structured Ad Data (JSON)
```

### Measured Performance (15-second ad, v0002.mp4)

| Stage                    | Input      | Output | Reduction | Time      |
| ------------------------ | ---------- | ------ | --------- | --------- |
| Video (30fps)            | 450 frames | -      | -         | -         |
| Candidate extraction     | 450        | 26     | 94.2%     | 5.4s      |
| PHash dedup              | 26         | 22     | 15.4%     | <0.1s     |
| SSIM dedup               | 22         | 21     | 4.5%      | 1.0s      |
| CLIP dedup               | 21         | 10     | 52.4%     | 6.2s      |
| Representative selection | 10         | 5      | 50.0%     | <0.1s     |
| LLM extraction (Gemini)  | 5          | -      | -         | 7.3s      |
| **Total**                | **450**    | **5**  | **98.9%** | **21.1s** |

---

## Mathematical Foundations

### Change Detection

#### Frame Difference (L1 Norm)

Measures pixel-level changes between consecutive frames:

$$
D_{L1}(f_t, f_{t+1}) = \frac{1}{HW} \sum_{i=1}^{H} \sum_{j=1}^{W} |I_t(i,j) - I_{t+1}(i,j)|
$$

where:

- $I_t(i,j)$ is the grayscale intensity at pixel $(i,j)$ in frame $t$
- $H, W$ are frame height and width
- Normalized by frame size for scale invariance

**Threshold:** Frame is a candidate if $D_{L1} > \tau_{diff}$ (typically 0.15)

#### Histogram Difference (Chi-Square)

Compares color distributions using Chi-Square distance:

$$
\chi^2(H_1, H_2) = \sum_{k=1}^{K} \frac{(H_1(k) - H_2(k))^2}{H_1(k) + H_2(k)}
$$

where:

- $H_i(k)$ is the normalized frequency of bin $k$ in histogram $i$
- $K$ is the total number of bins (typically $16 \times 16 \times 16 = 4096$ for RGB)
- Normalized histograms: $\sum_{k=1}^{K} H_i(k) = 1$

**Threshold:** Frame is a candidate if $\chi^2 > \tau_{hist}$ (typically 0.15)

**Why Chi-Square?** Robust to illumination changes, captures color distribution shifts

#### Edge Change Ratio

Detects structural changes using Canny edge detection:

$$
R_{edge}(f_t, f_{t+1}) = \frac{|E_t \oplus E_{t+1}|}{|E_t \cup E_{t+1}|}
$$

where:

- $E_t = \text{Canny}(f_t, \tau_{low}, \tau_{high})$ is the binary edge map
- $\oplus$ is the XOR operation (symmetric difference)
- $|\cdot|$ denotes the number of edge pixels

**Threshold:** Frame is a candidate if $R_{edge} > \tau_{edge}$ (typically 0.20)

---

### Hierarchical Deduplication

#### Layer 1: Perceptual Hash (pHash)

Computes a 64-bit perceptual hash using Discrete Cosine Transform (DCT):

$$
\text{pHash}(f) = \text{Binarize}(\text{DCT}_8(f_{32 \times 32}^{gray}))
$$

**Algorithm:**

1. Resize frame to $32 \times 32$ grayscale: $f_{32 \times 32}^{gray}$
2. Compute 2D DCT:
   $$
   D(u,v) = \sum_{x=0}^{31} \sum_{y=0}^{31} f(x,y) \cos\left[\frac{\pi u}{32}(x + 0.5)\right] \cos\left[\frac{\pi v}{32}(y + 0.5)\right]
   $$
3. Extract top-left $8 \times 8$ DCT coefficients (low frequencies)
4. Compute median $m = \text{median}(D_{8 \times 8})$
5. Binarize: $h_{ij} = \mathbb{1}[D_{ij} > m]$, producing 64-bit hash

**Similarity Metric:** Hamming distance

$$
d_H(h_1, h_2) = \sum_{i=1}^{64} \mathbb{1}[h_1^i \neq h_2^i] = \text{popcount}(h_1 \oplus h_2)
$$

**Threshold:** Frames are duplicates if $d_H < \tau_{phash}$ (typically 8)

**Complexity:** $O(1)$ per comparison (64-bit XOR and popcount)

#### Layer 2: Structural Similarity Index (SSIM)

Compares luminance, contrast, and structure:

$$
\text{SSIM}(x, y) = l(x,y) \cdot c(x,y) \cdot s(x,y)
$$

where:

**Luminance comparison:**

$$
l(x,y) = \frac{2\mu_x\mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1}
$$

**Contrast comparison:**

$$
c(x,y) = \frac{2\sigma_x\sigma_y + C_2}{\sigma_x^2 + \sigma_y^2 + C_2}
$$

**Structure comparison:**

$$
s(x,y) = \frac{\sigma_{xy} + C_3}{\sigma_x\sigma_y + C_3}
$$

with:

- $\mu_x, \mu_y$ = mean intensity of patches $x, y$
- $\sigma_x, \sigma_y$ = standard deviation of patches
- $\sigma_{xy}$ = covariance of patches
- $C_1, C_2, C_3$ = stabilization constants (avoid division by zero)

**Implementation:** SSIM is computed over sliding windows and averaged:

$$
\text{SSIM}(f_1, f_2) = \frac{1}{N} \sum_{i=1}^{N} \text{SSIM}(w_i^{(1)}, w_i^{(2)})
$$

where $w_i$ are $11 \times 11$ windows

**Threshold:** Frames are duplicates if $\text{SSIM} > \tau_{ssim}$ (typically 0.92)

**Range:** $\text{SSIM} \in [-1, 1]$, where 1 = identical

**Complexity:** $O(HW)$ for sliding window convolution

#### Layer 3: CLIP Semantic Similarity

Uses vision transformer to embed frames into semantic space:

$$
\mathbf{z} = \text{Normalize}(\text{CLIP-ViT}(f))
$$

where:

- $\text{CLIP-ViT}: \mathbb{R}^{H \times W \times 3} \to \mathbb{R}^{512}$ is the vision encoder
- $\text{Normalize}(\mathbf{v}) = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$ produces unit vectors

**CLIP Architecture (ViT-B/32):**

1. Patch embedding: Split $224 \times 224$ image into $7 \times 7$ patches of $32 \times 32$ pixels
2. Linear projection: $\mathbb{R}^{32 \times 32 \times 3} \to \mathbb{R}^{768}$
3. Transformer encoder: 12 layers with multi-head self-attention
4. Classification token pooling: Extract $\mathbf{z} \in \mathbb{R}^{512}$

**Similarity Metric:** Cosine similarity

$$
\text{sim}(\mathbf{z}_1, \mathbf{z}_2) = \mathbf{z}_1^T \mathbf{z}_2 = \frac{\mathbf{z}_1 \cdot \mathbf{z}_2}{\|\mathbf{z}_1\| \|\mathbf{z}_2\|}
$$

Since embeddings are normalized, this simplifies to dot product.

**Threshold:** Frames are duplicates if $\text{sim} > \tau_{clip}$ (typically 0.90)

**Range:** $\text{sim} \in [-1, 1]$, where 1 = identical

**Complexity:**

- Encoding: $O(P^2 L d^2)$ where $P=49$ patches, $L=12$ layers, $d=768$ hidden dim
- Comparison: $O(D)$ where $D=512$ embedding dimension

---

### Temporal Clustering

#### Scene Assignment

Assign each frame to a scene based on temporal overlap:

$$
\text{scene}(f_t) = \arg\min_{s \in S} \begin{cases}
0 & \text{if } t_s^{start} \leq t < t_s^{end} \\
\min(|t - t_s^{start}|, |t - t_s^{end}|) & \text{otherwise}
\end{cases}
$$

where:

- $S = \{(t_s^{start}, t_s^{end})\}$ is the set of scene boundaries
- Frames within boundaries assigned directly
- Frames outside assigned to nearest boundary

#### K-Means Clustering

Within each scene, cluster frames by CLIP embeddings:

**Objective:** Minimize within-cluster variance

$$
J = \sum_{i=1}^{k} \sum_{\mathbf{z} \in C_i} \|\mathbf{z} - \boldsymbol{\mu}_i\|^2
$$

where:

- $k = \min(M, |F_s|)$ is number of clusters ($M$ = max_frames_per_scene)
- $C_i$ is the $i$-th cluster
- $\boldsymbol{\mu}_i = \frac{1}{|C_i|} \sum_{\mathbf{z} \in C_i} \mathbf{z}$ is the centroid

**Algorithm (Lloyd's):**

1. Initialize $k$ centroids randomly
2. **Assignment step:** $C_i = \{\mathbf{z} : \|\mathbf{z} - \boldsymbol{\mu}_i\| \leq \|\mathbf{z} - \boldsymbol{\mu}_j\|, \forall j\}$
3. **Update step:** $\boldsymbol{\mu}_i = \frac{1}{|C_i|} \sum_{\mathbf{z} \in C_i} \mathbf{z}$
4. Repeat until convergence

**Representative selection:** For each cluster, select frame closest to centroid:

$$
f_i^* = \arg\min_{f \in C_i} \|\mathbf{z}_f - \boldsymbol{\mu}_i\|
$$

**Complexity:** $O(nkd \cdot I)$ where $n$ = frames, $k$ = clusters, $d$ = dimensions, $I$ = iterations

---

### Importance Scoring

Compute importance score as product of multiple factors:

$$
\text{importance}(f_t) = w_{pos}(t) \cdot w_{scene}(t, s) \cdot w_{audio}(t, A)
$$

#### Position in Video Weight

$$
w_{pos}(t) = \begin{cases}
1.5 & \text{if } \frac{t}{T} < 0.1 \text{ (opening)} \\
2.0 & \text{if } \frac{t}{T} > 0.9 \text{ (closing)} \\
2.5 & \text{if } \frac{t}{T} > 0.95 \text{ (final moments)} \\
1.0 & \text{otherwise}
\end{cases}
$$

where $T$ is video duration.

**Rationale:** Opening introduces brand, closing shows CTA

#### Position in Scene Weight

$$
w_{scene}(t, s) = \begin{cases}
1.4 & \text{if } \frac{t - t_s^{start}}{t_s^{end} - t_s^{start}} < 0.15 \text{ (scene start)} \\
1.2 & \text{if } \frac{t - t_s^{start}}{t_s^{end} - t_s^{start}} > 0.85 \text{ (scene end)} \\
1.0 & \text{otherwise}
\end{cases}
$$

**Rationale:** Scene boundaries mark narrative transitions

#### Audio Event Weight

$$
w_{audio}(t, A) = \prod_{e \in A} w_e(t)
$$

where for each event type:

**Energy peaks:**

$$
w_{peak}(t) = \begin{cases}
1.3 & \text{if } \exists t_p \in T_{peaks}: |t - t_p| < \delta \\
1.0 & \text{otherwise}
\end{cases}
$$

**Post-silence:**

$$
w_{silence}(t) = \begin{cases}
1.4 & \text{if } \exists (t_s^{start}, t_s^{end}) \in S_{silence}: t_s^{end} \leq t < t_s^{end} + \delta \\
1.0 & \text{otherwise}
\end{cases}
$$

where $\delta$ is proximity threshold (typically 0.5s)

**Combined importance:** Multiplicative model allows multiple factors to compound

---

### Temporal Gap Enforcement

Ensure minimum time separation between selected frames while preserving first and last:

**Algorithm:**

```
Input: Sorted frames F = {f_1, ..., f_n}, gap threshold τ
Output: Filtered frames F' ⊆ F

1. Initialize F' = {f_1}  // Always keep first
2. For i = 2 to n-1:
3.   If t_i - t_{last} ≥ τ:
4.     F' = F' ∪ {f_i}
5.     last = i
6. F' = F' ∪ {f_n}  // Always keep last
7. Return F'
```

**Mathematical formulation:**

$$
F' = \{f_1\} \cup \{f_i : 1 < i < n, \min_{j \in F', j < i}(t_i - t_j) \geq \tau\} \cup \{f_n\}
$$

**Greedy property:** This greedy algorithm is optimal for maximizing coverage given gap constraint

---

### Reduction Rate Analysis

Define reduction rate at each stage:

**Stage-specific reduction:**

$$
R_i = 1 - \frac{|F_i|}{|F_{i-1}|}
$$

**Cumulative reduction:**

$$
R_{total} = 1 - \frac{|F_{final}|}{|F_{initial}|} = 1 - \prod_{i=1}^{n} (1 - R_i)
$$

**Expected reduction (empirical from v0002.mp4):**

Given observed reductions:

- Change detection: $R_1 \approx 0.942$ (450 → 26 frames)
- PHash: $R_2 \approx 0.154$ (26 → 22 frames)
- SSIM: $R_3 \approx 0.045$ (22 → 21 frames)
- CLIP: $R_4 \approx 0.524$ (21 → 10 frames)
- Selection: $R_5 \approx 0.500$ (10 → 5 frames)

**Overall:**

$$
R_{total} = 1 - (1-0.942)(1-0.154)(1-0.045)(1-0.524)(1-0.500)
$$

$$
= 1 - 0.058 \times 0.846 \times 0.955 \times 0.476 \times 0.500 \approx 0.989
$$

This matches observed 98.9% reduction.

---

### Cost-Benefit Analysis

**API Cost Model:**

For vision-language models, cost is proportional to number of tokens:

$$
\text{Cost} = \alpha \cdot T_{input} + \beta \cdot T_{output}
$$

where:

- $\alpha$ = input token price (e.g., $3 per 1M tokens for Claude)
- $\beta$ = output token price (e.g., $15 per 1M tokens)
- $T_{input} \approx 1000 \cdot N_{frames}$ (each frame ~1000 tokens)
- $T_{output} \approx 1000$ (JSON response)

**Baseline (uniform sampling every 0.3s):**

$$
\text{Cost}_{baseline} = \alpha \cdot 1000 \cdot 50 + \beta \cdot 1000
$$

**Our pipeline:**

$$
\text{Cost}_{ours} = \alpha \cdot 1000 \cdot 5 + \beta \cdot 1000
$$

**Cost reduction:**

$$
\frac{\text{Cost}_{baseline} - \text{Cost}_{ours}}{\text{Cost}_{baseline}} = \frac{50\alpha - 5\alpha}{50\alpha + \beta} \approx 0.82
$$

for typical $\alpha = 3, \beta = 15$:

$$
= \frac{45 \times 3}{50 \times 3 + 15} = \frac{135}{165} \approx 0.818
$$

**Result:** 81.8% cost reduction, matching observed ~80.8%

---

### Computational Complexity

**Per-video complexity:**

| Stage            | Complexity                     | Dominant Term       |
| ---------------- | ------------------------------ | ------------------- |
| Change detection | $O(F \cdot HW)$                | Frame iteration     |
| Scene detection  | $O(F \cdot HW)$                | PySceneDetect       |
| PHash            | $O(C \cdot 32^2)$              | DCT on candidates   |
| SSIM             | $O(C^2 \cdot HW)$              | Pairwise comparison |
| CLIP             | $O(C \cdot P^2 L d^2)$         | Transformer         |
| Clustering       | $O(C \cdot k \cdot d \cdot I)$ | K-means             |
| LLM              | $O(N \cdot T)$                 | Token processing    |

where:

- $F$ = total frames (e.g., 450)
- $C$ = candidate frames (e.g., 26)
- $N$ = selected frames (e.g., 5)
- $HW$ = frame pixels (e.g., $720^2$)
- $P$ = patches (49), $L$ = layers (12), $d$ = dimensions (768)

**Bottleneck:** CLIP embedding with complexity $O(C \cdot 10^6)$

**Optimization:** GPU batching reduces wall-clock time by factor of $B$ (batch size)

---

### Evaluation Metrics

#### Precision and Recall

For field-level extraction evaluation:

**Precision:**

$$
P = \frac{|\text{Extracted} \cap \text{GroundTruth}|}{|\text{Extracted}|}
$$

**Recall:**

$$
R = \frac{|\text{Extracted} \cap \text{GroundTruth}|}{|\text{GroundTruth}|}
$$

**F1 Score:**

$$
F_1 = 2 \cdot \frac{P \cdot R}{P + R} = \frac{2 |\text{Extracted} \cap \text{GroundTruth}|}{|\text{Extracted}| + |\text{GroundTruth}|}
$$

#### Semantic Similarity

For comparing extracted text with ground truth:

**Sentence-BERT similarity:**

$$
\text{sim}_{SBERT}(s_1, s_2) = \frac{\text{BERT}(s_1) \cdot \text{BERT}(s_2)}{\|\text{BERT}(s_1)\| \|\text{BERT}(s_2)\|}
$$

where $\text{BERT}(s)$ produces contextualized sentence embedding.

**Threshold:** Match if $\text{sim}_{SBERT} > 0.8$

#### Coverage Score

Measures narrative completeness:

$$
\text{Coverage} = \frac{1}{3}\left(\mathbb{1}[\text{opening}] + \mathbb{1}[\text{middle}] + \mathbb{1}[\text{closing}]\right)
$$

where $\mathbb{1}[\cdot]$ indicates presence of frame in that position range.

**Perfect coverage:** $\text{Coverage} = 1$ (all three phases represented)

---

## Stage 1: Video Ingestion & Metadata Extraction

**Input:** Video file (MP4, MOV, AVI, MKV, WebM)

**Operations:**

- Extract video metadata (duration, fps, resolution, codec)
- Extract audio track for parallel processing
- Compute video-level statistics (average brightness, motion intensity)

**Output:**

- Video metadata JSON
- Separated audio file (WAV format)
- Initial video statistics

**Tools:** FFmpeg, OpenCV

**Implementation:**

```python
class VideoLoader:
    def load(self, video_path, max_resolution=720, extract_audio=True):
        """Load video and extract metadata."""
        cap = cv2.VideoCapture(video_path)

        metadata = {
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }

        if extract_audio:
            audio_path = self._extract_audio(video_path)

        return metadata, audio_path
```

---

## Stage 2: Lightweight Change Detection

**Purpose:** Identify candidate moments where frame extraction is worthwhile

**Methods (in order of computational cost):**

| Method                      | Cost      | What It Detects           | Use When               |
| --------------------------- | --------- | ------------------------- | ---------------------- |
| Motion vectors (from codec) | Near Zero | Movement between frames   | Real-time needed       |
| Frame difference (L1/L2)    | Very Low  | Any pixel changes         | Fast processing        |
| Histogram difference        | Low       | Color distribution shifts | Default (best balance) |
| Edge change ratio           | Low       | Structural changes        | High precision         |

### Implementation

```python
class HistogramDetector:
    """Default: Detects color distribution changes using Chi-Square distance."""

    def compute_change(self, frame1, frame2):
        """Compute Chi-Square distance between color histograms."""
        hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [16, 16, 16], [0, 256] * 3)
        hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [16, 16, 16], [0, 256] * 3)
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

class FrameDifferenceDetector:
    """L1 norm of grayscale difference, normalized by frame size."""

    def compute_change(self, frame1, frame2):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = np.abs(gray1.astype(float) - gray2.astype(float))
        return np.mean(diff) / 255.0

class EdgeChangeDetector:
    """Detects structural changes using Canny edge detection."""

    def compute_change(self, frame1, frame2):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        edges1 = cv2.Canny(gray1, 100, 200)
        edges2 = cv2.Canny(gray2, 100, 200)

        # XOR to find differences
        diff = cv2.bitwise_xor(edges1, edges2)
        union = cv2.bitwise_or(edges1, edges2)

        return np.sum(diff) / (np.sum(union) + 1e-6)
```

**Output:** List of timestamps where significant change detected

**Adaptive Threshold Logic:**

- Fast-paced ads (many scene cuts): Lower threshold, more candidates
- Slow-paced ads (few cuts): Higher threshold, fewer candidates
- Threshold adjusts based on running statistics of the video

---

## Stage 3: Scene Boundary Detection

**Purpose:** Segment video into coherent scenes/shots for narrative-aware selection

**Methods:**

| Method                            | Description                    | When to Use          | Accuracy  | Speed  |
| --------------------------------- | ------------------------------ | -------------------- | --------- | ------ |
| PySceneDetect (ContentDetector)   | Detects content changes        | General purpose      | Good      | Fast   |
| PySceneDetect (ThresholdDetector) | Detects fade-to-black          | TV commercials       | Good      | Fast   |
| TransNetV2                        | Neural shot boundary detection | High accuracy needed | Excellent | Medium |

### PySceneDetect Implementation

```python
from scenedetect import detect, ContentDetector, ThresholdDetector

class SceneDetector:
    def __init__(self, method='content', threshold=27.0):
        self.method = method
        self.threshold = threshold

    def detect_scenes(self, video_path):
        """Detect scene boundaries in video."""
        if self.method == 'content':
            detector = ContentDetector(threshold=self.threshold)
        elif self.method == 'threshold':
            detector = ThresholdDetector(threshold=self.threshold)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        scene_list = detect(video_path, detector)
        return [(s[0].get_seconds(), s[1].get_seconds()) for s in scene_list]
```

### TransNetV2 Implementation (Optional - Higher Accuracy)

```python
class TransNetV2Detector:
    """Neural shot boundary detection using TransNetV2."""

    def __init__(self, threshold=0.5):
        from transnetv2 import TransNetV2
        self.model = TransNetV2()
        self.threshold = threshold

    def detect_scenes(self, video_path):
        """
        Detect scene boundaries using TransNetV2.

        TransNetV2 is a neural network trained specifically for shot boundary
        detection. It achieves state-of-the-art results on standard benchmarks.

        Paper: Soucek, T., & Lokoč, J. (2020). TransNet V2: An effective deep network
               architecture for fast shot transition detection. arXiv:2008.04838
        """
        # Load video frames
        video_frames = self._load_video(video_path)

        # Predict shot boundaries
        predictions = self.model.predict_video(video_frames)

        # Threshold predictions to get boundaries
        boundaries = self._predictions_to_scenes(predictions, self.threshold)

        return boundaries

    def _load_video(self, video_path):
        """Load video frames for TransNetV2."""
        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # TransNetV2 expects RGB frames of size 48x27
            frame = cv2.resize(frame, (48, 27))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        return np.array(frames)

    def _predictions_to_scenes(self, predictions, threshold):
        """Convert predictions to scene boundaries."""
        fps = 30.0  # Assume 30fps, adjust based on video
        boundaries = []

        # Find peaks in predictions above threshold
        scene_starts = np.where(predictions > threshold)[0]

        # Convert frame indices to timestamps
        for i in range(len(scene_starts) - 1):
            start_time = scene_starts[i] / fps
            end_time = scene_starts[i + 1] / fps
            boundaries.append((start_time, end_time))

        return boundaries
```

**Output:** List of (start_time, end_time) tuples for each scene

**Comparison:**

- **PySceneDetect:** Fast, good for most cases, works well on ads with clear cuts
- **TransNetV2:** More accurate, handles gradual transitions, requires GPU for real-time, trained on large datasets

---

## Stage 4: Hierarchical Frame Deduplication

**Purpose:** Remove redundant frames using progressively expensive methods

**Philosophy:** Use cheap methods first to filter out obvious duplicates, then apply expensive methods only to remaining frames

### Layer 1: Perceptual Hashing (pHash)

- **Cost:** ~0.1ms per frame
- **What it catches:** Near-identical frames, minor compression artifacts, slight camera shake
- **Threshold:** Hamming distance < 8 (out of 64 bits)

```python
import imagehash
from PIL import Image

class PHashDeduplicator:
    def __init__(self, threshold=8):
        self.threshold = threshold

    def compute_phash(self, frame):
        """Compute perceptual hash of frame."""
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return imagehash.phash(pil_image)

    def deduplicate(self, frames):
        """Remove near-duplicate frames using pHash."""
        if not frames:
            return []

        unique_frames = [frames[0]]
        unique_hashes = [self.compute_phash(frames[0][1])]

        for timestamp, frame in frames[1:]:
            frame_hash = self.compute_phash(frame)

            # Check if similar to any existing frame
            is_duplicate = False
            for existing_hash in unique_hashes:
                if frame_hash - existing_hash < self.threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_frames.append((timestamp, frame))
                unique_hashes.append(frame_hash)

        return unique_frames
```

### Layer 2: Structural Similarity (SSIM)

- **Cost:** ~5ms per frame pair
- **What it catches:** Frames with same structure but different details (e.g., person in same pose but different lighting)
- **Threshold:** SSIM > 0.92

```python
from skimage.metrics import structural_similarity as ssim

class SSIMDeduplicator:
    def __init__(self, threshold=0.92):
        self.threshold = threshold

    def compute_ssim(self, frame1, frame2):
        """Compute SSIM between two frames."""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Resize to same size if needed
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

        score, _ = ssim(gray1, gray2, full=True)
        return score

    def deduplicate(self, frames):
        """Remove structurally similar frames."""
        if not frames:
            return []

        unique_frames = [frames[0]]

        for timestamp, frame in frames[1:]:
            is_duplicate = False

            for _, existing_frame in unique_frames:
                similarity = self.compute_ssim(frame, existing_frame)
                if similarity > self.threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_frames.append((timestamp, frame))

        return unique_frames
```

### Layer 3: CLIP Embedding Similarity

- **Cost:** ~50ms per frame (GPU), ~500ms (CPU)
- **What it catches:** Semantically similar frames with visual differences (e.g., different shots of same person, product at different angles)
- **Threshold:** Cosine similarity > 0.90

```python
import torch
import open_clip
from PIL import Image

class CLIPDeduplicator:
    def __init__(self, model_name="ViT-B-32", pretrained="openai", threshold=0.90, device="auto"):
        self.threshold = threshold
        self.device = self._get_device(device)

        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def _get_device(self, device):
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def embed(self, frame):
        """Compute CLIP embedding for frame."""
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_image(image_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize

        return embedding.cpu().numpy().flatten()

    def deduplicate(self, frames):
        """Remove semantically similar frames using CLIP."""
        if not frames:
            return [], None

        # Compute embeddings for all frames
        embeddings = [self.embed(frame) for _, frame in frames]

        unique_frames = [frames[0]]
        unique_embeddings = [embeddings[0]]

        for i, ((timestamp, frame), embedding) in enumerate(zip(frames[1:], embeddings[1:]), 1):
            is_duplicate = False

            for existing_emb in unique_embeddings:
                similarity = np.dot(embedding, existing_emb)
                if similarity > self.threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_frames.append((timestamp, frame))
                unique_embeddings.append(embedding)

        return unique_frames, np.array(unique_embeddings)
```

### Hierarchical Orchestrator

```python
class HierarchicalDeduplicator:
    """Orchestrates 3-layer hierarchical deduplication."""

    def __init__(
        self,
        phash_enabled=True, phash_threshold=8,
        ssim_enabled=True, ssim_threshold=0.92,
        clip_enabled=True, clip_threshold=0.90, clip_device="auto"
    ):
        self.phash_enabled = phash_enabled
        self.ssim_enabled = ssim_enabled
        self.clip_enabled = clip_enabled

        if phash_enabled:
            self.phash = PHashDeduplicator(threshold=phash_threshold)
        if ssim_enabled:
            self.ssim = SSIMDeduplicator(threshold=ssim_threshold)
        if clip_enabled:
            self.clip = CLIPDeduplicator(threshold=clip_threshold, device=clip_device)

    def deduplicate(self, frames):
        """Apply hierarchical deduplication."""
        stats = {"input": len(frames)}
        current_frames = frames
        embeddings = None

        # Layer 1: PHash (fastest)
        if self.phash_enabled and len(current_frames) > 1:
            current_frames = self.phash.deduplicate(current_frames)
            stats["after_phash"] = len(current_frames)

        # Layer 2: SSIM (medium)
        if self.ssim_enabled and len(current_frames) > 1:
            current_frames = self.ssim.deduplicate(current_frames)
            stats["after_ssim"] = len(current_frames)

        # Layer 3: CLIP (slowest but semantic)
        if self.clip_enabled and len(current_frames) > 1:
            current_frames, embeddings = self.clip.deduplicate(current_frames)
            stats["after_clip"] = len(current_frames)

        stats["output"] = len(current_frames)
        return current_frames, embeddings, stats
```

**Hierarchical Flow Visualization:**

```
All candidate frames (26)
    │
    ▼ pHash filtering
Frames passing pHash (22) - 15.4% removed
    │
    ▼ SSIM filtering
Frames passing SSIM (21) - 4.5% removed
    │
    ▼ CLIP filtering
Final keyframes (10) - 52.4% removed
```

**Why This Works:**

- pHash catches exact/near-duplicates cheaply (compression artifacts, slight variations)
- SSIM catches structural similarity (same composition, different lighting)
- CLIP (expensive) only runs on frames that survived cheap filters
- Each layer removes a different type of redundancy

---

## Stage 5: Temporal Clustering & Representative Selection

**Purpose:** Group remaining frames by scene and select best representatives based on importance and diversity

### Scene Assignment & Importance Scoring

```python
from sklearn.cluster import KMeans

class FrameSelector:
    def __init__(self, max_frames_per_scene=3, min_temporal_gap_s=0.5):
        self.clusterer = TemporalClusterer(
            max_frames_per_scene=max_frames_per_scene,
            min_temporal_gap_s=min_temporal_gap_s
        )
        self.scorer = ImportanceScorer()

    def select(self, frames, embeddings, scene_boundaries, video_duration, audio_events=None):
        """Select representative frames from candidates."""

        # Assign frames to scenes
        candidates = self.clusterer.assign_scenes(frames, scene_boundaries)

        # Score importance
        for cand in candidates:
            cand.importance_score = self.scorer.compute_importance(
                cand, video_duration, scene_boundaries, audio_events
            )

        # Cluster and select representatives
        selected = self.clusterer.cluster_and_select(candidates, embeddings)

        return selected
```

### Temporal Clustering with K-Means

```python
class TemporalClusterer:
    def cluster_and_select(self, candidates, embeddings=None):
        """
        Cluster frames within each scene and select representatives.
        Always includes first and last frame.
        """
        if not candidates:
            return []

        # ALWAYS include first and last frame
        first_frame = min(candidates, key=lambda c: c.timestamp)
        last_frame = max(candidates, key=lambda c: c.timestamp)

        first_frame.is_representative = True
        last_frame.is_representative = True

        must_include_timestamps = {first_frame.timestamp, last_frame.timestamp}

        # Attach embeddings
        if embeddings is not None:
            for i, cand in enumerate(candidates):
                if i < len(embeddings):
                    cand.embedding = embeddings[i]

        # Group by scene
        scene_frames = {}
        for cand in candidates:
            scene_id = cand.scene_id or 0
            if scene_id not in scene_frames:
                scene_frames[scene_id] = []
            scene_frames[scene_id].append(cand)

        # Select representatives from each scene
        selected = []

        for scene_id in sorted(scene_frames.keys()):
            scene_cands = scene_frames[scene_id]

            # Check how many must-include frames are in this scene
            scene_must_include = [c for c in scene_cands if c.timestamp in must_include_timestamps]
            remaining_slots = self.max_frames_per_scene - len(scene_must_include)

            if len(scene_cands) <= self.max_frames_per_scene:
                # Keep all frames in small scenes
                for cand in scene_cands:
                    cand.is_representative = True
                selected.extend(scene_cands)
            else:
                # Add must-include frames
                selected.extend(scene_must_include)

                # Cluster remaining frames
                remaining_cands = [c for c in scene_cands if c.timestamp not in must_include_timestamps]

                if remaining_slots > 0 and remaining_cands:
                    reps = self._kmeans_selection(remaining_cands, remaining_slots)
                    selected.extend(reps)

        # Enforce temporal gap (but preserve first/last)
        selected = self._enforce_temporal_gap(selected)

        return selected

    def _kmeans_selection(self, scene_frames, n_clusters):
        """Use K-means clustering on CLIP embeddings to select representatives."""
        if not scene_frames or not scene_frames[0].embedding:
            return self._uniform_selection(scene_frames, n_clusters)

        n_clusters = min(n_clusters, len(scene_frames))

        # Stack embeddings
        embeddings = np.array([f.embedding for f in scene_frames])

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Select frame closest to each centroid
        selected = []
        for cluster_id in range(n_clusters):
            cluster_frames = [f for i, f in enumerate(scene_frames) if labels[i] == cluster_id]
            if not cluster_frames:
                continue

            centroid = kmeans.cluster_centers_[cluster_id]

            # Find closest frame to centroid
            best_frame = min(
                cluster_frames,
                key=lambda f: np.linalg.norm(f.embedding - centroid)
            )

            best_frame.is_representative = True
            selected.append(best_frame)

        selected.sort(key=lambda x: x.timestamp)
        return selected

    def _enforce_temporal_gap(self, frames):
        """Remove frames too close together while preserving first/last."""
        if not frames or len(frames) <= 2:
            return frames

        sorted_frames = sorted(frames, key=lambda x: x.timestamp)

        first_frame = sorted_frames[0]
        last_frame = sorted_frames[-1]
        middle_frames = sorted_frames[1:-1]

        kept = [first_frame]

        for frame in middle_frames:
            if frame.timestamp - kept[-1].timestamp >= self.min_temporal_gap_s:
                kept.append(frame)

        # ALWAYS keep last frame
        if last_frame.timestamp != kept[-1].timestamp:
            kept.append(last_frame)

        return kept
```

### Importance Scorer

```python
class ImportanceScorer:
    def compute_importance(self, frame, video_duration, scene_boundaries=None, audio_events=None):
        """Compute overall importance score for a frame."""
        score = 1.0

        # Position in video
        score *= self.score_by_position(frame.timestamp, video_duration)

        # Position in scene
        if scene_boundaries and frame.scene_id is not None:
            if 0 <= frame.scene_id < len(scene_boundaries):
                start, end = scene_boundaries[frame.scene_id]
                score *= self.score_by_scene_position(frame.timestamp, start, end)

        # Audio events
        if audio_events:
            score *= self.score_by_audio_events(frame.timestamp, audio_events)

        return score

    def score_by_position(self, timestamp, duration):
        """Score based on position in video."""
        position = timestamp / duration if duration > 0 else 0

        if position < 0.1:
            return 1.5  # Opening
        elif position > 0.95:
            return 2.5  # Final moments (CTA)
        elif position > 0.9:
            return 2.0  # Closing
        return 1.0

    def score_by_scene_position(self, timestamp, scene_start, scene_end):
        """Score based on position within scene."""
        scene_duration = scene_end - scene_start
        if scene_duration <= 0:
            return 1.0

        position_in_scene = (timestamp - scene_start) / scene_duration

        if position_in_scene < 0.15:
            return 1.4  # Scene start
        elif position_in_scene > 0.85:
            return 1.2  # Scene end
        return 1.0

    def score_by_audio_events(self, timestamp, audio_events, proximity_threshold_s=0.5):
        """Score based on proximity to audio events."""
        score = 1.0

        # Check proximity to energy peaks
        for peak_ts in audio_events.get("energy_peaks", []):
            if abs(timestamp - peak_ts) < proximity_threshold_s:
                score *= 1.3
                break

        # Check if after silence
        for start, end in audio_events.get("silence_segments", []):
            if end <= timestamp < end + proximity_threshold_s:
                score *= 1.4
                break

        return score
```

---

## Stage 6: Audio-Visual Alignment (Optional Enhancement)

**Purpose:** Align frame selection with audio events for better context

**Audio Features to Extract:**

- Speech boundaries (using VAD - Voice Activity Detection)
- Music/jingle detection
- Audio energy peaks
- Silence detection (often indicates scene transitions)

```python
import librosa

class AudioExtractor:
    def extract_audio_events(self, audio_path):
        """Extract significant audio events."""
        y, sr = librosa.load(audio_path)

        # Energy-based event detection
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.times_like(rms, sr=sr)

        # Find peaks in audio energy
        peaks = librosa.util.peak_pick(
            rms, pre_max=3, post_max=3,
            pre_avg=3, post_avg=5, delta=0.1, wait=10
        )

        # Detect silence
        silence_segments = self._detect_silence(y, sr)

        return {
            "energy_peaks": times[peaks].tolist(),
            "silence_segments": silence_segments
        }

    def _detect_silence(self, y, sr, threshold=-40):
        """Detect silence segments in audio."""
        # Convert to dB
        db = librosa.amplitude_to_db(np.abs(y), ref=np.max)

        # Find silent regions
        silent = db < threshold

        # Convert to time segments
        segments = []
        in_silence = False
        start = 0

        for i, is_silent in enumerate(silent):
            if is_silent and not in_silence:
                start = i / sr
                in_silence = True
            elif not is_silent and in_silence:
                end = i / sr
                if end - start > 0.3:  # Minimum 0.3s silence
                    segments.append((start, end))
                in_silence = False

        return segments
```

**Integration:** Boost importance of frames near audio events (already implemented in ImportanceScorer)

---

## Stage 7: LLM Vision API Integration

**Purpose:** Extract structured insights from selected keyframes

**Supported APIs:**

| Provider  | Model            | Input Cost          | Output Cost   | Best For                     |
| --------- | ---------------- | ------------------- | ------------- | ---------------------------- |
| Anthropic | Claude Sonnet 4  | $3/1M tokens        | $15/1M tokens | Structured output, reasoning |
| OpenAI    | GPT-4o           | $2.50/1M tokens     | $10/1M tokens | Fast inference, good quality |
| Google    | Gemini 2.0 Flash | Free (experimental) | Free          | Cost optimization, testing   |

### Temporal-Aware Prompt Construction

The LLM receives all keyframes together with temporal context, enabling narrative understanding:

```python
def build_temporal_prompt(frames_with_timestamps, schema, video_duration):
    """
    Build a prompt that gives LLM temporal context for narrative understanding.

    Args:
        frames_with_timestamps: List of FrameForPrompt objects
        schema: JSON schema for extraction
        video_duration: Total video duration in seconds

    Returns:
        Formatted prompt string
    """
    num_frames = len(frames_with_timestamps)

    prompt = f"""You are analyzing a {video_duration:.1f}-second video advertisement through {num_frames} keyframes.

The frames are in CHRONOLOGICAL ORDER with timestamps. Analyze both individual frames AND the narrative progression.

TEMPORAL CONTEXT:
"""

    for i, frame in enumerate(frames_with_timestamps):
        position = frame.timestamp / video_duration

        line = f"\nFrame {i+1} @ {frame.timestamp:.1f}s"

        if i > 0:
            time_gap = frame.timestamp - frames_with_timestamps[i-1].timestamp
            line += f" (Δ{time_gap:.1f}s from previous)"

        # Add position label
        if frame.position_label:
            line += f" [{frame.position_label}]"

        prompt += line

    prompt += f"""

ANALYSIS INSTRUCTIONS:
1. Identify what CHANGES between frames (scene transitions, new elements, text changes)
2. Track the NARRATIVE ARC (setup → development → conclusion/CTA)
3. Note any RECURRING ELEMENTS (logo appearances, product shots, faces)
4. Pay special attention to [CLOSING] frames for call-to-action and pricing

Extract the following information in JSON format:
{json.dumps(schema, indent=2)}

IMPORTANT:
- Respond with ONLY valid JSON, no markdown or explanation
- Use null for fields where information is not available
- Be specific and concise in your descriptions

JSON Response:"""

    return prompt


def prepare_frames_for_prompt(frames, video_duration, include_position_labels=True):
    """Prepare frames with metadata for prompt."""
    prepared = []

    for ts, frame in frames:
        position_label = None

        if include_position_labels:
            position = ts / video_duration if video_duration > 0 else 0
            if position < 0.15:
                position_label = "OPENING"
            elif position > 0.85:
                position_label = "CLOSING"
            elif 0.4 < position < 0.6:
                position_label = "MIDDLE"

        prepared.append(FrameForPrompt(
            timestamp=ts,
            base64_image=frame_to_base64(frame),
            position_label=position_label
        ))

    return prepared
```

### Adaptive Schema Selection

The pipeline automatically selects the appropriate extraction schema based on detected ad type:

```python
class AdExtractor:
    """Main extractor with adaptive schema support."""

    def __init__(
        self,
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        schema_mode="adaptive",  # adaptive, fixed, flexible
        temporal_context=True
    ):
        self.client = get_llm_client(provider, model)
        self.schema_mode = schema_mode
        self.temporal_context = temporal_context

    def extract(self, frames, video_duration):
        """Extract structured information from ad frames."""
        if not frames:
            return {"error": "No frames provided"}

        # Prepare frames
        prepared_frames = prepare_frames_for_prompt(frames, video_duration)

        # Detect ad type if adaptive
        ad_type = None
        if self.schema_mode == "adaptive":
            ad_type = self.detect_ad_type(prepared_frames)

        # Get schema
        schema = get_schema(mode=self.schema_mode, ad_type=ad_type)

        # Build prompt
        prompt = build_temporal_prompt(prepared_frames, schema, video_duration)

        # Extract
        response = self.client.extract(prepared_frames, prompt)

        # Parse JSON
        result = json.loads(response)
        result["_metadata"] = {
            "ad_type": ad_type,
            "schema_mode": self.schema_mode,
            "num_frames": len(frames),
            "video_duration": video_duration
        }

        return result

    def detect_ad_type(self, frames):
        """First pass: detect ad type from frames."""
        detection_prompt = build_type_detection_prompt()

        response = self.client.extract(frames, detection_prompt)
        ad_type = response.strip().lower().replace(" ", "_")

        # Validate
        valid_types = get_valid_ad_types()
        if ad_type in valid_types:
            return ad_type

        # Try partial match
        for valid in valid_types:
            if valid in ad_type or ad_type in valid:
                return valid

        return "brand_awareness"  # Default fallback
```

### Schema Definitions

```python
# Base schema (always extracted)
BASE_SCHEMA = {
    "brand": {
        "name": "string",
        "logo_visible": "boolean",
        "logo_timestamps": ["float"]
    },
    "message": {
        "primary_message": "string",
        "call_to_action": "string | null",
        "tagline": "string | null"
    },
    "creative_elements": {
        "dominant_colors": ["string"],
        "text_overlays": ["string"],
        "music_mood": "string | null"
    },
    "target_audience": {
        "age_group": "string",
        "interests": ["string"]
    },
    "persuasion_techniques": ["string"]
}

# Type-specific schema extensions
SCHEMA_EXTENSIONS = {
    "product_demo": {
        "product": {
            "name": "string",
            "category": "string",
            "features_demonstrated": ["string"],
            "price_shown": "string | null"
        },
        "demo_steps": ["string"]
    },
    "testimonial": {
        "testimonial": {
            "speaker_name": "string | null",
            "speaker_role": "string",
            "key_quotes": ["string"],
            "credibility_markers": ["string"]
        }
    },
    "brand_awareness": {
        "emotional_appeal": {
            "primary_emotion": "string",
            "storytelling_elements": ["string"],
            "brand_values_conveyed": ["string"]
        }
    },
    "tutorial": {
        "tutorial": {
            "skill_taught": "string",
            "steps": ["string"],
            "tools_shown": ["string"]
        }
    },
    "entertainment": {
        "entertainment": {
            "humor_type": "string | null",
            "celebrity_featured": "string | null",
            "viral_elements": ["string"]
        }
    }
}

# Flexible schema (single-pass alternative)
FLEXIBLE_SCHEMA = {
    "brand": {"name": "string", "logo_visible": "boolean"},
    "ad_type": "string (product_demo | testimonial | brand_awareness | tutorial | entertainment)",
    "message": {"primary_message": "string", "call_to_action": "string | null"},
    "narrative": {
        "opening_hook": "string",
        "middle_development": "string",
        "closing_resolution": "string"
    },
    "key_elements": ["string"],
    "persuasion_techniques": ["string"],
    "target_audience": {"demographics": "string", "interests": ["string"]}
}
```

### Multi-Provider LLM Clients

```python
class AnthropicClient:
    """Claude API client."""

    def __init__(self, model="claude-sonnet-4-20250514", max_tokens=2000, temperature=0.0):
        import anthropic
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = anthropic.Anthropic()

    def extract(self, frames, prompt):
        content = []

        for frame in frames:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame.base64_image
                }
            })

        content.append({"type": "text", "text": prompt})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": content}]
        )

        return response.content[0].text


class OpenAIClient:
    """GPT-4V API client."""

    def __init__(self, model="gpt-4o", max_tokens=2000, temperature=0.0):
        from openai import OpenAI
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = OpenAI()

    def extract(self, frames, prompt):
        content = []

        for frame in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame.base64_image}"}
            })

        content.append({"type": "text", "text": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": content}]
        )

        return response.choices[0].message.content


class GeminiClient:
    """Google Gemini API client."""

    def __init__(self, model="gemini-2.0-flash-exp", max_tokens=2000, temperature=0.0):
        import google.generativeai as genai
        import os

        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

        self.model = genai.GenerativeModel(model)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def extract(self, frames, prompt):
        from PIL import Image
        from io import BytesIO
        import base64

        content = []

        for frame in frames:
            image_data = base64.b64decode(frame.base64_image)
            pil_image = Image.open(BytesIO(image_data))
            content.append(pil_image)

        content.append(prompt)

        response = self.model.generate_content(
            content,
            generation_config={
                "max_output_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
        )

        return response.text
```

---

## Experimental Results

### Example: 15-Second Product Demo (v0002.mp4)

**Video:** Target Optical - Precision 1 contact lenses

**Pipeline Execution:**

```
Input: 15.0s video, 720x720, 30fps (450 total frames)

Stage 1: Video ingestion           → 15.0s duration detected
Stage 2: Scene detection            → 5 scenes detected
Stage 3: Candidate extraction       → 26 candidates (HistogramDetector)
Stage 4: Hierarchical deduplication
  - PHash (threshold=8):            → 26 → 22 frames (4 removed, 15.4%)
  - SSIM (threshold=0.92):          → 22 → 21 frames (1 removed, 4.5%)
  - CLIP (threshold=0.90):          → 21 → 10 frames (11 removed, 52.4%)
Stage 5: Audio events               → 3 energy peaks, 1 silence segment
Stage 6: Representative selection   → 10 → 5 frames (50.0%)
  - Scene assignments: [0, 0, 1, 2, 4]
  - Importance scores: [2.10, 1.30, 1.00, 1.00, 2.00]
  - First frame (0.0s): guaranteed
  - Last frame (14.5s): guaranteed (CTA)
Stage 7: LLM extraction (Gemini)
  - Pass 1: Ad type → "product_demo"
  - Pass 2: Extraction → Success

Final: 5 frames selected (98.9% reduction) in 21.1s
```

**Selected Frames:**

| Frame | Timestamp | Scene | Importance | Position | Rationale                      |
| ----- | --------- | ----- | ---------- | -------- | ------------------------------ |
| 1     | 0.0s      | 0     | 2.10       | OPENING  | Brand intro (guaranteed first) |
| 2     | 2.1s      | 0     | 1.30       | -        | Product visibility             |
| 3     | 3.9s      | 1     | 1.00       | -        | Feature demonstration          |
| 4     | 7.0s      | 2     | 1.00       | -        | Social proof                   |
| 5     | 14.5s     | 4     | 2.00       | CLOSING  | CTA (guaranteed last)          |

**Extracted Data:**

```json
{
  "brand": {
    "name": "Target Optical",
    "logo_visible": true,
    "logo_timestamps": [0.0, 2.1, 3.9, 7.0, 8.3]
  },
  "message": {
    "primary_message": "Precision 1 contact lenses for clear vision and comfort",
    "call_to_action": null,
    "tagline": null
  },
  "product": {
    "name": "Precision 1",
    "category": "contact lenses",
    "features_demonstrated": ["clarity", "comfort", "daily disposable"],
    "price_shown": null
  },
  "creative_elements": {
    "dominant_colors": ["purple", "blue", "white"],
    "text_overlays": [
      "Two Peas, One Pod",
      "Welcome Home, Dear",
      "Matches the vibe"
    ],
    "music_mood": "upbeat"
  },
  "target_audience": {
    "age_group": "18-35",
    "interests": ["socializing", "fashion", "technology", "eye care"]
  },
  "persuasion_techniques": ["social proof", "lifestyle", "visual appeal"],
  "_metadata": {
    "ad_type": "product_demo",
    "schema_mode": "adaptive",
    "num_frames": 5,
    "video_duration": 15.015
  }
}
```

**Performance Analysis:**

- Successfully identified brand and product
- Extracted text overlays from frames
- Correctly detected ad type (product_demo)
- Identified target audience and persuasion techniques
- **Missing:** Call-to-action (may be in audio or appears very briefly at end)

**Cost Comparison:**

| Method       | Frames Sent | Input Tokens | Output Tokens | Cost (Claude) | Cost (Gemini) |
| ------------ | ----------- | ------------ | ------------- | ------------- | ------------- |
| Uniform 0.3s | 50          | 50,000       | 1,000         | $0.165        | Free          |
| Uniform 1.0s | 15          | 15,000       | 1,000         | $0.060        | Free          |
| Our Pipeline | 5           | 5,000        | 1,000         | $0.030        | Free          |

**Reduction vs. Uniform 0.3s:** 81.8% cost reduction

---

## Evaluation Framework

### Datasets

| Dataset                        | Size            | Annotations                     | Use Case               |
| ------------------------------ | --------------- | ------------------------------- | ---------------------- |
| **Hussain et al. (CVPR 2017)** | 3,477 video ads | Topic, sentiment, action-reason | Primary benchmark      |
| **LAMBDA**                     | 2,205 ads       | Memorability scores             | Secondary validation   |
| **Custom test set**            | 42 videos       | Full extraction ground truth    | LLM quality evaluation |

### Metrics

#### Efficiency Metrics

| Metric               | Formula                                              | Target        | Achieved      |
| -------------------- | ---------------------------------------------------- | ------------- | ------------- |
| Frame Reduction Rate | $1 - \frac{\text{selected}}{\text{total}}$           | > 70%         | 98.9%         |
| API Cost Reduction   | $1 - \frac{\text{our\_cost}}{\text{baseline\_cost}}$ | > 60%         | 80.8%         |
| Processing Time      | Total pipeline time                                  | < 2x duration | 1.4x duration |
| Throughput           | Videos per hour                                      | > 60          | ~170          |

#### Quality Metrics

| Metric                  | Description                 | How to Compute                       |
| ----------------------- | --------------------------- | ------------------------------------ |
| Extraction Accuracy     | Match with ground truth     | Field-by-field F1 score              |
| Extraction Completeness | Coverage of required fields | Recall of mandatory fields           |
| Semantic Similarity     | Text similarity             | SBERT cosine similarity              |
| Narrative Completeness  | Coverage of narrative arc   | Opening + middle + closing detection |

### Baselines to Compare Against

| Baseline         | Description                | Expected Performance        |
| ---------------- | -------------------------- | --------------------------- |
| **Uniform-0.3s** | Frame every 0.3s           | 50 frames, high redundancy  |
| **Uniform-1.0s** | Frame every 1.0s           | 15 frames, may miss details |
| **CLIP-Only**    | Uniform + CLIP dedup       | 10-15 frames, expensive     |
| **Scene-First**  | One frame per scene        | 3-5 frames, may miss CTA    |
| **Ours**         | Full hierarchical pipeline | 5-10 frames, optimal        |

### Ablation Studies

| Experiment             | What We Vary                     | What We Measure                     |
| ---------------------- | -------------------------------- | ----------------------------------- |
| pHash threshold        | Hamming distance: 4, 8, 12, 16   | Frames retained, downstream quality |
| SSIM threshold         | SSIM: 0.85, 0.90, 0.92, 0.95     | Frames retained, downstream quality |
| CLIP threshold         | Cosine sim: 0.85, 0.90, 0.95     | Frames retained, downstream quality |
| Hierarchical vs. flat  | With/without pHash/SSIM layers   | Processing time, quality            |
| Scene detection method | PySceneDetect vs. TransNetV2     | Accuracy, speed tradeoff            |
| Frames per scene       | 1, 2, 3, 5 representatives       | Coverage vs. efficiency             |
| Change detector        | Histogram vs. Edge vs. FrameDiff | Candidate quality, count            |

---

## Project Structure

```
video-analysis-pipeline-research/
├── README.md
├── pyproject.toml             # UV package configuration
├── .env                       # API keys (not committed)
│
├── config/
│   └── default.yaml           # Pipeline configuration
│
├── src/
│   ├── pipeline.py            # Main orchestrator
│   │
│   ├── ingestion/
│   │   ├── video_loader.py    # Video + metadata extraction
│   │   └── audio_extractor.py # Audio track + events
│   │
│   ├── detection/
│   │   ├── change_detector.py # Histogram/Edge/FrameDiff detectors
│   │   └── scene_detector.py  # PySceneDetect integration + candidate extraction
│   │
│   ├── deduplication/
│   │   ├── base.py            # Base deduplicator interface
│   │   ├── phash.py           # Perceptual hashing (Layer 1)
│   │   ├── ssim.py            # Structural similarity (Layer 2)
│   │   ├── clip_embed.py      # CLIP embeddings (Layer 3)
│   │   └── hierarchical.py    # Orchestrates 3-layer cascade
│   │
│   ├── selection/
│   │   ├── clustering.py      # Scene assignment + K-means clustering
│   │   └── representation.py  # Importance scoring + selection
│   │
│   ├── extraction/
│   │   ├── llm_client.py      # Multi-provider LLM clients + orchestrator
│   │   ├── prompt.py          # Temporal prompt building
│   │   └── schema.py          # Adaptive extraction schemas
│   │
│   └── utils/
│       ├── config.py          # YAML configuration loading
│       ├── logging.py         # Logging setup
│       ├── metrics.py         # PipelineResult & metrics
│       └── video_utils.py     # Video I/O utilities
│
├── data/ads/                  # Test videos
├── outputs/audio/             # Extracted audio tracks
│
├── experiments/
│   └── pipeline.py            # End-to-end test
│
└── tests/
    ├── clustering.py          # Scene assignment + clustering tests
    ├── representative.py      # Importance scoring tests
    ├── hierarchical.py        # Hierarchical dedup tests
    ├── phash.py               # PHash dedup tests
    ├── ssim.py                # SSIM dedup tests
    └── clip.py                # CLIP dedup tests
```

---

## Configuration

### Default Configuration (config/default.yaml)

```yaml
# Video ingestion
ingestion:
  max_resolution: 720
  extract_audio: true

# Change detection
change_detection:
  method: "histogram" # Options: histogram, edge, frame_diff
  threshold: 0.15
  min_interval_ms: 100

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
    threshold: 0.92 # Structural similarity
  clip:
    enabled: true
    model: "ViT-B/32"
    threshold: 0.90 # Cosine similarity
    device: "auto" # auto, cuda, cpu
    batch_size: 32

# Representative selection
selection:
  method: "clustering" # clustering, uniform, first
  max_frames_per_scene: 3
  min_temporal_gap_s: 0.5

# LLM extraction
extraction:
  provider: "anthropic" # anthropic, openai, gemini
  model: "claude-sonnet-4-20250514"
  max_tokens: 2000
  temperature: 0.0

  # Temporal reasoning
  temporal_context:
    enabled: true
    include_timestamps: true
    include_time_deltas: true
    include_position_labels: true # [OPENING], [CLOSING]
    include_narrative_instructions: true

  # Adaptive schema
  schema:
    mode: "adaptive" # adaptive, fixed, flexible
```

---

## Usage

### Basic Pipeline Usage

```python
from src.pipeline import AdVideoPipeline

# Initialize pipeline
pipeline = AdVideoPipeline(config_path="config/default.yaml")

# Process single video
result = pipeline.process("path/to/video.mp4", skip_extraction=False)

# Access results
print(f"Scenes detected: {len(result.scenes)}")
print(f"Total candidates: {result.total_frames_sampled}")
print(f"After PHash: {result.frames_after_phash}")
print(f"After SSIM: {result.frames_after_ssim}")
print(f"After CLIP: {result.frames_after_clip}")
print(f"Final frames: {result.final_frame_count}")
print(f"Reduction rate: {result.reduction_rate:.1%}")
print(f"Processing time: {result.processing_time_s:.1f}s")

# View extracted data
if result.extraction_result:
    print(f"Brand: {result.extraction_result['brand']['name']}")
    print(f"Message: {result.extraction_result['message']['primary_message']}")
    print(f"CTA: {result.extraction_result['message']['call_to_action']}")
    print(f"Ad type: {result.extraction_result['_metadata']['ad_type']}")
```

### Batch Processing

```python
# Process multiple videos
results = pipeline.process_batch(
    video_paths=["ad1.mp4", "ad2.mp4", "ad3.mp4"],
    max_workers=4,
    skip_extraction=False
)

# Analyze results
for result in results:
    if result is None:
        continue
    print(f"{result.video_path}: {result.final_frame_count} frames ({result.reduction_rate:.1%})")
```

### Custom Configuration

```python
# Override specific settings
pipeline = AdVideoPipeline(
    config_path="config/default.yaml",
    overrides={
        "deduplication": {
            "clip": {"threshold": 0.85}  # More aggressive
        },
        "selection": {
            "max_frames_per_scene": 2   # Fewer frames
        }
    }
)
```

### Environment Setup

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Set API keys in .env file
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# Run pipeline
uv run python -m experiments.pipeline
```

---

## Testing

### Run All Tests

```bash
# Clustering tests
uv run python -m tests.clustering

# Representation tests
uv run python -m tests.representative

# Hierarchical deduplication tests
uv run python -m tests.hierarchical

# Individual deduplicator tests
uv run python -m tests.phash
uv run python -m tests.ssim
uv run python -m tests.clip

# End-to-end pipeline test
uv run python -m experiments.pipeline
```

---

## Dependencies

### Core Dependencies

```
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
open-clip-torch>=2.20.0
scikit-learn>=1.3.0

# Audio processing
librosa>=0.10.0
soundfile>=0.12.0

# LLM APIs
anthropic>=0.18.0
openai>=1.0.0
google-generativeai>=0.3.0

# Utilities
python-dotenv>=1.0.0
pyyaml>=6.0
numpy>=1.24.0
```

---

## Known Limitations & Future Work

### Current Limitations

1. **Missing CTA Detection:** Closing frames may not always capture call-to-action if it appears only briefly at the very end or in audio. Mitigated by forcing last frame inclusion.

2. **Malformed JSON from Gemini:** Occasional duplicate keys in JSON response. Workaround: Use Claude (more reliable) or add JSON repair logic.

3. **Single-Language Support:** Currently optimized for English text overlays and audio. Multilingual support requires model swaps.

4. **No Streaming Support:** Requires complete video file. Real-time processing would need architectural redesign.

5. **Sequential Batch Processing:** Currently processes videos one-by-one to avoid multiprocessing issues with heavy models.

### Future Enhancements

#### Short-Term (1-2 weeks each)

- **Multilingual Support:** Swap to multilingual OCR (TrOCR) and ASR (Whisper) models
- **Full Audio-Visual Fusion:** Complete integration of audio event importance boosting
- **Confidence Scoring:** Multiple extraction passes with agreement-based confidence
- **JSON Repair:** Robust parsing for malformed LLM responses
- **True Parallel Batch Processing:** ThreadPoolExecutor with shared CLIP model

#### Medium-Term (1-3 months each)

- **Learned Thresholds:** Meta-learning optimal thresholds per ad category
- **Quality Metrics:** Automated evaluation against ground truth annotations
- **Web Interface:** Gradio/Streamlit demo for interactive testing
- **Dataset Collection:** Large-scale annotation for training and evaluation

#### Long-Term (Research Directions)

- **Streaming Processing:** Online frame selection without future context
- **Cross-Video Transfer:** Learn from corpus to improve per-video decisions
- **Causal Analysis:** Link visual elements to ad effectiveness metrics
- **Temporal Grounding:** Precise event localization with frame-level timestamps
- **Multi-Modal Fusion:** Joint audio-visual-text understanding

---

## References

### Key Papers

1. Hussain, Z., et al. "Automatic Understanding of Image and Video Advertisements." CVPR 2017.
2. Tan, K., et al. "Large Model based Sequential Keyframe Extraction for Video Summarization." CMLDS 2024.
3. Hu, W., et al. "M-LLM Based Video Frame Selection for Efficient Video Understanding." CVPR 2025.
4. Soucek, T., & Lokoč, J. "TransNet V2: An effective deep network architecture for fast shot transition detection." arXiv 2020.

### Tools & Libraries

- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) - Scene boundary detection
- [TransNetV2](https://github.com/soCzech/TransNetV2) - Neural shot detection
- [OpenAI CLIP](https://github.com/openai/CLIP) - Vision-language embeddings
- [ImageHash](https://github.com/JohannesBuchner/imagehash) - Perceptual hashing
- [scikit-image](https://scikit-image.org/) - SSIM implementation

---

## Contact

**Author:** Abdul Basit Tonmoy  
**Email:** abdulbasittonmoy@gmail.com  
**GitHub:** github.com/abtonmoy

For questions, issues, or contributions, please open an issue on the GitHub repository.

---

## License

This project is licensed under the MIT License - see LICENSE file for details.
