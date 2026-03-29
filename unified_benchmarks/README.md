# Unified Benchmarks for AdaFrame

A standardized benchmarking suite for comparing frame selection methods in video advertisement analysis. Outputs results in the same format as `main_results` and `new_experiments`.

## Overview

This benchmark suite compares 7 baseline frame selection methods against the AdaFrame pipeline:

1. **Uniform 1 FPS** - Industry standard uniform sampling
2. **Random** - Random frame selection
3. **Histogram** - HSV histogram change detection
4. **ORB** - ORB feature matching
5. **Optical Flow** - Motion peak detection
6. **CLIP Only** - CLIP embedding sequential dedup
7. **K-Means** - K-means clustering on CLIP embeddings

## Features

- ✅ **Same data structure** as `main_results` and `new_experiments`
- ✅ **Comprehensive metrics**: frames, cost, accuracy, efficiency
- ✅ **Error handling**: Retry queue with automatic retries
- ✅ **Figure generation**: Automatic visualization
- ✅ **Well documented**: Full API documentation

## Installation

```bash
# From project root
cd unified_benchmarks

# Install dependencies (if not already installed)
pip install -r ../requirements.txt
pip install matplotlib seaborn scikit-learn
```

## Quick Start

### 1. Run Benchmark

```bash
python scripts/run_benchmark.py \
    --video_dir /path/to/videos \
    --pipeline_results /path/to/filtered_results.json \
    --output_dir ./results
```

### 2. Generate Figures Only

```bash
python scripts/run_benchmark.py \
    --generate_figures \
    --output_dir ./results
```

### 3. Run Specific Methods

```bash
python scripts/run_benchmark.py \
    --video_dir /path/to/videos \
    --pipeline_results /path/to/results.json \
    --methods uniform_1fps random histogram \
    --output_dir ./results
```

### 4. Frame Selection Only (No VLM Calls)

```bash
python scripts/run_benchmark.py \
    --video_dir /path/to/videos \
    --pipeline_results /path/to/results.json \
    --selection_only \
    --output_dir ./results
```

## Output Structure

Results are saved in **per-method folders** plus a combined folder:

```
results/
├── uniform_1fps/              # Per-method folder
│   ├── results.json          # Results for this method only
│   ├── summary.json          # Aggregated stats for this method
│   └── results.csv           # CSV for this method
├── random/
│   ├── results.json
│   ├── summary.json
│   └── results.csv
├── histogram/
│   └── ...
├── orb/
├── optical_flow/
├── clip_only/
├── kmeans/
├── combined/                  # All methods together
│   ├── all_results.json      # All videos × all methods
│   ├── all_summary.json      # Aggregated stats for all methods
│   └── all_results.csv
└── figures/                   # All figures
    ├── method_comparison.png
    ├── accuracy_vs_cost.png
    ├── frame_distribution.png
    ├── compression_ratio.png
    └── info_density.png
```

## Data Format

### Per-Video Results (results.json)

```json
{
  "metadata": {
    "timestamp": "2026-03-25T...",
    "n_videos": 500,
    "n_methods": 7
  },
  "results": [
    {
      "status": "success",
      "video_path": "...",
      "video_name": "...",
      "processed_at": "...",
      "method": "uniform_1fps",
      "metadata": {
        "duration": 20.02,
        "fps": 29.97,
        "total_frames": 600
      },
      "pipeline_stats": {
        "final_frame_count": 20,
        "total_frames_sampled": 600,
        "reduction_rate": 0.967,
        "compression_ratio": 30.0,
        "selection_latency_s": 0.02,
        "info_density": 0.31,
        "cost_usd": 0.30
      },
      "extraction": {...},
      "comparison": {
        "topic_match": true,
        "brand_match": true,
        "cta_match": false
      }
    }
  ]
}
```

### Summary (summary.json)

```json
{
  "n_videos": 500,
  "n_success": 484,
  "n_failed": 16,
  "n_methods": 7,
  "methods": {
    "uniform_1fps": {
      "n_videos": 484,
      "frames": {
        "mean": 43.9,
        "median": 42.0,
        "std": 12.3,
        "min": 5,
        "max": 105
      },
      "cost_usd": {
        "mean": 0.504,
        "median": 0.483,
        "total": 243.94
      },
      "accuracy": {
        "topic_accuracy": 84.3,
        "super_category_accuracy": 89.9,
        "brand_detection_rate": 92.1,
        "cta_detection_rate": 81.8
      },
      "efficiency": {
        "mean_compression_ratio": 24.5,
        "mean_info_density": 0.267
      }
    }
  }
}
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# VLM settings
provider: "gemini"
model: "gemini-2.0-flash-exp"
max_tokens: 4000

# Method thresholds
thresholds:
  histogram_correlation: 0.95
  orb_good_matches: 40
  optical_flow_percentile: 85
  clip_cosine: 0.92

# Retry settings
retry:
  max_immediate_retries: 2
  retry_delay: 5.0
```

## Error Handling

The benchmark uses a retry queue system:

1. **Immediate retries**: 2 attempts with exponential backoff
2. **Retry queue**: Failed videos saved to `retry_queue/{video_id}.json`
3. **Failure log**: All failures logged to `failed.log` (JSONL format)

To retry failed videos:

```python
from src.retry_utils import process_retry_queue

process_retry_queue(
    retry_queue_dir="./retry_queue",
    process_fn=your_processing_function,
    max_attempts=3
)
```

## Metrics

### Frame Selection Metrics

- `final_frame_count` - Number of frames selected
- `total_frames_sampled` - Total frames in video
- `reduction_rate` - Percentage of frames eliminated
- `compression_ratio` - Total frames / selected frames
- `selection_latency_s` - Time to select frames
- `info_density` - Mean pairwise CLIP distance (diversity)

### Cost Metrics

- `cost_usd` - VLM cost per video (frames × $0.015)

### Accuracy Metrics

- `topic_accuracy` - % correct topic classification
- `super_category_accuracy` - % correct super-category
- `brand_detection_rate` - % brand correctly identified
- `cta_detection_rate` - % CTA correctly detected

## API Reference

### Methods

```python
from src.methods import get_method

# Get method instance
method = get_method("uniform_1fps")

# Select frames
frames, latency = method.run_timed(video_path)
```

### Runner

```python
from src.runner import BenchmarkRunner

runner = BenchmarkRunner(
    config=config,
    pipeline_results_path="path/to/results.json",
    output_dir="./results",
    methods=["uniform_1fps", "random"],
)

summary = runner.run(video_paths)
```

### Metrics

```python
from src.metrics import compute_selection_metrics, compare_extractions

# Compute selection metrics
metrics = compute_selection_metrics(
    selected_frames, total_frames, latency, clip_embeddings
)

# Compare against reference
comparison = compare_extractions(result, reference)
```

### Visualization

```python
from src.visualization import generate_all_figures, generate_all_figures_from_combined

# Generate from combined results (recommended)
generate_all_figures_from_combined("./results")

# Or generate from specific paths
generate_all_figures(
    summary_path="./results/combined/all_summary.json",
    results_path="./results/combined/all_results.json",
    output_dir="./results/figures"
)
```

## Troubleshooting

### No videos found

Check video directory path and file extensions:
```bash
ls /path/to/videos/*.mp4
```

### CLIP not available

GPU methods (CLIP Only, K-Means) require CLIP. Use `--skip_gpu` to skip:
```bash
python scripts/run_benchmark.py ... --skip_gpu
```

### Out of memory

Reduce batch size in config:
```yaml
clip:
  batch_size: 16  # Reduce from 32
```

### API rate limits

Increase throttle time:
```yaml
throttle_seconds: 8.0  # Increase from 4.0
```

## Analyzing Results

### View Single Method Results

```bash
# View summary for uniform_1fps
cat ./results/uniform_1fps/summary.json | python3 -m json.tool

# View results for random method
cat ./results/random/results.json | python3 -m json.tool

# View CSV for histogram method
head ./results/histogram/results.csv
```

### View Combined Results

```bash
# View all methods summary
cat ./results/combined/all_summary.json | python3 -m json.tool

# View all results
cat ./results/combined/all_results.json | python3 -m json.tool
```

### Compare Methods

```python
import json

# Load combined summary
with open('./results/combined/all_summary.json') as f:
    summary = json.load(f)

# Compare frame counts
for method, stats in summary['methods'].items():
    frames = stats['frames']['mean']
    cost = stats['cost_usd']['mean']
    acc = stats['accuracy']['topic_accuracy']
    print(f"{method:20s}: {frames:5.1f} frames, ${cost:.4f}, {acc:.1f}% acc")
```

## Development

### Adding a New Method

1. Create class in `ub_src/methods.py`:

```python
class MyMethod(BaseMethod):
    def __init__(self):
        super().__init__(name="my_method", requires_gpu=False)

    def select_frames(self, video_path: str, **kwargs):
        # Your implementation
        return [(timestamp, frame), ...]
```

2. Register in `ALL_METHODS`:

```python
ALL_METHODS = {
    ...
    "my_method": MyMethod,
}
```

3. Run benchmark:

```bash
python scripts/run_benchmark.py ... --methods my_method
```

## License

Same as parent project.

## Citation

If you use this benchmark suite, please cite:

```bibtex
@inproceedings{adaframe2026,
  title={AdaFrame: Hierarchical Multimodal Deduplication with Adaptive Information Budgeting},
  author={...},
  year={2026}
}
```
