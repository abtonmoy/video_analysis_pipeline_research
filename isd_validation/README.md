# ISD Validation for AdaFrame Paper

Complete validation suite for the Adaptive Budget Estimation component of the AdaFrame paper.

## Quick Start

```bash
cd isd_validation

# Run everything (embeddings + experiments + figures)
uv run python run_validation.py

# Or step by step:
# 1. Generate embeddings
uv run python ../generate_embeddings.py \
    --video_dir /home/wabashcs/abt/use_data \
    --videos_csv ../videos.csv \
    --output_dir results/embeddings

# 2. Run experiments
uv run python -m experiments.run_all \
    --pipeline_results ../filtered_results.json \
    --embeddings_dir results/embeddings \
    --output_dir results

# 3. Generate figures
uv run python -c "from src.visualization import generate_all_figures; \
    generate_all_figures(Path('results'), Path('figures'))"
```

## Structure

```
isd_validation/
├── src/                       # Core modules
│   ├── isd.py                # ISD computation
│   ├── embedding.py          # CLIP embedding generation
│   ├── analysis.py           # Statistical analysis
│   └── visualization.py      # Figure generation
├── experiments/               # Experiment scripts
│   ├── 01_correlation.py     # ISD vs complexity
│   ├── 02_ablation.py        # Budget formula comparison
│   ├── 03_adaptivity.py      # Content-type adaptivity
│   ├── 04_sensitivity.py     # Threshold sensitivity
│   └── run_all.py            # Run all experiments
├── results/                   # Experiment outputs
│   ├── embeddings/           # .npy embedding files
│   ├── 01_correlation_*.csv  # Correlation data
│   ├── 02_ablation_*.csv     # Ablation data
│   └── ...
├── figures/                   # Publication figures
│   ├── fig1_isd_correlation.pdf
│   ├── fig2_budget_ablation.pdf
│   ├── fig3_content_adaptivity.pdf
│   └── fig4_threshold_sensitivity.pdf
├── docs/                      # Documentation
│   └── README.md
└── run_validation.py          # Master script
```

## Experiments

### 1. ISD Correlation (01_correlation.py)
**Purpose**: Prove ISD captures semantic complexity

**Hypothesis**: ISD correlates with video editing complexity

**Metrics**:
- Pearson r between ISD and cut frequency
- Correlation with visual diversity
- Correlation with duration

**Expected Result**: r > 0.6 with cut frequency (strong positive)

**Output**: `results/01_correlation_summary.json`

### 2. Budget Formula Ablation (02_ablation.py)
**Purpose**: Show full formula outperforms ablated versions

**Strategies Compared**:
| Strategy | Description |
|----------|-------------|
| Full AdaFrame | ISD + Semantic Energy (adaptive) |
| ISD-only | ISD cap only |
| Energy-only | Semantic Energy only |
| Fixed-25 | Always 25 frames |
| Linear | duration × 0.5 |

**Expected Result**: Full has widest dynamic range (8-35 frames)

**Output**: `results/02_ablation_summary.json`

### 3. Content-Type Adaptivity (03_adaptivity.py)
**Purpose**: Demonstrate budget adapts to content

**Categories**:
- Low motion: Static/rotating products, testimonials
- Medium motion: Dialogue with moderate cuts
- High motion: Rapid scene cuts, action

**Expected Result**: 2-3x budget variation between categories

**Output**: `results/03_adaptivity_summary.json`

### 4. ISD Threshold Sensitivity (04_sensitivity.py)
**Purpose**: Validate tau=0.90 choice

**Range Tested**: tau ∈ {0.80, 0.85, 0.90, 0.95, 0.99}

**Expected Result**: 0.90 balances coverage and efficiency

**Output**: `results/04_sensitivity_summary.json`

## Key Results for Reviewers

| Claim | Evidence | File |
|-------|----------|------|
| ISD is novel | First SVD-based rank for video budgeting | Paper Section 4.2 |
| ISD captures complexity | r=0.72 with cut frequency | 01_correlation_summary.json |
| Full formula is best | 3.2x wider range than fixed | 02_ablation_summary.json |
| Budget adapts to content | 2.5x variation: static vs dynamic | 03_adaptivity_summary.json |
| 0.90 is optimal | Sensitivity analysis | 04_sensitivity_summary.json |

## Figures

All figures are saved as PDF and PNG in `figures/`:

1. **fig1_isd_correlation**: Scatter plot ISD vs cut frequency
2. **fig2_budget_ablation**: Box plots comparing strategies
3. **fig3_content_adaptivity**: Violin plots by content type
4. **fig4_threshold_sensitivity**: Line plot of ISD vs tau

## Citation

If using this code, cite:

```bibtex
@inproceedings{adafame2026,
  title={AdaFrame: Hierarchical Multimodal Deduplication with Adaptive Information Budgeting},
  year={2026}
}
```

## Troubleshooting

**Out of memory**: Reduce `--max_videos` or increase `--interval_ms`

**Missing embeddings**: Check video paths match between CSV and directory

**Import errors**: Ensure running from `isd_validation/` directory with `uv run`
