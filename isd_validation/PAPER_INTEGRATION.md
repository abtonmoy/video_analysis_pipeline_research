# ISD Validation - Paper Integration Summary

## Changes Made to Paper

### 1. New Section Added: "Adaptive Budget Estimation Validation"
**Location**: Section 7 (Results), after "Content-Type Analysis" and before "Failure Case Analysis"

**Content**: Four subsections with figures validating the ISD component:

#### a) ISD Correlates with Semantic Complexity (Figure 1)
- **Key Result**: ISD shows strong correlation with number of scenes (r=0.784, p<0.001)
- **Figure**: `fig1_isd_correlation.pdf` - Scatter plot with regression line
- **Caption**: Explains that ISD captures semantic complexity

#### b) Budget Formula Ablation (Figure 2)
- **Key Result**: Full AdaFrame has 100-frame dynamic range vs Fixed-25 has 0
- **Figure**: `fig2_budget_ablation.pdf` - Box plots comparing 5 strategies
- **Caption**: Shows full formulation provides optimal adaptivity

#### c) Content-Type Adaptivity (Figure 3)
- **Key Result**: 1.55x adaptivity ratio (ISD=17.7 low-motion vs ISD=27.3 high-motion)
- **Figure**: `fig3_content_adaptivity.pdf` - Violin plots by motion category
- **Caption**: Demonstrates automatic adjustment to content

#### d) ISD Threshold Sensitivity (Figure 4)
- **Key Result**: τ=0.90 is optimal (23.4 frames), balancing coverage vs efficiency
- **Figure**: `fig4_threshold_sensitivity.pdf` - Line plot across τ values
- **Caption**: Validates threshold choice

### 2. Updated Table: Budget Formula Ablation
**Location**: Section 7, Ablation subsection

**Changes**:
- Updated with actual validation results (N=500)
- Added columns: Mean, Std, Range, Max
- New values:
  - Full AdaFrame: 25.8 ± 16.9, range 100
  - ISD Only: 56.3 ± 30.1, range 173
  - Fixed-25: 25.0 ± 0.0, range 0
  - Linear Duration: 24.7 ± 13.4, range 55

### 3. Updated Text
**Location**: Paragraph before Table 2

**Changes**:
- Now references actual dynamic range (100 frames)
- Explains Fixed-25 has zero variance
- Mentions ISD-only over-budgets
- Adds reference to new Section \ref{sec:isd_validation}

## Files Added to Paper Directory

### Figures (symlink or copy from isd_validation/figures/)
```
paper/
└── isd_validation_figures/
    ├── fig1_isd_correlation.pdf
    ├── fig2_budget_ablation.pdf
    ├── fig3_content_adaptivity.pdf
    └── fig4_threshold_sensitivity.pdf
```

### LaTeX Integration
The figures are referenced using relative paths:
```latex
\includegraphics[width=\columnwidth]{../../isd_validation/figures/fig1_isd_correlation.pdf}
```

## Key Claims Supported

| Claim | Evidence | Location |
|-------|----------|----------|
| ISD captures complexity | r=0.784 with num scenes | Fig 1, Table |
| Full formula is best | 100-frame range vs 0 (Fixed) | Fig 2, Table |
| Budget adapts to content | 1.55x ratio low/high motion | Fig 3 |
| τ=0.90 is optimal | 23.4 frames vs 13-64 alternatives | Fig 4 |

## Reviewer Response Points

### For "Isn't this just K-means?"
**Response**: No, ISD uses SVD-based rank estimation, not clustering. Figure 1 shows ISD correlates with complexity (r=0.784), and Figure 2 shows full formula has wider range than K-means alone.

### For "Does it actually adapt?"
**Response**: Yes, Figure 3 shows 1.55x variation between content types, and Table shows 100-frame dynamic range vs Fixed-25's 0 range.

### For "Why not just use fixed budgets?"
**Response**: Fixed-25 has zero adaptivity (Table). Figure 3 shows content varies 1.55x, so fixed budgets under-allocate for complex videos and over-allocate for simple ones.

### For "How do you know 90% variance is right?"
**Response**: Figure 4 shows sensitivity analysis. τ=0.90 yields 23.4 frames, while τ=0.80 is too low (13.2) and τ=0.99 over-budgets (63.8).

## Compilation Notes

To compile the paper with new figures:
```bash
cd paper
# Ensure figures are accessible
ln -s ../isd_validation/figures isd_validation_figures

# Compile
pdflatex mm2026.tex
bibtex mm2026
pdflatex mm2026.tex
pdflatex mm2026.tex
```

## Validation Data Available

All raw data available in:
- `isd_validation/results/01_correlation_summary.json`
- `isd_validation/results/02_ablation_summary.json`
- `isd_validation/results/03_adaptivity_summary.json`
- `isd_validation/results/04_sensitivity_summary.json`

CSV files also available for detailed analysis.
