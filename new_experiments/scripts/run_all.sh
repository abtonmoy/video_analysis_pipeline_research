#!/bin/bash
# run_all.sh — Execute all experiments in order
# Run from: new_experiments/
# Usage: bash scripts/run_all.sh

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(dirname "$(pwd)")"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

echo "=== AdaFrame Experiment Suite ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Phase 1: No API calls
echo "====================================="
echo "PHASE 1: Data Analysis (no API calls)"
echo "====================================="

echo ""
echo "--- Experiment 4: Error-Free Intersection ---"
uv run python scripts/exp4_error_intersection.py \
    --input "$PROJECT_ROOT/test_results/benchmark_results.json" \
    --gt "$PROJECT_ROOT/data/" \
    --output results/error_intersection/

echo ""
echo "--- Experiment 5: Cost Breakdown ---"
uv run python scripts/exp5_cost_breakdown.py \
    --input "$PROJECT_ROOT/main_results/processing_results.json" \
    --output results/cost_breakdown/

# Phase 2: Some VLM calls
echo ""
echo "============================="
echo "PHASE 2: Budget Ablation"
echo "============================="

echo ""
echo "--- Experiment 1: Budget Ablation ---"
uv run python scripts/exp1_budget_ablation.py \
    --input "$PROJECT_ROOT/main_results/processing_results.json" \
    --output results/budget_ablation/

# Phase 3: API-heavy
echo ""
echo "=============================="
echo "PHASE 3: API-Heavy Experiments"
echo "=============================="

echo ""
echo "--- Generate shared 50 video IDs ---"
uv run python scripts/generate_shared_50.py

echo ""
echo "--- Experiment 2: PySceneDetect ---"
uv run python scripts/exp2_pyscenedetect.py \
    --benchmark "$PROJECT_ROOT/test_results/benchmark_results.json" \
    --output results/pyscenedetect/

echo ""
echo "--- Experiment 6: Claude (50 videos) ---"
uv run python scripts/exp6_claude.py \
    --video-ids scripts/shared_50_video_ids.json \
    --input "$PROJECT_ROOT/main_results/processing_results.json" \
    --output results/claude_50/

echo ""
echo "--- Experiment 7: Native Video (50 videos) ---"
uv run python scripts/exp7_native_video.py \
    --video-ids scripts/shared_50_video_ids.json \
    --output results/native_video_50/

echo ""
echo "--- Experiment 3: DSNet (optional) ---"
uv run python scripts/exp3_dsnet.py --output results/dsnet/ || true

echo ""
echo "--- Experiment 8: Limitations Update ---"
uv run python scripts/exp8_limitations.py \
    --claude-results results/claude_50/claude_results.json \
    --native-results results/native_video_50/native_video_results.json \
    --pipeline-results "$PROJECT_ROOT/main_results/processing_results.json" \
    --gt "$PROJECT_ROOT/data/annotations_videos/video/cleaned_result/video_Topics_clean.json" \
    --output results/

# Final assembly
echo ""
echo "========================="
echo "ASSEMBLING PAPER UPDATES"
echo "========================="
uv run python scripts/assemble_paper_updates.py

echo ""
echo "=== ALL EXPERIMENTS COMPLETE ==="
echo "Results: new_experiments/results/"
echo "Paper updates: new_experiments/paper_updates_final.tex"
