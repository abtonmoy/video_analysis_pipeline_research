#!/bin/bash
# run_tests.sh — Execute all experiments in TEST mode with 5 videos
# Run from: new_experiments/

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(dirname "$(pwd)")"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Select 5 consistent test videos
TEST_VIDEOS_DIR="/home/wabashcs/abt/use_data"
TEST_COUNT=5

echo "=== AdaFrame Experiment Suite - TEST MODE ==="
echo "Using $TEST_COUNT videos from $TEST_VIDEOS_DIR"

# Phase 1: No API calls
echo ""
echo "--- Experiment 4: Error-Free Intersection (TEST) ---"
uv run python scripts/exp4_error_intersection.py \
    --test --n $TEST_COUNT \
    --input "$PROJECT_ROOT/test_results/benchmark_results.json" \
    --gt "$PROJECT_ROOT/data/" \
    --output results/test/error_intersection/

echo ""
echo "--- Experiment 5: Cost Breakdown (TEST) ---"
uv run python scripts/exp5_cost_breakdown.py \
    --test --n $TEST_COUNT \
    --input "$PROJECT_ROOT/main_results/processing_results.json" \
    --output results/test/cost_breakdown/

# Phase 2: Some VLM calls
echo ""
echo "--- Experiment 1: Budget Ablation (TEST) ---"
uv run python scripts/exp1_budget_ablation.py \
    --test \
    --input "$PROJECT_ROOT/main_results/processing_results.json" \
    --output results/test/budget_ablation/ \
    --videos "$TEST_VIDEOS_DIR"

# Phase 3: API-heavy
echo ""
echo "--- Generate shared test video IDs (TEST) ---"
# Create a dummy shared list for the tests
mkdir -p results/test/
ls "$TEST_VIDEOS_DIR" | grep '\.mp4$' | head -n $TEST_COUNT > results/test/temp_vids.txt
uv run python -c "import json; vids = [line.strip() for line in open('results/test/temp_vids.txt')]; json.dump(vids, open('results/test/test_5_video_ids.json', 'w'))"
rm results/test/temp_vids.txt

echo ""
echo "--- Experiment 2: PySceneDetect (TEST) ---"
uv run python scripts/exp2_pyscenedetect.py \
    --test --n $TEST_COUNT \
    --videos "$TEST_VIDEOS_DIR" \
    --output results/test/pyscenedetect/

echo ""
echo "--- Experiment 6: Claude (TEST) ---"
uv run python scripts/exp6_claude.py \
    --test --n $TEST_COUNT \
    --videos "$TEST_VIDEOS_DIR" \
    --input "$PROJECT_ROOT/main_results/processing_results.json" \
    --output results/test/claude/ \
    --video-ids results/test/test_5_video_ids.json

echo ""
echo "--- Experiment 7: Native Video (TEST) ---"
uv run python scripts/exp7_native_video.py \
    --test --n $TEST_COUNT \
    --videos "$TEST_VIDEOS_DIR" \
    --output results/test/native_video/ \
    --video-ids results/test/test_5_video_ids.json

echo ""
echo "--- Experiment 3: DSNet (TEST) ---"
uv run python scripts/exp3_dsnet.py \
    --test \
    --videos "$TEST_VIDEOS_DIR" \
    --output results/test/dsnet/ || true

echo ""
echo "--- Experiment 8: Limitations Update (TEST) ---"
uv run python scripts/exp8_limitations.py \
    --claude-results results/test/claude/claude_results.json \
    --native-results results/test/native_video/native_video_results.json \
    --pipeline-results "$PROJECT_ROOT/main_results/processing_results.json" \
    --gt "$PROJECT_ROOT/data/annotations_videos/video/cleaned_result/video_Topics_clean.json" \
    --output results/test/

echo ""
echo "=== TEST RUN COMPLETE ==="
echo "Check new_experiments/results/test/ for outputs."
