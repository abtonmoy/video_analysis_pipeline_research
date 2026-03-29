#!/usr/bin/env python3
"""
Test script to verify unified_benchmarks structure.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from src import retry_utils
        print("✓ retry_utils")
    except Exception as e:
        print(f"✗ retry_utils: {e}")
        return False

    try:
        from src import methods
        print("✓ methods")
    except Exception as e:
        print(f"✗ methods: {e}")
        return False

    try:
        from src import metrics
        print("✓ metrics")
    except Exception as e:
        print(f"✗ metrics: {e}")
        return False

    try:
        from src import extraction
        print("✓ extraction")
    except Exception as e:
        print(f"✗ extraction: {e}")
        return False

    try:
        from src import runner
        print("✓ runner")
    except Exception as e:
        print(f"✗ runner: {e}")
        return False

    try:
        from src import aggregation
        print("✓ aggregation")
    except Exception as e:
        print(f"✗ aggregation: {e}")
        return False

    try:
        from src import visualization
        print("✓ visualization")
    except Exception as e:
        print(f"✗ visualization: {e}")
        return False

    return True


def test_methods():
    """Test that all methods are registered."""
    print("\nTesting methods...")

    from ub_src.methods import ALL_METHODS

    expected_methods = [
        "uniform_1fps",
        "random",
        "histogram",
        "orb",
        "optical_flow",
        "clip_only",
        "kmeans",
    ]

    for method in expected_methods:
        if method in ALL_METHODS:
            print(f"✓ {method}")
        else:
            print(f"✗ {method} not found")
            return False

    return True


def test_data_structure():
    """Test that output structure matches main_results."""
    print("\nTesting data structure...")

    # Expected keys in per-video result
    expected_keys = [
        "status",
        "video_path",
        "video_name",
        "processed_at",
        "method",
        "metadata",
        "pipeline_stats",
    ]

    print("Expected keys in per-video result:")
    for key in expected_keys:
        print(f"  ✓ {key}")

    # Expected keys in pipeline_stats
    expected_stats = [
        "final_frame_count",
        "total_frames_sampled",
        "reduction_rate",
        "compression_ratio",
        "selection_latency_s",
        "info_density",
        "cost_usd",
    ]

    print("\nExpected keys in pipeline_stats:")
    for key in expected_stats:
        print(f"  ✓ {key}")

    return True


def test_directory_structure():
    """Test that directory structure is correct."""
    print("\nTesting directory structure...")

    base_dir = Path(__file__).parent.parent

    required_dirs = [
        "src",
        "scripts",
        "results",
        "results/per_video",
        "results/figures",
        "retry_queue",
    ]

    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} missing")
            return False

    required_files = [
        "config.yaml",
        "README.md",
        "src/__init__.py",
        "src/retry_utils.py",
        "src/methods.py",
        "src/metrics.py",
        "src/extraction.py",
        "src/runner.py",
        "src/aggregation.py",
        "src/visualization.py",
        "scripts/run_benchmark.py",
    ]

    print("\nRequired files:")
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} missing")
            return False

    return True


def main():
    print("="*60)
    print("Unified Benchmarks Structure Test")
    print("="*60)

    all_passed = True

    if not test_directory_structure():
        all_passed = False

    if not test_imports():
        all_passed = False

    if not test_methods():
        all_passed = False

    if not test_data_structure():
        all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
