#!/usr/bin/env python3
"""
Aggregation utilities for unified benchmarks.

Computes summary statistics from per-video results.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def load_results(results_path: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def aggregate_by_method(results: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate results by method.

    Args:
        results: List of per-video results

    Returns:
        Dict with aggregated statistics per method
    """
    by_method = {}

    for result in results:
        method = result.get("method", "unknown")
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(result)

    summary = {
        "n_videos": len(set(r.get("video_name") for r in results)),
        "n_success": len([r for r in results if r.get("status") == "success"]),
        "n_failed": len([r for r in results if r.get("status") == "error"]),
        "n_methods": len(by_method),
        "methods": {},
    }

    for method, method_results in by_method.items():
        summary["methods"][method] = _aggregate_single_method(method_results)

    return summary


def _aggregate_single_method(results: List[Dict]) -> Dict[str, Any]:
    """Aggregate statistics for a single method."""

    # Extract pipeline stats
    frame_counts = []
    costs = []
    latencies = []
    info_densities = []
    compression_ratios = []
    reduction_rates = []

    for r in results:
        stats = r.get("pipeline_stats", {})
        frame_counts.append(stats.get("final_frame_count", 0))
        costs.append(stats.get("cost_usd", 0.0))
        latencies.append(stats.get("selection_latency_s", 0.0))

        if stats.get("info_density", 0.0) > 0:
            info_densities.append(stats.get("info_density", 0.0))
        if stats.get("compression_ratio", 0.0) > 0:
            compression_ratios.append(stats.get("compression_ratio", 0.0))
        if stats.get("reduction_rate", 0.0) > 0:
            reduction_rates.append(stats.get("reduction_rate", 0.0))

    # Compute accuracy metrics
    accuracy = _compute_accuracy(results)

    return {
        "n_videos": len(results),
        "frames": {
            "mean": round(np.mean(frame_counts), 1) if frame_counts else 0.0,
            "median": round(np.median(frame_counts), 1) if frame_counts else 0.0,
            "std": round(np.std(frame_counts), 2) if frame_counts else 0.0,
            "min": int(np.min(frame_counts)) if frame_counts else 0,
            "max": int(np.max(frame_counts)) if frame_counts else 0,
        },
        "cost_usd": {
            "mean": round(np.mean(costs), 4) if costs else 0.0,
            "median": round(np.median(costs), 4) if costs else 0.0,
            "total": round(np.sum(costs), 2) if costs else 0.0,
        },
        "latency_s": {
            "mean": round(np.mean(latencies), 3) if latencies else 0.0,
            "median": round(np.median(latencies), 3) if latencies else 0.0,
        },
        "efficiency": {
            "mean_compression_ratio": round(np.mean(compression_ratios), 1) if compression_ratios else 0.0,
            "mean_info_density": round(np.mean(info_densities), 3) if info_densities else 0.0,
            "mean_reduction_rate": round(np.mean(reduction_rates), 3) if reduction_rates else 0.0,
        },
        "accuracy": accuracy,
    }


def _compute_accuracy(results: List[Dict]) -> Dict[str, float]:
    """Compute accuracy metrics from results."""

    metrics = {
        "topic_accuracy": 0.0,
        "super_category_accuracy": 0.0,
        "brand_detection_rate": 0.0,
        "cta_detection_rate": 0.0,
    }

    topic_matches = 0
    super_matches = 0
    brand_matches = 0
    cta_matches = 0
    total = 0

    for result in results:
        comparison = result.get("comparison", {})
        if not comparison:
            continue

        total += 1
        if comparison.get("topic_match"):
            topic_matches += 1
        if comparison.get("super_category_match"):
            super_matches += 1
        if comparison.get("brand_match"):
            brand_matches += 1
        if comparison.get("cta_match"):
            cta_matches += 1

    if total > 0:
        metrics["topic_accuracy"] = round(100.0 * topic_matches / total, 1)
        metrics["super_category_accuracy"] = round(100.0 * super_matches / total, 1)
        metrics["brand_detection_rate"] = round(100.0 * brand_matches / total, 1)
        metrics["cta_detection_rate"] = round(100.0 * cta_matches / total, 1)

    return metrics


def compare_methods(summary: Dict, reference_method: str = "adaframe") -> Dict[str, Any]:
    """
    Compare all methods against a reference method.

    Args:
        summary: Summary dict with all methods
        reference_method: Name of reference method to compare against

    Returns:
        Dict with pairwise comparisons
    """
    methods = summary.get("methods", {})

    if reference_method not in methods:
        logger.warning(f"Reference method {reference_method} not found")
        return {}

    reference = methods[reference_method]
    comparisons = {}

    for method_name, method_stats in methods.items():
        if method_name == reference_method:
            continue

        comparisons[method_name] = {
            "frame_diff": round(method_stats["frames"]["mean"] - reference["frames"]["mean"], 1),
            "cost_diff_usd": round(method_stats["cost_usd"]["mean"] - reference["cost_usd"]["mean"], 4),
            "topic_acc_diff": round(method_stats["accuracy"]["topic_accuracy"] - reference["accuracy"]["topic_accuracy"], 1),
        }

    return comparisons


def generate_summary_table(summary: Dict) -> str:
    """Generate markdown table from summary."""

    lines = [
        "# Benchmark Summary\n",
        f"**Total Videos:** {summary['n_videos']}\n",
        f"**Successful:** {summary['n_success']}\n",
        f"**Failed:** {summary['n_failed']}\n",
        f"**Methods:** {summary['n_methods']}\n\n",
        "## Results by Method\n",
        "| Method | Frames (mean) | Cost ($) | Topic Acc (%) | Super-Cat Acc (%) |",
        "|--------|---------------|----------|---------------|-------------------|",
    ]

    for method_name, stats in summary.get("methods", {}).items():
        frames = stats["frames"]["mean"]
        cost = stats["cost_usd"]["mean"]
        topic = stats["accuracy"]["topic_accuracy"]
        super_cat = stats["accuracy"]["super_category_accuracy"]

        lines.append(f"| {method_name} | {frames:.1f} | {cost:.4f} | {topic:.1f} | {super_cat:.1f} |")

    return "\n".join(lines)


def save_summary(summary: Dict, output_path: str):
    """Save summary to JSON and markdown."""

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = Path(output_path).with_suffix(".json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary JSON to {json_path}")

    # Markdown table
    md_path = Path(output_path).with_suffix(".md")
    with open(md_path, 'w') as f:
        f.write(generate_summary_table(summary))
    logger.info(f"Saved summary markdown to {md_path}")
