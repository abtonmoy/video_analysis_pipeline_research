#!/usr/bin/env python3
"""
Visualization utilities for unified benchmarks.

Generates figures comparing baseline methods.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def load_summary(summary_path: str) -> Dict[str, Any]:
    """Load summary JSON."""
    with open(summary_path) as f:
        return json.load(f)


def plot_method_comparison(summary: Dict, output_dir: str):
    """
    Create bar chart comparing methods on key metrics.

    Args:
        summary: Summary dict with method statistics
        output_dir: Directory to save figures
    """
    methods = list(summary.get("methods", {}).keys())
    if not methods:
        logger.warning("No methods to plot")
        return

    # Extract data
    frame_means = [summary["methods"][m]["frames"]["mean"] for m in methods]
    costs = [summary["methods"][m]["cost_usd"]["mean"] for m in methods]
    topic_accs = [summary["methods"][m]["accuracy"]["topic_accuracy"] for m in methods]

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Frames
    axes[0].bar(methods, frame_means, color='steelblue')
    axes[0].set_ylabel('Mean Frames')
    axes[0].set_title('Frames per Video')
    axes[0].tick_params(axis='x', rotation=45)

    # Cost
    axes[1].bar(methods, costs, color='coral')
    axes[1].set_ylabel('Cost (USD)')
    axes[1].set_title('VLM Cost per Video')
    axes[1].tick_params(axis='x', rotation=45)

    # Accuracy
    axes[2].bar(methods, topic_accs, color='seagreen')
    axes[2].set_ylabel('Topic Accuracy (%)')
    axes[2].set_title('Topic Classification Accuracy')
    axes[2].set_ylim([0, 100])
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    output_path = Path(output_dir) / "method_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved method comparison to {output_path}")


def plot_accuracy_vs_cost(summary: Dict, output_dir: str):
    """
    Create scatter plot of accuracy vs cost (efficiency frontier).

    Args:
        summary: Summary dict
        output_dir: Directory to save figures
    """
    methods = list(summary.get("methods", {}).keys())
    if not methods:
        return

    costs = [summary["methods"][m]["cost_usd"]["mean"] for m in methods]
    topic_accs = [summary["methods"][m]["accuracy"]["topic_accuracy"] for m in methods]

    plt.figure(figsize=(10, 6))

    # Scatter plot
    plt.scatter(costs, topic_accs, s=200, alpha=0.6, c='steelblue')

    # Annotate points
    for i, method in enumerate(methods):
        plt.annotate(method, (costs[i], topic_accs[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9)

    plt.xlabel('Cost (USD per video)')
    plt.ylabel('Topic Accuracy (%)')
    plt.title('Accuracy vs Cost Trade-off')
    plt.grid(True, alpha=0.3)

    output_path = Path(output_dir) / "accuracy_vs_cost.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved accuracy vs cost plot to {output_path}")


def plot_frame_distribution(results: List[Dict], output_dir: str):
    """
    Create box plot of frame count distribution per method.

    Args:
        results: List of per-video results
        output_dir: Directory to save figures
    """
    # Group by method
    by_method = {}
    for r in results:
        method = r.get("method", "unknown")
        if method not in by_method:
            by_method[method] = []
        stats = r.get("pipeline_stats", {})
        by_method[method].append(stats.get("final_frame_count", 0))

    if not by_method:
        return

    methods = list(by_method.keys())
    data = [by_method[m] for m in methods]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=methods)
    plt.ylabel('Frame Count')
    plt.title('Frame Count Distribution by Method')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')

    output_path = Path(output_dir) / "frame_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved frame distribution to {output_path}")


def plot_compression_ratio(summary: Dict, output_dir: str):
    """
    Create bar chart of compression ratios.

    Args:
        summary: Summary dict
        output_dir: Directory to save figures
    """
    methods = list(summary.get("methods", {}).keys())
    if not methods:
        return

    ratios = [summary["methods"][m]["efficiency"]["mean_compression_ratio"] for m in methods]

    plt.figure(figsize=(10, 6))
    plt.bar(methods, ratios, color='teal')
    plt.ylabel('Compression Ratio')
    plt.title('Mean Compression Ratio by Method')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')

    output_path = Path(output_dir) / "compression_ratio.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved compression ratio plot to {output_path}")


def plot_info_density(summary: Dict, output_dir: str):
    """
    Create bar chart of information density.

    Args:
        summary: Summary dict
        output_dir: Directory to save figures
    """
    methods = list(summary.get("methods", {}).keys())
    if not methods:
        return

    densities = [summary["methods"][m]["efficiency"]["mean_info_density"] for m in methods]

    plt.figure(figsize=(10, 6))
    plt.bar(methods, densities, color='purple')
    plt.ylabel('Information Density')
    plt.title('Mean Information Density by Method')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')

    output_path = Path(output_dir) / "info_density.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved info density plot to {output_path}")


def generate_all_figures(summary_path: str, results_path: str, output_dir: str):
    """
    Generate all figures from summary and results.

    Args:
        summary_path: Path to summary.json (can be in combined/ or any method folder)
        results_path: Path to results.json
        output_dir: Directory to save figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    summary = load_summary(summary_path)

    with open(results_path) as f:
        results_data = json.load(f)
    results = results_data.get("results", [])

    # Generate figures
    logger.info("Generating figures...")

    plot_method_comparison(summary, output_dir)
    plot_accuracy_vs_cost(summary, output_dir)
    plot_frame_distribution(results, output_dir)
    plot_compression_ratio(summary, output_dir)
    plot_info_density(summary, output_dir)

    logger.info(f"All figures saved to {output_dir}")


def generate_all_figures_from_combined(output_dir: str):
    """
    Generate all figures from combined results (convenience function).

    Args:
        output_dir: Directory containing combined/ subfolder
    """
    output_dir = Path(output_dir)
    combined_dir = output_dir / "combined"

    summary_path = combined_dir / "all_summary.json"
    results_path = combined_dir / "all_results.json"

    if not summary_path.exists() or not results_path.exists():
        logger.error(f"Combined results not found in {combined_dir}")
        return

    figures_dir = output_dir / "figures"
    generate_all_figures(str(summary_path), str(results_path), str(figures_dir))
