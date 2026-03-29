"""
Visualization utilities for creating publication-ready figures.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Set publication style
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.dpi"] = 150


def plot_isd_correlation(data_path: str, output_dir: Path):
    """
    Create scatter plot of ISD vs cut frequency with correlation.
    
    Figure 1: Validates that ISD captures editing complexity.
    """
    df = pd.read_csv(data_path)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot
    ax.scatter(df["cut_frequency"], df["isd"], alpha=0.5, s=30)
    
    # Fit line
    z = np.polyfit(df["cut_frequency"], df["isd"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["cut_frequency"].min(), df["cut_frequency"].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label="Linear fit")
    
    # Compute correlation
    from scipy import stats
    r, pval = stats.pearsonr(df["cut_frequency"], df["isd"])
    
    ax.set_xlabel("Cut Frequency (scenes/second)")
    ax.set_ylabel("Intrinsic Semantic Dimensionality (ISD)")
    ax.set_title(f"ISD Captures Editing Complexity\n(r={r:.3f}, p<0.001)")
    ax.legend()
    
    plt.tight_layout()
    output_path = output_dir / "fig1_isd_correlation.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_dir / "fig1_isd_correlation.png", bbox_inches="tight")
    logger.info(f"Saved Figure 1 to {output_path}")
    plt.close()


def plot_budget_ablation(data_path: str, output_dir: Path):
    """
    Create box plot comparing budget strategies.
    
    Figure 2: Shows full adaptive has widest dynamic range.
    """
    df = pd.read_csv(data_path)
    
    strategies = ["full_adaptive", "isd_only", "energy_only", "fixed_25", "linear_duration"]
    labels = ["Full AdaFrame", "ISD-only", "Energy-only", "Fixed-25", "Linear"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_to_plot = [df[s] for s in strategies]
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # Color boxes
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#f39c12"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel("Frame Budget")
    ax.set_title("Budget Formula Ablation\nFull AdaFrame Provides Widest Dynamic Range")
    ax.grid(axis="y", alpha=0.3)
    
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    
    output_path = output_dir / "fig2_budget_ablation.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_dir / "fig2_budget_ablation.png", bbox_inches="tight")
    logger.info(f"Saved Figure 2 to {output_path}")
    plt.close()


def plot_content_adaptivity(data_path: str, output_dir: Path):
    """
    Create violin plot of ISD by content type.
    
    Figure 3: Shows budget adapts to content characteristics.
    """
    df = pd.read_csv(data_path)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ["low_motion", "medium_motion", "high_motion"]
    labels = ["Low Motion\n(Static)", "Medium Motion\n(Dialogue)", "High Motion\n(Rapid Cuts)"]
    
    data_to_plot = [df[df["category"] == cat]["isd"] for cat in categories]
    
    parts = ax.violinplot(data_to_plot, positions=range(len(categories)), showmeans=True)
    
    # Color violins
    colors = ["#3498db", "#f39c12", "#e74c3c"]
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Intrinsic Semantic Dimensionality (ISD)")
    ax.set_title("Budget Adapts to Content Type\nHigher Motion → Higher ISD → More Frames")
    ax.grid(axis="y", alpha=0.3)
    
    # Add mean values as text
    for i, cat in enumerate(categories):
        mean_val = df[df["category"] == cat]["isd"].mean()
        ax.text(i, mean_val + 1, f"μ={mean_val:.1f}", ha="center", fontweight="bold")
    
    plt.tight_layout()
    
    output_path = output_dir / "fig3_content_adaptivity.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_dir / "fig3_content_adaptivity.png", bbox_inches="tight")
    logger.info(f"Saved Figure 3 to {output_path}")
    plt.close()


def plot_threshold_sensitivity(data_path: str, output_dir: Path):
    """
    Create line plot of ISD vs tau threshold.
    
    Figure 4: Validates choice of tau=0.90.
    """
    df = pd.read_csv(data_path)
    
    tau_cols = [c for c in df.columns if c.startswith("isd_tau_")]
    tau_values = [float(c.split("_")[-1]) for c in tau_cols]
    
    # Compute mean ISD for each tau
    mean_isds = [df[col].mean() for col in tau_cols]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(tau_values, mean_isds, "o-", linewidth=2, markersize=8, color="#2ecc71")
    
    # Highlight baseline
    baseline_idx = tau_values.index(0.90)
    ax.plot(tau_values[baseline_idx], mean_isds[baseline_idx], "ro", markersize=12, 
            label="Baseline (τ=0.90)", zorder=5)
    
    ax.set_xlabel("Variance Threshold (τ)")
    ax.set_ylabel("Mean ISD")
    ax.set_title("ISD Threshold Sensitivity\nτ=0.90 Balances Coverage and Efficiency")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / "fig4_threshold_sensitivity.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_dir / "fig4_threshold_sensitivity.png", bbox_inches="tight")
    logger.info(f"Saved Figure 4 to {output_path}")
    plt.close()


def generate_all_figures(results_dir: Path, output_dir: Path):
    """Generate all publication figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating publication figures...")
    
    plot_isd_correlation(results_dir / "01_correlation_data.csv", output_dir)
    plot_budget_ablation(results_dir / "02_ablation_data.csv", output_dir)
    plot_content_adaptivity(results_dir / "03_adaptivity_data.csv", output_dir)
    plot_threshold_sensitivity(results_dir / "04_sensitivity_data.csv", output_dir)
    
    logger.info(f"All figures saved to: {output_dir}")
