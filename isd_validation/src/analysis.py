"""
Statistical analysis utilities for ISD validation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def load_pipeline_results(results_path: str) -> Dict:
    """Load pipeline results from JSON."""
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Handle both formats
    if "results" in data:
        return {r["video_name"]: r for r in data["results"] if "video_name" in r}
    return data


def compute_correlations(df: pd.DataFrame) -> Dict:
    """
    Compute Pearson correlations between ISD and video characteristics.
    
    Returns:
        Dictionary with correlation results
    """
    correlations = {}
    
    pairs = [
        ("isd", "cut_frequency", "ISD vs Cut Frequency"),
        ("isd", "visual_diversity", "ISD vs Visual Diversity"),
        ("isd", "duration", "ISD vs Duration"),
        ("isd", "num_scenes", "ISD vs Num Scenes"),
    ]
    
    for col1, col2, name in pairs:
        if col1 in df.columns and col2 in df.columns:
            r, p = stats.pearsonr(df[col1], df[col2])
            correlations[name] = {
                "r": float(r),
                "p": float(p),
                "significant": bool(p < 0.05),
                "strength": _correlation_strength(r)
            }
    
    return correlations


def _correlation_strength(r: float) -> str:
    """Classify correlation strength."""
    abs_r = abs(r)
    if abs_r >= 0.7:
        return "strong"
    elif abs_r >= 0.5:
        return "moderate"
    elif abs_r >= 0.3:
        return "weak"
    return "negligible"


def compute_budget_strategies(
    isd: int,
    semantic_velocity: float,
    duration: float,
    num_scenes: int,
    density: float = 0.25
) -> Dict[str, int]:
    """
    Compute budget using different strategies.
    
    Returns:
        Dictionary mapping strategy name to budget
    """
    base = max(5, num_scenes + 1)
    
    # Strategy 1: Full AdaFrame
    isd_cap = base + int(1.5 * isd)
    sem_energy = semantic_velocity * 1.0  # Simplified attention yield
    energy_scaled = max(base, int(base + duration * density * sem_energy))
    full_budget = min(isd_cap, energy_scaled)
    
    # Strategy 2: ISD-only
    isd_budget = base + int(1.5 * isd)
    
    # Strategy 3: Energy-only
    energy_budget = max(base, int(base + duration * density * sem_energy))
    
    # Strategy 4: Fixed-25
    fixed_budget = 25
    
    # Strategy 5: Linear duration
    linear_budget = max(5, int(duration * 0.5))
    
    return {
        "full_adaptive": full_budget,
        "isd_only": isd_budget,
        "energy_only": energy_budget,
        "fixed_25": fixed_budget,
        "linear_duration": linear_budget,
    }


def categorize_by_motion(cut_frequency: float) -> str:
    """Categorize video by motion level."""
    if cut_frequency < 0.17:
        return "low_motion"
    elif cut_frequency < 0.47:
        return "medium_motion"
    return "high_motion"


def save_results(results: Dict, output_dir: Path, prefix: str):
    """Save results to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_dir / f"{prefix}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {json_path}")


def create_summary_table(results_list: List[Dict]) -> pd.DataFrame:
    """Create summary DataFrame from results."""
    return pd.DataFrame(results_list)
