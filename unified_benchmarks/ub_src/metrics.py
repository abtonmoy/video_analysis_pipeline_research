#!/usr/bin/env python3
"""
Metric computation for unified benchmarks.

Computes frame selection metrics, cost metrics, and accuracy metrics.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


# Cost per frame for VLM (Gemini Flash)
COST_PER_FRAME_USD = 0.015


def compute_selection_metrics(
    selected_frames: List[Tuple[float, np.ndarray]],
    total_frames: int,
    latency_s: float,
    clip_embeddings: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute frame selection metrics.

    Args:
        selected_frames: List of (timestamp, frame) tuples
        total_frames: Total number of frames in video
        latency_s: Time taken to select frames
        clip_embeddings: Optional CLIP embeddings for info density

    Returns:
        Dict with selection metrics
    """
    n_selected = len(selected_frames)

    if n_selected == 0:
        return {
            "final_frame_count": 0,
            "total_frames_sampled": total_frames,
            "reduction_rate": 0.0,
            "compression_ratio": 0.0,
            "selection_latency_s": latency_s,
            "info_density": 0.0,
            "cost_usd": 0.0,
        }

    reduction_rate = (total_frames - n_selected) / total_frames if total_frames > 0 else 0.0
    compression_ratio = total_frames / n_selected if n_selected > 0 else 0.0
    cost_usd = n_selected * COST_PER_FRAME_USD

    # Compute info density if embeddings provided
    info_density = 0.0
    if clip_embeddings is not None and len(clip_embeddings) >= 2:
        info_density = compute_info_density(clip_embeddings)

    return {
        "final_frame_count": n_selected,
        "total_frames_sampled": total_frames,
        "reduction_rate": round(reduction_rate, 4),
        "compression_ratio": round(compression_ratio, 2),
        "selection_latency_s": round(latency_s, 3),
        "info_density": round(info_density, 5),
        "cost_usd": round(cost_usd, 4),
    }


def compute_info_density(embeddings: np.ndarray) -> float:
    """
    Compute information density as mean pairwise cosine distance.
    Higher = more diverse frames.

    Args:
        embeddings: Array of shape (N, D) with N embeddings

    Returns:
        Mean pairwise distance (0-1 range)
    """
    if len(embeddings) < 2:
        return 0.0

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)

    # Compute cosine similarity matrix
    sim_matrix = normalized @ normalized.T

    # Get upper triangle (excluding diagonal)
    n = len(embeddings)
    upper_tri_indices = np.triu_indices(n, k=1)
    similarities = sim_matrix[upper_tri_indices]

    # Return mean distance (1 - similarity)
    return float(1.0 - np.mean(similarities))


def compare_extractions(
    result: Dict[str, Any],
    reference: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare extraction result against reference (AdaFrame pipeline).

    Args:
        result: Extraction result from baseline method
        reference: Reference extraction from AdaFrame pipeline

    Returns:
        Dict with comparison metrics
    """
    comparison = {
        "brand_match": False,
        "product_match": False,
        "topic_match": False,
        "super_category_match": False,
        "cta_match": False,
        "effectiveness_diff": None,
    }

    # Extract fields with safe navigation
    result_brand = _extract_brand(result)
    reference_brand = _extract_brand(reference)
    comparison["brand_match"] = _match_brands(result_brand, reference_brand)

    result_topic = _extract_topic(result)
    reference_topic = _extract_topic(reference)
    comparison["topic_match"] = str(result_topic).lower() == str(reference_topic).lower() if result_topic and reference_topic else False

    result_super = _extract_super_category(result)
    reference_super = _extract_super_category(reference)
    comparison["super_category_match"] = str(result_super).lower() == str(reference_super).lower() if result_super and reference_super else False

    result_cta = _extract_cta(result)
    reference_cta = _extract_cta(reference)
    comparison["cta_match"] = result_cta == reference_cta

    # Effectiveness score difference
    result_eff = _extract_effectiveness(result)
    reference_eff = _extract_effectiveness(reference)
    if result_eff is not None and reference_eff is not None:
        comparison["effectiveness_diff"] = round(result_eff - reference_eff, 2)

    return comparison


def _extract_brand(extraction: Dict) -> str:
    """Extract brand name from extraction."""
    brand = extraction.get("brand", {})
    if isinstance(brand, dict):
        return brand.get("brand_name_text", "") or brand.get("name", "")
    return str(brand)


def _extract_topic(extraction: Dict) -> str:
    """Extract topic from extraction."""
    topic = extraction.get("topic", {})
    if isinstance(topic, dict):
        topic_id = topic.get("topic_id", "")
        topic_name = topic.get("name", "")
        # Convert to string in case topic_id is an integer
        return str(topic_id) if topic_id is not None else str(topic_name)
    return str(topic)


def _extract_super_category(extraction: Dict) -> str:
    """Extract super category from extraction."""
    topic = extraction.get("topic", {})
    if isinstance(topic, dict):
        super_cat = topic.get("super_category", "") or topic.get("category", "")
        return str(super_cat) if super_cat is not None else ""
    return ""


def _extract_cta(extraction: Dict) -> bool:
    """Extract CTA presence from extraction."""
    cta = extraction.get("call_to_action", {})
    if isinstance(cta, dict):
        return cta.get("cta_present", False) or cta.get("present", False)
    return bool(cta)


def _extract_effectiveness(extraction: Dict) -> Optional[float]:
    """Extract effectiveness score from extraction."""
    engagement = extraction.get("engagement_metrics", {})
    if isinstance(engagement, dict):
        score = engagement.get("effectiveness_score") or engagement.get("effectiveness")
        if score is not None:
            try:
                return float(score)
            except (ValueError, TypeError):
                return None
    return None


def _match_brands(brand1: str, brand2: str) -> bool:
    """Check if two brand names match (case-insensitive, partial)."""
    if not brand1 or not brand2:
        return False
    b1 = brand1.lower().strip()
    b2 = brand2.lower().strip()
    return b1 == b2 or b1 in b2 or b2 in b1


def compute_accuracy_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute accuracy metrics from a list of results.

    Args:
        results: List of result dicts with 'comparison' field

    Returns:
        Dict with accuracy percentages
    """
    if not results:
        return {
            "topic_accuracy": 0.0,
            "super_category_accuracy": 0.0,
            "brand_detection_rate": 0.0,
            "cta_detection_rate": 0.0,
        }

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


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate metrics across all results.

    Args:
        results: List of per-video results

    Returns:
        Dict with aggregated statistics
    """
    if not results:
        return {}

    # Extract numeric fields
    frame_counts = [r.get("final_frame_count", 0) for r in results]
    costs = [r.get("cost_usd", 0.0) for r in results]
    latencies = [r.get("selection_latency_s", 0.0) for r in results]
    info_densities = [r.get("info_density", 0.0) for r in results if r.get("info_density", 0.0) > 0]
    compression_ratios = [r.get("compression_ratio", 0.0) for r in results if r.get("compression_ratio", 0.0) > 0]

    # Compute statistics
    aggregated = {
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
        },
    }

    # Add accuracy metrics
    accuracy = compute_accuracy_metrics(results)
    aggregated["accuracy"] = accuracy

    return aggregated
