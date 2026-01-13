# experiments/compare_clustering_methods.py
"""
Compare NMS, K-means, and Hybrid frame selection methods on a video.
Runs deduplication ONCE, then tests all three selection methods.
Saves results to experiments/results/<video_name>.json
"""

from dotenv import load_dotenv
load_dotenv()

import json
import time
import argparse
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from src.pipeline import AdVideoPipeline
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.ingestion.video_loader import VideoLoader
from src.detection.scene_detector import SceneDetector, CandidateFrameExtractor
from src.detection.change_detector import get_change_detector
from src.deduplication.hierarchical import create_deduplicator
from src.selection.representative import FrameSelector
from src.extraction.llm_client import create_extractor


def run_shared_pipeline(video_path: str, config: Dict) -> Dict[str, Any]:
    """
    Run the shared pipeline stages (loading, scenes, dedup, audio) once.
    Returns all data needed for selection stage.
    """
    print(f"\n{'='*60}")
    print("SHARED PIPELINE (runs once)")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Stage 1: Load video
    print("\nStage 1: Loading video...")
    loader = VideoLoader(
        max_resolution=config.get("ingestion", {}).get("max_resolution", 720),
        extract_audio=config.get("ingestion", {}).get("extract_audio", True)
    )
    metadata, audio_path = loader.load(video_path)
    print(f"  Loaded: {metadata.duration:.1f}s, {metadata.width}x{metadata.height}")
    
    # Stage 2: Detect scenes
    print("\nStage 2: Detecting scenes...")
    scene_config = config.get("scene_detection", {})
    scene_detector = SceneDetector(
        method=scene_config.get("method", "content"),
        threshold=scene_config.get("threshold", 27.0),
        min_scene_length_s=scene_config.get("min_scene_length_s", 0.5)
    )
    scenes = scene_detector.detect_scenes(video_path)
    
    # Handle no scenes detected
    if not scenes:
        fallback = scene_config.get("fallback", {})
        if fallback.get("enabled", True):
            print("  No scenes detected, retrying with lower threshold...")
            scene_detector.threshold = fallback.get("threshold", 15.0)
            scenes = scene_detector.detect_scenes(video_path)
        
        if not scenes and fallback.get("artificial_chunks", True):
            print("  Creating artificial scene chunks...")
            chunk_size = fallback.get("chunk_size_s", 10.0)
            scenes = []
            current = 0.0
            while current < metadata.duration:
                end = min(current + chunk_size, metadata.duration)
                scenes.append((current, end))
                current = end
    
    print(f"  Detected {len(scenes)} scenes")
    
    # Stage 3: Extract candidate frames
    print("\nStage 3: Extracting candidate frames...")
    change_config = config.get("change_detection", {})
    change_detector = get_change_detector(change_config.get("method", "histogram"))
    
    extractor = CandidateFrameExtractor(
        change_detector=change_detector,
        threshold=change_config.get("threshold", 0.15),
        min_interval_ms=change_config.get("min_interval_ms", 100)
    )
    candidates = extractor.extract_candidates(
        video_path,
        max_resolution=config.get("ingestion", {}).get("max_resolution", 720)
    )
    print(f"  Extracted {len(candidates)} candidate frames")
    
    # Stage 4: Hierarchical deduplication
    print("\nStage 4: Hierarchical deduplication...")
    deduplicator = create_deduplicator(config)
    deduped_frames, embeddings, dedup_stats = deduplicator.deduplicate(candidates)
    print(f"  PHash: {dedup_stats.get('input', 0)} -> {dedup_stats.get('after_phash', 0)}")
    print(f"  SSIM: {dedup_stats.get('after_phash', 0)} -> {dedup_stats.get('after_ssim', 0)}")
    print(f"  CLIP: {dedup_stats.get('after_ssim', 0)} -> {dedup_stats.get('after_clip', 0)}")
    print(f"  Final: {len(deduped_frames)} frames")
    
    # Stage 5: Extract audio context (same as pipeline.py)
    print("\nStage 5: Extracting audio context...")
    audio_context = None
    if audio_path and config.get("audio_analysis", {}).get("enabled", True):
        try:
            from src.ingestion.audio_extractor import AudioExtractor
            audio_config = config.get("audio_analysis", {})
            audio_extractor = AudioExtractor(sample_rate=16000)
            audio_context = audio_extractor.extract_full_context(
                audio_path,
                transcribe=audio_config.get("transcription", {}).get("enabled", True),
                model_size=audio_config.get("transcription", {}).get("model", "base")
            )
            if audio_context:
                print(f"  Transcribed {len(audio_context.get('transcription', []))} segments")
                print(f"  Found {len(audio_context.get('key_phrases', []))} key phrases")
                print(f"  Mood: {audio_context.get('mood', 'unknown')}")
        except Exception as e:
            print(f"  Audio extraction failed: {e}")
    
    shared_time = time.time() - start_time
    print(f"\nShared pipeline completed in {shared_time:.1f}s")
    
    return {
        "video_path": video_path,
        "metadata": metadata,
        "scenes": scenes,
        "deduped_frames": deduped_frames,
        "embeddings": embeddings,
        "dedup_stats": dedup_stats,
        "audio_context": audio_context,
        "audio_path": audio_path,
        "shared_time": shared_time,
        "total_candidates": len(candidates)
    }


def run_selection_and_extraction(
    shared_data: Dict[str, Any],
    method: str,
    config: Dict
) -> Dict[str, Any]:
    """
    Run selection and extraction with a specific method.
    Uses pre-computed deduplication results.
    """
    print(f"\n{'-'*40}")
    print(f"Selection Method: {method.upper()}")
    print(f"{'-'*40}")
    
    start_time = time.time()
    
    # Get shared data
    metadata = shared_data["metadata"]
    scenes = shared_data["scenes"]
    deduped_frames = shared_data["deduped_frames"]
    embeddings = shared_data["embeddings"]
    audio_context = shared_data["audio_context"]
    
    # Build audio events for importance scoring (same as pipeline.py)
    audio_events = None
    if audio_context:
        audio_events = {
            "speech_segments": [(s["start"], s["end"]) for s in audio_context.get("transcription", [])],
            "key_phrases": audio_context.get("key_phrases", []),
            "energy_peaks": audio_context.get("energy_peaks", []),
            "silence_segments": audio_context.get("silence_segments", [])
        }
    
    # Create selector with specific method
    selection_config = config.get("selection", {})
    selector = FrameSelector(
        target_frame_density=selection_config.get("target_frame_density", 0.25),
        min_frames_per_scene=selection_config.get("min_frames_per_scene", 2),
        max_frames_per_scene=selection_config.get("max_frames_per_scene", 10),
        min_temporal_gap_s=selection_config.get("min_temporal_gap_s", 0.5),
        clustering_method=method,  # Override method here
        adaptive_density=selection_config.get("adaptive_density", True),
        use_importance_scoring=True
    )
    
    # Select frames
    selected = selector.select(
        deduped_frames,
        embeddings,
        scenes,
        metadata.duration,
        audio_events
    )
    
    selection_time = time.time() - start_time
    print(f"  Selected {len(selected)} frames in {selection_time:.2f}s")
    print(f"  Importance range: [{min(f.importance_score for f in selected):.2f}, {max(f.importance_score for f in selected):.2f}]")
    
    # Extract with LLM
    extraction_start = time.time()
    extractor = create_extractor(config)
    
    # Prepare frames for extraction
    frames_for_extraction = [(f.timestamp, f.frame) for f in selected]
    
    # Build audio context for prompt (same as pipeline.py)
    prompt_audio_context = None
    if audio_context and config.get("extraction", {}).get("audio_context", {}).get("enabled", True):
        prompt_audio_context = {
            "transcription": audio_context.get("transcription", []),
            "key_phrases": audio_context.get("key_phrases", []),
            "mood": audio_context.get("mood")
        }
    
    extraction_result = extractor.extract(
        frames_for_extraction,
        metadata.duration,
        prompt_audio_context
    )
    
    extraction_time = time.time() - extraction_start
    print(f"  Extraction completed in {extraction_time:.1f}s")
    
    total_time = time.time() - start_time
    
    return {
        "method": method,
        "final_frames": len(selected),
        "importance_range": [
            min(f.importance_score for f in selected),
            max(f.importance_score for f in selected)
        ],
        "selection_time": selection_time,
        "extraction_time": extraction_time,
        "total_time": total_time,
        "extraction": extraction_result
    }


def extract_key_fields(extraction: Dict) -> Dict[str, Any]:
    """Extract key fields from extraction result for comparison."""
    if not extraction:
        return {}
    
    return {
        "ad_type": extraction.get("_metadata", {}).get("ad_type"),
        "brand_name": extraction.get("brand", {}).get("brand_name_text"),
        "product_name": extraction.get("product", {}).get("product_name"),
        "industry": extraction.get("product", {}).get("industry"),
        "primary_message": extraction.get("message", {}).get("primary_message"),
        "cta_type": extraction.get("call_to_action", {}).get("cta_type"),
        "text_overlays": extraction.get("visual_elements", {}).get("text_overlays", []),
        "text_overlays_count": len(extraction.get("visual_elements", {}).get("text_overlays", [])),
        "persuasion_techniques": extraction.get("persuasion_techniques", []),
        "promo_present": extraction.get("promotion", {}).get("promo_present"),
        "promo_text": extraction.get("promotion", {}).get("promo_text"),
    }


def print_comparison_table(results: List[Dict[str, Any]], shared_data: Dict[str, Any]):
    """Print a side-by-side comparison table."""
    
    print(f"\n{'='*100}")
    print("COMPARISON TABLE")
    print(f"{'='*100}")
    
    # Pipeline metrics
    print(f"\n{'PIPELINE METRICS':-^100}")
    print(f"{'Field':<25} | {results[0]['method']:^22} | {results[1]['method']:^22} | {results[2]['method']:^22}")
    print("-" * 100)
    
    # Shared metrics
    print(f"{'Shared Pipeline Time':<25} | {shared_data['shared_time']:^22.1f} | {shared_data['shared_time']:^22.1f} | {shared_data['shared_time']:^22.1f}")
    print(f"{'Candidates After Dedup':<25} | {len(shared_data['deduped_frames']):^22} | {len(shared_data['deduped_frames']):^22} | {len(shared_data['deduped_frames']):^22}")
    
    # Method-specific metrics
    values = [f"{r['selection_time']:.2f}s" for r in results]
    print(f"{'Selection Time':<25} | {values[0]:^22} | {values[1]:^22} | {values[2]:^22}")
    
    values = [f"{r['extraction_time']:.1f}s" for r in results]
    print(f"{'Extraction Time':<25} | {values[0]:^22} | {values[1]:^22} | {values[2]:^22}")
    
    values = [f"{shared_data['shared_time'] + r['total_time']:.1f}s" for r in results]
    print(f"{'Total Time':<25} | {values[0]:^22} | {values[1]:^22} | {values[2]:^22}")
    
    values = [str(r['final_frames']) for r in results]
    print(f"{'Final Frames':<25} | {values[0]:^22} | {values[1]:^22} | {values[2]:^22}")
    
    values = [f"[{r['importance_range'][0]:.2f}, {r['importance_range'][1]:.2f}]" for r in results]
    print(f"{'Importance Range':<25} | {values[0]:^22} | {values[1]:^22} | {values[2]:^22}")
    
    # Extraction comparison
    print(f"\n{'EXTRACTION RESULTS':-^100}")
    
    extractions = [extract_key_fields(r.get("extraction", {})) for r in results]
    
    extraction_fields = [
        ("Ad Type", "ad_type"),
        ("Brand Name", "brand_name"),
        ("Product Name", "product_name"),
        ("Industry", "industry"),
        ("CTA Type", "cta_type"),
        ("Text Overlays Count", "text_overlays_count"),
        ("Promo Present", "promo_present"),
        ("Promo Text", "promo_text"),
        ("Persuasion Techniques", "persuasion_techniques"),
    ]
    
    print(f"{'Field':<25} | {results[0]['method']:^22} | {results[1]['method']:^22} | {results[2]['method']:^22}")
    print("-" * 100)
    
    for label, key in extraction_fields:
        values = []
        for e in extractions:
            val = e.get(key, "N/A")
            if isinstance(val, list):
                val = str(len(val)) + " items" if len(str(val)) > 20 else str(val)
            if val is None:
                val = "null"
            val = str(val)[:22]
            values.append(val)
        print(f"{label:<25} | {values[0]:^22} | {values[1]:^22} | {values[2]:^22}")
    
    # Primary message
    print(f"\n{'PRIMARY MESSAGE':-^100}")
    for r, e in zip(results, extractions):
        msg = e.get("primary_message", "N/A")
        print(f"{r['method']:<10}: {msg}")
    
    # Text overlays
    print(f"\n{'TEXT OVERLAYS':-^100}")
    for r, e in zip(results, extractions):
        overlays = e.get("text_overlays", [])
        print(f"{r['method']:<10}: {overlays}")


def save_results(
    video_path: str,
    shared_data: Dict[str, Any],
    results: List[Dict[str, Any]],
    output_dir: str = "experiments/results"
) -> str:
    """Save comparison results to JSON file."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    video_name = Path(video_path).stem
    output_path = os.path.join(output_dir, f"{video_name}.json")
    
    output = {
        "video": {
            "path": video_path,
            "name": video_name,
            "duration": shared_data["metadata"].duration,
            "resolution": f"{shared_data['metadata'].width}x{shared_data['metadata'].height}",
            "fps": shared_data["metadata"].fps
        },
        "shared_pipeline": {
            "time_seconds": shared_data["shared_time"],
            "scenes_detected": len(shared_data["scenes"]),
            "total_candidates": shared_data["total_candidates"],
            "after_phash": shared_data["dedup_stats"].get("after_phash", 0),
            "after_ssim": shared_data["dedup_stats"].get("after_ssim", 0),
            "after_clip": shared_data["dedup_stats"].get("after_clip", 0),
            "frames_after_dedup": len(shared_data["deduped_frames"]),
            "audio_segments": len(shared_data["audio_context"].get("transcription", [])) if shared_data["audio_context"] else 0,
            "key_phrases_found": len(shared_data["audio_context"].get("key_phrases", [])) if shared_data["audio_context"] else 0,
            "audio_mood": shared_data["audio_context"].get("mood") if shared_data["audio_context"] else None
        },
        "methods": {}
    }
    
    extractions = [extract_key_fields(r.get("extraction", {})) for r in results]
    
    for r in results:
        method_name = r["method"]
        output["methods"][method_name] = {
            "selection_time_seconds": r["selection_time"],
            "extraction_time_seconds": r["extraction_time"],
            "total_time_seconds": r["total_time"],
            "final_frames": r["final_frames"],
            "importance_range": r["importance_range"],
            "extraction": r["extraction"]
        }
    
    # Comparison summary
    ad_types = [e.get("ad_type") for e in extractions]
    messages = [e.get("primary_message") for e in extractions]
    overlay_counts = {r["method"]: e.get("text_overlays_count", 0) for r, e in zip(results, extractions)}
    
    output["comparison"] = {
        "ad_type_agreement": len(set(ad_types)) == 1,
        "ad_types": {r["method"]: e.get("ad_type") for r, e in zip(results, extractions)},
        "primary_message_agreement": len(set(messages)) == 1,
        "text_overlay_counts": overlay_counts,
        "fastest_method": min(results, key=lambda x: x["total_time"])["method"],
        "most_overlays_method": max(overlay_counts, key=overlay_counts.get)
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Compare frame selection methods on a video")
    parser.add_argument("video", nargs="?", default=r"data\hussain_videos\_wZSB0b9OCA.mp4",
                        help="Path to video file")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to config file")
    parser.add_argument("--full", action="store_true",
                        help="Print full extraction results")
    parser.add_argument("--methods", nargs="+", default=["nms", "kmeans", "hybrid"],
                        choices=["nms", "kmeans", "hybrid", "uniform"],
                        help="Methods to compare")
    parser.add_argument("--output-dir", default="experiments/results",
                        help="Directory to save results JSON")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results to file")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    
    print(f"{'='*100}")
    print(f"COMPARING FRAME SELECTION METHODS")
    print(f"{'='*100}")
    print(f"Video: {args.video}")
    print(f"Methods: {', '.join(args.methods)}")
    
    # Run shared pipeline ONCE
    shared_data = run_shared_pipeline(args.video, config)
    
    # Run each selection method
    print(f"\n{'='*60}")
    print("SELECTION & EXTRACTION (per method)")
    print(f"{'='*60}")
    
    results = []
    for method in args.methods:
        result = run_selection_and_extraction(shared_data, method, config)
        results.append(result)
    
    # Print comparison table
    print_comparison_table(results, shared_data)
    
    # Print full extractions if requested
    if args.full:
        for r in results:
            print(f"\n{'='*100}")
            print(f"FULL EXTRACTION - {r['method'].upper()}")
            print(f"{'='*100}")
            if r.get("extraction"):
                print(json.dumps(r["extraction"], indent=2))
    
    # Save results
    if not args.no_save:
        output_path = save_results(args.video, shared_data, results, args.output_dir)
        print(f"\nResults saved to: {output_path}")
    
    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    
    print(f"Shared pipeline time: {shared_data['shared_time']:.1f}s (dedup: {shared_data['total_candidates']} -> {len(shared_data['deduped_frames'])} frames)")
    
    extractions = [extract_key_fields(r.get("extraction", {})) for r in results]
    overlay_counts = [(r["method"], e.get("text_overlays_count", 0)) for r, e in zip(results, extractions)]
    best_overlays = max(overlay_counts, key=lambda x: x[1])
    fastest = min(results, key=lambda x: x["total_time"])
    
    print(f"Fastest selection+extraction: {fastest['method']} ({fastest['total_time']:.1f}s)")
    print(f"Most text overlays: {best_overlays[0]} ({best_overlays[1]} items)")
    
    ad_types = [e.get("ad_type") for e in extractions]
    if len(set(ad_types)) == 1:
        print(f"Ad type agreement: Yes ({ad_types[0]})")
    else:
        print(f"Ad type disagreement: {dict(zip(args.methods, ad_types))}")
    
    messages = [e.get("primary_message") for e in extractions]
    if len(set(messages)) == 1:
        print(f"Primary message agreement: Yes")
    else:
        print(f"Primary message disagreement: Methods produced different messages")
    
    # Time savings
    naive_time = shared_data['shared_time'] * len(args.methods) + sum(r['total_time'] for r in results)
    actual_time = shared_data['shared_time'] + sum(r['total_time'] for r in results)
    savings = naive_time - actual_time
    print(f"\nTime saved by sharing dedup: {savings:.1f}s ({savings/naive_time*100:.0f}% faster)")


if __name__ == "__main__":
    main()