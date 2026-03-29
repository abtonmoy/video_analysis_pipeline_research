#!/usr/bin/env python3
"""
Experiment 2: PySceneDetect Baseline

For each benchmark video: detect scenes with PySceneDetect, select first frame
of each scene + first frame of the video, then run VLM extraction with the
SAME prompt/schema as the main pipeline.

Results saved in same JSON format as main_results/processing_results.json.
"""

import json
import os
import sys
import argparse
import time
import base64
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'new_experiments', 'src_copy'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'new_experiments', 'scripts'))

from vlm_retry_utils import call_vlm_with_retry_queue, print_run_summary


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_frames_at_timestamps(video_path, timestamps):
    """Extract frames at specific timestamps from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    for ts in sorted(timestamps):
        frame_num = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frames.append((ts, frame))

    cap.release()
    return frames


def get_video_duration(video_path):
    """Get video duration in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return total_frames / fps if fps > 0 else 0


def detect_scenes_pyscenedetect(video_path):
    """Detect scenes using PySceneDetect ContentDetector."""
    from scenedetect import detect, ContentDetector

    scene_list = detect(video_path, ContentDetector())
    return scene_list


def run_vlm_extraction(frames, video_duration, extractor):
    """Run VLM extraction on selected frames."""
    result = extractor.extract(
        frames=frames,
        video_duration=video_duration,
        audio_context=None,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description='Experiment 2: PySceneDetect baseline')
    parser.add_argument('--benchmark', default='test_results/benchmark_results.json',
                        help='Benchmark results for video list')
    parser.add_argument('--output', default='new_experiments/results/pyscenedetect/',
                        help='Output directory')
    parser.add_argument('--test', action='store_true',
                        help='Test mode')
    parser.add_argument('--videos', default='/home/wabashcs/abt/use_data',
                        help='Video directory')
    parser.add_argument('--retry', default=None,
                        help='Retry queue directory')
    parser.add_argument('--n', type=int, default=10,
                        help='Number of videos in test mode')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    retry_queue_dir = os.path.join(args.output, 'retry_queue')
    failed_log_path = os.path.join(args.output, 'failed_videos.jsonl')

    # Set up extractor
    from extraction.llm_client import AdExtractor
    extractor = AdExtractor(
        provider="gemini",
        model="gemini-3-flash-preview",
        single_pass=True,
        schema_mode="fixed",
        temporal_context=True,
        include_timestamps=True,
        include_time_deltas=True,
        include_position_labels=True,
        include_narrative_instructions=True,
        max_tokens=8000,
    )

    if args.test:
        # Test mode: use videos from test directory
        video_dir = args.videos
        video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])[:args.n]
        print(f"Test mode: processing {len(video_files)} videos from {video_dir}")
    else:
        # Full run: use all videos from use_data (500 videos)
        video_dir = args.videos
        video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
        print(f"Full run: processing {len(video_files)} videos from {video_dir}")

    results = []
    processed_videos = set()
    output_path = os.path.join(args.output, 'pyscenedetect_results.json')
    
    if os.path.exists(output_path):
        try:
            old_data = load_json(output_path)
            results = old_data.get('results', [])
            processed_videos = {r['video_name'] for r in results if r.get('status') == 'success'}
            print(f"Resuming: {len(processed_videos)} videos already processed.")
        except Exception as e:
            print(f"Warning: Could not load existing results for resumption: {e}")

    succeeded = len(processed_videos)
    failed = 0
    total = len(video_files)

    from tqdm import tqdm
    for i, video_name in enumerate(tqdm(video_files, desc="Processing videos", unit="vid")):
        if video_name in processed_videos:
            continue

        video_path = os.path.join(video_dir, video_name)
        if not os.path.exists(video_path):
            # Use tqdm.write instead of print to avoid breaking the bar
            tqdm.write(f"  SKIP {video_name} — file not found")
            continue

        # tqdm.write(f"  Processing {video_name}...")

        try:
            # Step 1: Detect scenes
            scene_list = detect_scenes_pyscenedetect(video_path)
            
            # Select first frame of each scene
            # (PySceneDetect returns list of (start_time, end_time) pairs)
            timestamps = [0.0]  # Always start with 0.0
            for start, end in scene_list:
                timestamps.append(start.get_seconds())
            
            # Deduplicate and sort
            timestamps = sorted(list(set(timestamps)))
            duration = get_video_duration(video_path)

            # Extract frames
            frames = extract_frames_at_timestamps(video_path, timestamps)
            if not frames:
                tqdm.write(f"    WARNING: No frames extracted for {video_name}")
                failed += 1
                continue

            # Step 4: VLM extraction
            start_time = time.time()
            def do_extract():
                res = run_vlm_extraction(frames, duration, extractor)
                if res and "error" in res:
                    raise Exception(f"VLM Error: {res['error']}")
                return res

            result = call_vlm_with_retry_queue(
                video_id=video_name,
                extract_fn=do_extract,
                provider="gemini",
                model="gemini-3-flash-preview",
                retry_queue_dir=retry_queue_dir,
                failed_log_path=failed_log_path,
            )
            latency = time.time() - start_time

            if result is not None:
                results.append({
                    'status': 'success',
                    'video_name': video_name,
                    'duration': duration,
                    'n_frames': len(frames),
                    'latency_s': round(latency, 2),
                    'extraction': result,
                })
                succeeded += 1
                # tqdm.write(f"    ✓ Done ({len(frames)} frames, {latency:.1f}s)")
                
                # Incremental save
                output_path = os.path.join(args.output, 'pyscenedetect_results.json')
                output_data = {
                    'metadata': {
                        'experiment': 'exp2_pyscenedetect',
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'total_videos': total,
                        'succeeded': succeeded,
                        'failed': failed,
                        'partial': True
                    },
                    'results': results
                }
                with open(output_path, 'w') as f:
                    json.dump(output_data, f, indent=2)
            else:
                failed += 1

        except Exception as e:
            print(f"    ✗ Error: {e}")
            failed += 1
            continue

    # Save results
    output_path = os.path.join(args.output, 'pyscenedetect_results.json')
    output_data = {
        'metadata': {
            'experiment': 'exp2_pyscenedetect',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_videos': total,
            'succeeded': succeeded,
            'failed': failed
        },
        'results': results
    }
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print_run_summary(total, succeeded, failed, 'exp2_pyscenedetect', retry_queue_dir)
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
