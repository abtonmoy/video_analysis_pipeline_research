#!/usr/bin/env python3
"""
Experiment 6: Second VLM — Claude (50 videos)

Uses the SAME 50 videos as Experiment 7.
Loads AdaFrame's selected frames from main pipeline results, re-encodes
them as base64, and sends to Claude with the SAME prompt/schema.
"""

import json
import os
import sys
import argparse
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
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


def find_video_in_pipeline(video_name, pipeline_results):
    """Find a video's pipeline result by name."""
    for result in pipeline_results:
        if result.get('video_name') == video_name:
            return result
    return None


def main():
    parser = argparse.ArgumentParser(description='Experiment 6: Claude VLM (50 videos)')
    parser.add_argument('--video-ids', default='new_experiments/scripts/shared_50_video_ids.json',
                        help='JSON file with shared 50 video IDs')
    parser.add_argument('--input', default=os.path.join(PROJECT_ROOT, 'main_results/processing_results.json'),
                        help='Pipeline results for frame timestamps')
    parser.add_argument('--output', default='new_experiments/results/claude_50/',
                        help='Output directory')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--videos', default='/home/wabashcs/abt/use_data',
                        help='Video directory for test mode')
    parser.add_argument('--n', type=int, default=3,
                        help='Number of videos in test mode')
    parser.add_argument('--retry', default=None)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    retry_queue_dir = os.path.join(args.output, 'retry_queue')
    failed_log_path = os.path.join(args.output, 'failed_videos.jsonl')

    # Set up Claude extractor
    from extraction.llm_client import AdExtractor
    extractor = AdExtractor(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        single_pass=True,
        schema_mode="fixed",
        temporal_context=True,
        include_timestamps=True,
        include_time_deltas=True,
        include_position_labels=True,
        include_narrative_instructions=True,
        max_retries=3,
    )

    # Load pipeline results
    print("Loading pipeline results...")
    pipeline_data = load_json(args.input)
    pipeline_results = pipeline_data.get('results', [])

    # Load video list
    if args.test:
        # Test mode: use first N videos from test directory
        video_dir = args.videos
        video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])[:args.n]
        print(f"Test mode: {len(video_files)} videos from {video_dir}")
    else:
        # Try to load shared 50, but fall back to full directory if not found
        try:
            video_files = load_json(args.video_ids)
            video_dir = os.path.join(PROJECT_ROOT, 'data', 'hussain_videos')
            if not os.path.exists(video_dir):
                video_dir = args.videos
            print(f"Shared run: {len(video_files)} videos from {video_dir}")
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            video_dir = args.videos
            video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
            print(f"Full run: processing {len(video_files)} videos from {video_dir}")

    # Check for resume
    results_path = os.path.join(args.output, 'claude_results.json')
    processed_vids = set()
    existing_results = []
    if os.path.exists(results_path):
        try:
            old_data = load_json(results_path)
            existing_results = old_data.get('results', [])
            processed_vids = {r['video_name'] for r in existing_results}
            print(f"Resuming: {len(processed_vids)} videos already processed.")
        except Exception:
            pass

    results = existing_results
    succeeded = 0
    failed = 0
    total = len(video_files)

    for i, video_name in enumerate(video_files):
        if video_name in processed_vids:
            continue

        video_path = os.path.join(video_dir, video_name)
        if not os.path.exists(video_path):
            print(f"  [{i+1}/{total}] SKIP {video_name} — file not found")
            continue

        print(f"  [{i+1}/{total}] Processing {video_name}...")

        try:
            # Find pipeline result for this video to get selected frame timestamps
            pipeline_result = find_video_in_pipeline(video_name, pipeline_results)

            if pipeline_result and pipeline_result.get('selected_frames'):
                timestamps = [f['timestamp'] for f in pipeline_result['selected_frames']]
                duration = pipeline_result.get('metadata', {}).get('duration', 0)
            else:
                # Fall back to uniform 1fps
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = total_frames / fps if fps > 0 else 0
                cap.release()
                timestamps = [float(t) for t in range(int(duration))]

            # Extract frames
            frames = extract_frames_at_timestamps(video_path, timestamps)
            if not frames:
                print(f"    WARNING: No frames extracted")
                continue

            # Run Claude extraction
            start_time = time.time()

            def do_extract():
                return extractor.extract(
                    frames=frames,
                    video_duration=duration,
                    audio_context=None,
                )

            result = call_vlm_with_retry_queue(
                video_id=video_name,
                extract_fn=do_extract,
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
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
                print(f"    ✓ Done ({len(frames)} frames, {latency:.1f}s)")
                # Incremental save
                save_results(results, results_path, total, succeeded, failed)
            else:
                failed += 1

        except Exception as e:
            print(f"    ✗ Error: {e}")
            failed += 1

    # Save results
    output_path = os.path.join(args.output, 'claude_results.json')
    output_data = {
        'metadata': {
            'experiment': 'exp6_claude',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_videos': total,
            'succeeded': succeeded,
            'failed': failed
        },
        'results': results
    }
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print_run_summary(total, succeeded, failed, 'exp6_claude', retry_queue_dir)
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
