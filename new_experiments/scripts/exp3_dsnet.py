#!/usr/bin/env python3
"""
Experiment 3: DSNet Baseline (Optional)

Attempts to run DSNet video summarization. If it fails (common due to
dependency issues), produces fallback text for the paper.
"""

import json
import os
import sys
import argparse
import time
import torch
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_vlm_extraction(frames, video_duration, extractor):
    """Run VLM extraction on selected frames."""
    result = extractor.extract(
        frames=frames,
        video_duration=video_duration,
        audio_context=None,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description='Experiment 3: DSNet baseline (optional)')
    parser.add_argument('--output', default='new_experiments/results/dsnet/',
                        help='Output directory')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--videos', default='/home/wabashcs/abt/use_data')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Try to import DSNet
    dsnet_repo = os.path.join(PROJECT_ROOT, 'new_experiments', 'dsnet_repo')

    try:
        if not os.path.exists(dsnet_repo):
            raise ImportError(f"DSNet repo not found at {dsnet_repo}")

        sys.path.insert(0, os.path.join(dsnet_repo, 'src'))

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        
        # Import modules after adding to path
        import torch
        import numpy as np
        import cv2
        cv2.setNumThreads(0)

        from helpers import init_helper, vsumm_helper, bbox_helper, video_helper
        from modules.model_zoo import get_model

        from vlm_retry_utils import call_vlm_with_retry_queue
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'new_experiments', 'src_copy'))
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'new_experiments', 'scripts'))
        from extraction.llm_client import AdExtractor

        print("DSNet repo found and importable")

        # Configuration for DSNet anchor-based model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_kwargs = {
            'base_model': 'attention',
            'num_feature': 1024,
            'num_hidden': 128,
            'anchor_scales': [4, 8, 16, 32],
            'num_head': 8
        }
        ckpt_path = os.path.join(dsnet_repo, 'models/pretrain_ab_basic/checkpoint/tvsum.yml.0.pt')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
            
        print(f"Loading DSNet model from {ckpt_path} on {device}...")
        model = get_model('anchor-based', **model_kwargs)
        model = model.eval().to(device)
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)

        sample_rate = 15
        video_proc = video_helper.VideoPreprocessor(sample_rate)
        extractor = AdExtractor(
            provider="gemini",
            model="gemini-2.5-flash",
            single_pass=True,
            schema_mode="fixed",
            temporal_context=True,
            include_timestamps=True,
            include_time_deltas=True,
            include_position_labels=True,
            include_narrative_instructions=True,
            max_tokens=8000,
        )

        results = []
        retry_queue_dir = os.path.join(args.output, 'retry_queue')

        # Load video list
        if os.path.isdir(args.videos):
            video_files = [os.path.join(args.videos, f) for f in os.listdir(args.videos) if f.endswith('.mp4')]
        else:
            with open(args.videos, 'r') as f:
                video_files = [line.strip() for line in f if line.strip().endswith('.mp4')]

        video_files.sort()
        if args.test:
            video_files = video_files[:5]
            print(f"Test mode: processing {len(video_files)} videos from {args.videos}")

        # Load existing results for resumption
        results = []
        processed_videos = set()
        output_path = os.path.join(args.output, 'dsnet_results.json')
        if os.path.exists(output_path):
            try:
                old_data = load_json(output_path)
                results = old_data.get('results', [])
                processed_videos = {r['video_name'] for r in results if r.get('status') == 'success'}
                print(f"Resuming: {len(processed_videos)} videos already processed.")
            except Exception as e:
                print(f"Warning: Could not load existing results: {e}")

        succeeded = len(processed_videos)
        failed = 0

        from tqdm import tqdm
        for i, video_path in enumerate(tqdm(video_files, desc="DSNet Evaluation", unit="vid")):
            video_name = os.path.basename(video_path)
            if video_name in processed_videos:
                continue
            
            # Use closure to capture loop variables for the retry utility
            def extract_fn(vid_path=video_path, v_name=video_name):
                # 1. DSNet Inference
                print(f"    Extracting deep features for {v_name}...")
                n_frames, seq, cps, nfps, picks = video_proc.run(vid_path)
                seq_len = len(seq)

                with torch.no_grad():
                    seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)
                    pred_cls, pred_bboxes = model.predict(seq_torch)
                    pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)
                    pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, 0.4)
                    pred_summ = vsumm_helper.bbox2summary(
                        seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks)
                
                # Top frames selected by DSNet
                selected_indices = np.where(pred_summ)[0]
                
                # 2. Extract specific actual frames via cv2
                cap = cv2.VideoCapture(vid_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                sorted_indices = sorted(selected_indices)
                extracted_frames = []
                
                current_idx = 0
                frame_counter = 0
                while True:
                    ret, frame = cap.read()
                    if not ret or current_idx >= len(sorted_indices):
                        break
                    if frame_counter == sorted_indices[current_idx]:
                        timestamp = frame_counter / fps
                        extracted_frames.append((timestamp, frame))
                        current_idx += 1
                    frame_counter += 1
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = total_frames / fps if fps > 0 else 0
                cap.release()

                if not extracted_frames:
                    raise Exception("No frames selected by DSNet")

                print(f"    DSNet selected {len(extracted_frames)} frames. Querying VLM...")
                
                # 3. Query VLM
                result_json = run_vlm_extraction(extracted_frames, duration, extractor)
                if result_json and "error" in result_json:
                    raise Exception(f"VLM Error: {result_json['error']}")
                
                return {
                    'status': 'success',
                    'video_name': v_name,
                    'duration': duration,
                    'n_frames': len(extracted_frames),
                    'extraction': result_json
                }

            # Wrap in retry utility
            failed_log_path = os.path.join(args.output, 'failed_videos.jsonl')
            start_time = time.time()
            result = call_vlm_with_retry_queue(
                video_id=video_name,
                extract_fn=extract_fn,
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
                    'duration': result['duration'],
                    'n_frames': result['n_frames'],
                    'latency_s': round(latency, 2),
                    'extraction': result['extraction'],
                })
                succeeded += 1
                # tqdm.write(f"    ✓ Done ({len(extracted_frames)} frames, {latency:.1f}s)")

                # Incremental save
                output_path = os.path.join(args.output, 'dsnet_results.json')
                output_data = {
                    'metadata': {
                        'experiment': 'exp3_dsnet',
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'total_videos': len(video_files),
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

        print(f"\n=== exp3_dsnet Run Complete ===")
        print(f"Succeeded: {succeeded} / {len(video_files)}")
        print(f"Failed:    {failed} / {len(video_files)} (saved to retry_queue/)")

        # Save to JSON matching processing_results.json
        output_file = os.path.join(args.output, 'dsnet_results.json')
        output_data = {
            'metadata': {
                'experiment': 'exp3_dsnet',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'total_videos': len(video_files),
                'succeeded': succeeded,
                'failed': failed
            },
            'results': results
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {output_file}")

    except Exception as e:
        error_msg = str(e)
        import traceback
        traceback.print_exc()
        print(f"DSNet experiment failed: {error_msg}")
        print("Saving fallback text...")

        # Save failure record
        failed_path = os.path.join(args.output, 'FAILED.txt')
        with open(failed_path, 'w') as f:
            f.write(f"DSNet experiment failed at {datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"Error: {error_msg}\n\n")
            f.write("DSNet requires architecture-specific pretrained weights and\n")
            f.write("feature extraction (GoogLeNet features in HDF5 format).\n")
            f.write("This dependency chain is not easily reproducible.\n")

        # Save fallback LaTeX
        fallback_path = os.path.join(args.output, 'fallback_text.tex')
        with open(fallback_path, 'w') as f:
            f.write("%% DSNet Fallback Text\n")
            f.write("% Deep summarization baselines (DSNet) require architecture-specific\n")
            f.write("% inference pipelines and pretrained feature extractors that are\n")
            f.write("% not directly comparable to our frame-selection approach.\n")

        print(f"Saved to {failed_path} and {fallback_path}")


if __name__ == '__main__':
    main()
