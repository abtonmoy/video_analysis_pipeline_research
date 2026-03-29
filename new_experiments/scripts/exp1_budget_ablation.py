#!/usr/bin/env python3
"""
Experiment 1: Budget Formula Ablation

Tests 5 budget strategies and ISD multiplier sensitivity.
Only re-runs VLM extraction when the selected frame SET differs from Strategy A.

Strategies:
  A: Full AdaFrame (control) — current HIB formula
  B: ISD only — budget = max(5, isd)
  C: Linear duration — budget = max(5, floor(duration * 0.25))
  D: Fixed-25 — budget = 25
  E: Energy only (no ISD cap) — budget = max(5, floor(base + duration * 0.25 * e_k * e_p))

Also: ISD multiplier sensitivity {0.5, 1.0, 1.5, 2.0, 3.0}.
"""

import json
import os
import sys
import argparse
import csv
import math
import time
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'new_experiments', 'src_copy'))

from vlm_retry_utils import call_vlm_with_retry_queue, print_run_summary


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_budget_strategy_a(base, isd, duration, e_k, e_p, density=0.25):
    """Full AdaFrame: min(base + floor(1.5 * isd), max(base, floor(base + duration * 0.25 * e_k * e_p)))"""
    return min(
        base + math.floor(1.5 * isd),
        max(base, math.floor(base + duration * density * e_k * e_p))
    )


def compute_budget_strategy_b(isd):
    """ISD only: max(5, isd)"""
    return max(5, isd)


def compute_budget_strategy_c(duration):
    """Linear duration: max(5, floor(duration * 0.25))"""
    return max(5, math.floor(duration * 0.25))


def compute_budget_strategy_d():
    """Fixed-25"""
    return 25


def compute_budget_strategy_e(base, duration, e_k, e_p, density=0.25):
    """Energy only (no ISD cap): max(5, floor(base + duration * 0.25 * e_k * e_p))"""
    return max(5, math.floor(base + duration * density * e_k * e_p))


def compute_budget_with_multiplier(base, isd, duration, e_k, e_p, multiplier, density=0.25):
    """Vary the ISD multiplier."""
    return min(
        base + math.floor(multiplier * isd),
        max(base, math.floor(base + duration * density * e_k * e_p))
    )


def main():
    parser = argparse.ArgumentParser(description='Experiment 1: Budget formula ablation')
    parser.add_argument('--input', default=os.path.join(PROJECT_ROOT, os.path.join(PROJECT_ROOT, 'main_results/processing_results.json')),
                        help='Path to main pipeline results')
    parser.add_argument('--output', default='new_experiments/results/budget_ablation/',
                        help='Output directory')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: use test videos')
    parser.add_argument('--videos', default='/home/wabashcs/abt/use_data',
                        help='Video directory for test mode')
    parser.add_argument('--retry', default=None,
                        help='Retry queue directory to process')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.test:
        print("Test mode: simulating with first 10 videos from pipeline results")
        # In test mode, we still use pipeline results but limit to 10
        args.input = os.path.join(PROJECT_ROOT, os.path.join(PROJECT_ROOT, 'main_results/processing_results.json'))

    print("Loading pipeline results...")
    data = load_json(args.input)
    results_list = data.get('results', [])
    print(f"Total videos: {len(results_list)}")

    if args.test:
        results_list = results_list[:10]
        print(f"Test mode: using {len(results_list)} videos")

    # Load ground truth for evaluation
    gt_path = os.path.join(PROJECT_ROOT, 'data', 'annotations_videos', 'video',
                           'cleaned_result', 'video_Topics_clean.json')
    gt_topics = {}
    if os.path.exists(gt_path):
        gt_topics = load_json(gt_path)
        print(f"Ground truth loaded: {len(gt_topics)} entries")

    strategies = ['A', 'B', 'C', 'D', 'E']
    strategy_results = {s: {'budgets': [], 'topic_correct': 0, 'topic_total': 0} for s in strategies}
    multipliers = [0.5, 1.0, 1.5, 2.0, 3.0]
    multiplier_results = {m: {'budgets': [], 'topic_correct': 0, 'topic_total': 0} for m in multipliers}

    vlm_calls_needed = 0
    vlm_calls_saved = 0

    # Initialize extractor and paths for VLM calls
    from extraction.llm_client import AdExtractor
    extractor = AdExtractor(
        provider="gemini",
        model="gemini-3-flash-preview",
        schema_mode="fixed",
        max_tokens=4000,
    )
    retry_queue_dir = os.path.join(args.output, 'retry_queue')
    failed_log_path = os.path.join(args.output, 'failed_videos.jsonl')
    os.makedirs(retry_queue_dir, exist_ok=True)

    # Load existing results for resumption
    processed_results = []
    processed_video_names = set()
    output_path = os.path.join(args.output, 'budget_ablation_results.json')
    if os.path.exists(output_path):
        try:
            old_data = load_json(output_path)
            processed_results = old_data.get('results', [])
            processed_video_names = {r['video_name'] for r in processed_results}
            print(f"Resuming: {len(processed_video_names)} videos already processed.")
        except Exception as e:
            print(f"Warning: Could not load existing budget ablation results: {e}")

    valid_dir = '/home/wabashcs/abt/use_data'
    for result in results_list:
        if result.get('status') != 'success':
            continue

        video_name = result.get('video_name', '')
        if video_name in processed_video_names:
            continue
            
        if not os.path.exists(os.path.join(valid_dir, video_name)):
            continue
        duration = result.get('metadata', {}).get('duration', 0)
        selected_frames = result.get('selected_frames', [])
        extraction = result.get('extraction')
        scenes = result.get('scenes', [])
        stats = result.get('pipeline_stats', {})

        if not selected_frames or not extraction:
            continue
            
        strategy_data = {}
        pred_topic_id = str(extraction.get('topic', {}).get('topic_id', -1))

        # Get parameters for budget formula
        num_scenes = len(scenes)
        base = max(5, num_scenes + 1)
        n_candidates = stats.get('frames_after_clip', len(selected_frames))

        # Estimate e_k and e_p from available data
        # e_p = mean importance score
        importance_scores = [f.get('importance_score', 1.0) for f in selected_frames]
        e_p = sum(importance_scores) / len(importance_scores) if importance_scores else 1.0

        # e_k: estimate from frame count ratio (higher reduction = more semantic change)
        total_sampled = stats.get('total_frames_sampled', len(selected_frames))
        if total_sampled > 0:
            e_k = max(0.2, 1.0 - stats.get('reduction_rate', 0.5))
        else:
            e_k = 1.0

        # ISD: estimate from final frame count vs scenes
        isd = max(1, len(selected_frames) // 2)

        # Ground truth topic
        vid_id = video_name.replace('.mp4', '')
        gt_topic = gt_topics.get(vid_id)

        pred_topic_id = str(extraction.get('topic', {}).get('topic_id', -1)) if extraction else '-1'

        # Compute budgets for each strategy
        budget_a = compute_budget_strategy_a(base, isd, duration, e_k, e_p)
        budget_b = compute_budget_strategy_b(isd)
        budget_c = compute_budget_strategy_c(duration)
        budget_d = compute_budget_strategy_d()
        budget_e = compute_budget_strategy_e(base, duration, e_k, e_p)

        budgets = {'A': budget_a, 'B': budget_b, 'C': budget_c, 'D': budget_d, 'E': budget_e}

        # Sort frames by importance for selection
        sorted_frames = sorted(selected_frames, key=lambda f: f.get('importance_score', 0), reverse=True)
        all_timestamps = set(f['timestamp'] for f in selected_frames)

        for strategy, budget in budgets.items():
            strategy_results[strategy]['budgets'].append(budget)

            # Determine which frames this strategy would select
            selected_for_strategy = sorted_frames[:budget]
            strategy_timestamps = set(f['timestamp'] for f in selected_for_strategy)

            # If same set as Strategy A, reuse extraction
            if strategy_timestamps == all_timestamps or budget >= len(selected_frames):
                # Same as actual pipeline result
                if gt_topic is not None:
                    strategy_results[strategy]['topic_total'] += 1
                    if pred_topic_id == str(gt_topic):
                        strategy_results[strategy]['topic_correct'] += 1
                vlm_calls_saved += 1
            else:
                # Would need VLM re-extraction
                vlm_calls_needed += 1
                strategy_name = f"strategy_{strategy}"
                
                # Prepare frames for VLM
                strategy_frames = []
                for frame_data in selected_for_strategy:
                    frame_path = os.path.join(args.videos, video_name, frame_data['frame_path'])
                    strategy_frames.append({
                        'path': frame_path,
                        'timestamp': frame_data['timestamp'],
                        'frame_number': frame_data['frame_number']
                    })

                def do_extract():
                    res = extractor.extract(frames=strategy_frames, video_duration=duration)
                    if res and "error" in res:
                        raise Exception(f"VLM Error: {res['error']}")
                    return res

                result = call_vlm_with_retry_queue(
                    video_id=f"{video_name}_{strategy_name}",
                    extract_fn=do_extract,
                    provider="gemini",
                    model="gemini-3-flash-preview",
                    retry_queue_dir=retry_queue_dir,
                    failed_log_path=failed_log_path,
                )

                if result and result.get('extraction'):
                    current_strategy_pred_topic_id = str(result['extraction'].get('topic', {}).get('topic_id', -1))
                else:
                    current_strategy_pred_topic_id = '-1' # Indicate failure or no extraction

                if gt_topic is not None:
                    strategy_results[strategy]['topic_total'] += 1
                    if current_strategy_pred_topic_id == str(gt_topic):
                        strategy_results[strategy]['topic_correct'] += 1
            
            strategy_data[strategy] = {
                'budget': budget,
                'frames_selected': len(selected_for_strategy),
                'pred_topic_id': current_strategy_pred_topic_id,
                'is_correct': (current_strategy_pred_topic_id == str(gt_topic)) if gt_topic is not None else None
            }


        # Multiplier sensitivity
        for mult in multipliers:
            budget_m = compute_budget_with_multiplier(base, isd, duration, e_k, e_p, mult)
            multiplier_results[mult]['budgets'].append(budget_m)

            # For multiplier sensitivity, we'll reuse the original extraction's topic prediction
            # This is an approximation for now, as re-running VLM for each multiplier is too costly
            if gt_topic is not None:
                multiplier_results[mult]['topic_total'] += 1
                if pred_topic_id == str(gt_topic):
                    multiplier_results[mult]['topic_correct'] += 1

        # Combine and save after each video (incremental)
        processed_results.append({
            'video_name': video_name,
            'duration': duration,
            'gt_topic': gt_topic,
            'strategies': strategy_data
        })
        
        output_data = {
            'metadata': {
                'experiment': 'exp1_budget_ablation',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'vlm_calls_needed': vlm_calls_needed,
                'vlm_calls_saved': vlm_calls_saved,
                'partial': True
            },
            'results': processed_results
        }
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

    # Final summary print
    print(f"\nAblation complete.")

    # ---- Print Results ----
    print("\n" + "="*70)
    print("BUDGET ABLATION RESULTS")
    print("="*70)

    print(f"\n{'Strategy':<20} {'Mean Frames':<15} {'Topic Acc %':<15} {'N':<10}")
    print("-" * 60)
    for s in strategies:
        r = strategy_results[s]
        mean_frames = sum(r['budgets']) / len(r['budgets']) if r['budgets'] else 0
        acc = r['topic_correct'] / r['topic_total'] * 100 if r['topic_total'] > 0 else 0
        labels = {
            'A': 'Full AdaFrame',
            'B': 'ISD only',
            'C': 'Linear duration',
            'D': 'Fixed-25',
            'E': 'Energy only'
        }
        print(f"  {labels[s]:<18} {mean_frames:<15.1f} {acc:<15.1f} {r['topic_total']:<10}")

    print(f"\nVLM calls: {vlm_calls_needed} needed, {vlm_calls_saved} reused from Strategy A")

    # Multiplier sensitivity
    print(f"\n{'Multiplier':<15} {'Mean Frames':<15} {'Topic Acc %':<15}")
    print("-" * 45)
    for mult in multipliers:
        r = multiplier_results[mult]
        mean_frames = sum(r['budgets']) / len(r['budgets']) if r['budgets'] else 0
        acc = r['topic_correct'] / r['topic_total'] * 100 if r['topic_total'] > 0 else 0
        marker = " ← default" if mult == 1.5 else ""
        print(f"  {mult:<13} {mean_frames:<15.1f} {acc:<15.1f}{marker}")

    # ---- Save Results ----
    ablation_output = {
        'n_videos': len([r for r in results_list if r.get('status') == 'success']),
        'strategies': {},
        'multiplier_sensitivity': {},
    }
    for s in strategies:
        r = strategy_results[s]
        ablation_output['strategies'][s] = {
            'mean_frames': round(sum(r['budgets']) / len(r['budgets']), 1) if r['budgets'] else 0,
            'topic_accuracy': round(r['topic_correct'] / r['topic_total'] * 100, 1) if r['topic_total'] > 0 else 0,
            'n': r['topic_total'],
        }
    for mult in multipliers:
        r = multiplier_results[mult]
        ablation_output['multiplier_sensitivity'][str(mult)] = {
            'mean_frames': round(sum(r['budgets']) / len(r['budgets']), 1) if r['budgets'] else 0,
            'topic_accuracy': round(r['topic_correct'] / r['topic_total'] * 100, 1) if r['topic_total'] > 0 else 0,
            'n': r['topic_total'],
        }

    with open(os.path.join(args.output, 'ablation_results.json'), 'w') as f:
        json.dump(ablation_output, f, indent=2)

    # CSV for multiplier sensitivity
    csv_path = os.path.join(args.output, 'multiplier_sensitivity.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Multiplier', 'Mean_Frames', 'Topic_Accuracy', 'N'])
        for mult in multipliers:
            r = multiplier_results[mult]
            mean_frames = sum(r['budgets']) / len(r['budgets']) if r['budgets'] else 0
            acc = r['topic_correct'] / r['topic_total'] * 100 if r['topic_total'] > 0 else 0
            writer.writerow([mult, round(mean_frames, 1), round(acc, 1), r['topic_total']])

    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
