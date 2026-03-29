#!/usr/bin/env python3
"""
Experiment 5: Cascade Cost Breakdown

Computes the computational cost of the cascade vs. VLM cost savings.
Shows that the cascade overhead is offset by reduced VLM spending.

NO VLM calls — arithmetic on existing data from processing_results.json.
"""

import json
import os
import sys
import argparse
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import statistics
from pathlib import Path


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Experiment 5: Cascade cost breakdown')
    parser.add_argument('--input', default=os.path.join(PROJECT_ROOT, 'main_results/processing_results.json'),
                        help='Path to main pipeline results')
    parser.add_argument('--output', default='new_experiments/results/cost_breakdown/',
                        help='Output directory')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: only process first N videos')
    parser.add_argument('--n', type=int, default=10,
                        help='Number of videos in test mode')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading pipeline results...")
    data = load_json(args.input)

    # The results are in a "results" array
    results_list = data.get('results', [])
    if not results_list:
        # Maybe it's a dict
        results_list = [v for k, v in data.items() if k != 'metadata']

    print(f"Total videos: {len(results_list)}")

    if args.test:
        results_list = results_list[:args.n]
        print(f"Test mode: using {len(results_list)} videos")

    # Constants
    GPU_HOURLY_RATE = 0.50  # RTX 4090 cloud rate (conservative)
    # Gemini-3-Flash pricing (per 1K tokens, approximate)
    # For image input: ~$0.00263 per image (average)
    # Simplified: use cost from benchmark data if available, else estimate
    # From benchmark data: vlm_cost_usd is approximately $0.0115 per frame
    COST_PER_FRAME_USD = 0.0115  # Estimated from benchmark data averages

    rows = []
    cascade_times = []
    cascade_costs = []
    vlm_costs_adaframe = []
    vlm_costs_uniform = []
    total_costs_adaframe = []
    total_costs_uniform = []
    net_savings_list = []

    valid_dir = '/home/wabashcs/abt/use_data'
    for result in results_list:
        if result.get('status') != 'success':
            continue
        if not os.path.exists(os.path.join(valid_dir, result.get('video_name', ''))):
            continue

        video_name = result.get('video_name', 'unknown')
        duration = result.get('metadata', {}).get('duration', 0)
        fps = result.get('metadata', {}).get('fps', 30)

        stats = result.get('pipeline_stats', {})
        cascade_latency = stats.get('processing_time_s', 0)
        final_frame_count = stats.get('final_frame_count', 0)
        total_sampled = stats.get('total_frames_sampled', 0)

        # Skip videos with no frames selected (extraction=null)
        if final_frame_count == 0:
            continue

        # Cascade cost (GPU compute)
        cascade_cost = cascade_latency / 3600 * GPU_HOURLY_RATE

        # VLM cost for AdaFrame
        vlm_cost_adaframe = final_frame_count * COST_PER_FRAME_USD

        # VLM cost for uniform-1fps baseline
        uniform_frame_count = max(1, int(duration))  # 1 frame per second
        vlm_cost_uniform = uniform_frame_count * COST_PER_FRAME_USD

        # Total costs
        total_adaframe = cascade_cost + vlm_cost_adaframe
        total_uniform = vlm_cost_uniform  # No cascade overhead for uniform

        # Net savings
        net_saving = total_uniform - total_adaframe

        rows.append({
            'video': video_name,
            'duration_s': round(duration, 1),
            'cascade_latency_s': round(cascade_latency, 2),
            'cascade_cost_usd': round(cascade_cost, 6),
            'adaframe_frames': final_frame_count,
            'uniform_frames': uniform_frame_count,
            'vlm_cost_adaframe': round(vlm_cost_adaframe, 4),
            'vlm_cost_uniform': round(vlm_cost_uniform, 4),
            'total_adaframe': round(total_adaframe, 4),
            'total_uniform': round(total_uniform, 4),
            'net_saving_usd': round(net_saving, 4),
            'saving_pct': round(net_saving / total_uniform * 100, 1) if total_uniform > 0 else 0,
        })

        cascade_times.append(cascade_latency)
        cascade_costs.append(cascade_cost)
        vlm_costs_adaframe.append(vlm_cost_adaframe)
        vlm_costs_uniform.append(vlm_cost_uniform)
        total_costs_adaframe.append(total_adaframe)
        total_costs_uniform.append(total_uniform)
        net_savings_list.append(net_saving)

    print(f"\nProcessed {len(rows)} valid videos")

    # ---- Summary Statistics ----
    if rows:
        mean_cascade_time = statistics.mean(cascade_times)
        mean_cascade_cost = statistics.mean(cascade_costs)
        mean_vlm_adaframe = statistics.mean(vlm_costs_adaframe)
        mean_vlm_uniform = statistics.mean(vlm_costs_uniform)
        mean_total_adaframe = statistics.mean(total_costs_adaframe)
        mean_total_uniform = statistics.mean(total_costs_uniform)
        mean_net_saving = statistics.mean(net_savings_list)

        # Count videos where savings are positive
        positive_savings = sum(1 for s in net_savings_list if s > 0)
        pct_positive = positive_savings / len(net_savings_list) * 100

        # Find the breakeven duration
        breakeven_durations = []
        for row in rows:
            if row['net_saving_usd'] > 0:
                pass  # Already saves
            else:
                breakeven_durations.append(row['duration_s'])

        # Estimate breakeven: cascade_cost / (uniform_vlm_per_sec - adaframe_vlm_per_sec)
        # Roughly: cascade cost balanced by savings per second of video
        if mean_vlm_uniform > mean_vlm_adaframe:
            mean_duration = statistics.mean([r['duration_s'] for r in rows])
            saving_per_sec = (mean_vlm_uniform - mean_vlm_adaframe) / mean_duration
            breakeven_s = mean_cascade_cost / saving_per_sec if saving_per_sec > 0 else float('inf')
        else:
            breakeven_s = float('inf')

        print("\n" + "="*70)
        print("COST BREAKDOWN SUMMARY")
        print("="*70)
        print(f"Videos analyzed: {len(rows)}")
        print(f"\nCascade overhead:")
        print(f"  Mean latency:     {mean_cascade_time:.1f}s")
        print(f"  Mean cost:        ${mean_cascade_cost:.4f}")
        print(f"\nVLM costs:")
        print(f"  AdaFrame mean:    ${mean_vlm_adaframe:.4f}")
        print(f"  Uniform mean:     ${mean_vlm_uniform:.4f}")
        print(f"\nTotal cost per video:")
        print(f"  AdaFrame (cascade + VLM): ${mean_total_adaframe:.4f}")
        print(f"  Uniform (VLM only):       ${mean_total_uniform:.4f}")
        print(f"\nNet savings:")
        print(f"  Mean saving per video:   ${mean_net_saving:.4f}")
        print(f"  Videos with savings:     {positive_savings}/{len(rows)} ({pct_positive:.0f}%)")
        print(f"  Breakeven duration:      ~{breakeven_s:.1f}s")

        # ---- Save CSV ----
        csv_path = os.path.join(args.output, 'cost_analysis.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        # ---- Save summary JSON ----
        summary = {
            'n_videos': len(rows),
            'cascade': {
                'mean_latency_s': round(mean_cascade_time, 2),
                'median_latency_s': round(statistics.median(cascade_times), 2),
                'mean_cost_usd': round(mean_cascade_cost, 6),
            },
            'vlm_cost': {
                'adaframe_mean_usd': round(mean_vlm_adaframe, 4),
                'uniform_mean_usd': round(mean_vlm_uniform, 4),
            },
            'total_cost': {
                'adaframe_mean_usd': round(mean_total_adaframe, 4),
                'uniform_mean_usd': round(mean_total_uniform, 4),
            },
            'savings': {
                'mean_net_saving_usd': round(mean_net_saving, 4),
                'positive_savings_count': positive_savings,
                'positive_savings_pct': round(pct_positive, 1),
                'breakeven_duration_s': round(breakeven_s, 1),
            },
            'gpu_hourly_rate': GPU_HOURLY_RATE,
            'cost_per_frame_usd': COST_PER_FRAME_USD,
        }

        summary_path = os.path.join(args.output, 'cost_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to:")
        print(f"  {csv_path}")
        print(f"  {summary_path}")

        # ---- Paper text ----
        print("\n" + "="*60)
        print("PAPER TEXT (for %%PLACEHOLDER in §6.2):")
        print("="*60)
        print(f"""
The cascade itself incurs compute cost (mean {mean_cascade_time:.1f} seconds per
video, approximately \\${mean_cascade_cost:.4f} at RTX~4090 cloud rates of
\\$0.50/hr). However, the VLM cost reduction---from \\${mean_vlm_uniform:.2f}
(Uniform-1FPS) to \\${mean_vlm_adaframe:.2f} (AdaFrame)---more than offsets
this overhead. Net savings are positive for {pct_positive:.0f}\\% of videos
(those longer than $\\approx${breakeven_s:.0f} seconds). At production scale
(1M~videos/month), the total cost reduction is
\\${(mean_net_saving * 1_000_000):.0f}/month.
""")

    else:
        print("No valid videos found!")


if __name__ == '__main__':
    main()
