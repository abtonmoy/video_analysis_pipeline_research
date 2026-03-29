#!/usr/bin/env python3
"""
Experiment 8: Limitations Update

Reads results from Experiments 6 and 7, then generates the appropriate
paragraph for the limitations section based on what succeeded.
"""

import json
import os
import sys
import argparse
import math
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_accuracy_with_ci(results, gt_topics):
    """Compute topic accuracy with 95% CI using Wilson score interval."""
    correct = 0
    total = 0

    for vid_name, vid_data in results.items():
        extraction = vid_data.get('extraction', {})
        if not extraction or 'error' in extraction:
            continue

        pred_topic = str(extraction.get('topic', {}).get('topic_id', -1))
        vid_id = vid_name.replace('.mp4', '')

        gt = gt_topics.get(vid_id)
        if gt is None:
            continue

        total += 1
        if pred_topic == str(gt):
            correct += 1

    if total == 0:
        return 0, 0, 0, 0

    p = correct / total

    # Wilson score interval for 95% CI
    z = 1.96
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

    ci_low = max(0, center - margin) * 100
    ci_high = min(1, center + margin) * 100

    return p * 100, ci_low, ci_high, total


def main():
    parser = argparse.ArgumentParser(description='Experiment 8: Limitations update')
    parser.add_argument('--claude-results', default='new_experiments/results/claude_50/claude_results.json')
    parser.add_argument('--native-results', default='new_experiments/results/native_video_50/native_video_results.json')
    parser.add_argument('--pipeline-results', default='main_results/processing_results.json')
    parser.add_argument('--gt', default='data/annotations_videos/video/cleaned_result/video_Topics_clean.json')
    parser.add_argument('--output', default='new_experiments/results/')
    args = parser.parse_args()

    # Load ground truth
    gt_topics = load_json(args.gt)
    if gt_topics is None:
        gt_topics = {}
        print("WARNING: No ground truth found")

    # Load pipeline results for Gemini baseline accuracy on same videos
    pipeline_data = load_json(args.pipeline_results)
    pipeline_results = {}
    if pipeline_data:
        for r in pipeline_data.get('results', []):
            if r.get('extraction'):
                pipeline_results[r['video_name']] = r

    # Load Claude results
    claude_raw_data = load_json(args.claude_results)
    claude_results = {}
    if claude_raw_data:
        claude_list = claude_raw_data.get('results', [])
        claude_results = {row.get('video_name', ''): row for row in claude_list if row.get('video_name')}

    # Load native video results
    native_raw_data = load_json(args.native_results)
    native_results = {}
    if native_raw_data:
        native_list = native_raw_data.get('results', [])
        native_results = {row.get('video_name', ''): row for row in native_list if row.get('video_name')}

    # Check if experiments succeeded
    claude_ok = claude_results is not None and len(claude_results) > 0
    native_ok = native_results is not None and len(native_results) > 0

    print(f"Claude results: {'✓ ' + str(len(claude_results)) + ' videos' if claude_ok else '✗ not available'}")
    print(f"Native results: {'✓ ' + str(len(native_results)) + ' videos' if native_ok else '✗ not available'}")

    output_path = os.path.join(args.output, 'limitations_update.tex')

    if claude_ok and native_ok:
        # Both succeeded
        claude_acc, claude_lo, claude_hi, claude_n = compute_accuracy_with_ci(claude_results, gt_topics)
        native_acc, native_lo, native_hi, native_n = compute_accuracy_with_ci(native_results, gt_topics)

        # Get Gemini accuracy on same videos
        shared_vids = set(claude_results.keys()) & set(native_results.keys())
        gemini_subset = {k: {'extraction': pipeline_results[k].get('extraction', {})}
                        for k in shared_vids if k in pipeline_results}
        gemini_acc, _, _, gemini_n = compute_accuracy_with_ci(gemini_subset, gt_topics)

        text = f"""%% EXPERIMENT 8: Limitations Update — Both VLMs succeeded
We validated AdaFrame with Claude Haiku~4.5 on {claude_n} videos, achieving {claude_acc:.1f}\\%
topic accuracy (95\\% CI: [{claude_lo:.1f}\\%, {claude_hi:.1f}\\%]) compared to {gemini_acc:.1f}\\% for Gemini-3-Flash,
supporting provider-agnostic applicability. Gemini's native video input mode
achieves {native_acc:.1f}\\% topic accuracy (95\\% CI: [{native_lo:.1f}\\%, {native_hi:.1f}\\%])
on the same {native_n} videos, suggesting that frame-level analysis remains competitive
with native video understanding."""

        print(f"\nClaude accuracy: {claude_acc:.1f}% (N={claude_n})")
        print(f"Native video accuracy: {native_acc:.1f}% (N={native_n})")
        print(f"Gemini (same videos): {gemini_acc:.1f}% (N={gemini_n})")

    elif claude_ok:
        claude_acc, claude_lo, claude_hi, claude_n = compute_accuracy_with_ci(claude_results, gt_topics)

        gemini_subset = {k: {'extraction': pipeline_results[k].get('extraction', {})}
                        for k in claude_results.keys() if k in pipeline_results}
        gemini_acc, _, _, gemini_n = compute_accuracy_with_ci(gemini_subset, gt_topics)

        text = f"""%% EXPERIMENT 8: Limitations Update — Claude succeeded, native video failed
We validated AdaFrame with Claude Haiku~4.5 on {claude_n} videos, achieving {claude_acc:.1f}\\%
topic accuracy (95\\% CI: [{claude_lo:.1f}\\%, {claude_hi:.1f}\\%]) compared to {gemini_acc:.1f}\\% for Gemini-3-Flash,
supporting provider-agnostic applicability."""

    elif native_ok:
        native_acc, native_lo, native_hi, native_n = compute_accuracy_with_ci(native_results, gt_topics)

        text = f"""%% EXPERIMENT 8: Limitations Update — Native video succeeded, Claude failed
Gemini's native video input mode achieves {native_acc:.1f}\\% topic accuracy
(95\\% CI: [{native_lo:.1f}\\%, {native_hi:.1f}\\%]) on {native_n} videos. While the cascade is
provider-agnostic by design, multi-VLM evaluation beyond Gemini is a priority for future work."""

    else:
        # Neither succeeded
        text = """%% EXPERIMENT 8: Limitations Update — Neither VLM experiment succeeded
All experiments use Gemini-3-Flash. While AdaFrame's cascade is provider-agnostic
by design, multi-VLM evaluation is a priority for future work."""

    with open(output_path, 'w') as f:
        f.write(text + "\n")

    print(f"\nLimitations text saved to {output_path}")
    print("\n--- Generated Text ---")
    print(text)


if __name__ == '__main__':
    main()
