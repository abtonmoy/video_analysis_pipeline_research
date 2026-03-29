#!/usr/bin/env python3
"""
Experiment 4: Error-Free Intersection Accuracy

Finds the subset of benchmark videos where ALL methods produced valid
(non-error) extractions, then recomputes topic accuracy per method on
that intersection set. This ensures a fair apples-to-apples comparison.

NO VLM calls needed — pure analysis of existing data.
"""

import json
import os
import sys
import argparse
import csv
from pathlib import Path


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_ground_truth(gt_dir):
    """Load ground truth topic annotations."""
    gt_path = os.path.join(gt_dir, "annotations_videos", "video", "cleaned_result", "video_Topics_clean.json")
    if not os.path.exists(gt_path):
        # Try alternate path
        gt_path = os.path.join(gt_dir, "video_Topics_clean.json")
    return load_json(gt_path)


def normalize_vid_id(vid_name):
    """Strip .mp4 and normalize video ID to match ground truth keys."""
    vid = vid_name.replace('.mp4', '')
    return vid


def find_gt_key(vid, gt_topics):
    """Try multiple normalizations to find the GT key."""
    if vid in gt_topics:
        return vid
    # Try replacing first underscore with hyphen
    alt = vid.replace('_', '-', 1)
    if alt in gt_topics:
        return alt
    # Try last 11 chars
    if len(vid) > 11:
        suffix = vid[-11:]
        if suffix in gt_topics:
            return suffix
    return None


def main():
    parser = argparse.ArgumentParser(description='Experiment 4: Error-free intersection accuracy')
    parser.add_argument('--input', default='test_results/benchmark_results.json',
                        help='Path to benchmark results JSON')
    parser.add_argument('--gt', default='data/',
                        help='Path to ground truth directory')
    parser.add_argument('--dsnet', default='new_experiments/results/dsnet/dsnet_results.json',
                        help='Path to DSNet results JSON')
    parser.add_argument('--pyscenedetect', default='new_experiments/results/pyscenedetect/pyscenedetect_results.json',
                        help='Path to PySceneDetect results JSON')
    parser.add_argument('--output', default='new_experiments/results/error_intersection/',
                        help='Output directory for intersection results')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading benchmark results...")
    benchmark = load_json(args.input)
    per_video = benchmark.get('per_video', benchmark)

    # Remove metadata key if present
    per_video = {k: v for k, v in per_video.items() if k != 'metadata'}

    print(f"Total benchmark videos: {len(per_video)}")

    # Load ground truth
    print("Loading ground truth...")
    gt_topics = get_ground_truth(args.gt)
    print(f"Ground truth entries: {len(gt_topics)}")

    # Define the methods we compare (matching agents.md)
    methods = ['uniform_1fps', 'random', 'histogram', 'orb', 'optical_flow',
               'clip_only', 'kmeans', 'hib_pipeline']

    # Also check if static_pipeline exists
    sample_vid = list(per_video.values())[0]
    available_baselines = list(sample_vid.get('baselines', {}).keys())
    print(f"Available baselines: {available_baselines}")

    # Use only methods that actually exist in data
    methods = [m for m in methods if m in available_baselines]
    # Add hib_pipeline as our method (AdaFrame)
    if 'static_pipeline' in available_baselines and 'static_pipeline' not in methods:
        methods.append('static_pipeline')

    print(f"Methods to compare: {methods}")

    if args.test:
        video_ids = list(per_video.keys())[:args.n]
        per_video = {k: per_video[k] for k in video_ids}
        print(f"Test mode: using {len(per_video)} videos")

    # ---- Step 1: Find videos where ALL methods produced non-error extractions ----
    print("\nFinding error-free intersection...")

    # We'll check both bare_extraction and full_extraction
    intersection_vids = []
    error_counts = {m: 0 for m in methods}

    for vid_name, vid_data in per_video.items():
        baselines = vid_data.get('baselines', {})
        all_ok = True

        for method in methods:
            method_data = baselines.get(method, {})

            # Check if method has an error at top level
            if 'error' in method_data:
                error_counts[method] += 1
                all_ok = False
                continue

            # Check full_extraction for errors
            full_ext = method_data.get('full_extraction', {})
            if not full_ext or 'error' in full_ext:
                error_counts[method] += 1
                all_ok = False
                continue

            # Check if topic exists
            topic = full_ext.get('topic', {})
            if not topic or 'topic_id' not in topic:
                error_counts[method] += 1
                all_ok = False
                continue

        if all_ok:
            intersection_vids.append(vid_name)

    # ---- Step 1a: Merge and Intersect with External Baselines (DSNet/PySceneDetect) ----
    external_vids = {}
    if os.path.exists(args.dsnet):
        try:
            dsnet_data = load_json(args.dsnet)
            dsnet_results = {os.path.basename(r['video_name']): r for r in dsnet_data.get('results', [])}
            external_vids['dsnet'] = dsnet_results
            print(f"Loaded DSNet results: {len(dsnet_results)} videos")
        except Exception:
            print("Warning: Failed to load DSNet results")

    if os.path.exists(args.pyscenedetect):
        try:
            psd_data = load_json(args.pyscenedetect)
            psd_results = {os.path.basename(r['video_name']): r for r in psd_data.get('results', [])}
            external_vids['pyscenedetect'] = psd_results
            print(f"Loaded PySceneDetect results: {len(psd_results)} videos")
        except Exception:
            print("Warning: Failed to load PySceneDetect results")

    if external_vids:
        methods_with_external = methods + list(external_vids.keys())
        prev_count = len(intersection_vids)
        new_intersection = []
        for vid in intersection_vids:
            all_external_ok = True
            for m in external_vids:
                if vid not in external_vids[m] or external_vids[m][vid].get('status') != 'success':
                    all_external_ok = False
                    break
            if all_external_ok:
                new_intersection.append(vid)
        intersection_vids = new_intersection
        print(f"Intersection reduced from {prev_count} to {len(intersection_vids)} after adding {list(external_vids.keys())}")
        methods = methods_with_external
        # Update results dict to include external
        for m, count_data in external_vids.items():
            error_counts[m] = len(per_video) - len(count_data) # Approximate
    else:
        print("No external Phase 2 results found. Running on 8 traditional baselines.")

    print(f"\nError-free intersection: {len(intersection_vids)} / {len(per_video)} videos")
    print(f"\nPer-method error counts:")
    for method, count in error_counts.items():
        error_rate = count / len(per_video) * 100 if per_video else 0
        print(f"  {method}: {count} errors ({error_rate:.1f}%)")

    # ---- Step 2: Recompute topic accuracy on intersection ----
    print("\n--- Accuracy on error-free intersection ---")

    results = {}
    for method in methods:
        correct = 0
        total = 0
        valid_with_gt = 0

        for vid_name in intersection_vids:
            vid_data = per_video[vid_name]
            baselines = vid_data.get('baselines', {})
            # Map external results if needed
            if method in external_vids:
                method_data = external_vids[method].get(vid_name, {})
                full_ext = method_data.get('extraction', {})
            else:
                method_data = baselines.get(method, {})
                full_ext = method_data.get('full_extraction', {})
            
            if not full_ext or 'error' in full_ext:
                continue

            pred_topic_id = str(full_ext.get('topic', {}).get('topic_id', -1))

            # Find ground truth
            vid_id = normalize_vid_id(vid_name)
            gt_key = find_gt_key(vid_id, gt_topics)

            if gt_key is None:
                continue

            gt_topic = str(gt_topics[gt_key])
            valid_with_gt += 1

            if pred_topic_id == gt_topic:
                correct += 1
            total += 1

        accuracy = correct / total * 100 if total > 0 else 0

        results[method] = {
            'method': method,
            'intersection_n': len(intersection_vids),
            'valid_with_gt': valid_with_gt,
            'correct': correct,
            'total': total,
            'topic_accuracy': round(accuracy, 1)
        }

        print(f"  {method:20s}: {accuracy:5.1f}% ({correct}/{total}, N_gt={valid_with_gt})")

    # ---- Step 3: Also compute on full set for comparison ----
    print("\n--- Accuracy on FULL set (including errors) ---")

    full_results = {}
    for method in methods:
        correct = 0
        total = 0

        for vid_name, vid_data in per_video.items():
            baselines = vid_data.get('baselines', {})
            method_data = baselines.get(method, {})

            full_ext = method_data.get('full_extraction', {})
            if not full_ext or 'error' in full_ext:
                continue

            pred_topic_id = str(full_ext.get('topic', {}).get('topic_id', -1))

            vid_id = normalize_vid_id(vid_name)
            gt_key = find_gt_key(vid_id, gt_topics)

            if gt_key is None:
                continue

            gt_topic = str(gt_topics[gt_key])
            total += 1

            if pred_topic_id == gt_topic:
                correct += 1

        accuracy = correct / total * 100 if total > 0 else 0

        full_results[method] = {
            'method': method,
            'correct': correct,
            'total': total,
            'topic_accuracy': round(accuracy, 1)
        }

        print(f"  {method:20s}: {accuracy:5.1f}% ({correct}/{total})")

    # ---- Step 4: Save results ----
    output_data = {
        'intersection_size': len(intersection_vids),
        'total_videos': len(per_video),
        'intersection_video_ids': intersection_vids,
        'error_counts': error_counts,
        'intersection_accuracy': results,
        'full_accuracy': full_results,
    }

    output_path = os.path.join(args.output, 'intersection_accuracy.json')
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Also save CSV
    csv_path = os.path.join(args.output, 'intersection_accuracy.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Intersection_N', 'Valid_With_GT', 'Correct', 'Total',
                         'Intersection_Accuracy', 'Full_Accuracy'])
        for method in methods:
            r = results[method]
            fr = full_results[method]
            writer.writerow([
                method, r['intersection_n'], r['valid_with_gt'],
                r['correct'], r['total'], r['topic_accuracy'], fr['topic_accuracy']
            ])

    print(f"\nResults saved to:")
    print(f"  {output_path}")
    print(f"  {csv_path}")

    # Print summary for paper
    print("\n" + "="*60)
    print("PAPER TEXT (for %%PLACEHOLDER in §5.1):")
    print("="*60)

    rankings_changed = False
    int_methods_by_acc = sorted(results.items(), key=lambda x: x[1]['topic_accuracy'], reverse=True)
    full_methods_by_acc = sorted(full_results.items(), key=lambda x: x[1]['topic_accuracy'], reverse=True)
    int_ranking = [x[0] for x in int_methods_by_acc]
    full_ranking = [x[0] for x in full_methods_by_acc]

    if int_ranking[:3] != full_ranking[:3]:
        rankings_changed = True

    if rankings_changed:
        top_method = int_methods_by_acc[0][0]
        top_acc = int_methods_by_acc[0][1]['topic_accuracy']
        print(f"""
Restricting to the $N={len(intersection_vids)}$ videos where all methods
produced valid extractions \\textit{{changes}} the ranking: {top_method}
leads with {top_acc}\\% topic accuracy on this error-free subset.
""")
    else:
        our_int_acc = results.get('hib_pipeline', results.get('static_pipeline', {})).get('topic_accuracy', 0)
        print(f"""
Restricting to the $N={len(intersection_vids)}$ videos where all methods
produced valid extractions does not change the overall ranking.
AdaFrame achieves {our_int_acc}\\% topic accuracy on this error-free subset,
confirming that the main results are not biased by differential error rates.
""")


if __name__ == '__main__':
    main()
