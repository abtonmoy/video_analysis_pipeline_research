#!/usr/bin/env python3
"""Generate shared 50 video IDs for experiments 6 and 7."""

import random
import json
import os

def main():
    random.seed(42)

    # Load benchmark results to get video IDs
    with open('test_results/benchmark_results.json') as f:
        data = json.load(f)

    per_video = data.get('per_video', data)
    all_videos = [k for k in per_video.keys() if k != 'metadata']
    print(f"Total benchmark videos: {len(all_videos)}")

    selected_50 = random.sample(all_videos, min(50, len(all_videos)))
    print(f"Selected {len(selected_50)} videos")

    output_path = 'new_experiments/scripts/shared_50_video_ids.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(selected_50, f, indent=2)

    print(f"Saved to {output_path}")
    for vid in selected_50[:5]:
        print(f"  {vid}")
    print(f"  ... ({len(selected_50)} total)")


if __name__ == '__main__':
    main()
