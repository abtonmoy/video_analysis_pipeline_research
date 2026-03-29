#!/usr/bin/env python3
"""Extract results for videos listed in videos.csv from processing_results.json"""

import json
import csv
from pathlib import Path

# Read video names from CSV
videos_csv_path = Path("/home/wabashcs/abt/video_analysis_pipeline_research-main/videos.csv")
with open(videos_csv_path, 'r') as f:
    video_names = [line.strip() for line in f if line.strip()]

print(f"Looking for {len(video_names)} videos from CSV")

# Create a set for faster lookup
target_videos = set(video_names)

# Read the large JSON file
results_path = Path("/home/wabashcs/abt/video_analysis_pipeline_research-main/main_results/processing_results.json")
output_path = Path("/home/wabashcs/abt/video_analysis_pipeline_research-main/filtered_results.json")

# Load and filter
with open(results_path, 'r') as f:
    data = json.load(f)

# The JSON has structure: {"metadata": {...}, "results": [...]}
results_list = data.get('results', [])
print(f"Total entries in processing_results.json: {len(results_list)}")

# Filter results
filtered_results = []
for result in results_list:
    video_name = result.get('video_name', '')
    if video_name in target_videos:
        filtered_results.append(result)

print(f"Found {len(filtered_results)} matching videos")

# Save filtered results with same structure
output_data = {
    "metadata": {
        "source": "filtered from processing_results.json",
        "total_videos": len(filtered_results),
        "filtered_for": "use_data videos"
    },
    "results": filtered_results
}

with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Saved filtered results to: {output_path}")

# Also create a summary CSV
summary_path = Path("/home/wabashcs/abt/video_analysis_pipeline_research-main/video_results_summary.csv")
with open(summary_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['video_name', 'status', 'brand', 'product', 'ad_type', 'reduction_rate'])
    
    for result in sorted(filtered_results, key=lambda x: x.get('video_name', '')):
        video_name = result.get('video_name', '')
        status = result.get('status', 'unknown')
        
        if status == 'success':
            extraction = result.get('extraction', {})
            brand = extraction.get('brand', '')
            product = extraction.get('product', '')
            ad_type = extraction.get('ad_type', '')
            
            # Get reduction rate from pipeline_stats
            stats = result.get('pipeline_stats', {})
            reduction = stats.get('reduction_rate', '')
        else:
            brand = product = ad_type = ''
            reduction = ''
        
        writer.writerow([video_name, status, brand, product, ad_type, reduction])

print(f"Saved summary CSV to: {summary_path}")
