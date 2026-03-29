#!/usr/bin/env python3
"""
Experiment 7: Gemini Native Video Mode (500 videos)

Uses Gemini's native video input mode instead of sending extracted frames.
Uploads the full video to Gemini and uses the standard research prompt/schema.
"""

import json
import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'new_experiments', 'src_copy'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'new_experiments', 'scripts'))

from vlm_retry_utils import call_vlm_with_retry_queue, print_run_summary
from extraction.llm_client import AdExtractor, parse_json_response


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_video_duration(video_path):
    """Get video duration in seconds."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return total_frames / fps if fps > 0 else 0


def build_native_video_prompt(video_duration):
    """Build prompt for native video mode (no individual frame references)."""
    from extraction.prompts import get_topic_reference, get_sentiment_reference
    from extraction.schema import get_schema

    schema = get_schema(mode="fixed")
    enhanced_schema = {"ad_type": "string (one of: product_demo, testimonial, brand_awareness, tutorial, entertainment)"}
    enhanced_schema.update(schema)

    prompt = f"""You are analyzing a {video_duration:.1f}-second video advertisement.

Watch the entire video and extract structured information about this advertisement.

ANALYSIS APPROACH:
1. Identify the brand/product being advertised
2. Note key visual elements, text overlays, and calls to action
3. Assess the overall tone, target audience, and effectiveness
4. Classify the topic and sentiment

{get_topic_reference()}

{get_sentiment_reference()}

Extract the following information in JSON format:

{json.dumps(enhanced_schema, indent=2)}

IMPORTANT FORMATTING RULES:
- Respond with ONLY valid JSON, no markdown code blocks or explanations
- Use null for fields where information is not available
- Be specific and concise in your descriptions
- For topic_id and sentiment IDs, use INTEGER values

JSON Response:"""

    return prompt


def main():
    parser = argparse.ArgumentParser(description='Experiment 7: Gemini native video (500 videos)')
    parser.add_argument('--video-ids', default='new_experiments/scripts/shared_50_video_ids.json',
                        help='JSON file with shared 50 video IDs')
    parser.add_argument('--output', default='new_experiments/results/native_video_50/',
                        help='Output directory')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--videos', default='/home/wabashcs/abt/use_data',
                        help='Video directory')
    parser.add_argument('--n', type=int, default=3,
                        help='Number of videos in test mode')
    parser.add_argument('--retry', default=None)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    retry_queue_dir = os.path.join(args.output, 'retry_queue')
    failed_log_path = os.path.join(args.output, 'failed_videos.jsonl')

    # Load video list
    if args.test:
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
    results_path = os.path.join(args.output, 'native_video_results.json')
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
            duration = get_video_duration(video_path)
            prompt = build_native_video_prompt(duration)

            start_time = time.time()

            def do_extract():
                from google import genai
                from google.genai import types
                import os
                import time

                api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                client = genai.Client(api_key=api_key)

                # Upload video
                video_file = client.files.upload(file=video_path)
                
                # Wait for processing
                while video_file.state.name == "PROCESSING":
                    time.sleep(2)
                    video_file = client.files.get(name=video_file.name)

                if video_file.state.name == "FAILED":
                    raise RuntimeError(f"Video processing failed: {video_file.state.name}")

                config = types.GenerateContentConfig(
                    max_output_tokens=4000,
                    temperature=0.0,
                    response_mime_type="application/json",
                    safety_settings=[
                        types.SafetySetting(category=cat, threshold="BLOCK_NONE")
                        for cat in [
                            "HARM_CATEGORY_HARASSMENT",
                            "HARM_CATEGORY_HATE_SPEECH",
                            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "HARM_CATEGORY_CIVIC_INTEGRITY",
                        ]
                    ]
                )

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[video_file, prompt],
                    config=config
                )

                # Check for safety blocks
                if not response.candidates:
                    return {"error": "SAFETY_BLOCKED", "message": "No candidates (possible block)"}

                # Clean up immediately
                try:
                    client.files.delete(name=video_file.name)
                except Exception:
                    pass

                # Use the robust parser from llm_client
                return parse_json_response(response.text)

            result = call_vlm_with_retry_queue(
                video_id=video_name,
                extract_fn=do_extract,
                provider="gemini-native",
                model="gemini-2.5-flash",
                retry_queue_dir=retry_queue_dir,
                failed_log_path=failed_log_path,
            )

            latency = time.time() - start_time

            if result is not None:
                results.append({
                    'status': 'success',
                    'video_name': video_name,
                    'duration': round(duration, 2),
                    'latency_s': round(latency, 2),
                    'extraction': result,
                })
                succeeded += 1
                print(f"    ✓ Done ({latency:.1f}s)")
                
                # Incremental save
                save_results(results, results_path, total, succeeded, failed)
            else:
                failed += 1

        except Exception as e:
            print(f"    ✗ Error: {e}")
            failed += 1

    # Final save
    save_results(results, results_path, total, succeeded, failed)
    print_run_summary(total, succeeded, failed, 'exp7_native_video', retry_queue_dir)
    print(f"Results saved to {results_path}")


def save_results(results, path, total, succeeded, failed):
    output_data = {
        'metadata': {
            'experiment': 'exp7_native_video',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_videos': total,
            'succeeded': succeeded,
            'failed': failed
        },
        'results': results
    }
    with open(path, 'w') as f:
        json.dump(output_data, f, indent=2)


if __name__ == '__main__':
    main()
