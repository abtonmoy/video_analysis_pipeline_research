import os
import pandas as pd
import yt_dlp
import time
import random
import json

# --- CONFIGURATION ---
CSV_PATH = "data/final_video_id_list.csv"
OUTPUT_DIR = "data/hussain_videos"
LOG_FILE = "download_log.txt"
PROGRESS_FILE = "download_progress.json"  # NEW: Track progress
MIN_DELAY = 5.0
MAX_DELAY = 15.0
BOT_DETECTION_PAUSE = 300

# Create output directory (and any parent directories)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_progress():
    """Load the last processed index from progress file."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                data = json.load(f)
                return data.get('last_index', -1)
        except:
            return -1
    return -1

def save_progress(index):
    """Save the current progress index."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({'last_index': index}, f)

def scan_existing_videos(output_dir):
    """Scan the output directory and return a set of already downloaded video IDs."""
    existing_ids = set()
    
    if not os.path.exists(output_dir):
        return existing_ids
    
    for filename in os.listdir(output_dir):
        if filename.endswith('.mp4'):
            # Extract video ID (filename without extension)
            video_id = os.path.splitext(filename)[0]
            existing_ids.add(video_id)
    
    print(f"Existing videos sample: {list(existing_ids)[:5]}")
    return existing_ids

def clean_video_id(raw_id):
    """Clean video ID by removing quotes and whitespace."""
    if pd.isna(raw_id):
        return None
    
    cleaned = str(raw_id).strip()
    # Remove surrounding quotes (both single and double)
    cleaned = cleaned.strip("'").strip('"')
    return cleaned

def download_videos():
    # 1. Load last progress
    last_index = load_progress()
    print(f"Last processed index: {last_index}")
    if last_index >= 0:
        print(f"Resuming from index {last_index + 1}\n")
    else:
        print("Starting fresh download\n")
    
    # 2. Scan existing videos
    print("Scanning existing videos...")
    existing_videos = scan_existing_videos(OUTPUT_DIR)
    print(f"Found {len(existing_videos)} already downloaded videos\n")
    
    # 3. Load the CSV (no headers, just video IDs)
    try:
        df = pd.read_csv(CSV_PATH, header=None)
        df.columns = ['video_id']
        
        # Clean all video IDs (remove quotes)
        df['clean_id'] = df['video_id'].apply(clean_video_id)
        
        print(f"Loaded CSV. Found {len(df)} video entries.")
        print(f"First few IDs: {df['clean_id'].head(5).tolist()}\n")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # 4. Configure yt-dlp
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': f'{OUTPUT_DIR}/%(id)s.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
        'sleep_interval': 5,
        'max_sleep_interval': 15,
    }

    success_count = 0
    fail_count = 0
    skip_count = 0
    rate_limit_count = 0
    bot_detection_count = 0
    download_count = 0
    
    print("Starting download sequence...")
    print(f"Using delays between {MIN_DELAY}-{MAX_DELAY} seconds to avoid rate limiting\n")
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for index, row in df.iterrows():
            # Skip videos we've already processed
            if index <= last_index:
                continue
            
            video_id = row['clean_id']
            
            if not video_id or len(video_id) != 11:
                print(f"[{index+1}/{len(df)}] Skipping invalid ID: {video_id}")
                save_progress(index)  # Save progress even for invalid IDs
                continue
            
            # Skip if already in our scanned list
            if video_id in existing_videos:
                skip_count += 1
                success_count += 1
                print(f"[{index+1}/{len(df)}] Skipping {video_id} (Already exists)")
                save_progress(index)  # Save progress
                continue
            
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            expected_path = os.path.join(OUTPUT_DIR, f"{video_id}.mp4")

            try:
                print(f"[{index+1}/{len(df)}] Downloading {video_id}...", end="\r")
                ydl.download([video_url])
                
                # Check if file was actually created
                if os.path.exists(expected_path):
                    print(f"[{index+1}/{len(df)}] Downloaded {video_id}      ")
                    success_count += 1
                    download_count += 1
                    existing_videos.add(video_id)
                    save_progress(index)  # Save progress after successful download
                else:
                    print(f"[{index+1}/{len(df)}] Failed {video_id} (Unavailable)")
                    fail_count += 1
                    with open(LOG_FILE, "a") as f:
                        f.write(f"{video_id},unavailable\n")
                    save_progress(index)  # Save progress
            
            except Exception as e:
                error_msg = str(e)
                
                # Check for bot detection
                if "not a bot" in error_msg.lower() or "sign in to confirm" in error_msg.lower():
                    bot_detection_count += 1
                    print(f"\n[{index+1}/{len(df)}] BOT DETECTION TRIGGERED!")
                    print(f"Pausing for {BOT_DETECTION_PAUSE} seconds ({BOT_DETECTION_PAUSE//60} minutes)...")
                    print(f"Bot detection hits: {bot_detection_count}")
                    time.sleep(BOT_DETECTION_PAUSE)
                    fail_count += 1
                    
                elif "rate-limited" in error_msg.lower():
                    rate_limit_count += 1
                    print(f"\n[{index+1}/{len(df)}] RATE LIMITED - Pausing for 120 seconds...")
                    print(f"Rate limit hits: {rate_limit_count}")
                    time.sleep(120)
                    fail_count += 1
                    
                else:
                    print(f"[{index+1}/{len(df)}] Error {video_id}: {error_msg[:50]}")
                    fail_count += 1
                
                with open(LOG_FILE, "a") as f:
                    f.write(f"{video_id},error:{error_msg[:100]}\n")
                
                save_progress(index)  # Save progress even on error
            
            # Rate limiting - increased delay
            delay = random.uniform(MIN_DELAY, MAX_DELAY)
            time.sleep(delay)

    print(f"\n\nDone!")
    print(f"Total Videos in CSV: {len(df)}")
    print(f"Already Existed (Skipped): {skip_count}")
    print(f"Newly Downloaded: {download_count}")
    print(f"Failed: {fail_count}")
    print(f"Rate limit warnings: {rate_limit_count}")
    print(f"Bot detection warnings: {bot_detection_count}")
    print(f"Total Success: {success_count}")

if __name__ == "__main__":
    download_videos()