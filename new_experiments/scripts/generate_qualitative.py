import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def load_frames_at_timestamps(video_path, timestamps, max_frames=8):
    if not os.path.exists(video_path):
        return []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    
    # Sort and slice
    timestamps = sorted(timestamps)[:max_frames]
    
    for ts in timestamps:
        frame_idx = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize
            h, w = frame.shape[:2]
            scale = 120 / h
            frame = cv2.resize(frame, (int(w * scale), 120))
            frames.append(frame)
        else:
            print(f"Failed to read frame at {ts}s for {video_path}")
            
    cap.release()
    return frames

def build_filmstrip(frames, highlight_redundant=False):
    # Padding config
    padding = 5
    h, w = frames[0].shape[:2]
    
    strip = np.zeros((h + 2*padding, (w + padding)*len(frames) + padding, 3), dtype=np.uint8)
    strip.fill(255) # white bg
    
    for i, frame in enumerate(frames):
        x = padding + i*(w + padding)
        y = padding
        strip[y:y+h, x:x+w] = frame
        
        if highlight_redundant and i > 0:
            # Highlight redundant frames arbitrarily (just for illustration if needed)
            if i % 3 == 0:
                cv2.rectangle(strip, (x, y), (x+w, y+h), (255, 0, 0), 4)

    return strip

def main():
    bench_file = "test_results/benchmark_results.json"
    data_dir = "/home/wabashcs/abt/use_data"
    
    if not os.path.exists(bench_file):
        print(f"Data file not found: {bench_file}")
        return
        
    with open(bench_file, 'r') as f:
        data = json.load(f)
        
    results = data.get('results', [])
    
    fast_cut_video = None
    slow_paced_video = None
    
    # 1. Select the videos programmatically
    for r in results:
        vid_name = os.path.basename(r.get('video_name', ''))
        vid_path = os.path.join(data_dir, vid_name)
        if not os.path.exists(vid_path):
            continue
            
        stats = r.get('pipeline_stats', {})
        uniform_frames = r.get('uniform_1fps', {}).get('selected_frames', [])
        hib_frames = r.get('hib_pipeline', {}).get('selected_frames', [])
        num_scenes = len(r.get('scenes', []))
        
        if not uniform_frames or not hib_frames:
            continue
            
        # Fast cut: high scenes, highly active
        if num_scenes >= 15 and fast_cut_video is None:
            fast_cut_video = r
            
        # Slow pacing: low scenes but decent duration
        if 2 <= num_scenes <= 6 and slow_paced_video is None:
            slow_paced_video = r
            
        if fast_cut_video and slow_paced_video:
            break
            
    if not fast_cut_video or not slow_paced_video:
        print("Could not find contrasting videos matching the criteria.")
        return
        
    print(f"Selected Fast-Cut: {fast_cut_video['video_name']}")
    print(f"Selected Slow-Paced: {slow_paced_video['video_name']}")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    
    # 2. Process Fast-cut
    fast_vid_name = os.path.basename(fast_cut_video.get('video_name', ''))
    path_fast = os.path.join(data_dir, fast_vid_name)
    uni_ts_fast = [f['timestamp'] for f in fast_cut_video['uniform_1fps']['selected_frames']]
    ada_ts_fast = [f['timestamp'] for f in fast_cut_video['hib_pipeline']['selected_frames']]
    
    uni_frames_fast = load_frames_at_timestamps(path_fast, uni_ts_fast, 8)
    ada_frames_fast = load_frames_at_timestamps(path_fast, ada_ts_fast, 8)
    
    strip_fast_uni = build_filmstrip(uni_frames_fast, highlight_redundant=True)
    strip_fast_ada = build_filmstrip(ada_frames_fast)
    
    # Combine vertically for Fast-cut
    pad = 10
    h_uni, w_uni = strip_fast_uni.shape[:2]
    h_ada, w_ada = strip_fast_ada.shape[:2]
    w_max = max(w_uni, w_ada)
    
    combo_fast = np.ones((h_uni + h_ada + pad, w_max, 3), dtype=np.uint8) * 255
    combo_fast[0:h_uni, 0:w_uni] = strip_fast_uni
    combo_fast[h_uni+pad:h_uni+pad+h_ada, 0:w_ada] = strip_fast_ada
    
    axes[0].imshow(combo_fast)
    axes[0].set_title(f"Fast-Cut Video ({fast_cut_video['video_name']}) - Top: Uniform-1FPS, Bottom: AdaFrame", fontsize=14, loc='left')
    axes[0].axis('off')
    
    # 3. Process Slow-paced
    slow_vid_name = os.path.basename(slow_paced_video.get('video_name', ''))
    path_slow = os.path.join(data_dir, slow_vid_name)
    uni_ts_slow = [f['timestamp'] for f in slow_paced_video['uniform_1fps']['selected_frames']]
    ada_ts_slow = [f['timestamp'] for f in slow_paced_video['hib_pipeline']['selected_frames']]
    
    uni_frames_slow = load_frames_at_timestamps(path_slow, uni_ts_slow, 8)
    ada_frames_slow = load_frames_at_timestamps(path_slow, ada_ts_slow, 8)
    
    strip_slow_uni = build_filmstrip(uni_frames_slow, highlight_redundant=True)
    strip_slow_ada = build_filmstrip(ada_frames_slow)
    
    combo_slow = np.ones((h_uni + h_ada + pad, w_max, 3), dtype=np.uint8) * 255
    combo_slow[0:h_uni, 0:len(strip_slow_uni[0])] = strip_slow_uni
    combo_slow[h_uni+pad:h_uni+pad+h_ada, 0:len(strip_slow_ada[0])] = strip_slow_ada

    axes[1].imshow(combo_slow)
    axes[1].set_title(f"Slow-Paced Video ({slow_paced_video['video_name']}) - Top: Uniform-1FPS, Bottom: AdaFrame", fontsize=14, loc='left')
    axes[1].axis('off')
    
    plt.tight_layout()
    os.makedirs('new_experiments/figures', exist_ok=True)
    plt.savefig('new_experiments/figures/qualitative_comparison.pdf', bbox_inches='tight')
    print("Saved qualitative_comparison.pdf")

if __name__ == "__main__":
    main()
