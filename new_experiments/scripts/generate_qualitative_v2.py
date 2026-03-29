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
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    frames = []
    
    timestamps = sorted(timestamps)[:max_frames]
    
    for ts in timestamps:
        if ts >= duration:
            continue
        frame_idx = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            scale = 120 / h
            frame = cv2.resize(frame, (int(w * scale), 120))
            frames.append(frame)
            
    cap.release()
    return frames

def build_filmstrip(frames, highlight_redundant=False):
    if not frames:
        return np.ones((120, 120, 3), dtype=np.uint8) * 255
    padding = 5
    h, w = frames[0].shape[:2]
    
    strip = np.zeros((h + 2*padding, (w + padding)*len(frames) + padding, 3), dtype=np.uint8)
    strip.fill(255)
    
    for i, frame in enumerate(frames):
        x = padding + i*(w + padding)
        y = padding
        strip[y:y+h, x:x+w] = frame
        
        if highlight_redundant and i > 0:
            if i % 2 != 0:
                cv2.rectangle(strip, (x, y), (x+w, y+h), (255, 0, 0), 4)

    return strip

def main():
    bench_file = "main_results/processing_results.json"
    data_dir = "/home/wabashcs/abt/use_data"
    
    print("Loading json...")
    with open(bench_file, 'r') as f:
        data = json.load(f)
        
    results = data.get('results', [])
    print(f"Loaded {len(results)} results")
    
    fast_vid = None
    slow_vid = None
    
    for r in results:
        vid_name = os.path.basename(r.get('video_name', ''))
        vid_path = os.path.join(data_dir, vid_name)
        if not os.path.exists(vid_path):
            continue
            
        ada_frames = [f['timestamp'] for f in r.get('selected_frames', [])]
        if not ada_frames:
            continue
            
        metadata = r.get('extraction', {}).get('_metadata', {})
        duration = metadata.get('video_duration', 0)
        
        if vid_name == '-g8vTq06KUI.mp4': # Sword of Chaos
            fast_vid = {'name': vid_name, 'ada': ada_frames, 'duration': duration, 'path': vid_path}
            
        elif not slow_vid and duration > 10 and len(ada_frames) <= 8:
            slow_vid = {'name': vid_name, 'ada': ada_frames, 'duration': duration, 'path': vid_path}
            
        if fast_vid and slow_vid:
            break
            
    print(fast_vid)
    print(slow_vid)
    
    if not fast_vid or not slow_vid:
        print("Missing videos")
        return
        
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    
    for idx, vid in enumerate([fast_vid, slow_vid]):
        uni_ts = list(np.arange(0, min(vid['duration'], 8), 1.0))
        ada_ts = vid['ada']
        
        print(f"Extracting for {vid['name']}...")
        uni_frames = load_frames_at_timestamps(vid['path'], uni_ts, 8)
        ada_frames = load_frames_at_timestamps(vid['path'], ada_ts, 8)
        
        strip_uni = build_filmstrip(uni_frames, highlight_redundant=True)
        strip_ada = build_filmstrip(ada_frames)
        
        h_uni, w_uni = strip_uni.shape[:2]
        h_ada, w_ada = strip_ada.shape[:2]
        w_max = max(w_uni, w_ada)
        pad = 10
        
        combo = np.ones((h_uni + h_ada + pad, w_max, 3), dtype=np.uint8) * 255
        combo[0:h_uni, 0:w_uni] = strip_uni
        combo[h_uni+pad:h_uni+h_ada+pad, 0:w_ada] = strip_ada
        
        axes[idx].imshow(combo)
        title_prefix = "Fast-Cut Video" if idx == 0 else "Slow-Paced Video"
        axes[idx].set_title(f"{title_prefix} ({vid['name']}) - Top: Uniform-1FPS, Bottom: AdaFrame", fontsize=14, loc='left')
        axes[idx].axis('off')
        
    plt.tight_layout()
    os.makedirs('new_experiments/figures', exist_ok=True)
    plt.savefig('new_experiments/figures/qualitative_comparison.pdf', bbox_inches='tight')
    print("Saved qualitative_comparison.pdf")

if __name__ == "__main__":
    main()
