import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to sys.path so we can import src
sys.path.insert(0, '/home/wabashcs/abt/video_analysis_pipeline_research-main')

from src.deduplication.phash import PHashDeduplicator
from src.deduplication.lpips import LPIPSDeduplicator
from src.deduplication.clip_embed import CLIPDeduplicator

def get_pairs(dedup_instance, frames, max_pairs=4):
    if not frames:
        return [], []
        
    pairs = []
    
    if isinstance(dedup_instance, CLIPDeduplicator):
        frame_arrays = [f for _, f in frames]
        embeddings = dedup_instance.compute_signatures_batch(frame_arrays)
        signatures = [(ts, frame, embeddings[i]) for i, (ts, frame) in enumerate(frames)]
    else:
        signatures = [(ts, frame, dedup_instance.compute_signature(frame)) for ts, frame in frames]
        
    kept = [signatures[0]]
    
    for ts, frame, sig in signatures[1:]:
        is_duplicate = False
        for _, kept_frame, kept_sig in kept:
            if isinstance(dedup_instance, CLIPDeduplicator):
                similarity = np.dot(sig, kept_sig)
                if similarity > dedup_instance.threshold:
                    is_duplicate = True
            else:
                if dedup_instance.are_similar(sig, kept_sig):
                    is_duplicate = True
                    
            if is_duplicate:
                if len(pairs) < max_pairs:
                    pairs.append((kept_frame, frame))
                break
                
        if not is_duplicate:
            kept.append((ts, frame, sig))
            
    next_frames = [(ts, f) for ts, f, _ in kept]
    return next_frames, pairs

def main():
    video_path = "/home/wabashcs/abt/use_data/-g8vTq06KUI.mp4"
    if not os.path.exists(video_path):
        print(f"Video {video_path} not found.")
        return

    # Extract 1 keyframe every 5 frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if count % 5 == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            scale = 180 / h
            frame = cv2.resize(frame, (int(w * scale), 180))
            ts = count / fps
            frames.append((ts, frame))
        count += 1
    cap.release()
    print(f"Loaded {len(frames)} candidate frames.")

    phash = PHashDeduplicator(threshold=12)
    lpips_dedup = LPIPSDeduplicator(threshold=0.15, device="cuda")
    clip_dedup = CLIPDeduplicator(threshold=0.90, device="cuda")

    print("Running PHash...")
    frames_after_p, p_pairs = get_pairs(phash, frames, max_pairs=4)
    print(f"PHash pairs found: {len(p_pairs)}")

    print("Running LPIPS...")
    frames_after_l, l_pairs = get_pairs(lpips_dedup, frames_after_p, max_pairs=4)
    print(f"LPIPS pairs found: {len(l_pairs)}")

    print("Running CLIP...")
    frames_after_c, c_pairs = get_pairs(clip_dedup, frames_after_l, max_pairs=4)
    print(f"CLIP pairs found: {len(c_pairs)}")

    all_pairs = [p_pairs, l_pairs, c_pairs]
    row_labels = ["Tier 1: Hash Voting", "Tier 2: LPIPS", "Tier 3: CLIP Clustering"]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for row_idx, (pairs, label) in enumerate(zip(all_pairs, row_labels)):
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            if col_idx < len(pairs):
                kept, removed = pairs[col_idx]
                
                kept_bordered = cv2.copyMakeBorder(kept, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 255, 0])
                removed_bordered = cv2.copyMakeBorder(removed, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 0, 0])
                
                h, w = kept_bordered.shape[:2]
                combo = np.ones((h, w * 2 + 5, 3), dtype=np.uint8) * 255
                combo[0:h, 0:w] = kept_bordered
                combo[0:h, w+5:w*2+5] = removed_bordered
                
                ax.imshow(combo)
            else:
                ax.text(0.5, 0.5, "No Pair Found", ha='center', va='center')
                
            ax.axis('off')
            if col_idx == 0:
                ax.set_title(label + "\nKept (Green) \u2192 Removed (Red)", loc='left')

    plt.tight_layout()
    os.makedirs('new_experiments/figures', exist_ok=True)
    plt.savefig('new_experiments/figures/tier_examples.pdf', bbox_inches='tight')
    print("Saved tier_examples.pdf")

if __name__ == "__main__":
    main()
