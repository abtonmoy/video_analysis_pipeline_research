from src.detection.scene_detector import SceneDetector, CandidateFrameExtractor
from src.detection.change_detector import HistogramDetector, AdaptiveChangeDetector, EdgeChangeDetector
from src.deduplication.clip_embed import CLIPDeduplicator
import numpy as np


def main():
    video_path = "data/ads/ads/videos/v0002.mp4"
    
    # Extract candidate frames using your existing tools
    print("Extracting candidate frames... Histogram")
    change_detector = HistogramDetector()
    extractor = CandidateFrameExtractor(change_detector)
    candidates = extractor.extract_candidates(video_path=video_path, max_resolution=720)
    
    print(f"Extracted {len(candidates)} candidate frames")
    
    # Test CLIPDeduplicator on the candidates
    print("\nTesting CLIPDeduplicator...")
    deduplicator = CLIPDeduplicator(
        model_name="ViT-B-32",
        pretrained="openai",
        threshold=0.90,
        device="auto",
        batch_size=8
    )
    
    # Compute embeddings (this will batch process)
    print("\nComputing CLIP embeddings...")
    frame_arrays = [frame for _, frame in candidates]
    embeddings = deduplicator.compute_signatures_batch(frame_arrays)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Store with timestamps
    signatures = []
    for i, (timestamp, _) in enumerate(candidates):
        signatures.append((timestamp, embeddings[i]))
        print(f"Frame at {timestamp:.2f}s: embedding shape={embeddings[i].shape}")
    
    # Check for duplicates using cosine similarity
    print("\nChecking for similar frames (cosine similarity)...")
    
    for i in range(len(signatures) - 1):
        ts1, emb1 = signatures[i]
        ts2, emb2 = signatures[i + 1]
        
        # Compute cosine similarity
        similarity = np.dot(emb1, emb2)
        similar = deduplicator.are_similar(emb1, emb2)
        
        print(f"Frame {ts1:.2f}s vs {ts2:.2f}s: similarity={similarity:.4f}, similar={similar}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    similar_count = 0
    for i in range(len(signatures) - 1):
        ts1, emb1 = signatures[i]
        ts2, emb2 = signatures[i + 1]
        if deduplicator.are_similar(emb1, emb2):
            similar_count += 1
    
    print(f"Total frames: {len(candidates)}")
    print(f"Similar consecutive pairs: {similar_count}")
    print(f"Unique frames (estimate): {len(candidates) - similar_count}")
    
    # Test the built-in deduplicate method
    print("\n" + "="*60)
    print("USING BUILT-IN DEDUPLICATE METHOD")
    print("="*60)
    
    kept_frames, kept_embeddings = deduplicator.deduplicate(candidates)
    
    print(f"Original frames: {len(candidates)}")
    print(f"After deduplication: {len(kept_frames)}")
    print(f"Frames removed: {len(candidates) - len(kept_frames)}")
    print(f"Reduction: {(1 - len(kept_frames)/len(candidates))*100:.1f}%")
    
    print("\nKept frame timestamps:")
    for ts, _ in kept_frames:
        print(f"  {ts:.2f}s")


if __name__ == "__main__":
    main()