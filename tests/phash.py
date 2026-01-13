from src.detection.scene_detector import SceneDetector, CandidateFrameExtractor
from src.detection.change_detector import HistogramDetector, AdaptiveChangeDetector, EdgeChangeDetector
from src.deduplication.phash import PHashDeduplicator


def main():
    video_path = "data/ads/ads/videos/v0002.mp4"
    
    # Extract candidate frames using your existing tools
    print("Extracting candidate frames... Histogram")
    change_detector = HistogramDetector()
    extractor = CandidateFrameExtractor(change_detector)
    candidates = extractor.extract_candidates(video_path=video_path, max_resolution=720)
    
    print(f"Extracted {len(candidates)} candidate frames")
    
    # Test PHashDeduplicator on the candidates
    print("\nTesting PHashDeduplicator...")
    deduplicator = PHashDeduplicator(threshold=8)
    
    # Compute hashes
    hashes = []
    for timestamp, frame in candidates:
        hash_val = deduplicator.compute_signature(frame)
        hashes.append((timestamp, hash_val))
        print(f"Frame at {timestamp:.2f}s: hash={hash_val}")
    
    # Check for duplicates
    print("\nChecking for similar frames...")
    for i in range(len(hashes) - 1):
        ts1, hash1 = hashes[i]
        ts2, hash2 = hashes[i + 1]
        
        similar = deduplicator.are_similar(hash1, hash2)
        hamming_dist = hash1 - hash2
        
        print(f"Frame {ts1:.2f}s vs {ts2:.2f}s: distance={hamming_dist}, similar={similar}")


if __name__ == "__main__":
    main()