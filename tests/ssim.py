from src.detection.scene_detector import SceneDetector, CandidateFrameExtractor
from src.detection.change_detector import HistogramDetector, AdaptiveChangeDetector, EdgeChangeDetector
from src.deduplication.ssim import SSIMDeduplicator
from skimage.metrics import structural_similarity as ssim


def main():
    video_path = "data/ads/ads/videos/v0002.mp4"
    
    # Extract candidate frames using your existing tools
    print("Extracting candidate frames... Histogram")
    change_detector = HistogramDetector()
    # change_detector = AdaptiveChangeDetector()
    # change_detector = EdgeChangeDetector()
    extractor = CandidateFrameExtractor(change_detector)
    candidates = extractor.extract_candidates(video_path=video_path, max_resolution=720)
    
    print(f"Extracted {len(candidates)} candidate frames")
    
    # Test SSIMDeduplicator on the candidates
    print("\nTesting SSIMDeduplicator...")
    deduplicator = SSIMDeduplicator(threshold=0.92)  # SSIM threshold is 0.0-1.0
    
    # Compute signatures (grayscale resized frames)
    signatures = []
    for timestamp, frame in candidates:
        sig = deduplicator.compute_signature(frame)
        signatures.append((timestamp, sig))
        print(f"Frame at {timestamp:.2f}s: signature shape={sig.shape}")
    
    # Check for duplicates using SSIM
    print("\nChecking for similar frames...")
    
    for i in range(len(signatures) - 1):
        ts1, sig1 = signatures[i]
        ts2, sig2 = signatures[i + 1]
        
        # Compute actual SSIM score (NOT array subtraction!)
        score = ssim(sig1, sig2)
        similar = deduplicator.are_similar(sig1, sig2)
        
        print(f"Frame {ts1:.2f}s vs {ts2:.2f}s: SSIM={score:.4f}, similar={similar}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    similar_count = 0
    for i in range(len(signatures) - 1):
        ts1, sig1 = signatures[i]
        ts2, sig2 = signatures[i + 1]
        if deduplicator.are_similar(sig1, sig2):
            similar_count += 1
    
    print(f"Total frames: {len(candidates)}")
    print(f"Similar consecutive pairs: {similar_count}")
    print(f"Unique frames (estimate): {len(candidates) - similar_count}")


if __name__ == "__main__":
    main()