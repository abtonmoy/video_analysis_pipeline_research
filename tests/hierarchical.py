import yaml
from pathlib import Path
from src.detection.scene_detector import SceneDetector, CandidateFrameExtractor
from src.detection.change_detector import HistogramDetector, EdgeChangeDetector
from src.deduplication.hierarchical import HierarchicalDeduplicator


def load_config(config_path: str = "config/default.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_change_detector_from_config(config: dict):
    """Create change detector based on config."""
    method = config['change_detection']['method']
    
    if method == "histogram":
        from src.detection.change_detector import HistogramDetector
        return HistogramDetector()
    elif method == "edge":
        from src.detection.change_detector import EdgeChangeDetector
        return EdgeChangeDetector()
    elif method == "frame_diff":
        from src.detection.change_detector import FrameDifferenceDetector
        return FrameDifferenceDetector()
    else:
        raise ValueError(f"Unknown change detection method: {method}")


def create_deduplicator_from_config(config: dict):
    """Create hierarchical deduplicator based on config."""
    dedup_config = config['deduplication']
    
    return HierarchicalDeduplicator(
        phash_enabled=dedup_config['phash']['enabled'],
        phash_threshold=dedup_config['phash']['threshold'],
        ssim_enabled=dedup_config['ssim']['enabled'],
        ssim_threshold=dedup_config['ssim']['threshold'],
        clip_enabled=dedup_config['clip']['enabled'],
        clip_model=dedup_config['clip']['model'],
        clip_threshold=dedup_config['clip']['threshold'],
        clip_device=dedup_config['clip']['device'],
        clip_batch_size=dedup_config['clip']['batch_size']
    )


def test_with_config():
    """Test deduplication pipeline using config file."""
    
    # Load config
    config = load_config("config/default.yaml")
    
    print("="*80)
    print("DEDUPLICATION PIPELINE TEST - USING CONFIG FILE")
    print("="*80)
    
    # Display config settings
    print("\nConfiguration:")
    print(f"  Change Detection: {config['change_detection']['method']}")
    print(f"  Change Threshold: {config['change_detection']['threshold']}")
    print(f"  Min Interval: {config['change_detection']['min_interval_ms']}ms")
    print(f"\n  Deduplication Pipeline:")
    print(f"    PHash:  enabled={config['deduplication']['phash']['enabled']}, threshold={config['deduplication']['phash']['threshold']}")
    print(f"    SSIM:   enabled={config['deduplication']['ssim']['enabled']}, threshold={config['deduplication']['ssim']['threshold']}")
    print(f"    CLIP:   enabled={config['deduplication']['clip']['enabled']}, threshold={config['deduplication']['clip']['threshold']}")
    print(f"    Device: {config['deduplication']['clip']['device']}")
    print(f"    Batch:  {config['deduplication']['clip']['batch_size']}")
    
    # Video path
    video_path = "data/ads/ads/videos/v0002.mp4"
    
    # Step 1: Extract candidate frames
    print("\n" + "-"*80)
    print("STEP 1: Extracting candidate frames")
    print("-"*80)
    
    change_detector = create_change_detector_from_config(config)
    extractor = CandidateFrameExtractor(
        change_detector=change_detector,
        threshold=config['change_detection']['threshold'],
        min_interval_ms=config['change_detection']['min_interval_ms']
    )
    
    candidates = extractor.extract_candidates(
        video_path=video_path,
        max_resolution=config['ingestion']['max_resolution']
    )
    
    print(f"Extracted {len(candidates)} candidate frames")
    
    # Step 2: Hierarchical deduplication
    print("\n" + "-"*80)
    print("STEP 2: Hierarchical deduplication")
    print("-"*80)
    
    deduplicator = create_deduplicator_from_config(config)
    final_frames, embeddings, stats = deduplicator.deduplicate(candidates)
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\nPipeline stages:")
    print(f"  Input frames:     {stats['input']}")
    
    if 'after_phash' in stats:
        reduction = (1 - stats['after_phash']/stats['input']) * 100
        removed = stats['input'] - stats['after_phash']
        print(f"  After PHash:      {stats['after_phash']:3d} ({removed:2d} removed, {reduction:5.1f}% reduction)")
    
    if 'after_ssim' in stats:
        reduction = (1 - stats['after_ssim']/stats['after_phash']) * 100
        removed = stats['after_phash'] - stats['after_ssim']
        print(f"  After SSIM:       {stats['after_ssim']:3d} ({removed:2d} removed, {reduction:5.1f}% reduction)")
    
    if 'after_clip' in stats:
        reduction = (1 - stats['after_clip']/stats['after_ssim']) * 100
        removed = stats['after_ssim'] - stats['after_clip']
        print(f"  After CLIP:       {stats['after_clip']:3d} ({removed:2d} removed, {reduction:5.1f}% reduction)")
    
    total_reduction = (1 - stats['output']/stats['input']) * 100
    total_removed = stats['input'] - stats['output']
    print(f"\n  Final output:     {stats['output']:3d} frames")
    print(f"  Total removed:    {total_removed:3d} frames")
    print(f"  Total reduction:  {total_reduction:.1f}%")
    
    # Show final timestamps
    print(f"\nFinal frame timestamps:")
    for ts, _ in final_frames:
        print(f"  {ts:.2f}s")
    
    if embeddings is not None:
        print(f"\nCLIP embeddings shape: {embeddings.shape}")
    
    return final_frames, embeddings, stats


def test_config_variations():
    """Test with different detector settings."""
    
    print("\n" + "="*80)
    print("TESTING CONFIG VARIATIONS")
    print("="*80)
    
    video_path = "data/ads/ads/videos/v0002.mp4"
    
    # Test different change detection methods
    methods = ["histogram", "edge"]
    results = {}
    
    for method in methods:
        print(f"\n--- Testing with {method} detector ---")
        
        # Load and modify config
        config = load_config("config/default.yaml")
        config['change_detection']['method'] = method
        
        # Extract candidates
        change_detector = create_change_detector_from_config(config)
        extractor = CandidateFrameExtractor(
            change_detector=change_detector,
            threshold=config['change_detection']['threshold'],
            min_interval_ms=config['change_detection']['min_interval_ms']
        )
        candidates = extractor.extract_candidates(video_path=video_path, max_resolution=720)
        
        print(f"Extracted {len(candidates)} candidates")
        
        # Deduplicate
        deduplicator = create_deduplicator_from_config(config)
        final_frames, _, stats = deduplicator.deduplicate(candidates)
        
        results[method] = stats
        
        print(f"After dedup: {stats['output']} frames ({(1-stats['output']/stats['input'])*100:.1f}% reduction)")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    for method, stats in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Candidates:  {stats['input']}")
        if 'after_phash' in stats:
            print(f"  After PHash: {stats['after_phash']}")
        if 'after_ssim' in stats:
            print(f"  After SSIM:  {stats['after_ssim']}")
        if 'after_clip' in stats:
            print(f"  After CLIP:  {stats['after_clip']}")
        print(f"  Final:       {stats['output']} ({(1-stats['output']/stats['input'])*100:.1f}% reduction)")


def main():
    print("Deduplication Pipeline Test - Config-Based\n")
    
    # Test 1: Default config
    print("TEST 1: Using default configuration")
    print("="*80)
    test_with_config()
    
    # Test 2: Config variations
    print("\n\nTEST 2: Testing different detectors")
    print("="*80)
    test_config_variations()


if __name__ == "__main__":
    main()