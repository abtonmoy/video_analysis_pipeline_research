# tests/pipeline.py
from dotenv import load_dotenv
load_dotenv()

from src.pipeline import AdVideoPipeline
import json

def main():
    print("Test pipeline")
    
    pipeline = AdVideoPipeline(config_path="config/default.yaml")
    video = "data/ads/ads/videos/v0017.mp4"
    
    result = pipeline.process(
        video,
        skip_extraction=False
    )
    
    print(f"\n{'='*60}")
    print("PIPELINE RESULTS")
    print(f"{'='*60}")
    print(f"Video: {result.video_path}")
    print(f"Duration: {result.metadata.duration}s")
    print(f"Scenes: {len(result.scenes)}")
    print(f"Total candidates: {result.total_frames_sampled}")
    print(f"After PHash: {result.frames_after_phash}")
    print(f"After SSIM: {result.frames_after_ssim}")
    print(f"After CLIP: {result.frames_after_clip}")
    print(f"Final frames: {result.final_frame_count}")
    print(f"Reduction: {result.reduction_rate:.1%}")
    print(f"Processing time: {result.processing_time_s:.1f}s")
    
    if result.extraction_result:
        print(f"\n{'='*60}")
        print("EXTRACTED AD DATA")
        print(f"{'='*60}")
        print(json.dumps(result.extraction_result, indent=2))
    else:
        print("\n✗ No extraction result")

if __name__ == "__main__":
    main()