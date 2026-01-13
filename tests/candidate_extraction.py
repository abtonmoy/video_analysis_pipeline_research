from src.detection.scene_detector import SceneDetector, CandidateFrameExtractor
from src.detection.change_detector import (
    FrameDifferenceDetector,
    HistogramDetector,
    EdgeChangeDetector,
    AdaptiveChangeDetector
)
def main():
    video = "data/ads/ads/videos/v0002.mp4"
    
    
    print("Testing candidate detector")
    print("Test video:", video)
    change_detector = AdaptiveChangeDetector() # HistogramDetector
    test2 = CandidateFrameExtractor(change_detector)
    result2 = test2.extract_candidates(video_path=video, max_resolution=720)
    print("result: ", result2)
    

if __name__ == "__main__":
    main()