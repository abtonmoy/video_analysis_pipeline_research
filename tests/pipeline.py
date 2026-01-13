from src.pipeline import AdVideoPipeline, process_video

def main():
    video_path = "data/ads/ads/videos/v0002.mp4"
    print("Test pipeline")
    pipeline = process_video(video_path,"config\default.yaml")

if __name__=="__main__":
    main()
