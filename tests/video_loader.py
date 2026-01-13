from src.ingestion.video_loader import  VideoLoader

def main():
    video = "data/ads/ads/videos/v0002.mp4"
    print("Test video:", video)
    print("___"*20)
    video_loader = VideoLoader()
    load = video_loader.load(video)
    print("Loaded:",load)
    iterator = video_loader.get_frame_iterator(video)
    print("Iterator:", iterator)

if __name__ == "__main__":
    main()