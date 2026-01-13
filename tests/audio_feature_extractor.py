from src.ingestion.audio_extractor import AudioExtractor

def main():
    audio_file = "outputs/audio/v0002.wav"
    print("Test audio:", audio_file)
    print("__"*20)

    extractor = AudioExtractor()
    
    # load_audio returns (audio_array, sample_rate), not a boolean
    audio_data, sample_rate = extractor.load_audio(audio_file)
    
    print(f"Audio loaded successfully!")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {len(audio_data) / sample_rate:.2f} seconds")
    print(f"Audio shape: {audio_data.shape}")
    print("__"*20)
    
    # Extract audio events
    print("\nExtracting audio features...")
    events = extractor.get_audio_events(audio_file)
    
    print(f"\nEnergy peaks found: {len(events['energy_peaks'])}")
    print(f"Peak timestamps (first 10): {events['energy_peaks'][:10]}")
    
    print(f"\nSilence segments found: {len(events['silence_segments'])}")
    print(f"Silence segments: {events['silence_segments']}")
    
    print("\n" + "__"*20)
    print("All events:")
    print(events)


if __name__ == "__main__":
    main()