# Video & Audio Processing

Python modules for extracting and analyzing video frames and audio from video files.

## Features

### VideoLoader

- Load and validate video files (mp4, mov, avi, mkv, webm, m4v)
- Extract video metadata (duration, resolution, fps)
- Extract audio tracks to WAV format
- Iterate through video frames at specified intervals

### AudioExtractor

- **Speech Processing**: Transcribe audio with timestamps using Whisper
- **Audio Events**: Detect energy peaks and silence segments
- **Speech Detection**: Identify when people are talking
- **Music Analysis**: Extract tempo/BPM and classify mood
- **Key Phrase Extraction**: Find promotional keywords and timestamps

## Installation

```bash
pip install librosa openai-whisper webrtcvad
```

Requires `ffmpeg` for audio extraction:

```bash
# Ubuntu/Debian
apt-get install ffmpeg

# macOS
brew install ffmpeg
```

## Quick Start

### Load Video

```python
from video_loader import VideoLoader

loader = VideoLoader(max_resolution=720, extract_audio=True)
metadata, audio_path = loader.load("video.mp4")

# Iterate frames
with loader.get_frame_iterator("video.mp4", interval_ms=100) as frames:
    for timestamp, frame in frames:
        # Process frame
        pass
```

### Extract Audio Context

```python
from audio_extractor import AudioExtractor

extractor = AudioExtractor()

# Full analysis with transcription
context = extractor.extract_full_context(
    "audio.wav",
    transcribe=True,
    model_size="base"
)

# Access results
print(context["transcription"])  # Speech segments
print(context["key_phrases"])    # Important phrases
print(context["mood"])            # Music mood
print(context["tempo"]["bpm"])   # Beats per minute
```

### Transcribe Speech

```python
segments = extractor.transcribe_audio("audio.wav")
for seg in segments:
    print(f"{seg['start']:.1f}s: {seg['text']}")
```

### Detect Audio Events

```python
# Energy peaks (exciting moments)
peaks = extractor.extract_energy_peaks("audio.wav")

# Silence segments
silences = extractor.detect_silence("audio.wav")

# Speech segments (when talking)
speech = extractor.detect_speech_segments("audio.wav")
```

## Configuration

### Whisper Models

- `tiny`: Fastest, less accurate
- `base`: Balanced (default)
- `small`: Better accuracy
- `medium`, `large`: Highest accuracy, slower

### AudioExtractor Parameters

- `sample_rate`: Audio sample rate (default: 16000 Hz)
- `threshold_percentile`: Energy peak detection threshold (default: 90)
- `threshold_db`: Silence detection threshold (default: -40 dB)
- `min_silence_s`: Minimum silence duration (default: 0.3s)

## Output Structure

### Full Audio Context

```python
{
    "transcription": [...],      # Speech segments with timestamps
    "key_phrases": [...],         # Important phrases
    "speech_segments": [...],     # When speech occurs
    "has_speech": bool,
    "energy_peaks": [...],        # High-energy moments
    "silence_segments": [...],    # Quiet periods
    "tempo": {"bpm": float, ...},
    "mood": str                   # upbeat/calm/dramatic/energetic/melancholic
}
```

## Notes

- Audio extraction creates files in `outputs/audio/`
- Transcription can be slow for long videos (disable with `transcribe=False`)
- WebRTC VAD provides better speech detection but requires `webrtcvad` package
- Music analysis is heuristic-based (ML classification available as extension)
