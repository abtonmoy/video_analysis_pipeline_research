import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np


logger = logging.getLogger(__name__)

class AudioExtractor:
    """
    Extracts audio features for audio-visual alignment.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._librosa = None
    
    def _get_librosa(self):
        """Lazy load librosa."""
        if self._librosa is None:
            import librosa
            self._librosa = librosa
        return self._librosa
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        librosa = self._get_librosa()
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        return y, sr
    
    def extract_energy_peaks(
        self,
        audio_path: str,
        threshold_percentile: float = 90
    ) -> list:
        """
        Extract timestamps of energy peaks in audio.
        
        Args:
            audio_path: Path to audio file
            threshold_percentile: Percentile threshold for peak detection
            
        Returns:
            List of peak timestamps in seconds
        """
        librosa = self._get_librosa()
        
        y, sr = self.load_audio(audio_path)
        
        # Compute RMS energy
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.times_like(rms, sr=sr)
        
        # Find peaks above threshold
        threshold = np.percentile(rms, threshold_percentile)
        peak_indices = librosa.util.peak_pick(
            rms,
            pre_max=3, post_max=3,
            pre_avg=3, post_avg=5,
            delta=threshold * 0.1,
            wait=10
        )
        
        return times[peak_indices].tolist()
    
    def detect_silence(
        self,
        audio_path: str,
        threshold_db: float = -40,
        min_silence_s: float = 0.3
    ) -> list:
        """
        Detect silence segments in audio.
        
        Args:
            audio_path: Path to audio file
            threshold_db: Silence threshold in dB
            min_silence_s: Minimum silence duration
            
        Returns:
            List of (start, end) tuples for silence segments
        """
        librosa = self._get_librosa()
        
        y, sr = self.load_audio(audio_path)
        
        # Convert to dB
        rms = librosa.feature.rms(y=y)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        times = librosa.times_like(rms, sr=sr)
        
        # Find silence regions
        is_silence = rms_db < threshold_db
        
        # Group consecutive silence frames
        silences = []
        start = None
        
        for i, silent in enumerate(is_silence):
            if silent and start is None:
                start = times[i]
            elif not silent and start is not None:
                end = times[i]
                if end - start >= min_silence_s:
                    silences.append((start, end))
                start = None
        
        # Handle trailing silence
        if start is not None:
            end = times[-1]
            if end - start >= min_silence_s:
                silences.append((start, end))
        
        return silences
    
    def get_audio_events(self, audio_path: str) -> dict:
        """
        Extract all audio events for frame importance boosting.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with energy_peaks and silence_segments
        """
        return {
            "energy_peaks": self.extract_energy_peaks(audio_path),
            "silence_segments": self.detect_silence(audio_path)
        }
