# src\ingestion\audio_extractor.py
import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np


logger = logging.getLogger(__name__)

class AudioExtractor:
    """
    Extracts audio features for audio-visual alignment and LLM context.
    
    Features:
    - Speech transcription with timestamps
    - Energy peaks and silence detection
    - Music mood classification
    - Tempo/BPM analysis
    - Speech vs music separation
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._librosa = None
        self._whisper = None
    
    def _get_librosa(self):
        """Lazy load librosa."""
        if self._librosa is None:
            try:
                import librosa
                self._librosa = librosa
            except ImportError:
                raise ImportError(
                    "librosa not installed. Install with: pip install librosa"
                )
        return self._librosa
    
    def _get_whisper(self):
        """Lazy load whisper."""
        if self._whisper is None:
            try:
                import whisper
                self._whisper = whisper
            except ImportError:
                raise ImportError(
                    "whisper not installed. Install with: pip install openai-whisper"
                )
        return self._whisper
    
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
    
    # ========================================================================
    # Speech Transcription
    # ========================================================================
    
    def transcribe_audio(
        self,
        audio_path: str,
        model_size: str = "base",
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        """
        Transcribe audio using OpenAI Whisper.
        
        Args:
            audio_path: Path to audio file
            model_size: Whisper model size (tiny, base, small, medium, large)
            language: Language code (en, es, fr, etc.)
            
        Returns:
            List of transcription segments with timestamps:
            [
                {
                    "text": "Hello world",
                    "start": 0.5,
                    "end": 1.2,
                    "confidence": 0.95
                },
                ...
            ]
        """
        whisper = self._get_whisper()
        
        logger.info(f"Loading Whisper model: {model_size}")
        model = whisper.load_model(model_size)
        
        logger.info(f"Transcribing audio: {audio_path}")
        result = model.transcribe(
            audio_path,
            language=language,
            word_timestamps=False,  # Set to True for word-level timestamps
            verbose=False
        )
        
        segments = []
        for segment in result["segments"]:
            segments.append({
                "text": segment["text"].strip(),
                "start": segment["start"],
                "end": segment["end"],
                "confidence": segment.get("no_speech_prob", 0.0)
            })
        
        logger.info(f"Transcribed {len(segments)} segments")
        return segments
    
    def extract_key_phrases(
        self,
        transcription: List[Dict[str, Any]],
        keywords: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract key phrases from transcription.
        
        Args:
            transcription: Output from transcribe_audio()
            keywords: Optional list of keywords to search for
            
        Returns:
            List of key phrases with timestamps:
            [
                {
                    "text": "50% off",
                    "timestamp": 3.2,
                    "context": "Get 50% off today only"
                },
                ...
            ]
        """
        if keywords is None:
            # Default promotional/commercial keywords
            keywords = [
                "off", "sale", "discount", "free", "save", "deal",
                "limited", "now", "today", "buy", "get", "order",
                "call", "visit", "sign up", "download", "try",
                "percent", "%", "$", "price", "cost"
            ]
        
        key_phrases = []
        
        for segment in transcription:
            text_lower = segment["text"].lower()
            
            # Check if any keyword is in this segment
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    # Use middle of segment as timestamp
                    timestamp = (segment["start"] + segment["end"]) / 2
                    
                    key_phrases.append({
                        "text": keyword,
                        "timestamp": timestamp,
                        "context": segment["text"],
                        "start": segment["start"],
                        "end": segment["end"]
                    })
                    break  # Only add segment once
        
        logger.info(f"Found {len(key_phrases)} key phrases")
        return key_phrases
    
    # ========================================================================
    # Energy and Silence Detection
    # ========================================================================
    
    def extract_energy_peaks(
        self,
        audio_path: str,
        threshold_percentile: float = 90
    ) -> List[float]:
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
    ) -> List[Tuple[float, float]]:
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
    
    # ========================================================================
    # Speech Detection
    # ========================================================================
    
    def detect_speech_segments(
        self,
        audio_path: str,
        aggressiveness: int = 2
    ) -> List[Tuple[float, float]]:
        """
        Detect speech segments (when someone is talking).
        
        Args:
            audio_path: Path to audio file
            aggressiveness: VAD aggressiveness (0-3, higher = more aggressive)
            
        Returns:
            List of (start, end) tuples for speech segments
        """
        try:
            import webrtcvad
        except ImportError:
            logger.warning("webrtcvad not installed, using energy-based speech detection")
            return self._detect_speech_energy_based(audio_path)
        
        # Load audio at 16kHz (required for webrtcvad)
        y, sr = self.load_audio(audio_path)
        
        # Convert to 16-bit PCM
        audio_int16 = (y * 32767).astype(np.int16)
        
        vad = webrtcvad.Vad(aggressiveness)
        
        # Process in 30ms frames (480 samples at 16kHz)
        frame_duration = 30  # ms
        frame_size = int(self.sample_rate * frame_duration / 1000)
        
        speech_segments = []
        is_speech_active = False
        segment_start = None
        
        for i in range(0, len(audio_int16) - frame_size, frame_size):
            frame = audio_int16[i:i + frame_size].tobytes()
            timestamp = i / self.sample_rate
            
            try:
                is_speech = vad.is_speech(frame, self.sample_rate)
                
                if is_speech and not is_speech_active:
                    # Speech started
                    segment_start = timestamp
                    is_speech_active = True
                elif not is_speech and is_speech_active:
                    # Speech ended
                    speech_segments.append((segment_start, timestamp))
                    is_speech_active = False
            except:
                # Skip problematic frames
                continue
        
        # Close final segment if still active
        if is_speech_active and segment_start is not None:
            speech_segments.append((segment_start, len(audio_int16) / self.sample_rate))
        
        logger.info(f"Detected {len(speech_segments)} speech segments")
        return speech_segments
    
    def _detect_speech_energy_based(self, audio_path: str) -> List[Tuple[float, float]]:
        """Fallback speech detection using energy."""
        librosa = self._get_librosa()
        y, sr = self.load_audio(audio_path)
        
        # Compute RMS energy
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        times = librosa.times_like(rms, sr=sr, hop_length=512)
        
        # Threshold at median + std
        threshold = np.median(rms) + np.std(rms)
        
        # Find speech regions
        is_speech = rms > threshold
        
        segments = []
        start = None
        
        for i, speaking in enumerate(is_speech):
            if speaking and start is None:
                start = times[i]
            elif not speaking and start is not None:
                if times[i] - start > 0.3:  # Min 0.3s
                    segments.append((start, times[i]))
                start = None
        
        return segments
    
    # ========================================================================
    # Music Analysis
    # ========================================================================
    
    def analyze_tempo(self, audio_path: str) -> Dict[str, Any]:
        """
        Detect BPM and tempo changes.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with tempo analysis:
            {
                "bpm": 120.0,
                "beat_times": [0.5, 1.0, 1.5, ...],
                "tempo_changes": [(5.0, 140.0), ...]  # (time, new_bpm)
            }
        """
        librosa = self._get_librosa()
        
        y, sr = self.load_audio(audio_path)
        
        # Detect tempo and beats
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        return {
            "bpm": float(tempo),
            "beat_times": beat_times.tolist(),
            "tempo_changes": []  # Could be enhanced with librosa.beat.plp
        }
    
    def classify_mood(
        self,
        audio_path: str,
        use_ml: bool = False
    ) -> str:
        """
        Classify overall audio mood.
        
        Args:
            audio_path: Path to audio file
            use_ml: Whether to use ML model (requires additional dependencies)
            
        Returns:
            Mood string: "upbeat", "calm", "dramatic", "energetic", "melancholic"
        """
        if use_ml:
            return self._classify_mood_ml(audio_path)
        else:
            return self._classify_mood_heuristic(audio_path)
    
    def _classify_mood_heuristic(self, audio_path: str) -> str:
        """Simple heuristic-based mood classification."""
        librosa = self._get_librosa()
        
        y, sr = self.load_audio(audio_path)
        
        # Extract features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        rms = np.mean(librosa.feature.rms(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # Simple rules
        if tempo > 140 and rms > 0.05:
            return "energetic"
        elif tempo > 120 and spectral_centroid > 2000:
            return "upbeat"
        elif tempo < 80 and rms < 0.03:
            return "calm"
        elif rms > 0.08:
            return "dramatic"
        elif tempo < 90:
            return "melancholic"
        else:
            return "neutral"
    
    def _classify_mood_ml(self, audio_path: str) -> str:
        """ML-based mood classification (placeholder)."""
        # Could integrate with models like:
        # - essentia (music analysis)
        # - YAMNet (audio classification)
        # - openl3 (audio embeddings)
        logger.warning("ML mood classification not implemented, using heuristic")
        return self._classify_mood_heuristic(audio_path)
    
    # ========================================================================
    # Comprehensive Analysis
    # ========================================================================
    
    def extract_full_context(
        self,
        audio_path: str,
        transcribe: bool = True,
        model_size: str = "base"
    ) -> Dict[str, Any]:
        """
        Extract comprehensive audio context for LLM.
        
        Args:
            audio_path: Path to audio file
            transcribe: Whether to transcribe speech
            model_size: Whisper model size if transcribing
            
        Returns:
            Complete audio context dictionary ready for LLM
        """
        logger.info(f"Extracting full audio context from: {audio_path}")
        
        context = {}
        
        # Basic audio events (always extract)
        context["energy_peaks"] = self.extract_energy_peaks(audio_path)
        context["silence_segments"] = self.detect_silence(audio_path)
        
        # Speech detection
        speech_segments = self.detect_speech_segments(audio_path)
        context["speech_segments"] = speech_segments
        context["has_speech"] = len(speech_segments) > 0
        
        # Transcription (optional, slow)
        if transcribe and context["has_speech"]:
            try:
                transcription = self.transcribe_audio(audio_path, model_size=model_size)
                context["transcription"] = transcription
                context["key_phrases"] = self.extract_key_phrases(transcription)
            except Exception as e:
                logger.error(f"Transcription failed: {e}")
                context["transcription"] = []
                context["key_phrases"] = []
        else:
            context["transcription"] = []
            context["key_phrases"] = []
        
        # Music analysis
        try:
            tempo_info = self.analyze_tempo(audio_path)
            context["tempo"] = tempo_info
            context["mood"] = self.classify_mood(audio_path)
        except Exception as e:
            logger.warning(f"Music analysis failed: {e}")
            context["tempo"] = {"bpm": None, "beat_times": []}
            context["mood"] = "unknown"
        
        logger.info(f"Audio context extracted: {len(context['transcription'])} segments, "
                   f"{len(context['key_phrases'])} key phrases, mood: {context['mood']}")
        
        return context
    
    # ========================================================================
    # Legacy compatibility
    # ========================================================================
    
    def get_audio_events(self, audio_path: str) -> Dict[str, Any]:
        """
        Extract basic audio events (legacy method for backward compatibility).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with energy_peaks and silence_segments
        """
        return {
            "energy_peaks": self.extract_energy_peaks(audio_path),
            "silence_segments": self.detect_silence(audio_path)
        }