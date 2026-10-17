"""
Audio Module — Audio Signal Extraction & Speech/Noise Analysis.

This module handles:
- Audio signal loading, extraction, and preprocessing
- Noise level detection and environmental noise classification
- Privacy-preserving speech activity detection (Voice Activity Detection without content/speech recognition)
- Spectral feature extraction (RMS, Zero-Crossing Rate, Spectral Centroid)
- Rolling audio window analysis and engagement indicator estimation
"""

import os
import numpy as np
from config import AUDIO_CONFIG
from utils.audio_utils import (
    is_valid_audio_file,
    get_audio_duration,
    normalize_audio,
    calculate_rms_energy
)

# Optional librosa import with graceful fallback to scipy/numpy
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


class AudioExtractor:
    """
    Handles audio extraction, loading, and preprocessing from files or audio streams.
    """

    def __init__(self, target_sr=None):
        """
        Initialize the AudioExtractor.
        
        Args:
            target_sr (int, optional): Target sample rate. Defaults to AUDIO_CONFIG['sample_rate'].
        """
        self.target_sr = target_sr or AUDIO_CONFIG["sample_rate"]

    def load_audio(self, file_path, sr=None):
        """
        Load an audio file as a mono floating-point numpy array.
        
        Args:
            file_path (str): Path to audio file.
            sr (int, optional): Target sample rate.
            
        Returns:
            tuple[np.ndarray, int]: Audio signal array in range [-1.0, 1.0] and sample rate.
            
        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file cannot be decoded.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        target_sr = sr or self.target_sr

        if HAS_LIBROSA:
            try:
                y, actual_sr = librosa.load(file_path, sr=target_sr, mono=True)
                return y.astype(np.float32), actual_sr
            except Exception as e:
                pass  # Fallback to soundfile or wave

        if HAS_SOUNDFILE:
            try:
                data, actual_sr = sf.read(file_path, dtype="float32")
                if data.ndim > 1:
                    data = np.mean(data, axis=1)  # Convert to mono
                if actual_sr != target_sr and HAS_LIBROSA:
                    data = librosa.resample(data, orig_sr=actual_sr, target_sr=target_sr)
                    actual_sr = target_sr
                return data, actual_sr
            except Exception as e:
                pass

        # Native wave fallback for .wav files
        import wave
        try:
            with wave.open(file_path, "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                actual_sr = wf.getframerate()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)

                if sampwidth == 2:
                    dtype = np.int16
                    max_val = 32768.0
                elif sampwidth == 1:
                    dtype = np.uint8
                    max_val = 128.0
                elif sampwidth == 4:
                    dtype = np.int32
                    max_val = 2147483648.0
                else:
                    raise ValueError(f"Unsupported sample width: {sampwidth}")

                audio_data = np.frombuffer(raw_data, dtype=dtype).astype(np.float32) / max_val
                if n_channels > 1:
                    audio_data = audio_data.reshape(-1, n_channels).mean(axis=1)

                return audio_data, actual_sr
        except Exception as e:
            raise ValueError(f"Failed to load audio from {file_path}: {e}")

    def generate_synthetic_audio(self, duration_seconds=10.0, pattern="speech_and_silence"):
        """
        Generate synthetic audio signals for testing or demonstration without microphone.
        
        Args:
            duration_seconds (float): Duration in seconds.
            pattern (str): Type of pattern ('speech_and_silence', 'continuous_noise', 'silent').
            
        Returns:
            tuple[np.ndarray, int]: Synthetic signal and sample rate.
        """
        sr = self.target_sr
        total_samples = int(duration_seconds * sr)
        t = np.linspace(0, duration_seconds, total_samples, endpoint=False)

        if pattern == "silent":
            return np.zeros(total_samples, dtype=np.float32), sr

        if pattern == "continuous_noise":
            noise = np.random.normal(0, 0.05, total_samples).astype(np.float32)
            return noise, sr

        # Default pattern: speech-like harmonics interspersed with quiet periods
        signal = np.zeros(total_samples, dtype=np.float32)
        # Period 1 (0 to 3s): Speech burst (fundamental 150Hz + harmonics)
        mask1 = (t >= 0.5) & (t <= 3.5)
        signal[mask1] += 0.3 * np.sin(2 * np.pi * 150 * t[mask1])
        signal[mask1] += 0.15 * np.sin(2 * np.pi * 300 * t[mask1])
        signal[mask1] += 0.08 * np.sin(2 * np.pi * 600 * t[mask1])

        # Period 2 (3.5 to 6s): Pause / background noise
        mask2 = (t > 3.5) & (t < 6.0)
        signal[mask2] += np.random.normal(0, 0.005, np.sum(mask2)).astype(np.float32)

        # Period 3 (6 to 9s): Discussion / active interaction
        mask3 = (t >= 6.0) & (t <= 9.0)
        signal[mask3] += 0.25 * np.sin(2 * np.pi * 220 * t[mask3])
        signal[mask3] += 0.12 * np.sin(2 * np.pi * 440 * t[mask3])

        # Add subtle ambient floor noise
        signal += np.random.normal(0, 0.003, total_samples).astype(np.float32)
        return signal.astype(np.float32), sr


class AudioAnalyzer:
    """
    Analyzes environmental audio to detect noise levels, speech activity,
    and interaction dynamics without content transcription (privacy-preserving).
    """

    def __init__(self, sample_rate=None):
        """Initialize analyzer with audio configuration parameters."""
        self.sr = sample_rate or AUDIO_CONFIG["sample_rate"]
        self.noise_threshold_db = AUDIO_CONFIG["noise_threshold_db"]
        self.speech_energy_threshold = AUDIO_CONFIG["speech_energy_threshold"]
        self.silence_threshold_seconds = AUDIO_CONFIG["silence_threshold_seconds"]
        self.hop_length = AUDIO_CONFIG["hop_length"]
        self.n_fft = AUDIO_CONFIG["n_fft"]

    def compute_db(self, rms_energy, ref=1.0):
        """
        Convert RMS energy to decibels (dB).
        
        Args:
            rms_energy (float): RMS amplitude.
            ref (float): Reference amplitude.
            
        Returns:
            float: Decibel level (dB).
        """
        if rms_energy <= 1e-9:
            return -100.0
        return float(20.0 * np.log10(rms_energy / ref))

    def compute_zero_crossing_rate(self, signal):
        """
        Calculate zero crossing rate of an audio signal.
        
        Args:
            signal (np.ndarray): Audio signal array.
            
        Returns:
            float: Zero crossing rate between 0 and 1.
        """
        if len(signal) < 2:
            return 0.0
        zero_crossings = np.nonzero(np.diff(signal > 0))[0]
        return float(len(zero_crossings) / (len(signal) - 1))

    def compute_spectral_centroid(self, signal):
        """
        Calculate spectral centroid (frequency center of mass) in Hz.
        
        Args:
            signal (np.ndarray): Audio signal.
            
        Returns:
            float: Spectral centroid in Hz.
        """
        if len(signal) < 16:
            return 0.0

        if HAS_LIBROSA:
            try:
                centroids = librosa.feature.spectral_centroid(
                    y=signal, sr=self.sr, n_fft=min(len(signal), self.n_fft)
                )
                return float(np.mean(centroids))
            except Exception:
                pass

        # Pure numpy FFT fallback
        fft_vals = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), 1.0 / self.sr)
        sum_fft = np.sum(fft_vals)
        if sum_fft == 0:
            return 0.0
        return float(np.sum(freqs * fft_vals) / sum_fft)

    def detect_speech_activity(self, signal):
        """
        Detect speech activity in an audio chunk without content recognition.
        
        Combines RMS energy, zero crossing rate (typical for human speech 0.02 - 0.25),
        and spectral centroid (speech center usually 300Hz - 3500Hz).
        
        Args:
            signal (np.ndarray): Audio segment.
            
        Returns:
            dict: Analysis results containing:
                - is_speech (bool)
                - rms_energy (float)
                - db_level (float)
                - zcr (float)
                - spectral_centroid (float)
                - classification (str): 'silence', 'speech', 'background_noise', or 'loud_noise'
        """
        rms = calculate_rms_energy(signal)
        db = self.compute_db(rms)
        zcr = self.compute_zero_crossing_rate(signal)
        centroid = self.compute_spectral_centroid(signal)

        # Classification heuristics
        if db < self.noise_threshold_db or rms < 0.002:
            classification = "silence"
            is_speech = False
        elif rms >= self.speech_energy_threshold:
            # Human speech characteristics:
            # - Spectral centroid typically between 100Hz and 4500Hz
            # - ZCR typical for speech frames is moderate (0.005 to 0.40)
            if 120 <= centroid <= 4500 and 0.008 <= zcr <= 0.40:
                if db > -3.0:
                    classification = "loud_noise"
                    is_speech = False
                else:
                    classification = "speech"
                    is_speech = True
            elif db > -10.0:
                classification = "loud_noise"
                is_speech = False
            else:
                classification = "background_noise"
                is_speech = False
        else:
            classification = "background_noise"
            is_speech = False

        return {
            "is_speech": is_speech,
            "rms_energy": round(rms, 5),
            "db_level": round(db, 2),
            "zcr": round(zcr, 4),
            "spectral_centroid": round(centroid, 1),
            "classification": classification
        }

    def classify_interaction_level(self, speech_ratio, avg_db, noise_ratio):
        """
        Classify overall classroom interaction dynamic.
        
        Args:
            speech_ratio (float): Fraction of time speech was detected [0, 1].
            avg_db (float): Average decibel level.
            noise_ratio (float): Fraction of time non-speech noise was dominant.
            
        Returns:
            str: One of 'silent', 'passive_listening', 'active_discussion', 'disruptive_noise'.
        """
        if avg_db < -50 or (speech_ratio < 0.05 and noise_ratio < 0.05):
            return "silent"
        if noise_ratio > 0.40 and avg_db > -20:
            return "disruptive_noise"
        if speech_ratio >= 0.25:
            return "active_discussion"
        return "passive_listening"


class RollingAudioProcessor:
    """
    Processes audio streams or recorded signals in rolling windows
    to generate timeline trends of noise, speech, and classroom engagement.
    """

    def __init__(self, sample_rate=None, window_seconds=None):
        """
        Initialize rolling processor.
        
        Args:
            sample_rate (int, optional): Audio sample rate.
            window_seconds (float, optional): Window size in seconds.
        """
        self.sr = sample_rate or AUDIO_CONFIG["sample_rate"]
        self.window_seconds = window_seconds or AUDIO_CONFIG["analysis_window_seconds"]
        self.analyzer = AudioAnalyzer(sample_rate=self.sr)

    def process_timeline(self, audio_signal):
        """
        Divide audio signal into consecutive windows and analyze each.
        
        Args:
            audio_signal (np.ndarray): Full audio signal array.
            
        Returns:
            list[dict]: List of window analysis results.
        """
        window_size = int(self.window_seconds * self.sr)
        total_samples = len(audio_signal)
        
        if total_samples == 0:
            return []

        # Ensure at least 1 window
        num_windows = max(1, int(np.ceil(total_samples / window_size)))
        timeline = []

        sub_chunk_size = int(0.5 * self.sr)  # 0.5s sub-chunks to measure speech ratio

        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = min(start_idx + window_size, total_samples)
            window_data = audio_signal[start_idx:end_idx]

            start_sec = start_idx / self.sr
            end_sec = end_idx / self.sr

            # Analyze sub-chunks within window to compute speech/noise ratios
            sub_chunks = []
            for sub_start in range(0, len(window_data), sub_chunk_size):
                sub_end = min(sub_start + sub_chunk_size, len(window_data))
                sub_chunk = window_data[sub_start:sub_end]
                if len(sub_chunk) > 100:
                    sub_chunks.append(self.analyzer.detect_speech_activity(sub_chunk))

            if not sub_chunks:
                sub_chunks = [self.analyzer.detect_speech_activity(window_data)]

            speech_count = sum(1 for c in sub_chunks if c["is_speech"])
            noise_count = sum(1 for c in sub_chunks if c["classification"] in ("background_noise", "loud_noise"))
            silence_count = sum(1 for c in sub_chunks if c["classification"] == "silence")
            total_sub = len(sub_chunks)

            speech_ratio = speech_count / total_sub
            noise_ratio = noise_count / total_sub
            silence_ratio = silence_count / total_sub

            overall_analysis = self.analyzer.detect_speech_activity(window_data)
            interaction = self.analyzer.classify_interaction_level(
                speech_ratio, overall_analysis["db_level"], noise_ratio
            )

            # Audio engagement score for this window:
            # - Optimal engagement: Moderate speech (15-50%) + attentive listening (not dead silent or chaos)
            if interaction == "active_discussion":
                audio_score = min(1.0, 0.70 + speech_ratio * 0.3)
            elif interaction == "passive_listening":
                audio_score = 0.65  # Focused lecture listening
            elif interaction == "disruptive_noise":
                audio_score = max(0.1, 0.40 - noise_ratio * 0.3)
            else:  # silent
                audio_score = 0.40  # Inactive or independent reading

            timeline.append({
                "window_index": i,
                "start_time": round(start_sec, 2),
                "end_time": round(end_sec, 2),
                "duration": round(end_sec - start_sec, 2),
                "rms_energy": overall_analysis["rms_energy"],
                "db_level": overall_analysis["db_level"],
                "speech_ratio": round(speech_ratio, 3),
                "silence_ratio": round(silence_ratio, 3),
                "noise_ratio": round(noise_ratio, 3),
                "interaction_level": interaction,
                "audio_engagement_score": round(audio_score, 3)
            })

        return timeline


def process_audio(audio_source, sample_rate=None, is_synthetic=False, synthetic_pattern="speech_and_silence"):
    """
    Main entry point for the Audio Module.
    
    Processes an audio source (file path or synthetic generator) and returns
    comprehensive session analytics and timeline data.
    
    Args:
        audio_source (str or None): Path to audio file. If None or is_synthetic=True, generates synthetic audio.
        sample_rate (int, optional): Audio sampling rate.
        is_synthetic (bool): Force synthetic audio generation.
        synthetic_pattern (str): Synthetic pattern type if synthetic.
        
    Returns:
        dict: Complete audio session summary containing:
            - duration_seconds
            - sample_rate
            - average_db
            - speech_percentage
            - silence_percentage
            - noise_percentage
            - dominant_interaction
            - overall_audio_score
            - timeline (list of window dicts)
    """
    extractor = AudioExtractor(target_sr=sample_rate)

    if is_synthetic or audio_source is None:
        signal, sr = extractor.generate_synthetic_audio(duration_seconds=15.0, pattern=synthetic_pattern)
    else:
        signal, sr = extractor.load_audio(audio_source, sr=sample_rate)

    processor = RollingAudioProcessor(sample_rate=sr)
    timeline = processor.process_timeline(signal)

    duration = get_audio_duration(signal, sr)

    if timeline:
        avg_db = float(np.mean([w["db_level"] for w in timeline]))
        avg_speech = float(np.mean([w["speech_ratio"] for w in timeline]))
        avg_silence = float(np.mean([w["silence_ratio"] for w in timeline]))
        avg_noise = float(np.mean([w["noise_ratio"] for w in timeline]))
        avg_score = float(np.mean([w["audio_engagement_score"] for w in timeline]))

        # Find dominant interaction
        interactions = [w["interaction_level"] for w in timeline]
        dominant_interaction = max(set(interactions), key=interactions.count)
    else:
        avg_db = -100.0
        avg_speech = 0.0
        avg_silence = 1.0
        avg_noise = 0.0
        avg_score = 0.0
        dominant_interaction = "silent"

    return {
        "status": "success",
        "duration_seconds": round(duration, 2),
        "sample_rate": sr,
        "average_db": round(avg_db, 2),
        "speech_percentage": round(avg_speech * 100, 1),
        "silence_percentage": round(avg_silence * 100, 1),
        "noise_percentage": round(avg_noise * 100, 1),
        "dominant_interaction": dominant_interaction,
        "overall_audio_score": round(avg_score, 3),
        "timeline_windows_count": len(timeline),
        "timeline": timeline
    }
