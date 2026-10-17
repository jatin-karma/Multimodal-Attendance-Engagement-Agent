"""
Tests for the Audio Module.

Tests audio signal extraction, noise detection, speech activity classification,
spectral analysis, rolling window processing, and end-to-end pipeline execution.
"""

import os
import wave
import pytest
import numpy as np
from modules.audio import (
    AudioExtractor,
    AudioAnalyzer,
    RollingAudioProcessor,
    process_audio
)
from utils.audio_utils import is_valid_audio_file, calculate_rms_energy, normalize_audio


class TestAudioExtractor:
    """Test audio extraction and signal generation."""

    def setup_method(self):
        self.extractor = AudioExtractor(target_sr=16000)

    def test_synthetic_audio_generation_speech(self):
        """Test generating speech-like synthetic audio."""
        signal, sr = self.extractor.generate_synthetic_audio(duration_seconds=3.0, pattern="speech_and_silence")
        assert sr == 16000
        assert len(signal) == 3 * 16000
        assert isinstance(signal, np.ndarray)
        assert np.max(np.abs(signal)) > 0.05

    def test_synthetic_audio_generation_silent(self):
        """Test generating silent synthetic audio."""
        signal, sr = self.extractor.generate_synthetic_audio(duration_seconds=2.0, pattern="silent")
        assert len(signal) == 2 * 16000
        assert np.all(signal == 0.0)

    def test_load_audio_wav_file(self, tmp_path):
        """Test writing and reading a real WAV file."""
        wav_path = str(tmp_path / "test_sample.wav")
        sr = 16000
        # 1-second 440 Hz tone
        t = np.linspace(0, 1.0, sr, endpoint=False)
        waveform = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(waveform.tobytes())

        loaded_signal, loaded_sr = self.extractor.load_audio(wav_path, sr=sr)
        assert loaded_sr == sr
        assert len(loaded_signal) == sr
        assert abs(np.max(loaded_signal) - 1.0) < 0.05

    def test_load_audio_file_not_found(self):
        """Test error raised when audio file is missing."""
        with pytest.raises(FileNotFoundError):
            self.extractor.load_audio("non_existent_file.wav")


class TestAudioAnalyzer:
    """Test acoustic metrics and classification."""

    def setup_method(self):
        self.analyzer = AudioAnalyzer(sample_rate=16000)

    def test_compute_db_silence(self):
        """Test dB computation for zero or near-zero signal."""
        db = self.analyzer.compute_db(0.0)
        assert db == -100.0

    def test_compute_db_loud(self):
        """Test dB computation for full amplitude."""
        db = self.analyzer.compute_db(1.0)
        assert db == 0.0

    def test_zero_crossing_rate(self):
        """Test ZCR of alternating and flat signals."""
        alternating = np.array([1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32)
        zcr = self.analyzer.compute_zero_crossing_rate(alternating)
        assert zcr == 1.0

        constant = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        assert self.analyzer.compute_zero_crossing_rate(constant) == 0.0

    def test_spectral_centroid(self):
        """Test centroid returns positive Hz frequency for non-silent signal."""
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        signal = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        centroid = self.analyzer.compute_spectral_centroid(signal)
        # 1000 Hz tone centroid should be close to 1000 Hz
        assert 800 <= centroid <= 1200

    def test_detect_speech_activity_silence(self):
        """Test that silence is correctly classified."""
        silence = np.zeros(1600, dtype=np.float32)
        res = self.analyzer.detect_speech_activity(silence)
        assert res["is_speech"] is False
        assert res["classification"] == "silence"

    def test_detect_speech_activity_speech_signal(self):
        """Test synthetic speech tone within speech bounds."""
        t = np.linspace(0, 0.5, 8000, endpoint=False)
        # Speech-like mixture (150Hz fundamental + 300Hz harmonic + small noise)
        signal = (0.3 * np.sin(2 * np.pi * 150 * t) + 0.15 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        res = self.analyzer.detect_speech_activity(signal)
        assert res["is_speech"] is True
        assert res["classification"] == "speech"

    def test_classify_interaction_level(self):
        """Test classification of classroom interaction states."""
        assert self.analyzer.classify_interaction_level(0.01, -60, 0.01) == "silent"
        assert self.analyzer.classify_interaction_level(0.40, -25, 0.10) == "active_discussion"
        assert self.analyzer.classify_interaction_level(0.10, -30, 0.10) == "passive_listening"
        assert self.analyzer.classify_interaction_level(0.05, -15, 0.60) == "disruptive_noise"


class TestRollingAudioProcessor:
    """Test timeline segmentation and window aggregation."""

    def test_process_timeline(self):
        extractor = AudioExtractor(target_sr=16000)
        signal, sr = extractor.generate_synthetic_audio(duration_seconds=6.0, pattern="speech_and_silence")
        processor = RollingAudioProcessor(sample_rate=sr, window_seconds=2.0)
        timeline = processor.process_timeline(signal)

        assert len(timeline) == 3  # 6 seconds / 2s windows = 3 windows
        first_window = timeline[0]
        assert "start_time" in first_window
        assert "end_time" in first_window
        assert "interaction_level" in first_window
        assert "audio_engagement_score" in first_window
        assert 0.0 <= first_window["audio_engagement_score"] <= 1.0


class TestProcessAudioPipeline:
    """Test full process_audio pipeline."""

    def test_process_audio_synthetic(self):
        """Test end-to-end processing with synthetic input."""
        result = process_audio(audio_source=None, sample_rate=16000, is_synthetic=True)
        assert result["status"] == "success"
        assert result["duration_seconds"] > 0
        assert "dominant_interaction" in result
        assert "overall_audio_score" in result
        assert 0.0 <= result["overall_audio_score"] <= 1.0
        assert len(result["timeline"]) > 0

    def test_audio_utils_helpers(self):
        """Test helper functions in utils/audio_utils.py."""
        assert is_valid_audio_file("lecture.wav") is True
        assert is_valid_audio_file("lecture.mp3") is True
        assert is_valid_audio_file("document.pdf") is False

        raw = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
        norm = normalize_audio(raw)
        assert np.max(np.abs(norm)) == 1.0

        rms = calculate_rms_energy(np.array([1.0, -1.0, 1.0, -1.0]))
        assert abs(rms - 1.0) < 1e-4
