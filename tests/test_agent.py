"""
Tests for the Multimodal Agent Module.

Tests signal fusion, exponential smoothing, rule-based reasoning,
attention drop detection, interaction peaks, and session metrics aggregation.
"""

import pytest
import numpy as np
from modules.agent import (
    MultimodalFusionEngine,
    RuleBasedReasoningEngine,
    MultimodalAgent
)


class TestMultimodalFusionEngine:
    """Test multimodal signal fusion logic."""

    def setup_method(self):
        self.engine = MultimodalFusionEngine(visual_weight=0.6, audio_weight=0.4, smoothing_factor=0.3)

    def test_weight_normalization(self):
        """Test weights are properly normalized to sum to 1.0."""
        e = MultimodalFusionEngine(visual_weight=3.0, audio_weight=1.0)
        assert abs(e.visual_weight - 0.75) < 1e-4
        assert abs(e.audio_weight - 0.25) < 1e-4

    def test_calculate_visual_engagement_empty(self):
        """Test visual score with no faces detected."""
        score = self.engine.calculate_visual_engagement({"faces_detected": 0})
        assert score == 0.1

    def test_calculate_visual_engagement_all_forward(self):
        """Test visual score when all detected faces are looking forward."""
        frame_data = {
            "faces_detected": 4,
            "looking_forward_count": 4,
            "poses": [
                {"yaw": 0.05, "pitch": -0.02},
                {"yaw": -0.04, "pitch": 0.01},
                {"yaw": 0.02, "pitch": 0.03},
                {"yaw": -0.01, "pitch": -0.01}
            ]
        }
        score = self.engine.calculate_visual_engagement(frame_data)
        assert score > 0.85

    def test_fuse_signals_synchronization(self):
        """Test time alignment of visual and audio streams into intervals."""
        visual_timeline = [
            {"time": 1.0, "faces_detected": 5, "looking_forward_count": 4, "poses": []},
            {"time": 3.0, "faces_detected": 5, "looking_forward_count": 4, "poses": []},
            {"time": 6.0, "faces_detected": 5, "looking_forward_count": 3, "poses": []},
            {"time": 8.0, "faces_detected": 5, "looking_forward_count": 2, "poses": []}
        ]
        audio_timeline = [
            {
                "window_index": 0, "start_time": 0.0, "end_time": 5.0,
                "speech_ratio": 0.3, "db_level": -22.0, "interaction_level": "active_discussion",
                "audio_engagement_score": 0.75
            },
            {
                "window_index": 1, "start_time": 5.0, "end_time": 10.0,
                "speech_ratio": 0.05, "db_level": -35.0, "interaction_level": "passive_listening",
                "audio_engagement_score": 0.60
            }
        ]

        fused = self.engine.fuse_signals(visual_timeline, audio_timeline, interval_seconds=5.0)
        assert len(fused) == 2
        assert fused[0]["start_time"] == 0.0
        assert fused[0]["end_time"] == 5.0
        assert 0.0 <= fused[0]["engagement_score"] <= 1.0
        assert 0.0 <= fused[1]["engagement_score"] <= 1.0


class TestRuleBasedReasoningEngine:
    """Test rule-based heuristics and alerts."""

    def setup_method(self):
        self.reasoning = RuleBasedReasoningEngine(high_thresh=0.7, low_thresh=0.3)

    def test_classification(self):
        assert self.reasoning.classify_engagement_level(0.85) == "high"
        assert self.reasoning.classify_engagement_level(0.55) == "moderate"
        assert self.reasoning.classify_engagement_level(0.20) == "low"

    def test_attention_drop_detection(self):
        """Test alert when score drops sharply (> 0.18)."""
        timeline = [
            {"start_time": 0.0, "end_time": 5.0, "engagement_score": 0.75, "speech_ratio": 0.1, "looking_forward_ratio": 0.8},
            {"start_time": 5.0, "end_time": 10.0, "engagement_score": 0.45, "speech_ratio": 0.05, "looking_forward_ratio": 0.4}
        ]
        dynamics = self.reasoning.analyze_session_dynamics(timeline)
        assert dynamics["attention_drops_count"] == 1
        assert len(dynamics["attention_drops"]) == 1
        assert dynamics["attention_drops"][0]["drop_amount"] == 0.30

    def test_interaction_peak_detection(self):
        """Test peak detected when speech is high and faces look forward."""
        timeline = [
            {"start_time": 0.0, "end_time": 5.0, "engagement_score": 0.82, "speech_ratio": 0.40, "looking_forward_ratio": 0.75}
        ]
        dynamics = self.reasoning.analyze_session_dynamics(timeline)
        assert dynamics["interaction_peaks_count"] == 1

    def test_pacing_alert_on_sustained_low(self):
        """Test alert raised when low engagement is sustained."""
        timeline = [
            {"start_time": 0.0, "end_time": 5.0, "engagement_score": 0.25, "speech_ratio": 0.0, "looking_forward_ratio": 0.2},
            {"start_time": 5.0, "end_time": 10.0, "engagement_score": 0.22, "speech_ratio": 0.0, "looking_forward_ratio": 0.15}
        ]
        dynamics = self.reasoning.analyze_session_dynamics(timeline)
        assert dynamics["pacing_alerts_count"] >= 1


class TestMultimodalAgentIntegration:
    """Test end-to-end multimodal analysis."""

    def test_analyze_session_pipeline(self):
        agent = MultimodalAgent()

        mock_vision = {
            "attendance": {
                "session_id": "test_20261017",
                "total_detected": 10,
                "total_present": 9
            },
            "engagement_data": [
                {"time": 1.0, "faces_detected": 10, "looking_forward_count": 9, "poses": []},
                {"time": 4.0, "faces_detected": 10, "looking_forward_count": 8, "poses": []},
                {"time": 7.0, "faces_detected": 10, "looking_forward_count": 8, "poses": []}
            ]
        }

        mock_audio = {
            "dominant_interaction": "active_discussion",
            "speech_percentage": 35.0,
            "timeline": [
                {
                    "start_time": 0.0, "end_time": 5.0,
                    "audio_engagement_score": 0.78, "speech_ratio": 0.35,
                    "db_level": -20.0, "interaction_level": "active_discussion"
                },
                {
                    "start_time": 5.0, "end_time": 10.0,
                    "audio_engagement_score": 0.72, "speech_ratio": 0.30,
                    "db_level": -24.0, "interaction_level": "active_discussion"
                }
            ]
        }

        result = agent.analyze_session(mock_vision, mock_audio, interval_seconds=5.0)

        assert result["status"] == "success"
        metrics = result["session_metrics"]
        assert metrics["total_detected_students"] == 10
        assert metrics["confirmed_present"] == 9
        assert metrics["attendance_rate_percent"] == 90.0
        assert 0.0 <= metrics["average_engagement_score"] <= 1.0
        assert metrics["engagement_classification"] in ("high", "moderate", "low")
        assert len(result["fused_timeline"]) > 0
