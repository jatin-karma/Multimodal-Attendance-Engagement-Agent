"""
Tests for the Analytics and Recommendation Engine Module.

Tests teaching suggestion generation, student profiling, class distribution,
and reporting export in JSON and CSV formats.
"""

import os
import json
import csv
import pytest
from modules.analytics import (
    TeachingRecommendationEngine,
    StudentAnalyticsEngine,
    AnalyticsReportGenerator
)


class TestTeachingRecommendationEngine:
    """Test pedagogical recommendation logic."""

    def setup_method(self):
        # Initialize without Gemini to test offline heuristics
        self.engine = TeachingRecommendationEngine(use_gemini=False)

    def test_suggestions_on_attention_drop(self):
        """Test suggestion generated in response to an attention drop."""
        metrics = {"average_engagement_score": 0.50}
        dynamics = {
            "attention_drops": [{"time_range": "10s-15s", "reason": "Sharp pitch tilt downward"}],
            "pacing_alerts": [],
            "interaction_peaks": []
        }
        suggestions = self.engine.generate_heuristic_suggestions(metrics, dynamics)
        assert len(suggestions) > 0
        assert any(s["type"] == "immediate_action" for s in suggestions)
        assert "10s-15s" in suggestions[0]["suggestion"]

    def test_suggestions_on_pacing_alert(self):
        """Test suggestion generated for prolonged low engagement."""
        metrics = {"average_engagement_score": 0.25}
        dynamics = {
            "attention_drops": [],
            "pacing_alerts": [{"time_range": "15s-25s", "consecutive_intervals": 2}],
            "interaction_peaks": []
        }
        suggestions = self.engine.generate_heuristic_suggestions(metrics, dynamics)
        assert any(s["type"] == "pacing_adjustment" for s in suggestions)

    def test_suggestions_on_high_engagement(self):
        """Test reinforcement suggestion when engagement is high."""
        metrics = {"average_engagement_score": 0.85}
        dynamics = {"attention_drops": [], "pacing_alerts": [], "interaction_peaks": []}
        suggestions = self.engine.generate_heuristic_suggestions(metrics, dynamics)
        assert any(s["type"] == "positive_reinforcement" for s in suggestions)

    def test_get_ai_enhanced_suggestions_fallback(self):
        """Test that get_ai_enhanced_suggestions falls back gracefully offline."""
        metrics = {"average_engagement_score": 0.60}
        dynamics = {"attention_drops": [], "pacing_alerts": [], "interaction_peaks": []}
        res = self.engine.get_ai_enhanced_suggestions(metrics, dynamics)
        assert res["source"] == "rule_based_engine"
        assert res["suggestions_count"] > 0


class TestStudentAnalyticsEngine:
    """Test student profiling and participation classifications."""

    def setup_method(self):
        self.engine = StudentAnalyticsEngine(total_session_seconds=60.0)

    def test_compute_student_profiles(self):
        records = {
            "face_1": {
                "face_id": "face_1",
                "first_detected": 0.0,
                "last_detected": 55.0,
                "total_visible_seconds": 52.0,
                "attendance_marked": True
            },
            "face_2": {
                "face_id": "face_2",
                "first_detected": 10.0,
                "last_detected": 30.0,
                "total_visible_seconds": 8.0,
                "attendance_marked": False
            }
        }
        profiles = self.engine.compute_student_profiles(records, session_duration=60.0)
        assert len(profiles) == 2
        # face_1 was visible for 52s -> Highly Engaged
        assert profiles[0]["student_id"] == "face_1"
        assert profiles[0]["attendance_confirmed"] is True
        assert profiles[0]["classification"] == "Highly Engaged"

        # face_2 was visible for 8s (below 10s threshold) -> Needs Attention
        assert profiles[1]["student_id"] == "face_2"
        assert profiles[1]["attendance_confirmed"] is False
        assert profiles[1]["classification"] == "Needs Attention"

    def test_generate_class_distribution(self):
        profiles = [
            {"classification": "Highly Engaged"},
            {"classification": "Highly Engaged"},
            {"classification": "Moderately Engaged"},
            {"classification": "Passive Observer"}
        ]
        dist = self.engine.generate_class_distribution(profiles)
        assert dist["total_students"] == 4
        assert dist["counts"]["Highly Engaged"] == 2
        assert dist["percentages"]["Highly Engaged"] == 50.0


class TestAnalyticsReportGenerator:
    """Test JSON and CSV report exporting."""

    def test_export_json_report(self, tmp_path):
        generator = AnalyticsReportGenerator(output_dir=str(tmp_path))
        session_analysis = {
            "session_metrics": {"session_id": "test_session_123", "average_engagement_score": 0.75},
            "dynamics": {}
        }
        suggestions = {"structured_suggestions": []}
        student_profiles = [{"student_id": "face_1", "engagement_score": 0.9}]

        filepath = generator.export_json_report(session_analysis, suggestions, student_profiles)
        assert os.path.exists(filepath)

        with open(filepath, "r") as f:
            data = json.load(f)
        assert data["session_metrics"]["session_id"] == "test_session_123"

    def test_export_students_csv(self, tmp_path):
        generator = AnalyticsReportGenerator(output_dir=str(tmp_path))
        student_profiles = [
            {
                "student_id": "face_1",
                "attendance_confirmed": True,
                "visible_duration_seconds": 45.0,
                "presence_percentage": 90.0,
                "engagement_score": 0.88,
                "classification": "Highly Engaged",
                "student_feedback": "Great work"
            }
        ]
        filepath = generator.export_students_csv(student_profiles)
        assert os.path.exists(filepath)

        with open(filepath, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["student_id"] == "face_1"
        assert rows[0]["classification"] == "Highly Engaged"
