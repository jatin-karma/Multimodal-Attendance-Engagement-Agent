"""
End-to-End and Integration Tests for Epic 6.

Validates:
- Activity 6.1: Core application running & API endpoints
- Activity 6.2: Streamlit dashboard logic & synthetic benchmark
- Activity 6.3: System accuracy, edge cases (empty upload, zero students, pure silence, loud noise)
"""

import io
import os
import json
import pytest
from app import create_app
from dashboard import run_synthetic_session
from modules.agent import MultimodalAgent
from modules.audio import process_audio


@pytest.fixture
def app():
    """Create Flask application in testing mode."""
    application = create_app()
    application.config["TESTING"] = True
    return application


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestCoreApplicationEndpoints:
    """Validate all Flask API Gateway routes (Activity 6.1)."""

    def test_health_check(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "healthy"

    def test_api_status(self, client):
        res = client.get("/api/status")
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "operational"
        assert "modules" in data

    def test_web_routes(self, client):
        res_home = client.get("/")
        assert res_home.status_code == 200
        res_dash = client.get("/dashboard")
        assert res_dash.status_code == 200

    def test_upload_api_no_file(self, client):
        res = client.post("/api/upload")
        assert res.status_code == 400

    def test_upload_api_valid_audio(self, client):
        data = {
            "file": (io.BytesIO(b"RIFFdummydataWAVEfmt "), "sample.wav")
        }
        res = client.post("/api/upload", data=data, content_type="multipart/form-data")
        assert res.status_code == 201
        assert res.get_json()["status"] == "success"

    def test_full_pipeline_analyze_endpoint(self, client):
        """Test full pipeline analysis execution via API Gateway."""
        res = client.post("/api/analyze", json={"use_synthetic": True})
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "success"
        assert "session_metrics" in data
        assert "attendance" in data
        assert "dynamics" in data
        assert "teaching_suggestions" in data
        assert "student_profiles" in data
        assert len(data["fused_timeline"]) > 0

    def test_get_attendance_and_analytics_after_analyze(self, client):
        """Verify querying attendance and analytics after analysis."""
        client.post("/api/analyze", json={"use_synthetic": True})

        res_att = client.get("/api/attendance")
        assert res_att.status_code == 200
        assert "total_present" in res_att.get_json()

        res_ana = client.get("/api/analytics")
        assert res_ana.status_code == 200
        assert "teaching_suggestions" in res_ana.get_json()


class TestDashboardLogic:
    """Validate Streamlit dashboard logic (Activity 6.2)."""

    def test_synthetic_session_execution(self):
        result = run_synthetic_session(v_weight=0.6, a_weight=0.4)
        assert result is not None
        assert "session_intel" in result
        assert "suggestions" in result
        assert "student_profiles" in result
        assert len(result["student_profiles"]) > 0


class TestEdgeCasesAndRobustness:
    """Validate system robustness under extreme or abnormal inputs (Activity 6.3)."""

    def test_edge_case_zero_students_detected(self):
        """Pipeline handles zero detected faces gracefully without crashing."""
        agent = MultimodalAgent()
        vision_zero = {
            "attendance": {"session_id": "test_zero", "total_detected": 0, "total_present": 0, "records": {}},
            "engagement_data": []
        }
        audio_mock = process_audio(None, is_synthetic=True, synthetic_pattern="silent")
        res = agent.analyze_session(vision_zero, audio_mock)
        assert res["status"] == "success"
        assert res["session_metrics"]["total_detected_students"] == 0
        assert res["session_metrics"]["confirmed_present"] == 0

    def test_edge_case_pure_silence(self):
        """Audio module handles pure silence without division by zero."""
        silence_res = process_audio(None, is_synthetic=True, synthetic_pattern="silent")
        assert silence_res["status"] == "success"
        assert silence_res["silence_percentage"] == 100.0
        assert silence_res["speech_percentage"] == 0.0

    def test_edge_case_continuous_noise(self):
        """Audio module classifies continuous noise appropriately."""
        noise_res = process_audio(None, is_synthetic=True, synthetic_pattern="continuous_noise")
        assert noise_res["status"] == "success"
        assert noise_res["noise_percentage"] > 50.0
