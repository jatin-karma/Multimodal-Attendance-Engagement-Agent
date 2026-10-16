"""
Tests for the core application setup and configuration.

Validates that the Flask app initializes correctly, required directories
are created, and configuration values are properly loaded.
"""

import os
import pytest
from app import create_app
from config import (
    APP_NAME, UPLOAD_FOLDER, STORAGE_DIR,
    ATTENDANCE_LOG_DIR, GENERATED_AUDIO_DIR, GENERATED_CODE_DIR
)


@pytest.fixture
def app():
    """Create application for testing."""
    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestAppSetup:
    """Test application initialization and configuration."""

    def test_app_creates_successfully(self, app):
        """Test that Flask app initializes without errors."""
        assert app is not None

    def test_app_name_configured(self):
        """Test that app name is correctly configured."""
        assert APP_NAME == "Multimodal Attendance & Engagement Agent"

    def test_health_endpoint(self, client):
        """Test health check endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"

    def test_api_status_endpoint(self, client):
        """Test API status endpoint returns module info."""
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.get_json()
        assert "modules" in data
        assert data["modules"]["vision"] == "available"
        assert data["modules"]["audio"] == "available"
        assert data["modules"]["agent"] == "available"
        assert data["modules"]["analytics"] == "available"

    def test_index_page_loads(self, client):
        """Test that the index page renders successfully."""
        response = client.get("/")
        assert response.status_code == 200


class TestDirectorySetup:
    """Test that required directories are created."""

    def test_upload_directory_exists(self):
        """Test upload directory is created."""
        assert os.path.isdir(UPLOAD_FOLDER)

    def test_storage_directory_exists(self):
        """Test storage directory is created."""
        assert os.path.isdir(STORAGE_DIR)

    def test_attendance_log_directory_exists(self):
        """Test attendance log directory is created."""
        assert os.path.isdir(ATTENDANCE_LOG_DIR)

    def test_generated_audio_directory_exists(self):
        """Test generated audio directory is created."""
        assert os.path.isdir(GENERATED_AUDIO_DIR)

    def test_generated_code_directory_exists(self):
        """Test generated code directory is created."""
        assert os.path.isdir(GENERATED_CODE_DIR)
