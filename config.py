"""
Configuration settings for the Multimodal Attendance & Engagement Agent.

This module centralizes all configurable parameters including file paths,
model settings, detection thresholds, and API configurations.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================
# General Application Settings
# ============================================
APP_NAME = "Multimodal Attendance & Engagement Agent"
DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")
HOST = os.getenv("FLASK_HOST", "0.0.0.0")
PORT = int(os.getenv("FLASK_PORT", 5000))

# ============================================
# File & Directory Paths
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, os.getenv("UPLOAD_FOLDER", "uploads"))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
ATTENDANCE_LOG_DIR = os.path.join(STORAGE_DIR, "attendance_logs")
GENERATED_AUDIO_DIR = os.path.join(STORAGE_DIR, "generated_audio")
GENERATED_CODE_DIR = os.path.join(STORAGE_DIR, "generated_code")

# Allowed file extensions for uploads
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a"}

# ============================================
# Vision Module Settings
# ============================================
VISION_CONFIG = {
    "min_detection_confidence": 0.5,    # Minimum confidence for face detection
    "min_tracking_confidence": 0.5,     # Minimum confidence for face tracking
    "max_num_faces": 30,                # Maximum number of faces to detect
    "presence_threshold_seconds": 10,   # Seconds of sustained presence for attendance
    "frame_skip": 2,                    # Process every Nth frame for performance
    "attendance_check_interval": 5,     # Seconds between attendance checks
}

# ============================================
# Audio Module Settings
# ============================================
AUDIO_CONFIG = {
    "sample_rate": 22050,               # Audio sample rate in Hz
    "hop_length": 512,                  # Hop length for audio analysis
    "n_fft": 2048,                      # FFT window size
    "noise_threshold_db": -40,          # dB threshold for noise detection
    "speech_energy_threshold": 0.01,    # Energy threshold for speech detection
    "analysis_window_seconds": 5,       # Window size for rolling analysis
    "silence_threshold_seconds": 3,     # Seconds of silence to flag inactivity
}

# ============================================
# Engagement Scoring Settings
# ============================================
ENGAGEMENT_CONFIG = {
    "visual_weight": 0.6,               # Weight for visual engagement signals
    "audio_weight": 0.4,                # Weight for audio engagement signals
    "high_engagement_threshold": 0.7,   # Score >= this = high engagement
    "low_engagement_threshold": 0.3,    # Score <= this = low engagement
    "trend_window_minutes": 5,          # Window for trend calculation
    "score_smoothing_factor": 0.3,      # Exponential smoothing factor
}

# ============================================
# AI / GenAI Settings (Google Gemini)
# ============================================
GEMINI_CONFIG = {
    "api_key": os.getenv("GEMINI_API_KEY", ""),
    "model_name": "gemini-2.0-flash",
    "max_output_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
}

# ============================================
# Dashboard Settings
# ============================================
DASHBOARD_CONFIG = {
    "refresh_interval_seconds": 10,     # Auto-refresh interval
    "chart_theme": "plotly_dark",        # Plotly chart theme
    "max_display_students": 50,         # Max students shown in table
}


def ensure_directories():
    """Create all required directories if they don't exist."""
    directories = [
        UPLOAD_FOLDER,
        STORAGE_DIR,
        ATTENDANCE_LOG_DIR,
        GENERATED_AUDIO_DIR,
        GENERATED_CODE_DIR,
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# Ensure directories exist on import
ensure_directories()
