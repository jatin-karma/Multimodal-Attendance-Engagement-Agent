# ============================================
# Multimodal Attendance & Engagement Agent
# Production Dockerfile for Google Cloud Run / Container Deployment
# ============================================

FROM python:3.11-slim

# Avoid prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install essential system libraries for OpenCV, audio processing, and ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python packages and Gunicorn production WSGI server
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy application source code
COPY . .

# Ensure storage and upload directories exist
RUN mkdir -p uploads storage/attendance_logs storage/generated_audio storage/generated_code

# Cloud Run injects PORT environment variable (default 8080)
ENV PORT=8080

EXPOSE 8080

# Production launch with Gunicorn
CMD exec gunicorn --bind 0.0.0.0:${PORT} --workers 2 --threads 4 --timeout 120 "app:create_app()"
