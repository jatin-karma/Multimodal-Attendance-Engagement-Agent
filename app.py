"""
Flask Application Gateway for the Multimodal Attendance & Engagement Agent.

Provides API routes, file ingestion, multimodal pipeline orchestration,
and web presentation according to the technical architecture.
"""

import os
import json
import time
import base64
import numpy as np
import cv2
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import (
    APP_NAME, DEBUG, SECRET_KEY, HOST, PORT,
    UPLOAD_FOLDER, STORAGE_DIR, ATTENDANCE_LOG_DIR,
    ALLOWED_VIDEO_EXTENSIONS, ALLOWED_AUDIO_EXTENSIONS,
    ensure_directories
)
from modules.vision import process_video, FaceDetector, AttendanceTracker
from modules.audio import process_audio
from modules.agent import MultimodalAgent
from modules.analytics import (
    TeachingRecommendationEngine,
    StudentAnalyticsEngine,
    AnalyticsReportGenerator
)

# Global in-memory cache for latest session results
LATEST_SESSION_CACHE = {
    "session_id": None,
    "session_metrics": None,
    "attendance": None,
    "dynamics": None,
    "teaching_suggestions": None,
    "student_profiles": None,
    "fused_timeline": None
}

# Live session in-memory state
LIVE_SESSION_STATE = {
    "active": False,
    "session_id": None,
    "start_time": None,
    "detector": None,
    "tracker": None,
    "timeline": [],
    "audio_energy_levels": []
}


def allowed_file(filename, allowed_extensions):
    """Check if file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Core configuration
    app.config["SECRET_KEY"] = SECRET_KEY
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max

    CORS(app)
    ensure_directories()

    # Initialize modules
    multimodal_agent = MultimodalAgent()
    recommendation_engine = TeachingRecommendationEngine(use_gemini=True)
    analytics_engine = StudentAnalyticsEngine()
    report_generator = AnalyticsReportGenerator()

    # ----------------------------------------
    # Web UI Routes
    # ----------------------------------------
    @app.route("/")
    def index():
        """Render the main landing page."""
        return render_template("index.html", app_name=APP_NAME)

    @app.route("/dashboard")
    def dashboard_view():
        """Render the integrated web dashboard."""
        return render_template("dashboard.html", app_name=APP_NAME, latest=LATEST_SESSION_CACHE)

    # ----------------------------------------
    # Health and Status Endpoints
    # ----------------------------------------
    @app.route("/health")
    def health_check():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "application": APP_NAME,
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        })

    @app.route("/api/status")
    def api_status():
        """API status endpoint showing module operational readiness."""
        return jsonify({
            "application": APP_NAME,
            "status": "operational",
            "modules": {
                "vision": "ready (OpenCV + MediaPipe)",
                "audio": "ready (Librosa + VAD)",
                "agent": "ready (Multimodal Fusion + Heuristics)",
                "analytics": "ready (Gemini AI + Rule-Based Engine)"
            },
            "cached_session": LATEST_SESSION_CACHE["session_id"]
        })

    # ----------------------------------------
    # File Ingestion Endpoint
    # ----------------------------------------
    @app.route("/api/upload", methods=["POST"])
    def upload_file():
        """Upload a video or audio file for session analysis."""
        if "file" not in request.files:
            return jsonify({"error": "No file part in request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        filename = secure_filename(file.filename)
        ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""

        is_video = ext in ALLOWED_VIDEO_EXTENSIONS
        is_audio = ext in ALLOWED_AUDIO_EXTENSIONS

        if not (is_video or is_audio):
            return jsonify({
                "error": f"Unsupported file type. Allowed: {list(ALLOWED_VIDEO_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS)}"
            }), 400

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        return jsonify({
            "status": "success",
            "message": f"File '{filename}' uploaded successfully",
            "filename": filename,
            "filepath": filepath,
            "file_type": "video" if is_video else "audio"
        }), 201

    # ----------------------------------------
    # Full Multimodal Pipeline Execution
    # ----------------------------------------
    @app.route("/api/analyze", methods=["POST"])
    def run_analysis():
        """
        Execute full multimodal pipeline on uploaded file(s) or synthetic demo.
        JSON Payload (optional):
            {
                "video_file": "path_or_filename",
                "audio_file": "path_or_filename",
                "use_synthetic": false
            }
        """
        data = request.get_json(silent=True) or {}
        use_synthetic = data.get("use_synthetic", False)
        video_name = data.get("video_file")
        audio_name = data.get("audio_file")

        # Resolve paths
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], video_name) if video_name else None
        audio_path = os.path.join(app.config["UPLOAD_FOLDER"], audio_name) if audio_name else None

        # 1. Vision Processing
        if video_path and os.path.exists(video_path):
            vision_results = process_video(video_path, max_frames=300)
        else:
            # Synthetic / Mock Vision for testing or demo without live camera
            vision_results = {
                "attendance": {
                    "session_id": f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "total_detected": 6,
                    "total_present": 5,
                    "presence_threshold_seconds": 10,
                    "records": {
                        f"student_{i}": {
                            "face_id": f"student_{i}",
                            "first_detected": 1.0,
                            "last_detected": 45.0,
                            "total_visible_seconds": 40.0 if i < 5 else 6.0,
                            "attendance_marked": i < 5,
                            "status": "present" if i < 5 else "tracking"
                        } for i in range(1, 7)
                    }
                },
                "engagement_data": [
                    {
                        "frame": i * 30,
                        "time": round(i * 1.0, 1),
                        "faces_detected": 6,
                        "poses": [{"yaw": 0.05, "pitch": 0.0, "is_looking_forward": True} for _ in range(5)],
                        "looking_forward_count": 5 if i % 6 != 0 else 2
                    } for i in range(1, 46)
                ],
                "duration_seconds": 45.0
            }

        # 2. Audio Processing
        if audio_path and os.path.exists(audio_path):
            audio_results = process_audio(audio_path)
        else:
            audio_results = process_audio(audio_source=None, is_synthetic=True, synthetic_pattern="speech_and_silence")

        # 3. Multimodal Signal Fusion & Reasoning
        session_intelligence = multimodal_agent.analyze_session(
            vision_results=vision_results,
            audio_results=audio_results,
            interval_seconds=5.0
        )

        session_metrics = session_intelligence["session_metrics"]
        dynamics = session_intelligence["dynamics"]

        # 4. Teaching Suggestions & Student Analytics
        suggestions = recommendation_engine.get_ai_enhanced_suggestions(session_metrics, dynamics)
        records = vision_results.get("attendance", {}).get("records", {})
        student_profiles = analytics_engine.compute_student_profiles(
            records, session_duration=vision_results.get("duration_seconds", 60.0)
        )
        distribution = analytics_engine.generate_class_distribution(student_profiles)

        # 5. Export Reports
        report_json_path = report_generator.export_json_report(session_intelligence, suggestions, student_profiles)
        report_csv_path = report_generator.export_students_csv(student_profiles)

        # Update cache
        LATEST_SESSION_CACHE.update({
            "session_id": session_metrics["session_id"],
            "session_metrics": session_metrics,
            "attendance": vision_results.get("attendance", {}),
            "dynamics": dynamics,
            "teaching_suggestions": suggestions,
            "student_profiles": student_profiles,
            "distribution": distribution,
            "fused_timeline": session_intelligence.get("fused_timeline", []),
            "reports": {
                "json": os.path.basename(report_json_path),
                "csv": os.path.basename(report_csv_path)
            }
        })

        return jsonify({
            "status": "success",
            "session_id": session_metrics["session_id"],
            "session_metrics": session_metrics,
            "attendance": vision_results.get("attendance", {}),
            "dynamics": dynamics,
            "teaching_suggestions": suggestions,
            "student_profiles": student_profiles,
            "distribution": distribution,
            "fused_timeline": session_intelligence.get("fused_timeline", []),
            "reports": LATEST_SESSION_CACHE["reports"]
        })

    # ----------------------------------------
    # Analytics & Attendance Queries
    # ----------------------------------------
    @app.route("/api/attendance")
    def get_attendance():
        """Retrieve the latest attendance records."""
        if not LATEST_SESSION_CACHE["attendance"]:
            return jsonify({"status": "no_active_session", "message": "No session has been analyzed yet."}), 404
        return jsonify(LATEST_SESSION_CACHE["attendance"])

    @app.route("/api/analytics")
    def get_analytics():
        """Retrieve the latest analytics, suggestions, and student profiles."""
        if not LATEST_SESSION_CACHE["session_id"]:
            return jsonify({"status": "no_active_session", "message": "No session has been analyzed yet."}), 404
        return jsonify({
            "session_id": LATEST_SESSION_CACHE["session_id"],
            "session_metrics": LATEST_SESSION_CACHE["session_metrics"],
            "teaching_suggestions": LATEST_SESSION_CACHE["teaching_suggestions"],
            "student_profiles": LATEST_SESSION_CACHE["student_profiles"],
            "distribution": LATEST_SESSION_CACHE["distribution"]
        })

    @app.route("/api/reports/download/<filename>")
    def download_report(filename):
        """Download an exported attendance/analytics report."""
        clean_name = secure_filename(filename)
        # Search in STORAGE_DIR and ATTENDANCE_LOG_DIR
        for search_dir in [STORAGE_DIR, ATTENDANCE_LOG_DIR]:
            if os.path.exists(os.path.join(search_dir, clean_name)):
                return send_from_directory(search_dir, clean_name, as_attachment=True)
        return jsonify({"error": "Report file not found"}), 404

    # ----------------------------------------
    # Real-Time Live Streaming & In-Browser Processing
    # ----------------------------------------
    @app.route("/api/live/start", methods=["POST"])
    def live_start():
        """Initialize and start a real-time live attendance & engagement session."""
        detector = FaceDetector()
        tracker = AttendanceTracker()
        session_id = tracker.start_session("live_web")

        LIVE_SESSION_STATE.update({
            "active": True,
            "session_id": session_id,
            "start_time": time.time(),
            "detector": detector,
            "tracker": tracker,
            "timeline": [],
            "audio_energy_levels": []
        })

        return jsonify({
            "status": "started",
            "session_id": session_id,
            "presence_threshold_seconds": tracker.presence_threshold
        })

    @app.route("/api/live/process_frame", methods=["POST"])
    def live_process_frame():
        """Process a single video frame sent from the browser webcam in real time."""
        if not LIVE_SESSION_STATE["active"]:
            return jsonify({"error": "No live session is currently active. Call /api/live/start first."}), 400

        data = request.get_json(silent=True) or {}
        frame_data = data.get("frame")
        audio_level = data.get("audio_level", 0.0)

        if not frame_data:
            return jsonify({"error": "Missing frame data"}), 400

        try:
            # Strip data URL prefix if present (e.g., 'data:image/jpeg;base64,')
            if "," in frame_data:
                frame_data = frame_data.split(",", 1)[1]
            raw_bytes = base64.b64decode(frame_data)
            np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                return jsonify({"error": "Could not decode image frame"}), 400

            elapsed = time.time() - LIVE_SESSION_STATE["start_time"]
            detector = LIVE_SESSION_STATE["detector"]
            tracker = LIVE_SESSION_STATE["tracker"]

            # Detect & Track Faces
            detections = detector.detect_faces(frame)
            tracked = detector.track_faces(detections, elapsed)
            tracker.update_attendance(tracked)

            # Head Pose Estimation for engagement
            poses = detector.get_head_pose(frame)
            looking_forward_count = sum(1 for p in poses if p.get("is_looking_forward", True))

            # Record timeline entry
            LIVE_SESSION_STATE["timeline"].append({
                "time": round(elapsed, 2),
                "faces_detected": len(detections),
                "poses": poses,
                "looking_forward_count": looking_forward_count
            })

            # Record audio energy
            LIVE_SESSION_STATE["audio_energy_levels"].append(audio_level)

            # Prepare real-time response for frontend canvas drawing
            faces_info = []
            for fid, info in tracked.items():
                if not info.get("is_present", False):
                    continue
                num_id = fid.replace("face_", "")
                faces_info.append({
                    "id": num_id,
                    "full_id": fid,
                    "bbox": info["bbox"],
                    "is_present": info["is_present"],
                    "visible_time": round(info["total_visible_time"], 1),
                    "attendance_marked": info["total_visible_time"] >= tracker.presence_threshold,
                    "state": info.get("engagement_state", "engaged"),
                    "yaw": info.get("yaw", 0.0)
                })

            return jsonify({
                "status": "ok",
                "elapsed": round(elapsed, 1),
                "detected_count": len(detections),
                "faces": faces_info,
                "attendance": tracker.get_summary()
            })

        except Exception as e:
            return jsonify({"error": f"Frame processing error: {str(e)}"}), 500

    @app.route("/api/live/stop", methods=["POST"])
    def live_stop():
        """Conclude active live session and generate complete multimodal AI insights."""
        if not LIVE_SESSION_STATE["active"]:
            return jsonify({"error": "No active live session to stop."}), 400

        total_duration = max(1.0, time.time() - LIVE_SESSION_STATE["start_time"])
        tracker = LIVE_SESSION_STATE["tracker"]
        summary = tracker.get_summary()

        # Save Attendance Logs
        log_json = tracker.save_log()
        log_csv = tracker.save_csv()

        # Process Audio Dynamics
        audio_results = process_audio(audio_source=None, is_synthetic=True, synthetic_pattern="speech_and_silence")

        # Multimodal Fusion & Reasoning
        vision_results = {
            "attendance": summary,
            "engagement_data": LIVE_SESSION_STATE["timeline"],
            "duration_seconds": total_duration
        }

        session_intelligence = multimodal_agent.analyze_session(
            vision_results=vision_results,
            audio_results=audio_results,
            interval_seconds=5.0
        )
        session_metrics = session_intelligence["session_metrics"]
        dynamics = session_intelligence["dynamics"]

        # Gemini AI Teaching Recommendations & Student Analytics
        suggestions = recommendation_engine.get_ai_enhanced_suggestions(session_metrics, dynamics)
        student_profiles = analytics_engine.compute_student_profiles(
            summary["records"], session_duration=total_duration
        )
        distribution = analytics_engine.generate_class_distribution(student_profiles)

        # Export Reports
        report_json_path = report_generator.export_json_report(session_intelligence, suggestions, student_profiles)
        report_csv_path = report_generator.export_students_csv(student_profiles)

        # Update Cache
        LATEST_SESSION_CACHE.update({
            "session_id": session_metrics["session_id"],
            "session_metrics": session_metrics,
            "attendance": summary,
            "dynamics": dynamics,
            "teaching_suggestions": suggestions,
            "student_profiles": student_profiles,
            "distribution": distribution,
            "fused_timeline": session_intelligence.get("fused_timeline", []),
            "reports": {
                "json": os.path.basename(report_json_path),
                "csv": os.path.basename(report_csv_path)
            }
        })

        LIVE_SESSION_STATE["active"] = False

        return jsonify({
            "status": "success",
            "session_id": session_metrics["session_id"],
            "session_metrics": session_metrics,
            "attendance": summary,
            "dynamics": dynamics,
            "teaching_suggestions": suggestions,
            "student_profiles": student_profiles,
            "distribution": distribution,
            "reports": LATEST_SESSION_CACHE["reports"]
        })

    @app.route("/api/live/status")
    def live_status():
        """Check active live session status."""
        active = LIVE_SESSION_STATE["active"]
        elapsed = round(time.time() - LIVE_SESSION_STATE["start_time"], 1) if active else 0.0
        summary = LIVE_SESSION_STATE["tracker"].get_summary() if active and LIVE_SESSION_STATE["tracker"] else None

        return jsonify({
            "active": active,
            "session_id": LIVE_SESSION_STATE["session_id"],
            "elapsed_seconds": elapsed,
            "attendance": summary
        })

    @app.route("/video_feed")
    def video_feed():
        """MJPEG streaming route from server-connected webcam."""
        def generate_frames():
            cap = cv2.VideoCapture(0)
            detector = FaceDetector()
            tracker = AttendanceTracker()
            tracker.start_session("stream")
            start = time.time()

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        # Yield black frame placeholder if camera cannot be accessed
                        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(placeholder, "Camera not available", (150, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                        _, buf = cv2.imencode(".jpg", placeholder)
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                        time.sleep(0.5)
                        continue

                    elapsed = time.time() - start
                    detections = detector.detect_faces(frame)
                    detector.track_faces(detections, elapsed)
                    annotated = detector.draw_detections(frame, show_ids=True)

                    _, buffer = cv2.imencode(".jpg", annotated)
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            finally:
                cap.release()

        return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


# ============================================
# Main Entry Point
# ============================================
if __name__ == "__main__":
    app = create_app()
    print(f"\n{'='*55}")
    print(f"  {APP_NAME} — API Gateway Active")
    print(f"  Web Interface & API: http://{HOST}:{PORT}")
    print(f"{'='*55}\n")
    app.run(host=HOST, port=PORT, debug=DEBUG)
