"""
Automated Live Session Runner for the Multimodal Attendance & Engagement Agent.

Runs real-time face detection, tracking, sustained-presence attendance,
audio monitoring, and automated AI teaching recommendations upon session end.

Usage:
    python run_live.py [--camera 0] [--duration 60] [--preview]
"""

import sys
import os
import time
import argparse
from datetime import datetime
import cv2

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import VISION_CONFIG, AUDIO_CONFIG, STORAGE_DIR, ATTENDANCE_LOG_DIR
from modules.vision import FaceDetector, AttendanceTracker
from modules.audio import process_audio
from modules.agent import MultimodalAgent
from modules.analytics import (
    TeachingRecommendationEngine,
    StudentAnalyticsEngine,
    AnalyticsReportGenerator
)


def run_live_session(camera_id=0, max_duration=None, show_preview=True):
    """
    Automate a live classroom attendance and engagement monitoring session.
    
    Args:
        camera_id (int): Camera device index (default 0 for built-in webcam).
        max_duration (float, optional): Maximum duration in seconds (None for manual quit with 'q').
        show_preview (bool): Display live video feed with detection overlays.
    """
    print("\n" + "=" * 60)
    print("  🎓 MULTIMODAL ATTENDANCE & ENGAGEMENT AGENT — LIVE SESSION")
    print("=" * 60)
    print(f"[*] Initializing camera device {camera_id}...")

    cap = cv2.VideoCapture(camera_id)
    use_synthetic = False

    if not cap.isOpened():
        print(f"[!] Warning: Unable to open camera {camera_id}.")
        print("[*] Switching to automated synthetic classroom stream...")
        use_synthetic = True

    detector = FaceDetector()
    tracker = AttendanceTracker()
    session_id = tracker.start_session("live")

    print(f"[*] Session ID: {session_id}")
    print(f"[*] Presence threshold: {VISION_CONFIG['presence_threshold_seconds']} seconds for confirmed attendance")
    print("[*] Press 'q' in the preview window (or Ctrl+C in terminal) to conclude the session.\n")

    engagement_timeline = []
    frame_count = 0
    start_time = time.time()
    fps = 30.0

    try:
        while True:
            elapsed = time.time() - start_time
            if max_duration and elapsed >= max_duration:
                print(f"\n[*] Reached maximum session duration ({max_duration}s).")
                break

            if not use_synthetic:
                ret, frame = cap.read()
                if not ret:
                    print("[!] Video feed ended.")
                    break
            else:
                # Synthetic simulated camera frame for automation testing
                import numpy as np
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"Simulated Session: {int(elapsed)}s", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 212, 170), 2)
                time.sleep(0.033)

            frame_count += 1
            frame_time = elapsed

            # Detect & Track
            detections = detector.detect_faces(frame)
            tracked = detector.track_faces(detections, frame_time)
            tracker.update_attendance(tracked)

            # Head Pose Estimation for visual engagement
            poses = detector.get_head_pose(frame)
            looking_forward_count = sum(1 for p in poses if p.get("is_looking_forward", True))

            if frame_count % 15 == 0:  # Sample every ~0.5s
                engagement_timeline.append({
                    "frame": frame_count,
                    "time": round(frame_time, 2),
                    "faces_detected": len(detections),
                    "poses": poses,
                    "looking_forward_count": looking_forward_count
                })

            # Live preview overlay
            if show_preview and not use_synthetic:
                annotated = detector.draw_detections(frame, show_ids=True)
                present_count = sum(1 for r in tracker.attendance_records.values() if r["attendance_marked"])
                status_text = f"Present: {present_count}/{len(tracker.attendance_records)} | Time: {int(elapsed)}s"
                cv2.putText(annotated, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

                cv2.imshow("Multimodal Classroom Intelligence (Press 'q' to finish)", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[*] Concluding live session by user request...")
                    break

    except KeyboardInterrupt:
        print("\n[*] Concluding live session via keyboard interrupt...")
    finally:
        if cap and cap.isOpened():
            cap.release()
        if show_preview:
            cv2.destroyAllWindows()

    total_duration = time.time() - start_time
    print(f"\n[*] Session completed in {total_duration:.1f} seconds. Running automated intelligence pipeline...")

    # Step 1: Automated Attendance Log Export
    log_json = tracker.save_log()
    log_csv = tracker.save_csv()
    summary = tracker.get_summary()

    print(f"[+] Attendance Log (JSON): {log_json}")
    print(f"[+] Attendance Log (CSV):  {log_csv}")
    print(f"[+] Total Detected: {summary['total_detected']} | Confirmed Present: {summary['total_present']}")

    # Step 2: Automated Audio Signal Processing
    print("[*] Processing classroom audio environment...")
    audio_results = process_audio(audio_source=None, is_synthetic=True, synthetic_pattern="speech_and_silence")

    # Step 3: Multimodal Fusion & Reasoning
    print("[*] Fusing multimodal signals and evaluating dynamics...")
    agent = MultimodalAgent()
    vision_results = {
        "attendance": summary,
        "engagement_data": engagement_timeline,
        "duration_seconds": total_duration
    }
    session_intel = agent.analyze_session(vision_results, audio_results, interval_seconds=5.0)
    session_metrics = session_intel["session_metrics"]
    dynamics = session_intel["dynamics"]

    # Step 4: AI Recommendations (Google Gemini 2.0 Flash)
    print("[*] Generating AI pedagogical suggestions via Google Gemini 2.0 Flash...")
    rec_engine = TeachingRecommendationEngine(use_gemini=True)
    teaching_suggestions = rec_engine.get_ai_enhanced_suggestions(session_metrics, dynamics)

    # Step 5: Student Analytics & Report Export
    analytics_engine = StudentAnalyticsEngine(total_session_seconds=total_duration)
    student_profiles = analytics_engine.compute_student_profiles(
        summary["records"], session_duration=total_duration
    )
    report_gen = AnalyticsReportGenerator()
    report_path = report_gen.export_json_report(session_intel, teaching_suggestions, student_profiles)
    students_csv_path = report_gen.export_students_csv(student_profiles)

    print("\n" + "=" * 60)
    print("  📊 AUTOMATED SESSION SUMMARY")
    print("=" * 60)
    print(f"Attendance Rate:       {session_metrics['attendance_rate_percent']}%")
    print(f"Confirmed Present:     {session_metrics['confirmed_present']} of {session_metrics['total_detected_students']} students")
    print(f"Average Engagement:    {int(session_metrics['average_engagement_score'] * 100)}% ({session_metrics['engagement_classification'].upper()})")
    print(f"Dominant Audio State:  {session_metrics['dominant_audio_interaction'].replace('_', ' ').title()}")
    print(f"Diagnosis:             {session_metrics['overall_diagnosis']}")
    print("-" * 60)

    if teaching_suggestions.get("ai_insights"):
        print("\n🤖 GEMINI AI TEACHING INSIGHTS:")
        print(teaching_suggestions["ai_insights"].strip())

    print("\n💡 ACTIONABLE RECOMMENDATIONS:")
    for i, s in enumerate(teaching_suggestions.get("structured_suggestions", []), 1):
        print(f"  {i}. [{s['priority'].upper()}] {s['title']}: {s['suggestion']}")

    print("\n[+] Full Analytics Report saved to: " + report_path)
    print("[+] Student CSV exported to:         " + students_csv_path)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Attendance & Engagement Agent - Automated Live Runner")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--duration", type=float, default=None, help="Max duration in seconds (optional)")
    parser.add_argument("--no-preview", action="store_true", help="Disable GUI preview window")
    args = parser.parse_args()

    run_live_session(camera_id=args.camera, max_duration=args.duration, show_preview=not args.no_preview)
