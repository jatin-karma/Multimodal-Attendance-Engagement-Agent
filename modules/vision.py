"""
Vision Module — Face Detection, Tracking & Attendance Logic.

This module handles:
- Face detection using MediaPipe Face Detection
- Face tracking across video frames with ID assignment
- Sustained presence logic for accurate attendance marking
- Timestamped attendance log generation
- Head pose estimation for engagement indicators
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from datetime import datetime
from config import VISION_CONFIG, ATTENDANCE_LOG_DIR


class FaceDetector:
    """
    Detects and tracks faces in video frames using MediaPipe.
    
    Assigns persistent IDs to detected faces based on spatial proximity
    across frames, enabling sustained presence tracking.
    """

    def __init__(self):
        """Initialize the face detector with MediaPipe Face Detection."""
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=VISION_CONFIG["min_detection_confidence"],
            model_selection=1  # 1 = full range model (better for classroom)
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=VISION_CONFIG["max_num_faces"],
            min_detection_confidence=VISION_CONFIG["min_detection_confidence"],
            min_tracking_confidence=VISION_CONFIG["min_tracking_confidence"],
            refine_landmarks=True
        )

        # Tracking state
        self.tracked_faces = {}      # face_id -> tracking info
        self.next_face_id = 1
        self.iou_threshold = 0.3     # IoU threshold for face matching

    def detect_faces(self, frame):
        """
        Detect all faces in a single video frame.
        
        Args:
            frame (np.ndarray): BGR video frame from OpenCV.
        
        Returns:
            list[dict]: List of detected face dictionaries containing:
                - bbox: (x, y, w, h) bounding box
                - confidence: detection confidence score
                - landmarks: facial landmark points (if available)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        results = self.face_detection.process(rgb_frame)
        detections = []
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                
                # Clamp to frame boundaries
                x = max(0, x)
                y = max(0, y)
                bw = min(bw, w - x)
                bh = min(bh, h - y)
                
                confidence = detection.score[0] if detection.score else 0.0
                
                detections.append({
                    "bbox": (x, y, bw, bh),
                    "confidence": float(confidence),
                    "center": (x + bw // 2, y + bh // 2)
                })
        
        return detections

    def get_head_pose(self, frame):
        """
        Estimate head poses for all detected faces using Face Mesh landmarks.
        
        Uses key facial landmarks (nose tip, chin, eye corners) to estimate
        the head orientation, which serves as an engagement indicator.
        
        Args:
            frame (np.ndarray): BGR video frame.
        
        Returns:
            list[dict]: List of head pose data per face, each containing:
                - yaw: left/right rotation (-1 to 1)
                - pitch: up/down rotation (-1 to 1)
                - is_looking_forward: boolean engagement indicator
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        results = self.face_mesh.process(rgb_frame)
        
        poses = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Key landmarks for pose estimation
                nose_tip = face_landmarks.landmark[1]       # Nose tip
                chin = face_landmarks.landmark[152]          # Chin
                left_eye = face_landmarks.landmark[33]       # Left eye outer
                right_eye = face_landmarks.landmark[263]     # Right eye outer
                forehead = face_landmarks.landmark[10]       # Forehead
                
                # Estimate yaw (left/right) from eye positions relative to nose
                eye_center_x = (left_eye.x + right_eye.x) / 2
                yaw = (nose_tip.x - eye_center_x) * 4  # Scale to [-1, 1] approx
                yaw = max(-1.0, min(1.0, yaw))
                
                # Estimate pitch (up/down) from nose-chin vs nose-forehead ratio
                nose_to_chin = chin.y - nose_tip.y
                nose_to_forehead = nose_tip.y - forehead.y
                if nose_to_forehead > 0:
                    pitch_ratio = nose_to_chin / nose_to_forehead
                    pitch = (pitch_ratio - 1.0) * 2  # Normalize around 0
                    pitch = max(-1.0, min(1.0, pitch))
                else:
                    pitch = 0.0
                
                # Determine if person is roughly looking forward
                is_looking_forward = abs(yaw) < 0.35 and abs(pitch) < 0.4
                
                poses.append({
                    "yaw": round(yaw, 3),
                    "pitch": round(pitch, 3),
                    "is_looking_forward": is_looking_forward,
                    "nose_position": (
                        int(nose_tip.x * w),
                        int(nose_tip.y * h)
                    )
                })
        
        return poses

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def _distance_between_centers(self, c1, c2):
        """Calculate Euclidean distance between two center points."""
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    def track_faces(self, detections, frame_time):
        """
        Match detected faces to existing tracked faces using spatial proximity.
        
        Assigns persistent IDs to faces across frames. New faces get new IDs,
        and faces that haven't been seen for a while are marked as departed.
        
        Args:
            detections (list[dict]): Face detections from detect_faces().
            frame_time (float): Timestamp of the current frame in seconds.
        
        Returns:
            dict: Updated tracked_faces dictionary mapping face_id to:
                - bbox: current bounding box
                - first_seen: timestamp of first detection
                - last_seen: timestamp of most recent detection
                - total_visible_time: cumulative seconds visible
                - is_present: whether face is currently detected
        """
        matched_ids = set()
        unmatched_detections = list(range(len(detections)))
        
        # Try to match each detection to existing tracked faces
        for face_id, track_info in list(self.tracked_faces.items()):
            best_match_idx = None
            best_distance = float('inf')
            
            for det_idx in unmatched_detections:
                det = detections[det_idx]
                distance = self._distance_between_centers(
                    track_info["center"], det["center"]
                )
                iou = self._calculate_iou(track_info["bbox"], det["bbox"])
                
                # Match by proximity or IoU
                if (distance < best_distance and 
                    (iou > self.iou_threshold or distance < 100)):
                    best_distance = distance
                    best_match_idx = det_idx
            
            if best_match_idx is not None:
                det = detections[best_match_idx]
                self.tracked_faces[face_id].update({
                    "bbox": det["bbox"],
                    "center": det["center"],
                    "confidence": det["confidence"],
                    "last_seen": frame_time,
                    "is_present": True,
                })
                # Accumulate visible time
                if track_info.get("_prev_frame_time"):
                    dt = frame_time - track_info["_prev_frame_time"]
                    self.tracked_faces[face_id]["total_visible_time"] += dt
                self.tracked_faces[face_id]["_prev_frame_time"] = frame_time
                
                matched_ids.add(face_id)
                unmatched_detections.remove(best_match_idx)
        
        # Mark unmatched tracked faces as not present
        for face_id in self.tracked_faces:
            if face_id not in matched_ids:
                self.tracked_faces[face_id]["is_present"] = False
                self.tracked_faces[face_id]["_prev_frame_time"] = None
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            face_id = f"face_{self.next_face_id}"
            self.next_face_id += 1
            
            self.tracked_faces[face_id] = {
                "bbox": det["bbox"],
                "center": det["center"],
                "confidence": det["confidence"],
                "first_seen": frame_time,
                "last_seen": frame_time,
                "total_visible_time": 0.0,
                "is_present": True,
                "_prev_frame_time": frame_time,
            }
        
        return self.tracked_faces

    def draw_detections(self, frame, show_ids=True, show_pose=False, poses=None):
        """
        Draw bounding boxes and labels on the frame for visualization.
        
        Args:
            frame (np.ndarray): BGR video frame to annotate.
            show_ids (bool): Whether to display face IDs.
            show_pose (bool): Whether to display head pose indicators.
            poses (list[dict]): Head pose data from get_head_pose().
        
        Returns:
            np.ndarray: Annotated frame.
        """
        annotated = frame.copy()
        
        for face_id, info in self.tracked_faces.items():
            if not info["is_present"]:
                continue
            
            x, y, w, h = info["bbox"]
            
            # Color based on presence duration
            visible_time = info["total_visible_time"]
            threshold = VISION_CONFIG["presence_threshold_seconds"]
            
            if visible_time >= threshold:
                color = (0, 200, 100)    # Green = attendance confirmed
                status = "PRESENT"
            else:
                color = (0, 165, 255)    # Orange = tracking, not yet confirmed
                status = f"{visible_time:.0f}s"
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            if show_ids:
                label = f"{face_id} [{status}]"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(
                    annotated,
                    (x, y - label_size[1] - 10),
                    (x + label_size[0] + 5, y),
                    color, -1
                )
                cv2.putText(
                    annotated, label,
                    (x + 2, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1
                )
        
        return annotated

    def reset(self):
        """Reset all tracking state."""
        self.tracked_faces = {}
        self.next_face_id = 1


class AttendanceTracker:
    """
    Manages attendance records based on sustained face presence.
    
    Attendance is marked only when a face has been continuously or
    cumulatively present for longer than the configured threshold,
    reducing false positives from brief appearances.
    """

    def __init__(self):
        """Initialize the attendance tracker."""
        self.attendance_records = {}  # face_id -> attendance info
        self.session_start = None
        self.session_id = None
        self.presence_threshold = VISION_CONFIG["presence_threshold_seconds"]

    def start_session(self, session_name=None):
        """
        Begin a new attendance tracking session.
        
        Args:
            session_name (str, optional): Optional name for the session.
        
        Returns:
            str: The generated session ID.
        """
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        if session_name:
            self.session_id = f"{self.session_id}_{session_name}"
        
        self.attendance_records = {}
        return self.session_id

    def update_attendance(self, tracked_faces):
        """
        Update attendance records based on current tracked face data.
        
        A face is marked as "present" (attendance confirmed) once its
        cumulative visible time exceeds the presence threshold.
        
        Args:
            tracked_faces (dict): Current tracked faces from FaceDetector.
        
        Returns:
            dict: Updated attendance records.
        """
        for face_id, info in tracked_faces.items():
            visible_time = info["total_visible_time"]
            
            if face_id not in self.attendance_records:
                self.attendance_records[face_id] = {
                    "face_id": face_id,
                    "first_detected": info["first_seen"],
                    "last_detected": info["last_seen"],
                    "total_visible_seconds": visible_time,
                    "attendance_marked": visible_time >= self.presence_threshold,
                    "marked_at": info["first_seen"] + self.presence_threshold 
                                 if visible_time >= self.presence_threshold else None,
                    "status": "present" if visible_time >= self.presence_threshold 
                              else "tracking"
                }
            else:
                record = self.attendance_records[face_id]
                record["last_detected"] = info["last_seen"]
                record["total_visible_seconds"] = visible_time
                
                if not record["attendance_marked"] and visible_time >= self.presence_threshold:
                    record["attendance_marked"] = True
                    record["marked_at"] = info["last_seen"]
                    record["status"] = "present"
        
        return self.attendance_records

    def get_summary(self):
        """
        Generate an attendance summary for the current session.
        
        Returns:
            dict: Summary containing:
                - session_id: unique session identifier
                - session_start: start timestamp
                - total_detected: total unique faces detected
                - total_present: faces confirmed as present
                - total_tracking: faces still being tracked (not yet confirmed)
                - records: detailed per-face attendance records
        """
        present_count = sum(
            1 for r in self.attendance_records.values() 
            if r["attendance_marked"]
        )
        tracking_count = sum(
            1 for r in self.attendance_records.values() 
            if not r["attendance_marked"]
        )
        
        return {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat() if self.session_start else None,
            "total_detected": len(self.attendance_records),
            "total_present": present_count,
            "total_tracking": tracking_count,
            "presence_threshold_seconds": self.presence_threshold,
            "records": self.attendance_records
        }

    def save_log(self, filename=None):
        """
        Save attendance records to a JSON file.
        
        Args:
            filename (str, optional): Custom filename. 
                Defaults to session_id based name.
        
        Returns:
            str: Path to the saved log file.
        """
        if filename is None:
            filename = f"attendance_{self.session_id}.json"
        
        filepath = os.path.join(ATTENDANCE_LOG_DIR, filename)
        
        summary = self.get_summary()
        
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        return filepath

    def save_csv(self, filename=None):
        """
        Save attendance records to a CSV file.
        
        Args:
            filename (str, optional): Custom filename.
        
        Returns:
            str: Path to the saved CSV file.
        """
        import csv
        
        if filename is None:
            filename = f"attendance_{self.session_id}.csv"
        
        filepath = os.path.join(ATTENDANCE_LOG_DIR, filename)
        
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "face_id", "first_detected", "last_detected",
                "total_visible_seconds", "attendance_marked", 
                "marked_at", "status"
            ])
            
            for record in self.attendance_records.values():
                writer.writerow([
                    record["face_id"],
                    record["first_detected"],
                    record["last_detected"],
                    round(record["total_visible_seconds"], 2),
                    record["attendance_marked"],
                    record["marked_at"],
                    record["status"]
                ])
        
        return filepath


def process_video(video_source, max_frames=None, show_preview=False):
    """
    Process a video file or camera feed for attendance tracking.
    
    This is the main entry point for the vision module. It processes
    each frame to detect, track, and mark attendance.
    
    Args:
        video_source: Path to a video file (str) or camera index (int, 0 for webcam).
        max_frames (int, optional): Maximum frames to process. None = all.
        show_preview (bool): Whether to display a live preview window.
    
    Returns:
        dict: Processing results containing:
            - attendance: attendance summary from AttendanceTracker
            - engagement_data: per-frame head pose data for engagement analysis
            - frame_count: total frames processed
            - duration_seconds: total processing duration
    """
    detector = FaceDetector()
    tracker = AttendanceTracker()
    tracker.start_session()
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {video_source}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_skip = VISION_CONFIG["frame_skip"]
    
    frame_count = 0
    processed_count = 0
    engagement_frames = []
    
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if max_frames and frame_count > max_frames:
                break
            
            # Skip frames for performance
            if frame_count % frame_skip != 0:
                continue
            
            processed_count += 1
            frame_time = frame_count / fps
            
            # Detect and track faces
            detections = detector.detect_faces(frame)
            tracked = detector.track_faces(detections, frame_time)
            
            # Update attendance
            tracker.update_attendance(tracked)
            
            # Get head pose data for engagement
            poses = detector.get_head_pose(frame)
            engagement_frames.append({
                "frame": frame_count,
                "time": round(frame_time, 2),
                "faces_detected": len(detections),
                "poses": poses,
                "looking_forward_count": sum(
                    1 for p in poses if p["is_looking_forward"]
                )
            })
            
            # Optional preview
            if show_preview:
                annotated = detector.draw_detections(frame, show_ids=True)
                cv2.imshow("Attendance Tracker", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    
    # Save attendance log
    log_path = tracker.save_log()
    csv_path = tracker.save_csv()
    
    return {
        "attendance": tracker.get_summary(),
        "engagement_data": engagement_frames,
        "frame_count": frame_count,
        "processed_frames": processed_count,
        "duration_seconds": round(elapsed, 2),
        "fps": fps,
        "log_file": log_path,
        "csv_file": csv_path
    }
