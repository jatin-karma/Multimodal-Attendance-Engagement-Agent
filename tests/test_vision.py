"""
Tests for the Vision Module.

Tests face detection, tracking, attendance logic, and head pose estimation.
Uses synthetic test data since actual camera/video may not be available in CI.
"""

import os
import json
import pytest
import numpy as np
from modules.vision import FaceDetector, AttendanceTracker
from config import VISION_CONFIG, ATTENDANCE_LOG_DIR


class TestFaceDetector:
    """Test the FaceDetector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = FaceDetector()

    def test_initialization(self):
        """Test FaceDetector initializes correctly."""
        assert self.detector is not None
        assert self.detector.next_face_id == 1
        assert self.detector.tracked_faces == {}

    def test_detect_faces_with_blank_frame(self):
        """Test that detection returns empty list for a blank frame."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = self.detector.detect_faces(blank)
        assert isinstance(detections, list)
        # Blank frame should have no face detections
        assert len(detections) == 0

    def test_iou_calculation_overlap(self):
        """Test IoU calculation with overlapping boxes."""
        bbox1 = (0, 0, 100, 100)
        bbox2 = (50, 50, 100, 100)
        iou = self.detector._calculate_iou(bbox1, bbox2)
        assert 0 < iou < 1

    def test_iou_calculation_no_overlap(self):
        """Test IoU calculation with non-overlapping boxes."""
        bbox1 = (0, 0, 50, 50)
        bbox2 = (200, 200, 50, 50)
        iou = self.detector._calculate_iou(bbox1, bbox2)
        assert iou == 0.0

    def test_iou_calculation_identical(self):
        """Test IoU calculation with identical boxes."""
        bbox = (10, 10, 100, 100)
        iou = self.detector._calculate_iou(bbox, bbox)
        assert iou == 1.0

    def test_distance_calculation(self):
        """Test center distance calculation."""
        c1 = (0, 0)
        c2 = (3, 4)
        dist = self.detector._distance_between_centers(c1, c2)
        assert dist == 5.0

    def test_track_faces_assigns_new_ids(self):
        """Test that unmatched detections get new face IDs."""
        detections = [
            {"bbox": (100, 100, 50, 50), "confidence": 0.9, "center": (125, 125)},
            {"bbox": (300, 100, 50, 50), "confidence": 0.85, "center": (325, 125)},
        ]
        tracked = self.detector.track_faces(detections, frame_time=0.0)
        assert len(tracked) == 2
        assert "face_1" in tracked
        assert "face_2" in tracked

    def test_track_faces_maintains_ids(self):
        """Test that tracked faces maintain their IDs across frames."""
        # Frame 1
        det1 = [
            {"bbox": (100, 100, 50, 50), "confidence": 0.9, "center": (125, 125)},
        ]
        self.detector.track_faces(det1, frame_time=0.0)
        
        # Frame 2 - face slightly moved
        det2 = [
            {"bbox": (105, 102, 50, 50), "confidence": 0.88, "center": (130, 127)},
        ]
        tracked = self.detector.track_faces(det2, frame_time=0.5)
        
        assert "face_1" in tracked
        assert tracked["face_1"]["is_present"] is True

    def test_track_faces_marks_departed(self):
        """Test that faces not in current frame are marked as not present."""
        # Frame 1
        det1 = [
            {"bbox": (100, 100, 50, 50), "confidence": 0.9, "center": (125, 125)},
        ]
        self.detector.track_faces(det1, frame_time=0.0)
        
        # Frame 2 - face disappeared
        tracked = self.detector.track_faces([], frame_time=0.5)
        assert tracked["face_1"]["is_present"] is False

    def test_reset(self):
        """Test that reset clears all tracking state."""
        det = [{"bbox": (100, 100, 50, 50), "confidence": 0.9, "center": (125, 125)}]
        self.detector.track_faces(det, frame_time=0.0)
        assert len(self.detector.tracked_faces) > 0
        
        self.detector.reset()
        assert len(self.detector.tracked_faces) == 0
        assert self.detector.next_face_id == 1

    def test_draw_detections_returns_frame(self):
        """Test that draw_detections returns a valid frame."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        det = [{"bbox": (100, 100, 50, 50), "confidence": 0.9, "center": (125, 125)}]
        self.detector.track_faces(det, frame_time=0.0)
        
        annotated = self.detector.draw_detections(blank)
        assert annotated is not None
        assert annotated.shape == blank.shape


class TestAttendanceTracker:
    """Test the AttendanceTracker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = AttendanceTracker()

    def test_start_session(self):
        """Test session initialization."""
        session_id = self.tracker.start_session("test_session")
        assert session_id is not None
        assert "test_session" in session_id
        assert self.tracker.session_start is not None

    def test_update_attendance_below_threshold(self):
        """Test that attendance is not marked below the presence threshold."""
        self.tracker.start_session()
        
        tracked = {
            "face_1": {
                "first_seen": 0.0,
                "last_seen": 3.0,
                "total_visible_time": 3.0,
                "is_present": True
            }
        }
        
        records = self.tracker.update_attendance(tracked)
        assert records["face_1"]["attendance_marked"] is False
        assert records["face_1"]["status"] == "tracking"

    def test_update_attendance_above_threshold(self):
        """Test that attendance is marked when threshold is exceeded."""
        self.tracker.start_session()
        threshold = VISION_CONFIG["presence_threshold_seconds"]
        
        tracked = {
            "face_1": {
                "first_seen": 0.0,
                "last_seen": threshold + 5,
                "total_visible_time": threshold + 5,
                "is_present": True
            }
        }
        
        records = self.tracker.update_attendance(tracked)
        assert records["face_1"]["attendance_marked"] is True
        assert records["face_1"]["status"] == "present"

    def test_get_summary(self):
        """Test summary generation."""
        self.tracker.start_session("summary_test")
        threshold = VISION_CONFIG["presence_threshold_seconds"]
        
        tracked = {
            "face_1": {
                "first_seen": 0.0, "last_seen": threshold + 5,
                "total_visible_time": threshold + 5, "is_present": True
            },
            "face_2": {
                "first_seen": 2.0, "last_seen": 5.0,
                "total_visible_time": 3.0, "is_present": True
            }
        }
        
        self.tracker.update_attendance(tracked)
        summary = self.tracker.get_summary()
        
        assert summary["total_detected"] == 2
        assert summary["total_present"] == 1
        assert summary["total_tracking"] == 1

    def test_save_log_json(self):
        """Test saving attendance log as JSON."""
        self.tracker.start_session("json_test")
        threshold = VISION_CONFIG["presence_threshold_seconds"]
        
        tracked = {
            "face_1": {
                "first_seen": 0.0, "last_seen": threshold + 1,
                "total_visible_time": threshold + 1, "is_present": True
            }
        }
        self.tracker.update_attendance(tracked)
        
        filepath = self.tracker.save_log()
        assert os.path.exists(filepath)
        
        with open(filepath) as f:
            data = json.load(f)
        assert data["total_present"] == 1
        
        # Cleanup
        os.remove(filepath)

    def test_save_csv(self):
        """Test saving attendance log as CSV."""
        self.tracker.start_session("csv_test")
        threshold = VISION_CONFIG["presence_threshold_seconds"]
        
        tracked = {
            "face_1": {
                "first_seen": 0.0, "last_seen": threshold + 1,
                "total_visible_time": threshold + 1, "is_present": True
            }
        }
        self.tracker.update_attendance(tracked)
        
        filepath = self.tracker.save_csv()
        assert os.path.exists(filepath)
        
        # Cleanup
        os.remove(filepath)
