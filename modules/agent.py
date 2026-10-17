"""
Multimodal Agent — Signal Fusion & Engagement Scoring.

This module serves as the core intelligence layer of the system:
- Aggregates and time-aligns outputs from both Vision and Audio modules
- Executes multimodal signal fusion with configurable weighting and exponential smoothing
- Applies interpretable, rule-based reasoning to analyze classroom dynamics
- Identifies attention drops, interaction peaks, and session pacing bottlenecks
"""

import numpy as np
from datetime import datetime
from config import ENGAGEMENT_CONFIG


class MultimodalFusionEngine:
    """
    Fuses vision and audio signals into synchronized temporal intervals
    and computes weighted multimodal engagement scores.
    """

    def __init__(self, visual_weight=None, audio_weight=None, smoothing_factor=None):
        """
        Initialize the signal fusion engine.
        
        Args:
            visual_weight (float, optional): Weight for visual signal (0-1).
            audio_weight (float, optional): Weight for audio signal (0-1).
            smoothing_factor (float, optional): Exponential smoothing factor (0-1).
        """
        self.visual_weight = visual_weight if visual_weight is not None else ENGAGEMENT_CONFIG["visual_weight"]
        self.audio_weight = audio_weight if audio_weight is not None else ENGAGEMENT_CONFIG["audio_weight"]
        self.smoothing_factor = smoothing_factor if smoothing_factor is not None else ENGAGEMENT_CONFIG["score_smoothing_factor"]

        # Normalize weights so they sum to 1.0
        total_w = self.visual_weight + self.audio_weight
        if total_w > 0:
            self.visual_weight /= total_w
            self.audio_weight /= total_w

    def calculate_visual_engagement(self, frame_data):
        """
        Calculate visual engagement score for a single frame or time window.
        
        Evaluates:
        - Ratio of faces looking forward (yaw & pitch within active cone)
        - Face detection stability (presence of participants)
        
        Args:
            frame_data (dict): Frame data containing 'faces_detected', 'poses', 'looking_forward_count'.
            
        Returns:
            float: Visual score between 0.0 and 1.0.
        """
        faces_detected = frame_data.get("faces_detected", 0)
        if faces_detected == 0:
            return 0.1  # Minimal score if no students detected

        looking_forward = frame_data.get("looking_forward_count", 0)
        forward_ratio = looking_forward / faces_detected

        # Average pose alignment penalty if poses available
        poses = frame_data.get("poses", [])
        if poses:
            pose_penalties = []
            for p in poses:
                yaw_dev = abs(p.get("yaw", 0.0))
                pitch_dev = abs(p.get("pitch", 0.0))
                penalty = (yaw_dev + pitch_dev) / 2.0
                pose_penalties.append(penalty)
            avg_penalty = float(np.mean(pose_penalties))
            visual_score = (forward_ratio * 0.75) + ((1.0 - min(1.0, avg_penalty)) * 0.25)
        else:
            visual_score = forward_ratio

        return float(np.clip(visual_score, 0.0, 1.0))

    def fuse_signals(self, visual_timeline, audio_timeline, interval_seconds=5.0):
        """
        Time-align and fuse visual and audio timelines into unified intervals.
        
        Args:
            visual_timeline (list[dict]): Visual frame data with timestamps.
            audio_timeline (list[dict]): Audio window data with start_time/end_time.
            interval_seconds (float): Length of each synchronized interval.
            
        Returns:
            list[dict]: List of synchronized, fused multimodal intervals.
        """
        # Determine maximum duration across both modalities
        max_v_time = visual_timeline[-1].get("time", 0.0) if visual_timeline else 0.0
        max_a_time = audio_timeline[-1].get("end_time", 0.0) if audio_timeline else 0.0
        max_time = max(max_v_time, max_a_time)

        if max_time <= 0:
            return []

        num_intervals = int(np.ceil(max_time / interval_seconds))
        fused_timeline = []
        smoothed_score = 0.5  # Neutral baseline initialization

        for i in range(num_intervals):
            int_start = i * interval_seconds
            int_end = (i + 1) * interval_seconds

            # Gather visual data within interval
            v_frames = [
                vf for vf in visual_timeline
                if int_start <= vf.get("time", 0.0) < int_end
            ]
            if v_frames:
                v_scores = [self.calculate_visual_engagement(vf) for vf in v_frames]
                visual_score = float(np.mean(v_scores))
                avg_faces = float(np.mean([vf.get("faces_detected", 0) for vf in v_frames]))
                looking_ratio = float(np.mean([
                    (vf.get("looking_forward_count", 0) / max(1, vf.get("faces_detected", 1)))
                    for vf in v_frames
                ]))
            else:
                visual_score = 0.5
                avg_faces = 0.0
                looking_ratio = 0.5

            # Gather audio data matching interval
            a_windows = [
                aw for aw in audio_timeline
                if not (aw.get("end_time", 0.0) <= int_start or aw.get("start_time", 0.0) >= int_end)
            ]
            if a_windows:
                audio_score = float(np.mean([aw.get("audio_engagement_score", 0.5) for aw in a_windows]))
                speech_ratio = float(np.mean([aw.get("speech_ratio", 0.0) for aw in a_windows]))
                db_level = float(np.mean([aw.get("db_level", -50.0) for aw in a_windows]))
                interactions = [aw.get("interaction_level", "passive_listening") for aw in a_windows]
                dominant_interaction = max(set(interactions), key=interactions.count)
            else:
                audio_score = 0.5
                speech_ratio = 0.0
                db_level = -50.0
                dominant_interaction = "passive_listening"

            # Weighted fusion
            raw_fused_score = (self.visual_weight * visual_score) + (self.audio_weight * audio_score)

            # Apply exponential smoothing
            smoothed_score = (self.smoothing_factor * raw_fused_score) + ((1.0 - self.smoothing_factor) * smoothed_score)

            fused_timeline.append({
                "interval_index": i,
                "start_time": round(int_start, 2),
                "end_time": round(int_end, 2),
                "visual_score": round(visual_score, 3),
                "audio_score": round(audio_score, 3),
                "raw_fused_score": round(raw_fused_score, 3),
                "engagement_score": round(smoothed_score, 3),
                "average_faces": round(avg_faces, 1),
                "looking_forward_ratio": round(looking_ratio, 3),
                "speech_ratio": round(speech_ratio, 3),
                "db_level": round(db_level, 2),
                "interaction_level": dominant_interaction
            })

        return fused_timeline


class RuleBasedReasoningEngine:
    """
    Applies explainable, rule-based reasoning on fused classroom signals
    to interpret session dynamics, detect attention anomalies, and identify pacing issues.
    """

    def __init__(self, high_thresh=None, low_thresh=None):
        """
        Initialize reasoning engine.
        
        Args:
            high_thresh (float, optional): High engagement boundary (default 0.7).
            low_thresh (float, optional): Low engagement boundary (default 0.3).
        """
        self.high_thresh = high_thresh if high_thresh is not None else ENGAGEMENT_CONFIG["high_engagement_threshold"]
        self.low_thresh = low_thresh if low_thresh is not None else ENGAGEMENT_CONFIG["low_engagement_threshold"]

    def classify_engagement_level(self, score):
        """Classify score into High, Moderate, or Low."""
        if score >= self.high_thresh:
            return "high"
        if score <= self.low_thresh:
            return "low"
        return "moderate"

    def analyze_session_dynamics(self, fused_timeline):
        """
        Evaluate full session dynamics for attention drops, interaction peaks, and pacing issues.
        
        Rules:
        1. Attention Drop: Score drops by >= 0.20 compared to previous interval.
        2. Interaction Peak: High speech ratio (>= 0.30) with looking forward ratio >= 0.60.
        3. Prolonged Inactivity: >= 2 consecutive intervals with engagement <= low_thresh.
        4. Classroom State: Overall session health diagnosis.
        
        Args:
            fused_timeline (list[dict]): Fused multimodal timeline.
            
        Returns:
            dict: Comprehensive reasoning output with triggered rules and alerts.
        """
        if not fused_timeline:
            return {
                "overall_state": "insufficient_data",
                "average_engagement": 0.0,
                "attention_drops": [],
                "interaction_peaks": [],
                "pacing_alerts": [],
                "rule_triggers": []
            }

        scores = [item["engagement_score"] for item in fused_timeline]
        avg_score = float(np.mean(scores))

        attention_drops = []
        interaction_peaks = []
        pacing_alerts = []
        rule_triggers = []

        consecutive_low = 0

        for i, item in enumerate(fused_timeline):
            current_score = item["engagement_score"]
            t_label = f"{item['start_time']}s-{item['end_time']}s"

            # Rule 1: Attention Drop Detection
            if i > 0:
                prev_score = fused_timeline[i - 1]["engagement_score"]
                delta = current_score - prev_score
                if delta <= -0.18:
                    drop_info = {
                        "interval": i,
                        "time_range": t_label,
                        "drop_amount": round(abs(delta), 3),
                        "score_before": prev_score,
                        "score_after": current_score,
                        "reason": "Sudden drop in visual attention or audio participation"
                    }
                    attention_drops.append(drop_info)
                    rule_triggers.append(f"Rule: Attention Drop detected at {t_label} (delta: {round(delta, 2)})")

            # Rule 2: Interaction Peak Detection
            if item["speech_ratio"] >= 0.25 and item["looking_forward_ratio"] >= 0.55:
                peak_info = {
                    "interval": i,
                    "time_range": t_label,
                    "speech_ratio": item["speech_ratio"],
                    "looking_ratio": item["looking_forward_ratio"],
                    "description": "Active classroom discussion with focused visual engagement"
                }
                interaction_peaks.append(peak_info)
                rule_triggers.append(f"Rule: Interaction Peak at {t_label} (speech: {int(item['speech_ratio']*100)}%)")

            # Rule 3: Prolonged Inactivity / Pacing Alert
            if current_score <= self.low_thresh:
                consecutive_low += 1
                if consecutive_low >= 2:
                    pacing_alerts.append({
                        "interval": i,
                        "time_range": t_label,
                        "consecutive_intervals": consecutive_low,
                        "issue": "Prolonged low engagement detected; instructor intervention recommended"
                    })
                    rule_triggers.append(f"Rule: Low engagement sustained for {consecutive_low} intervals around {t_label}")
            else:
                consecutive_low = 0

        # Overall diagnosis
        overall_level = self.classify_engagement_level(avg_score)
        if overall_level == "high":
            overall_diagnosis = "Highly attentive and actively engaged session"
        elif overall_level == "low":
            overall_diagnosis = "Significantly disengaged session requiring pacing and activity adjustments"
        else:
            overall_diagnosis = "Steady baseline engagement with normal periodic fluctuations"

        return {
            "overall_state": overall_level,
            "overall_diagnosis": overall_diagnosis,
            "average_engagement": round(avg_score, 3),
            "attention_drops_count": len(attention_drops),
            "attention_drops": attention_drops,
            "interaction_peaks_count": len(interaction_peaks),
            "interaction_peaks": interaction_peaks,
            "pacing_alerts_count": len(pacing_alerts),
            "pacing_alerts": pacing_alerts,
            "rule_triggers": rule_triggers
        }


class MultimodalAgent:
    """
    Central AI agent integrating vision, audio, and reasoning modules
    to provide holistic attendance and engagement intelligence.
    """

    def __init__(self, visual_weight=None, audio_weight=None):
        """Initialize the multimodal agent."""
        self.fusion_engine = MultimodalFusionEngine(
            visual_weight=visual_weight,
            audio_weight=audio_weight
        )
        self.reasoning_engine = RuleBasedReasoningEngine()

    def analyze_session(self, vision_results, audio_results, interval_seconds=5.0):
        """
        Process both vision and audio results to produce complete multimodal session intelligence.
        
        Args:
            vision_results (dict): Results from modules.vision.process_video().
            audio_results (dict): Results from modules.audio.process_audio().
            interval_seconds (float): Fusion interval step in seconds.
            
        Returns:
            dict: Integrated session analysis containing:
                - attendance_summary: confirmed attendance records
                - fused_timeline: time-synchronized multimodal timeline
                - dynamics: attention drops, peaks, and pacing alerts
                - session_metrics: aggregated attendance, engagement, and participation statistics
        """
        visual_timeline = vision_results.get("engagement_data", [])
        audio_timeline = audio_results.get("timeline", [])

        # Step 1: Multimodal Signal Fusion
        fused_timeline = self.fusion_engine.fuse_signals(
            visual_timeline, audio_timeline, interval_seconds=interval_seconds
        )

        # Step 2: Rule-Based Reasoning & Dynamics Analysis
        dynamics = self.reasoning_engine.analyze_session_dynamics(fused_timeline)

        # Step 3: Aggregate Session Metrics
        attendance = vision_results.get("attendance", {})
        total_detected = attendance.get("total_detected", 0)
        total_present = attendance.get("total_present", 0)
        attendance_rate = (total_present / max(1, total_detected)) * 100.0

        session_metrics = {
            "session_id": attendance.get("session_id", datetime.now().strftime("%Y%m%d_%H%M%S")),
            "total_detected_students": total_detected,
            "confirmed_present": total_present,
            "attendance_rate_percent": round(attendance_rate, 1),
            "average_engagement_score": dynamics["average_engagement"],
            "engagement_classification": dynamics["overall_state"],
            "overall_diagnosis": dynamics["overall_diagnosis"],
            "total_intervals": len(fused_timeline),
            "attention_drops": dynamics["attention_drops_count"],
            "interaction_peaks": dynamics["interaction_peaks_count"],
            "pacing_alerts": dynamics["pacing_alerts_count"],
            "dominant_audio_interaction": audio_results.get("dominant_interaction", "passive_listening"),
            "speech_percentage": audio_results.get("speech_percentage", 0.0)
        }

        return {
            "status": "success",
            "session_metrics": session_metrics,
            "attendance_summary": attendance,
            "fused_timeline": fused_timeline,
            "dynamics": dynamics
        }
