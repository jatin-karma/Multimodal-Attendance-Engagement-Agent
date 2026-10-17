"""
Analytics & Recommendation Engine.

This module translates low-level engagement and attendance data into actionable insights:
- Context-aware pedagogical suggestions for instructors (rule-based + Google Gemini 2.0 Flash)
- Per-student session analytics, engagement profiles, and constructive self-reflection feedback
- Session-level trend reporting and attendance health metrics
"""

import os
import json
import csv
from datetime import datetime
from config import STORAGE_DIR
from utils.genai_utils import initialize_gemini, generate_teaching_suggestion, generate_student_summary


class TeachingRecommendationEngine:
    """
    Generates actionable, context-aware pedagogical suggestions for instructors.
    Supports both offline rule-based heuristics and Google Gemini 2.0 Flash AI reasoning.
    """

    def __init__(self, use_gemini=True):
        """
        Initialize the recommendation engine.
        
        Args:
            use_gemini (bool): Whether to attempt AI generation via Google Gemini.
        """
        self.use_gemini = use_gemini
        self.gemini_model = None
        if use_gemini:
            try:
                self.gemini_model = initialize_gemini()
            except Exception:
                self.gemini_model = None

    def generate_heuristic_suggestions(self, session_metrics, dynamics):
        """
        Generate offline, explainable rule-based teaching recommendations.
        
        Args:
            session_metrics (dict): High-level session metrics from MultimodalAgent.
            dynamics (dict): Classroom dynamics from RuleBasedReasoningEngine.
            
        Returns:
            list[dict]: List of categorized recommendations with priority and rationale.
        """
        suggestions = []
        avg_score = session_metrics.get("average_engagement_score", 0.5)
        drops = dynamics.get("attention_drops", [])
        pacing_alerts = dynamics.get("pacing_alerts", [])
        peaks = dynamics.get("interaction_peaks", [])
        dominant_audio = session_metrics.get("dominant_audio_interaction", "passive_listening")

        # Suggestion 1: Response to Attention Drops
        if drops:
            latest_drop = drops[-1]
            suggestions.append({
                "type": "immediate_action",
                "priority": "high",
                "title": "Address Recent Attention Drop",
                "suggestion": f"An attention drop was detected around {latest_drop['time_range']}. Consider pausing the monologue to ask a quick comprehension question or conduct a 60-second think-pair-share.",
                "rationale": latest_drop.get("reason", "Sharp drop in visual orientation")
            })

        # Suggestion 2: Pacing Adjustments
        if pacing_alerts:
            suggestions.append({
                "type": "pacing_adjustment",
                "priority": "high",
                "title": "Adjust Lecture Pace & Activity",
                "suggestion": "Students have exhibited low engagement over consecutive intervals. Break up the current segment with an interactive problem, live code demonstration, or group poll.",
                "rationale": "Sustained below-threshold engagement detected"
            })
        elif dominant_audio == "silent" and avg_score < 0.6:
            suggestions.append({
                "type": "pacing_adjustment",
                "priority": "medium",
                "title": "Stimulate Classroom Dialogue",
                "suggestion": "Classroom audio has remained predominantly silent. Invite student questions or ask an open-ended question to spark discussion.",
                "rationale": "Low verbal interaction observed over the session"
            })

        # Suggestion 3: Reinforcing Interaction Peaks
        if peaks:
            suggestions.append({
                "type": "reinforcement",
                "priority": "medium",
                "title": "Capitalize on Discussion Momentum",
                "suggestion": "Productive interaction peaks occurred during the session. Summarize the key ideas voiced by participants and connect them directly to upcoming concepts.",
                "rationale": f"Detected {len(peaks)} high-interaction discussion moments"
            })

        # Suggestion 4: General High/Moderate State Guidance
        if avg_score >= 0.7:
            suggestions.append({
                "type": "positive_reinforcement",
                "priority": "low",
                "title": "Maintain High Engagement Rhythm",
                "suggestion": "Class engagement is strong! Transition smoothly into higher-order problem solving or challenging application exercises.",
                "rationale": f"Classroom average engagement is {int(avg_score * 100)}%"
            })
        elif not suggestions:
            suggestions.append({
                "type": "maintenance",
                "priority": "low",
                "title": "Continue Balanced Delivery",
                "suggestion": "Engagement is stable. Continue pacing with periodic checkpoints and visual aids.",
                "rationale": "Session dynamics show steady moderate engagement"
            })

        return suggestions

    def get_ai_enhanced_suggestions(self, session_metrics, dynamics):
        """
        Generate recommendations using Google Gemini 2.0 Flash when available,
        falling back seamlessly to rule-based heuristics.
        
        Args:
            session_metrics (dict): High-level session metrics.
            dynamics (dict): Classroom dynamics.
            
        Returns:
            dict: Teaching suggestions containing heuristic items and optional Gemini AI insights.
        """
        heuristics = self.generate_heuristic_suggestions(session_metrics, dynamics)
        ai_narrative = None

        if self.gemini_model:
            try:
                engagement_payload = {
                    "avg_score": f"{session_metrics.get('average_engagement_score', 0.0) * 100:.1f}%",
                    "trend": session_metrics.get("engagement_classification", "moderate"),
                    "active_count": session_metrics.get("confirmed_present", 0),
                    "total_count": session_metrics.get("total_detected_students", 0),
                    "low_periods": f"{dynamics.get('pacing_alerts_count', 0)} intervals",
                    "audio_activity": session_metrics.get("dominant_audio_interaction", "passive_listening")
                }
                ai_narrative = generate_teaching_suggestion(self.gemini_model, engagement_payload)
            except Exception as e:
                ai_narrative = f"AI suggestions offline: {str(e)}"

        return {
            "source": "gemini_ai" if ai_narrative and not ai_narrative.startswith("AI suggestions offline") else "rule_based_engine",
            "ai_insights": ai_narrative,
            "structured_suggestions": heuristics,
            "suggestions_count": len(heuristics)
        }


class StudentAnalyticsEngine:
    """
    Computes per-student attendance and engagement profiles,
    participation classifications, and personalized reflection summaries.
    """

    def __init__(self, total_session_seconds=None):
        """
        Initialize student analytics engine.
        
        Args:
            total_session_seconds (float, optional): Total duration of classroom session.
        """
        self.total_session_seconds = total_session_seconds or 60.0

    def compute_student_profiles(self, attendance_records, session_duration=None):
        """
        Compute individual student metrics from attendance records.
        
        Args:
            attendance_records (dict): Records dictionary from AttendanceTracker.
            session_duration (float, optional): Total session duration.
            
        Returns:
            list[dict]: List of student analytics profile dictionaries.
        """
        duration = session_duration or self.total_session_seconds
        profiles = []

        for face_id, record in attendance_records.items():
            visible_secs = record.get("total_visible_seconds", 0.0)
            is_present = record.get("attendance_marked", False)
            presence_pct = min(100.0, (visible_secs / max(1.0, duration)) * 100.0)

            # Estimate individual engagement score based on visibility consistency
            # and sustained participation (clamped 0.1 to 1.0)
            individual_score = min(1.0, max(0.1, (visible_secs / max(10.0, duration * 0.75))))

            if individual_score >= 0.75 and is_present:
                classification = "Highly Engaged"
                feedback = "Outstanding consistency and sustained presence throughout the session."
            elif individual_score >= 0.50 and is_present:
                classification = "Moderately Engaged"
                feedback = "Good attendance and steady participation. Try to stay actively involved during group checkpoints."
            elif is_present:
                classification = "Passive Observer"
                feedback = "Attendance confirmed, but visible engagement dipped periodically. Encourage active note-taking."
            else:
                classification = "Needs Attention"
                feedback = "Presence duration was below attendance threshold. Verify webcam positioning or connectivity."

            profiles.append({
                "student_id": face_id,
                "attendance_confirmed": is_present,
                "first_detected_seconds": round(record.get("first_detected", 0.0), 1),
                "last_detected_seconds": round(record.get("last_detected", 0.0), 1),
                "visible_duration_seconds": round(visible_secs, 1),
                "presence_percentage": round(presence_pct, 1),
                "engagement_score": round(individual_score, 3),
                "classification": classification,
                "student_feedback": feedback
            })

        # Sort profiles by engagement score descending
        profiles.sort(key=lambda x: x["engagement_score"], reverse=True)
        return profiles

    def generate_class_distribution(self, student_profiles):
        """
        Compute aggregate distribution across engagement classifications.
        
        Args:
            student_profiles (list[dict]): Student profiles list.
            
        Returns:
            dict: Counts per category and percentage shares.
        """
        total = len(student_profiles)
        if total == 0:
            return {"total_students": 0, "counts": {}, "percentages": {}}

        counts = {
            "Highly Engaged": sum(1 for p in student_profiles if p["classification"] == "Highly Engaged"),
            "Moderately Engaged": sum(1 for p in student_profiles if p["classification"] == "Moderately Engaged"),
            "Passive Observer": sum(1 for p in student_profiles if p["classification"] == "Passive Observer"),
            "Needs Attention": sum(1 for p in student_profiles if p["classification"] == "Needs Attention"),
        }

        percentages = {
            k: round((v / total) * 100.0, 1) for k, v in counts.items()
        }

        return {
            "total_students": total,
            "counts": counts,
            "percentages": percentages
        }


class AnalyticsReportGenerator:
    """
    Exports comprehensive analytics reports to disk in JSON, CSV, and Markdown formats.
    """

    def __init__(self, output_dir=None):
        """Initialize the report generator with storage directory."""
        self.output_dir = output_dir or STORAGE_DIR

    def export_json_report(self, session_analysis, suggestions, student_profiles, filename=None):
        """
        Save complete session intelligence as a JSON report.
        
        Args:
            session_analysis (dict): Output from MultimodalAgent.analyze_session().
            suggestions (dict): Output from TeachingRecommendationEngine.
            student_profiles (list[dict]): Output from StudentAnalyticsEngine.
            filename (str, optional): Target filename.
            
        Returns:
            str: Absolute filepath of saved report.
        """
        session_id = session_analysis.get("session_metrics", {}).get("session_id", "session")
        if filename is None:
            filename = f"analytics_report_{session_id}.json"

        filepath = os.path.join(self.output_dir, filename)
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "session_metrics": session_analysis.get("session_metrics", {}),
            "teaching_suggestions": suggestions,
            "student_profiles": student_profiles,
            "dynamics": session_analysis.get("dynamics", {})
        }

        with open(filepath, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        return filepath

    def export_students_csv(self, student_profiles, filename=None):
        """
        Save student analytics to a CSV file.
        
        Args:
            student_profiles (list[dict]): Student profiles list.
            filename (str, optional): Target filename.
            
        Returns:
            str: Absolute filepath of saved CSV.
        """
        if filename is None:
            filename = f"students_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        filepath = os.path.join(self.output_dir, filename)
        if not student_profiles:
            return filepath

        fieldnames = [
            "student_id", "attendance_confirmed", "visible_duration_seconds",
            "presence_percentage", "engagement_score", "classification", "student_feedback"
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for profile in student_profiles:
                writer.writerow(profile)

        return filepath
