"""
Streamlit Dashboard for the Multimodal Attendance & Engagement Agent.

Provides an interactive interface for:
- Video and audio file upload & analysis
- Real-time and session-level attendance tracking
- Multimodal engagement trends visualization (visual, audio, fused score)
- AI-driven teaching suggestions powered by Google Gemini 2.0 Flash
- Individual student participation profiles and downloadable analytics reports

Run with:
    streamlit run dashboard.py
"""

import os
import json
import pandas as pd
import numpy as np

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    st = None
    HAS_STREAMLIT = False

# Optional Plotly for rich interactive charts
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from config import (
    APP_NAME, UPLOAD_FOLDER, STORAGE_DIR,
    ATTENDANCE_LOG_DIR, VISION_CONFIG, AUDIO_CONFIG, ENGAGEMENT_CONFIG
)
from modules.vision import process_video
from modules.audio import process_audio
from modules.agent import MultimodalAgent
from modules.analytics import (
    TeachingRecommendationEngine,
    StudentAnalyticsEngine,
    AnalyticsReportGenerator
)


def load_custom_css():
    """Apply styling for the Streamlit dashboard."""
    if not HAS_STREAMLIT:
        return
    st.markdown("""
    <style>
    .metric-card {
        background-color: #1a1a2e;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid rgba(108, 99, 255, 0.2);
    }
    .stAlert {
        border-radius: 10px;
    }
    .suggestion-card {
        background: #16213e;
        border-left: 4px solid #00d4aa;
        padding: 15px;
        border-radius: 0 10px 10px 0;
        margin-bottom: 12px;
    }
    </style>
    """, unsafe_allow_html=True)


def run_synthetic_session(v_weight, a_weight):
    """Generate and process a rich synthetic classroom session."""
    agent = MultimodalAgent(visual_weight=v_weight, audio_weight=a_weight)
    rec_engine = TeachingRecommendationEngine(use_gemini=True)
    analytics_engine = StudentAnalyticsEngine(total_session_seconds=60.0)

    # Synthetic vision results with multiple students and head poses
    num_students = 8
    duration = 60.0
    mock_attendance = {
        "session_id": f"demo_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
        "total_detected": num_students,
        "total_present": 7,
        "presence_threshold_seconds": 10,
        "records": {
            f"student_{i}": {
                "face_id": f"student_{i}",
                "first_detected": 1.0,
                "last_detected": 58.0,
                "total_visible_seconds": 55.0 if i != 8 else 8.0,
                "attendance_marked": i != 8,
                "status": "present" if i != 8 else "tracking"
            } for i in range(1, num_students + 1)
        }
    }

    mock_visual_timeline = []
    for sec in range(1, 61, 2):
        # Simulate slight drop around 25-35s
        is_dip = 24 <= sec <= 34
        mock_visual_timeline.append({
            "time": float(sec),
            "faces_detected": num_students,
            "looking_forward_count": 3 if is_dip else 7,
            "poses": [
                {"yaw": 0.4 if is_dip else 0.05, "pitch": -0.4 if is_dip else 0.0, "is_looking_forward": not is_dip}
                for _ in range(num_students)
            ]
        })

    vision_results = {
        "attendance": mock_attendance,
        "engagement_data": mock_visual_timeline,
        "duration_seconds": duration
    }

    # Synthetic audio with active speech and quiet pauses
    audio_results = process_audio(audio_source=None, is_synthetic=True, synthetic_pattern="speech_and_silence")

    # Multimodal Agent Analysis
    session_intel = agent.analyze_session(vision_results, audio_results, interval_seconds=5.0)
    suggestions = rec_engine.get_ai_enhanced_suggestions(session_intel["session_metrics"], session_intel["dynamics"])
    student_profiles = analytics_engine.compute_student_profiles(mock_attendance["records"], session_duration=duration)
    distribution = analytics_engine.generate_class_distribution(student_profiles)

    return {
        "session_intel": session_intel,
        "suggestions": suggestions,
        "student_profiles": student_profiles,
        "distribution": distribution,
        "audio_results": audio_results
    }


def main():
    """Main Streamlit application layout and routing."""
    if not HAS_STREAMLIT:
        print("Streamlit is not installed. Run: pip install streamlit")
        return

    st.set_page_config(
        page_title="Multimodal Attendance & Engagement Agent",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    load_custom_css()

    st.title("🎓 Multimodal Attendance & Engagement Agent")
    st.markdown("*Privacy-preserving classroom intelligence system integrating Computer Vision, Audio Signal Processing, and AI reasoning.*")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("⚙️ Configuration & Input")
    input_mode = st.sidebar.radio(
        "Choose Analysis Mode:",
        ["🚀 Quick Benchmark (Demo)", "📁 Upload Files (Video/Audio)"]
    )

    st.sidebar.subheader("Engagement Weights")
    v_weight = st.sidebar.slider("Visual Weight", 0.0, 1.0, 0.6, 0.05)
    a_weight = st.sidebar.slider("Audio Weight", 0.0, 1.0, 0.4, 0.05)

    if input_mode == "📁 Upload Files (Video/Audio)":
        uploaded_video = st.sidebar.file_uploader("Upload Classroom Video", type=["mp4", "avi", "mov", "webm"])
        uploaded_audio = st.sidebar.file_uploader("Upload Audio Feed (optional)", type=["wav", "mp3", "flac"])
        run_btn = st.sidebar.button("Run Pipeline Analysis", type="primary")
    else:
        uploaded_video = None
        uploaded_audio = None
        run_btn = st.sidebar.button("Run Live Benchmark", type="primary")

    # Session State Cache
    if "session_data" not in st.session_state:
        st.session_state["session_data"] = None

    if run_btn:
        with st.spinner("Processing multimodal signals..."):
            if input_mode == "🚀 Quick Benchmark (Demo)":
                st.session_state["session_data"] = run_synthetic_session(v_weight, a_weight)
                st.success("Benchmark completed successfully!")
            else:
                # Handle uploaded files
                if uploaded_video:
                    video_path = os.path.join(UPLOAD_FOLDER, uploaded_video.name)
                    with open(video_path, "wb") as f:
                        f.write(uploaded_video.read())
                    vision_res = process_video(video_path, max_frames=300)
                else:
                    vision_res = None

                if uploaded_audio:
                    audio_path = os.path.join(UPLOAD_FOLDER, uploaded_audio.name)
                    with open(audio_path, "wb") as f:
                        f.write(uploaded_audio.read())
                    audio_res = process_audio(audio_path)
                else:
                    audio_res = process_audio(None, is_synthetic=True)

                if vision_res is None:
                    st.warning("No video uploaded. Running with audio-only and synthetic vision benchmark.")
                    st.session_state["session_data"] = run_synthetic_session(v_weight, a_weight)
                else:
                    agent = MultimodalAgent(visual_weight=v_weight, audio_weight=a_weight)
                    intel = agent.analyze_session(vision_res, audio_res, interval_seconds=5.0)
                    rec_engine = TeachingRecommendationEngine(use_gemini=True)
                    analytics_engine = StudentAnalyticsEngine()
                    sugg = rec_engine.get_ai_enhanced_suggestions(intel["session_metrics"], intel["dynamics"])
                    profiles = analytics_engine.compute_student_profiles(vision_res["attendance"]["records"])
                    dist = analytics_engine.generate_class_distribution(profiles)
                    st.session_state["session_data"] = {
                        "session_intel": intel,
                        "suggestions": sugg,
                        "student_profiles": profiles,
                        "distribution": dist,
                        "audio_results": audio_res
                    }
                    st.success("File processing completed successfully!")

    # Display Dashboard Content
    data = st.session_state["session_data"]

    if data is None:
        st.info("👈 Select an input mode in the sidebar and click **Run** to launch the classroom analysis.")
        return

    metrics = data["session_intel"]["session_metrics"]
    dynamics = data["session_intel"]["dynamics"]
    suggestions = data["suggestions"]
    profiles = data["student_profiles"]
    fused_timeline = data["session_intel"]["fused_timeline"]

    # Top KPI Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Attendance Rate", f"{metrics['attendance_rate_percent']}%", f"{metrics['confirmed_present']}/{metrics['total_detected_students']} Present")
    with c2:
        st.metric("Average Engagement", f"{round(metrics['average_engagement_score'] * 100, 1)}%", metrics["engagement_classification"].upper())
    with c3:
        st.metric("Attention Drops", metrics["attention_drops"], "Drop Anomalies")
    with c4:
        st.metric("Dominant Interaction", metrics["dominant_audio_interaction"].replace("_", " ").title())

    st.markdown("---")

    # Visualizations Row
    col_chart1, col_chart2 = st.columns([2, 1])

    with col_chart1:
        st.subheader("📈 Multimodal Engagement Timeline")
        if fused_timeline and HAS_PLOTLY:
            df_timeline = pd.DataFrame(fused_timeline)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_timeline["start_time"], y=df_timeline["engagement_score"],
                mode="lines+markers", name="Fused Engagement", line=dict(color="#00d4aa", width=3)
            ))
            fig.add_trace(go.Scatter(
                x=df_timeline["start_time"], y=df_timeline["visual_score"],
                mode="lines", name="Visual Score", line=dict(color="#6c63ff", dash="dash")
            ))
            fig.add_trace(go.Scatter(
                x=df_timeline["start_time"], y=df_timeline["audio_score"],
                mode="lines", name="Audio Score", line=dict(color="#ffa500", dash="dot")
            ))
            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Time (seconds)",
                yaxis_title="Score (0.0 to 1.0)",
                yaxis=dict(range=[0, 1.05]),
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        elif fused_timeline:
            df_t = pd.DataFrame(fused_timeline)[["start_time", "engagement_score", "visual_score", "audio_score"]].set_index("start_time")
            st.line_chart(df_t)

    with col_chart2:
        st.subheader("🎯 Class Distribution")
        dist = data["distribution"]
        if dist["counts"] and HAS_PLOTLY:
            fig_donut = go.Figure(data=[go.Pie(
                labels=list(dist["counts"].keys()),
                values=list(dist["counts"].values()),
                hole=0.45,
                marker=dict(colors=["#00d4aa", "#6c63ff", "#ffa500", "#ff6b6b"])
            )])
            fig_donut.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.json(dist["percentages"])

    st.markdown("---")

    # Tabs for Details
    tab1, tab2, tab3 = st.tabs(["👥 Student Analytics", "🤖 Teaching Suggestions", "⚠️ Dynamics & Alerts"])

    with tab1:
        st.subheader("Individual Student Profiles")
        df_students = pd.DataFrame(profiles)
        if not df_students.empty:
            display_cols = ["student_id", "attendance_confirmed", "visible_duration_seconds", "presence_percentage", "engagement_score", "classification", "student_feedback"]
            st.dataframe(df_students[display_cols], use_container_width=True)

            csv_data = df_students.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Student Analytics (CSV)", csv_data, "student_analytics.csv", "text/csv")

    with tab2:
        st.subheader("Pedagogical Insights for Instructors")
        st.caption(f"Generated via: **{suggestions['source'].replace('_', ' ').title()}**")

        if suggestions.get("ai_insights"):
            st.markdown(f"**Gemini AI Summary:**\n> {suggestions['ai_insights']}")

        for s in suggestions.get("structured_suggestions", []):
            priority_color = "#ff6b6b" if s["priority"] == "high" else ("#ffa500" if s["priority"] == "medium" else "#00d4aa")
            st.markdown(f"""
            <div style="background:#16213e; border-left: 4px solid {priority_color}; padding: 12px; border-radius: 0 8px 8px 0; margin-bottom: 10px;">
                <strong style="color: {priority_color};">[{s['priority'].upper()}] {s['title']}</strong><br/>
                <span style="font-size: 0.95em;">{s['suggestion']}</span><br/>
                <small style="color: #888;">Rationale: {s['rationale']}</small>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.subheader("Classroom Dynamics & Rule-Based Triggers")
        st.write(f"**Overall Diagnosis:** {dynamics['overall_diagnosis']}")

        triggers = dynamics.get("rule_triggers", [])
        if triggers:
            for t in triggers:
                st.warning(f"⚠️ {t}")
        else:
            st.success("✅ No critical alerts triggered during this session.")


if __name__ == "__main__":
    main()
