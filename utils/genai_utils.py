"""
Google Gemini API Integration Utilities.

Provides helper functions for interacting with the Google Gemini 2.0 Flash API
for generating teaching suggestions, engagement insights, and analytics reports.
"""

import google.generativeai as genai
from config import GEMINI_CONFIG


def initialize_gemini():
    """
    Initialize the Google Gemini API client.
    
    Returns:
        genai.GenerativeModel: Configured Gemini model instance.
        
    Raises:
        ValueError: If GEMINI_API_KEY is not set.
    """
    api_key = GEMINI_CONFIG["api_key"]
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Please set it in your .env file."
        )
    
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel(
        model_name=GEMINI_CONFIG["model_name"],
        generation_config={
            "max_output_tokens": GEMINI_CONFIG["max_output_tokens"],
            "temperature": GEMINI_CONFIG["temperature"],
            "top_p": GEMINI_CONFIG["top_p"],
            "top_k": GEMINI_CONFIG["top_k"],
        }
    )
    
    return model


def generate_teaching_suggestion(model, engagement_data):
    """
    Generate AI-driven teaching suggestions based on engagement data.
    
    Args:
        model: Initialized Gemini model instance.
        engagement_data (dict): Dictionary containing engagement metrics
            including scores, trends, and session dynamics.
    
    Returns:
        str: AI-generated teaching suggestion text.
    """
    prompt = f"""
    You are an AI teaching assistant analyzing classroom engagement data.
    Based on the following engagement metrics, provide specific, actionable 
    teaching suggestions to improve student engagement and learning outcomes.
    
    Engagement Data:
    - Average Engagement Score: {engagement_data.get('avg_score', 'N/A')}
    - Engagement Trend: {engagement_data.get('trend', 'N/A')}
    - Active Participants: {engagement_data.get('active_count', 'N/A')}
    - Total Participants: {engagement_data.get('total_count', 'N/A')}
    - Low Engagement Periods: {engagement_data.get('low_periods', 'N/A')}
    - Audio Activity Level: {engagement_data.get('audio_activity', 'N/A')}
    
    Provide 3-5 specific, actionable suggestions. Focus on:
    1. Immediate actions the instructor can take
    2. Session pacing adjustments
    3. Interactive activity recommendations
    
    Keep suggestions concise and practical.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Unable to generate suggestions: {str(e)}"


def generate_student_summary(model, student_data):
    """
    Generate an AI-powered summary for individual student analytics.
    
    Args:
        model: Initialized Gemini model instance.
        student_data (dict): Dictionary containing per-student metrics.
    
    Returns:
        str: AI-generated student session summary.
    """
    prompt = f"""
    You are an AI teaching assistant providing a brief session summary for a student.
    Based on the following data, provide a constructive, encouraging summary.
    
    Student Session Data:
    - Attendance Duration: {student_data.get('attendance_duration', 'N/A')}
    - Engagement Score: {student_data.get('engagement_score', 'N/A')}
    - Participation Level: {student_data.get('participation', 'N/A')}
    - Attention Pattern: {student_data.get('attention_pattern', 'N/A')}
    
    Provide a 2-3 sentence constructive summary. Be encouraging and specific.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Unable to generate summary: {str(e)}"
