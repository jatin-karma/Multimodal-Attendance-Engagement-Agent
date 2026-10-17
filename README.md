# Multimodal Attendance & Engagement Agent

An AI-powered, privacy-preserving classroom intelligence system that integrates computer vision, audio signal processing, and agent-based reasoning to automatically track attendance, analyze engagement trends, generate AI-driven teaching suggestions, and provide student session analytics.

## 🎯 Key Features

- **Automated Attendance Tracking** — Face detection and sustained presence logic for accurate, automated roll calls
- **Engagement Analysis** — Multimodal fusion of visual and audio signals to estimate real-time engagement
- **AI Teaching Suggestions** — Context-aware recommendations powered by Google Gemini 2.0 Flash
- **Student Analytics** — Per-student session summaries with engagement trends and participation scores
- **Privacy-Preserving** — Designed for offline/local execution with no cloud-based speech recognition

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend Layer                        │
│  Flask API Gateway │ HTML5 Templates │ CSS3 │ JS        │
│  Route Handlers │ GenAI Utils │ Audio Utils              │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────┴─────────────────────────────────┐
│                   AI Service Layer                       │
│  Google Gemini 2.0 Flash │ Engagement Scoring Engine     │
│  Teaching Suggestion Generator │ Analytics Engine        │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────┴─────────────────────────────────┐
│              Processing Modules                          │
│  Vision Module (OpenCV + MediaPipe)                      │
│  Audio Module (Librosa)                                  │
│  Multimodal Agent (Signal Fusion + Reasoning)            │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────┴─────────────────────────────────┐
│                   Storage Layer                          │
│  Attendance Logs │ Generated Audio │ Uploaded Files       │
└─────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
├── app.py                  # Main Flask application
├── config.py               # Configuration settings
├── dashboard.py            # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── modules/
│   ├── vision.py           # Face detection & attendance tracking
│   ├── audio.py            # Audio signal processing
│   ├── agent.py            # Multimodal engagement agent
│   └── analytics.py        # Analytics & recommendation engine
├── utils/
│   ├── genai_utils.py      # Google Gemini API integration
│   └── audio_utils.py      # Audio helper utilities
├── templates/              # Flask HTML templates
├── static/                 # CSS, JS, assets
├── storage/                # Generated output files
├── uploads/                # Uploaded video/audio input files
└── tests/                  # Unit & integration tests
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Webcam or video file (for vision module)
- Microphone or audio file (optional, for audio module)

### Installation

```bash
# Clone the repository
git clone https://github.com/jatin-karma/Multimodal-Attendance-Engagement-Agent.git
cd Multimodal-Attendance-Engagement-Agent

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_google_gemini_api_key
FLASK_SECRET_KEY=your_secret_key
UPLOAD_FOLDER=uploads
```

### Running the Application

```bash
# Run the Flask application
python app.py

# Run the Streamlit dashboard (optional)
streamlit run dashboard.py
```

## 📋 Milestones

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Requirements & Environment Setup | ✅ Complete |
| 2 | Vision Module Development | ✅ Complete |
| 3 | Audio Module Development | ✅ Complete |
| 4 | Multimodal Agent Implementation | ✅ Complete |
| 5 | Analytics & Recommendation Engine | ⬜ Pending |
| 6 | Interface, Testing & Deployment | ⬜ Pending |

## 🛠️ Technology Stack

| Category | Technology |
|----------|-----------|
| Web Framework | Flask |
| Computer Vision | OpenCV, MediaPipe |
| Audio Processing | Librosa |
| AI/GenAI | Google Gemini 2.0 Flash |
| Data Analytics | NumPy, Pandas, SciPy |
| Visualization | Matplotlib, Plotly |
| Dashboard | Streamlit |
| Testing | Pytest |

## 📄 License

This project is developed for educational purposes as part of the GenAI with Google Cloud program.
