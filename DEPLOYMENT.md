# Deployment Guide — Multimodal Attendance & Engagement Agent

This guide covers deployment options for the Multimodal Attendance & Engagement Agent, including **Google Cloud Run (Serverless)**, **Docker / Container Deployment**, and **Edge / Local Classroom Setup**.

---

## ☁️ Option 1: Google Cloud Run (Recommended for Cloud)

Google Cloud Run is a fully managed serverless platform that automatically scales containers and provides automatic HTTPS (essential for browser camera/microphone permissions).

### Prerequisites
1. [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/docs/install) installed and initialized.
2. A GCP project with billing enabled.
3. Your Google Gemini API Key.

### Step-by-Step Cloud Run Deployment

1. **Authenticate with Google Cloud:**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Enable Required Google Cloud Services:**
   ```bash
   gcloud services enable run.googleapis.com \
                          cloudbuild.googleapis.com \
                          artifactregistry.googleapis.com \
                          secretmanager.googleapis.com
   ```

3. **Store your Gemini API Key in Google Secret Manager (Secure):**
   ```bash
   echo -n "YOUR_GEMINI_API_KEY" | gcloud secrets create gemini-api-key --data-file=-
   ```

4. **Deploy directly from source to Cloud Run:**
   ```bash
   gcloud run deploy multimodal-agent \
     --source . \
     --region us-central1 \
     --platform managed \
     --allow-unauthenticated \
     --memory 2Gi \
     --cpu 2 \
     --set-secrets GEMINI_API_KEY=gemini-api-key:latest \
     --set-env-vars FLASK_SECRET_KEY=change-in-production,GEMINI_MODEL=gemini-3.8-flash
   ```

5. **Access Your Application:**
   Cloud Run will output your live HTTPS URL:
   ```text
   Service URL: https://multimodal-agent-xyz-uc.a.run.app
   ```
   Open `https://multimodal-agent-xyz-uc.a.run.app/dashboard` in any browser!

> [!NOTE]
> When accessed over HTTPS, browsers allow full access to the webcam and microphone (`navigator.mediaDevices.getUserMedia`). Students and educators can use live facial tracking and audio analysis directly from their device.

---

## 🐳 Option 2: Docker Container (Local / Any Cloud VPS / GCE)

You can run the entire system inside a Docker container locally or on any cloud virtual machine (e.g., Google Compute Engine, AWS EC2, DigitalOcean).

### Quick Launch with Docker Compose:
```bash
# 1. Ensure your .env file has your GEMINI_API_KEY
# 2. Build and start the container
docker compose up --build -d

# 3. View container logs
docker compose logs -f

# 4. Open in browser:
# http://localhost:5000/dashboard
```

### Manual Docker Build & Run:
```bash
# Build the Docker image
docker build -t multimodal-agent:latest .

# Run the container
docker run -d -p 5000:8080 \
  -e GEMINI_API_KEY="your_api_key_here" \
  -e FLASK_SECRET_KEY="your_secret_key" \
  -v "$(pwd)/storage:/app/storage" \
  --name classroom-agent \
  multimodal-agent:latest
```

---

## 🏫 Option 3: Local / Edge Classroom Deployment (Smart Podium)

In physical classroom environments with a local PC connected to a podium webcam and boundary microphone:

### 1. Launch via Windows Batch File:
Double-click `start.bat` or run:
```powershell
.\start.bat
```

### 2. Launch Dedicated Real-Time Video & Audio Tracker:
```powershell
python run_live.py --camera 0
```
- Automatically monitors attendance via the connected webcam.
- Evaluates sustained face presence (10s threshold).
- Press `q` to finish: automatically saves CSV/JSON logs and fetches AI suggestions from Google Gemini.

### 3. Launch the Optional Streamlit Analytics Dashboard:
```powershell
python -m streamlit run dashboard.py
```

---

## 🔒 Security & Privacy Best Practices

- **Zero Content Transcription:** The audio module analyzes volume (RMS), Zero-Crossing Rate, and noise characteristics without speech-to-text or conversation transcription, ensuring full student privacy.
- **Local Face Processing:** Facial landmarks and head poses are computed locally or on your container; raw facial biometric matrices are not transmitted to third-party servers.
- **Environment Isolation:** Keep `.env` out of version control (enforced by `.gitignore`). In cloud environments, use Google Secret Manager.
