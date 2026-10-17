@echo off
title Multimodal Attendance & Engagement Agent
echo ========================================================
echo   Launching Multimodal Attendance & Engagement Agent...
echo ========================================================

REM Activate virtual environment if present
if exist "venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check if dependencies are installed
python -c "import flask" 2>nul
if %errorlevel% neq 0 (
    echo [*] Installing required dependencies...
    pip install -r requirements.txt
)

echo [*] Starting Flask API Gateway and Dashboard on http://localhost:5000
start http://localhost:5000/dashboard
python app.py
pause
