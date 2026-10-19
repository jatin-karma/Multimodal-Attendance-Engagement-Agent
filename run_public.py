"""
Public HTTPS Tunnel Runner for Multimodal Attendance & Engagement Agent.

Starts the Flask web application and creates a secure public HTTPS tunnel
(via pyngrok or localtunnel) so you can access the live camera & mic dashboard
from any device, phone, or remote browser without cloud billing.

Usage:
    python run_public.py
"""

import sys
import os
import time
import threading

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from config import APP_NAME, PORT, HOST


def run_flask_server():
    """Run Flask server in background thread."""
    app = create_app()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)


def start_tunnel():
    """Start HTTPS tunnel and print public link."""
    # Start Flask
    server_thread = threading.Thread(target=run_flask_server, daemon=True)
    server_thread.start()
    time.sleep(2)  # Wait for Flask to bind

    public_url = None

    # Try pyngrok first
    try:
        from pyngrok import ngrok
        tunnel = ngrok.connect(PORT, "http")
        public_url = tunnel.public_url
        if public_url.startswith("http://"):
            public_url = public_url.replace("http://", "https://")
    except Exception as e:
        print(f"[!] pyngrok note: {e}")

    # Fallback to localtunnel via npx if ngrok requires authtoken
    if not public_url:
        print("[*] Starting localtunnel fallback...")
        import subprocess
        try:
            cmd = "npx -y localtunnel --port " + str(PORT)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True
            )
            for line in proc.stdout:
                if "url is:" in line:
                    public_url = line.split("url is:")[-1].strip()
                    break
        except Exception as e:
            print(f"[!] localtunnel error: {e}")

    print("\n" + "=" * 65)
    print(f"  🎓 {APP_NAME} — LIVE PUBLIC HTTPS ACCESS")
    print("=" * 65)
    print(f"[*] Local URL:      http://localhost:{PORT}/dashboard")
    if public_url:
        print(f"[*] Public HTTPS:   {public_url}/dashboard")
        print("\n  👉 Open the Public HTTPS link on ANY phone, tablet, or laptop!")
        print("     Camera & microphone permissions will work seamlessly over HTTPS.")
    else:
        print("[!] Could not generate tunnel. You can still access locally via:")
        print(f"    http://localhost:{PORT}/dashboard")
    print("=" * 65)
    print("[*] Press Ctrl+C to terminate the session.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Terminating tunnel and server...")
        try:
            from pyngrok import ngrok
            ngrok.kill()
        except Exception:
            pass


if __name__ == "__main__":
    start_tunnel()
