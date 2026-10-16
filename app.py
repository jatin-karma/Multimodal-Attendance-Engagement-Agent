"""
Flask Application Entry Point for the Multimodal Attendance & Engagement Agent.

This is the main application file that initializes the Flask server,
registers routes, and starts the web application.
"""

from flask import Flask, render_template, jsonify
from flask_cors import CORS
from config import (
    APP_NAME, DEBUG, SECRET_KEY, HOST, PORT,
    UPLOAD_FOLDER, ensure_directories
)

# ============================================
# Application Factory
# ============================================
def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Core configuration
    app.config["SECRET_KEY"] = SECRET_KEY
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max upload

    # Enable CORS
    CORS(app)

    # Ensure all required directories exist
    ensure_directories()

    # ----------------------------------------
    # Register Routes
    # ----------------------------------------
    @app.route("/")
    def index():
        """Render the main application page."""
        return render_template("index.html", app_name=APP_NAME)

    @app.route("/health")
    def health_check():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "application": APP_NAME,
            "version": "1.0.0"
        })

    @app.route("/api/status")
    def api_status():
        """API status endpoint showing module availability."""
        return jsonify({
            "application": APP_NAME,
            "modules": {
                "vision": "available",
                "audio": "available",
                "agent": "available",
                "analytics": "available"
            },
            "status": "operational"
        })

    return app


# ============================================
# Run Application
# ============================================
if __name__ == "__main__":
    app = create_app()
    print(f"\n{'='*50}")
    print(f"  {APP_NAME}")
    print(f"  Running on http://{HOST}:{PORT}")
    print(f"{'='*50}\n")
    app.run(host=HOST, port=PORT, debug=DEBUG)
