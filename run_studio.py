"""
Novel Visualiser — Studio Launcher

Launches the unified fullstack studio:
- FastAPI Backend & REST API
- Production React Studio Web App
- 3-GPU Cluster Telemetry Monitor
- Static media server for rendered shots & plates

Usage:
    python run_studio.py
"""

import os
import sys
import webbrowser
from pathlib import Path

# Add novel_visualiser to sys.path
root_dir = Path(__file__).resolve().parent
app_dir = root_dir / "novel_visualiser"
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

import uvicorn
from app.config import get_settings


def main():
    settings = get_settings()
    host = "127.0.0.1"
    port = 8000
    studio_url = f"http://{host}:{port}"

    print("=" * 72)
    print("🎬 NOVEL VISUALISER — AUTONOMOUS ANIME PRODUCTION STUDIO")
    print("=" * 72)
    print(f"🚀 Studio Web UI  : {studio_url}")
    print(f"📖 OpenAPI Docs   : {studio_url}/docs")
    print("-" * 72)
    print(f"🟢 Colab GPU      : {settings.comfyui.colab_url}")
    print(f"🟢 Kaggle GPU 0   : {settings.comfyui.gpu0_url}")
    print(f"🟢 Kaggle GPU 1   : {settings.comfyui.gpu1_url}")
    print("=" * 72)

    # Launch browser after a brief delay
    try:
        webbrowser.open(studio_url)
    except Exception:
        pass

    # Start FastAPI server
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
        app_dir=str(app_dir),
        log_level="info",
    )


if __name__ == "__main__":
    main()
