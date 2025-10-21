#!/usr/bin/env python3
"""
Simple script to run the Kubernetes Copilot UI.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit UI."""
    ui_path = Path(__file__).parent / "k8s_copilot" / "ui" / "app.py"
    
    if not ui_path.exists():
        print(f"❌ UI file not found: {ui_path}")
        sys.exit(1)
    
    print("🚀 Starting Kubernetes Copilot UI...")
    print("📖 The UI will open in your browser automatically.")
    print("🛑 Press Ctrl+C to stop the server.")
    
    try:
        subprocess.run([
            "streamlit", "run", str(ui_path),
            "--server.address", "localhost",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down Kubernetes Copilot UI.")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it with: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()






