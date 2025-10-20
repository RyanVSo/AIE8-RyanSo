#!/usr/bin/env python3
"""
Main application runner for the K8s RAG Copilot.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the Streamlit application."""
    
    # Check if we're in the right directory
    current_dir = Path(__file__).parent
    app_file = current_dir / "k8s_rag" / "ui" / "app.py"
    
    if not app_file.exists():
        print(f"❌ Application file not found: {app_file}")
        print("Please make sure you're running this from the correct directory.")
        sys.exit(1)
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    print("🚀 Starting K8s RAG Copilot...")
    print(f"📁 Application directory: {current_dir}")
    print(f"🌐 Starting Streamlit server...")
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_file),
            "--server.port", "8501",
            "--server.address", "localhost"
        ], cwd=current_dir)
    except KeyboardInterrupt:
        print("\n👋 Shutting down K8s RAG Copilot...")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
