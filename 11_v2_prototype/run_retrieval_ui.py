#!/usr/bin/env python3
"""
Launch script for the enhanced Kubernetes Copilot UI with retrieval method comparison.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the enhanced Streamlit UI with retrieval comparison."""
    ui_path = Path(__file__).parent / "k8s_copilot" / "ui" / "retrieval_comparison_ui.py"
    
    if not ui_path.exists():
        print(f"❌ UI file not found: {ui_path}")
        sys.exit(1)
    
    # Set environment variables if needed
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. The UI may not function properly.")
        print("   Set it with: export OPENAI_API_KEY='your-api-key-here'")
    
    print("🚀 Starting Enhanced Kubernetes Copilot UI with Retrieval Comparison...")
    print("📖 The UI will open in your browser automatically.")
    print("🔍 Features: Compare different retrieval methods, performance analysis, interactive testing")
    print("🛑 Press Ctrl+C to stop the server.")
    
    try:
        subprocess.run([
            "streamlit", "run", str(ui_path),
            "--server.address", "localhost",
            "--server.port", "8502"  # Different port to avoid conflicts
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down Enhanced Kubernetes Copilot UI.")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it with: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
