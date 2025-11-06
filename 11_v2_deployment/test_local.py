#!/usr/bin/env python3
"""
Local testing script for the Vercel-ready Kubernetes Copilot.
This script helps you test the application locally before deploying to Vercel.
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path
import signal
import threading

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        print("✅ FastAPI dependencies found")
        return True
    except ImportError:
        print("❌ Missing dependencies. Please run: pip install -r requirements.txt")
        return False

def start_api_server():
    """Start the FastAPI server."""
    api_path = Path(__file__).parent / "api"
    os.chdir(api_path)
    
    print("🚀 Starting FastAPI server on http://localhost:8000")
    process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "main:app", 
        "--reload", "--host", "0.0.0.0", "--port", "8000"
    ])
    return process

def start_frontend_server():
    """Start a simple HTTP server for the frontend."""
    public_path = Path(__file__).parent / "public"
    os.chdir(public_path)
    
    print("🌐 Starting frontend server on http://localhost:3000")
    process = subprocess.Popen([
        sys.executable, "-m", "http.server", "3000"
    ])
    return process

def main():
    """Main testing function."""
    print("🧪 Kubernetes Copilot - Local Testing")
    print("=" * 50)
    
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        print()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Start servers
    try:
        api_process = start_api_server()
        time.sleep(2)  # Give API server time to start
        
        frontend_process = start_frontend_server()
        time.sleep(1)  # Give frontend server time to start
        
        print("\n🎉 Servers started successfully!")
        print("📊 API Documentation: http://localhost:8000/docs")
        print("🖥️  Frontend Application: http://localhost:3000")
        print("🔍 API Health Check: http://localhost:8000/api/health")
        print("\n💡 The frontend will try to connect to the API at localhost:8000")
        print("   Make sure both servers are running for full functionality.")
        print("\n🛑 Press Ctrl+C to stop both servers")
        
        # Optionally open browser
        try:
            webbrowser.open("http://localhost:3000")
        except:
            pass
        
        # Wait for user interruption
        def signal_handler(sig, frame):
            print("\n\n🛑 Shutting down servers...")
            api_process.terminate()
            frontend_process.terminate()
            print("👋 Servers stopped. Goodbye!")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down servers...")
        api_process.terminate()
        frontend_process.terminate()
        print("👋 Servers stopped. Goodbye!")
        return 0
    except Exception as e:
        print(f"❌ Error starting servers: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
