#!/bin/bash
# Simple script to demonstrate the A2A Client Agent

echo "🚀 A2A Client Agent Demo"
echo "========================"
echo ""

# Check if server is running
echo "🔍 Checking if A2A server is running..."
if curl -s http://localhost:10000/.well-known/agent_card > /dev/null 2>&1; then
    echo "✅ A2A server is running on localhost:10000"
    echo ""
else
    echo "❌ A2A server is not running!"
    echo ""
    echo "Please start the server first:"
    echo "  Terminal 1: uv run python -m app"
    echo ""
    echo "Then run this demo again."
    exit 1
fi

# Run the demo
echo "🎬 Starting client demo..."
echo ""
uv run python demo_client.py
