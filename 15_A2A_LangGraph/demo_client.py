#!/usr/bin/env python3
"""
Demo script showing the Simple Client Agent in action with A2A protocol.

This script demonstrates:
1. How to run the client agent
2. Multiple example queries 
3. Real-time interaction with the A2A server
"""

import asyncio
import logging
import sys
import time
from typing import List

# Configure logging for demo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our client agent
try:
    from simple_client_agent import SimpleClientAgent
except ImportError as e:
    print(f"❌ Error importing SimpleClientAgent: {e}")
    print("Make sure simple_client_agent.py is in the current directory")
    sys.exit(1)


class ClientDemo:
    """Demo runner for the Simple Client Agent."""
    
    def __init__(self):
        self.client_agent = None
        
    async def initialize(self):
        """Initialize the client agent."""
        print("🔧 Initializing Simple Client Agent...")
        self.client_agent = SimpleClientAgent()
        print("✅ Client Agent ready!")
        
    async def run_demo_queries(self):
        """Run a series of demo queries to showcase the client."""
        if not self.client_agent:
            print("❌ Client agent not initialized")
            return
            
        # Demo queries showcasing different agent capabilities
        demo_queries = [
            {
                "query": "What are the latest developments in artificial intelligence?",
                "description": "Web search query to test Tavily integration"
            },
            {
                "query": "Find me recent papers on transformer architectures",
                "description": "Academic search query to test ArXiv integration"
            },
            {
                "query": "What information is available in the loaded documents about AI applications?",
                "description": "RAG query to test document retrieval"
            }
        ]
        
        print(f"\n🎯 Running {len(demo_queries)} demo queries...\n")
        
        for i, demo_item in enumerate(demo_queries, 1):
            query = demo_item["query"]
            description = demo_item["description"]
            
            print("=" * 80)
            print(f"📤 Demo Query {i}/{len(demo_queries)}")
            print(f"Description: {description}")
            print(f"Query: {query}")
            print("-" * 80)
            
            try:
                # Add a small delay for readability
                await asyncio.sleep(1)
                
                start_time = time.time()
                response = await self.client_agent.query(query)
                end_time = time.time()
                
                print(f"📥 A2A Server Response:")
                print(f"")
                print(f"{response}")
                print(f"")
                print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                logger.error(f"Query failed: {e}", exc_info=True)
            
            print("=" * 80)
            
            # Pause between queries for readability
            if i < len(demo_queries):
                print("\n⏳ Waiting 3 seconds before next query...\n")
                await asyncio.sleep(3)
    
    async def interactive_mode(self):
        """Allow user to interact with the client agent."""
        if not self.client_agent:
            print("❌ Client agent not initialized")
            return
            
        print("\n🎮 Interactive Mode")
        print("Type your questions and press Enter. Type 'quit' to exit.")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n💬 Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                    
                if not user_input:
                    print("Please enter a question.")
                    continue
                
                print("🔄 Processing your question...")
                start_time = time.time()
                response = await self.client_agent.query(user_input)
                end_time = time.time()
                
                print(f"\n📥 A2A Server Response:")
                print(f"")
                print(f"{response}")
                print(f"")
                print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
                
            except KeyboardInterrupt:
                print("\n👋 Demo interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"Interactive query failed: {e}", exc_info=True)


def print_banner():
    """Print demo banner."""
    print("🤖 Simple Client Agent - A2A Protocol Demo")
    print("=" * 60)
    print("This demo shows a LangGraph client agent communicating")
    print("with an A2A server through the Agent-to-Agent protocol.")
    print("=" * 60)


def print_prerequisites():
    """Print demo prerequisites."""
    print("\n📋 Prerequisites:")
    print("1. A2A Server must be running on localhost:10000")
    print("   Start it with: uv run python -m app")
    print("2. Environment variables must be configured")
    print("   Check with: uv run python check_env.py")
    print("3. Dependencies must be installed")
    print("   Install with: uv sync")


async def main():
    """Main demo function."""
    print_banner()
    print_prerequisites()
    
    # Ask user if they want to proceed
    print("\n❓ Is the A2A server running? (y/n): ", end="")
    user_input = input().strip().lower()
    
    if user_input not in ['y', 'yes']:
        print("Please start the A2A server first:")
        print("  uv run python -m app")
        print("Then run this demo again.")
        return
    
    try:
        # Initialize demo
        demo = ClientDemo()
        await demo.initialize()
        
        # Ask what type of demo to run
        print("\n🎯 Demo Options:")
        print("1. Run automated demo queries")
        print("2. Interactive mode (ask your own questions)")
        print("3. Both")
        
        choice = input("\nChoose an option (1/2/3): ").strip()
        
        if choice in ['1', '3']:
            await demo.run_demo_queries()
            
        if choice in ['2', '3']:
            await demo.interactive_mode()
            
        if choice not in ['1', '2', '3']:
            print("Invalid choice. Running automated demo...")
            await demo.run_demo_queries()
            
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.error(f"Demo failed: {e}", exc_info=True)
        print("\n🔧 Troubleshooting:")
        print("1. Make sure the A2A server is running: uv run python -m app")
        print("2. Check your environment: uv run python check_env.py")
        print("3. Check the logs above for specific error details")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
