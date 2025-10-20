"""
Configuration utilities for the K8s RAG system.
"""

import os
from typing import Dict, Any
from pathlib import Path

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed


def setup_environment() -> bool:
    """Setup environment variables and check API keys.
    
    Returns:
        True if setup successful, False otherwise
    """
    # Check required environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("💡 Set it with: export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    # Optional environment variables
    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        print("⚠️  COHERE_API_KEY not set - contextual compression will use LLM fallback")
        print("💡 For better reranking, set: export COHERE_API_KEY='your-cohere-key-here'")
    
    return True


def get_config() -> Dict[str, Any]:
    """Get system configuration.
    
    Returns:
        Configuration dictionary
    """
    return {
        "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
        "retrieval_k": int(os.getenv("RETRIEVAL_K", "5")),
        "debug": os.getenv("DEBUG", "false").lower() == "true",
    }
