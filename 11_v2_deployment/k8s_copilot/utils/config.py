"""Configuration utilities for the Kubernetes copilot system."""

import os
from pathlib import Path
from typing import Dict, Any

def get_config() -> Dict[str, Any]:
    """Get configuration settings for the application."""
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "vector_store_location": os.getenv("VECTOR_STORE_LOCATION", ":memory:"),
        "data_dir": Path(__file__).parent.parent / "data",
        "max_tokens": int(os.getenv("MAX_TOKENS", "4000")),
        "temperature": float(os.getenv("TEMPERATURE", "0")),
        "debug": os.getenv("DEBUG", "false").lower() == "true"
    }

def setup_environment() -> bool:
    """Set up the environment and check for required configurations."""
    config = get_config()
    
    if not config["openai_api_key"]:
        print("❌ OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        return False
    
    if not config["data_dir"].exists():
        print(f"❌ Data directory not found: {config['data_dir']}")
        print("Please run the data generation script first.")
        return False
    
    print("✅ Environment setup complete")
    return True

def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent.parent / "data"






