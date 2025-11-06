"""
FastAPI backend for Kubernetes Copilot - Vercel compatible.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Kubernetes Copilot API", version="1.0.0")

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import K8s components (handle import errors gracefully for Vercel)
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from k8s_copilot.vector_db.vector_store import K8sVectorStore
    from k8s_copilot.vector_db.data_loader import K8sDataLoader
    from k8s_copilot.agents.k8s_agent import K8sCopilotAgent, K8sRAGAgent
    
    K8S_IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"K8s imports not available: {e}")
    K8S_IMPORTS_AVAILABLE = False

# Global variables for caching (in production, use Redis or similar)
_vector_store = None
_copilot_agent = None
_rag_agent = None

class QueryRequest(BaseModel):
    query: str
    agent_type: str = "copilot"  # "copilot" or "rag"

class QueryResponse(BaseModel):
    response: str
    agent_type: str
    query: str
    success: bool
    error: Optional[str] = None

class SystemStats(BaseModel):
    total_documents: int
    document_types: Dict[str, int]
    initialized: bool

def initialize_system():
    """Initialize the K8s system with caching."""
    global _vector_store, _copilot_agent, _rag_agent
    
    if not K8S_IMPORTS_AVAILABLE:
        raise HTTPException(status_code=500, detail="K8s components not available")
    
    if _vector_store is None:
        logger.info("Initializing K8s system...")
        
        # Set API keys from environment variables
        if not os.getenv("OPENAI_API_KEY"):
            # For demo purposes - in production, always use environment variables
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
        
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        try:
            # Initialize vector store
            _vector_store = K8sVectorStore()
            
            # Load data
            data_dir = Path(__file__).parent.parent / "k8s_copilot" / "data"
            if not data_dir.exists():
                logger.error(f"Data directory not found: {data_dir}")
                raise HTTPException(status_code=500, detail="Data directory not found")
            
            data_loader = K8sDataLoader(data_dir)
            data_loader.load_all_data(_vector_store)
            
            # Initialize agents
            _copilot_agent = K8sCopilotAgent(_vector_store)
            _rag_agent = K8sRAGAgent(_vector_store)
            
            logger.info("K8s system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize K8s system: {e}")
            raise HTTPException(status_code=500, detail=f"System initialization failed: {str(e)}")
    
    return _vector_store, _copilot_agent, _rag_agent

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Kubernetes Copilot API is running", "status": "healthy"}

@app.get("/api/health")
async def health_check():
    """Detailed health check."""
    try:
        vector_store, _, _ = initialize_system()
        stats = vector_store.get_stats()
        return {
            "status": "healthy",
            "initialized": True,
            "total_documents": stats["total_documents"],
            "k8s_imports": K8S_IMPORTS_AVAILABLE
        }
    except Exception as e:
        return {
            "status": "error", 
            "initialized": False,
            "error": str(e),
            "k8s_imports": K8S_IMPORTS_AVAILABLE
        }

@app.get("/api/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system statistics."""
    try:
        vector_store, _, _ = initialize_system()
        stats = vector_store.get_stats()
        return SystemStats(
            total_documents=stats["total_documents"],
            document_types=stats["document_types"],
            initialized=True
        )
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query."""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        vector_store, copilot_agent, rag_agent = initialize_system()
        
        # Select agent based on type
        if request.agent_type.lower() == "rag":
            agent = rag_agent
            agent_display = "RAG Agent"
        else:
            agent = copilot_agent
            agent_display = "Copilot Agent"
        
        logger.info(f"Processing query with {agent_display}: {request.query[:50]}...")
        
        # Process the query
        response = agent.invoke(request.query.strip())
        
        return QueryResponse(
            response=response,
            agent_type=agent_display,
            query=request.query,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        return QueryResponse(
            response="",
            agent_type=request.agent_type,
            query=request.query,
            success=False,
            error=str(e)
        )

@app.get("/api/cost-data")
async def get_cost_data():
    """Get cost analysis data for visualization."""
    try:
        data_dir = Path(__file__).parent.parent / "k8s_copilot" / "data"
        
        # Load cost data files
        cost_files = {
            "deployment_costs": data_dir / "deployment_costs.csv",
            "node_costs": data_dir / "node_costs.csv",
            "cost_data": data_dir / "cost_data.json"
        }
        
        result = {}
        
        for file_type, file_path in cost_files.items():
            if file_path.exists():
                try:
                    if file_path.suffix == '.csv':
                        import pandas as pd
                        df = pd.read_csv(file_path)
                        result[file_type] = df.to_dict('records')
                    elif file_path.suffix == '.json':
                        with open(file_path, 'r') as f:
                            result[file_type] = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load {file_type}: {e}")
                    result[file_type] = None
            else:
                result[file_type] = None
        
        return {"success": True, "data": result}
        
    except Exception as e:
        logger.error(f"Failed to get cost data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/example-queries")
async def get_example_queries():
    """Get example queries for the frontend."""
    return {
        "queries": [
            "What are the costs of my Kubernetes deployments?",
            "How many GPUs does the ml-training deployment use?",
            "How can I improve resource utilization?",
            "Which deployments are using the most memory?",
            "Show me optimization opportunities",
            "What's the total cluster cost?",
            "Analyze the nginx-deployment resources",
            "What deployments are running in my cluster?",
            "How many total pods do I have?"
        ]
    }

# For Vercel deployment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
