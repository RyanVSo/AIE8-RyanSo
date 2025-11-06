"""
Lightweight FastAPI backend for Kubernetes Copilot - Vercel serverless deployment.
Minimal dependencies to stay under 250MB limit.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Kubernetes Copilot API", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to import OpenAI (minimal dependency)
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    OPENAI_AVAILABLE = True
    logger.info("OpenAI integration available")
except ImportError as e:
    logger.warning(f"OpenAI not available: {e}")
    OPENAI_AVAILABLE = False

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    agent_type: str = "copilot"

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

# Mock data for demonstration (since we removed vector database)
MOCK_K8S_DATA = {
    "deployments": [
        {"name": "nginx-deployment", "replicas": 3, "cpu": "500m", "memory": "512Mi"},
        {"name": "api-server", "replicas": 2, "cpu": "1000m", "memory": "1Gi"},
        {"name": "ml-training", "replicas": 1, "cpu": "2000m", "memory": "4Gi", "gpus": 2},
        {"name": "database-postgresql", "replicas": 1, "cpu": "500m", "memory": "2Gi"},
        {"name": "frontend-react", "replicas": 2, "cpu": "250m", "memory": "256Mi"}
    ],
    "costs": {
        "nginx-deployment": {"monthly": 45.20, "cpu_cost": 25.50, "memory_cost": 19.70},
        "api-server": {"monthly": 89.40, "cpu_cost": 52.80, "memory_cost": 36.60},
        "ml-training": {"monthly": 234.60, "cpu_cost": 134.20, "memory_cost": 100.40},
        "database-postgresql": {"monthly": 78.30, "cpu_cost": 22.10, "memory_cost": 56.20},
        "frontend-react": {"monthly": 23.15, "cpu_cost": 12.50, "memory_cost": 10.65}
    },
    "cluster_stats": {
        "total_nodes": 5,
        "total_pods": 23,
        "total_deployments": 5,
        "total_gpus": 2,
        "cpu_utilization": "65%",
        "memory_utilization": "78%",
        "storage_utilization": "42%",
        "network_utilization": "23%"
    }
}

def get_openai_response(query: str, agent_type: str = "copilot") -> str:
    """Get response from OpenAI using minimal LangChain."""
    if not OPENAI_AVAILABLE:
        return "OpenAI integration not available. This is a mock response for your query: " + query
    
    if not os.getenv("OPENAI_API_KEY"):
        return "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
    
    try:
        # Initialize ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=1000
        )
        
        # Create system message based on agent type
        if agent_type == "rag":
            system_msg = """You are a simple Kubernetes assistant. Answer questions about Kubernetes deployments, costs, and resources based on the following mock data:

Deployments:
- nginx-deployment: 3 replicas, 500m CPU, 512Mi memory, monthly cost: $45.20
- api-server: 2 replicas, 1000m CPU, 1Gi memory, monthly cost: $89.40  
- ml-training: 1 replica, 2000m CPU, 4Gi memory, 2 GPUs, monthly cost: $234.60
- database-postgresql: 1 replica, 500m CPU, 2Gi memory, monthly cost: $78.30
- frontend-react: 2 replicas, 250m CPU, 256Mi memory, monthly cost: $23.15

Cluster Stats:
- Total nodes: 5, Total pods: 23, Total deployments: 5, Total GPUs: 2
- CPU utilization: 65%, Memory: 78%, Storage: 42%, Network: 23%

Answer directly and concisely."""
        else:
            system_msg = """You are an expert Kubernetes Copilot assistant with specialized knowledge of cluster management, cost optimization, and resource analysis.

You have access to the following cluster data:
- 5 deployments running across 5 nodes with 23 total pods
- 2 GPUs available in the ml-training deployment
- Total monthly cluster cost: $470.65
- Current utilization: CPU 65%, Memory 78%, Storage 42%, Network 23%

Key deployments and their costs:
- ml-training (most expensive): $234.60/month, 2 GPUs, 4Gi memory
- api-server: $89.40/month, high CPU usage
- database-postgresql: $78.30/month, memory intensive
- nginx-deployment: $45.20/month, load balancer
- frontend-react (least expensive): $23.15/month

Provide detailed, actionable insights about costs, optimization opportunities, resource utilization, and specific recommendations for Kubernetes best practices."""
        
        # Create messages
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=query)
        ]
        
        # Get response
        response = llm.invoke(messages)
        return response.content
        
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return f"Error processing query with OpenAI: {str(e)}"

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Kubernetes Copilot API is running", "status": "healthy", "version": "2.0.0-lightweight"}

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "initialized": True,
        "total_documents": 50,  # Mock data
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "cohere_configured": False,  # Removed to reduce size
        "mode": "lightweight",
        "vector_db": False,  # Disabled for size limits
        "full_agent_system": False  # Disabled for size limits
    }

@app.get("/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system statistics."""
    return SystemStats(
        total_documents=50,  # Mock data
        document_types={
            "deployments": 5,
            "services": 5,
            "configmaps": 10,
            "secrets": 8,
            "ingresses": 5,
            "cost_data": 12,
            "resource_metrics": 5
        },
        initialized=True
    )

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query using lightweight OpenAI integration."""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Processing query: {request.query[:50]}...")
        
        # Get response from OpenAI
        response = get_openai_response(request.query.strip(), request.agent_type)
        
        agent_display = "RAG Agent (Lightweight)" if request.agent_type == "rag" else "Copilot Agent (Lightweight)"
        
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

@app.get("/cost-data")
async def get_cost_data():
    """Get cost analysis data (mock data for lightweight deployment)."""
    try:
        # Generate mock cost data structure compatible with frontend
        deployment_costs = []
        for i, (deployment, cost_info) in enumerate(MOCK_K8S_DATA["costs"].items()):
            deployment_costs.append({
                "deployment": deployment,
                "date": f"2024-11-{str(i+1).zfill(2)}",
                "total_cost": cost_info["monthly"] / 30,  # Daily cost
                "cpu_cost": cost_info["cpu_cost"] / 30,
                "memory_cost": cost_info["memory_cost"] / 30,
                "storage_cost": (cost_info["monthly"] - cost_info["cpu_cost"] - cost_info["memory_cost"]) / 30 * 0.3,
                "network_cost": (cost_info["monthly"] - cost_info["cpu_cost"] - cost_info["memory_cost"]) / 30 * 0.1
            })
        
        return {
            "success": True,
            "data": {
                "deployment_costs": deployment_costs,
                "node_costs": None,  # Simplified for lightweight deployment
                "cost_data": MOCK_K8S_DATA["costs"]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get cost data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/example-queries")
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

# Export for Vercel serverless
from mangum import Mangum
handler = Mangum(app)

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)