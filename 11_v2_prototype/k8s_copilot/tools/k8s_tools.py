"""
Kubernetes-specific tools for the copilot system.
These tools provide specialized functionality for analyzing K8s resources, costs, and optimizations.
"""

import json
import yaml
from typing import List, Dict, Any, Optional, Annotated
from langchain_core.tools import tool
from langchain_core.documents import Document

from ..vector_db.vector_store import K8sVectorStore

class K8sManifestAnalyzer:
    """Analyzes Kubernetes manifests for resource usage and configuration."""
    
    def __init__(self, vector_store: K8sVectorStore):
        self.vector_store = vector_store
    
    def analyze_deployment_resources(self, deployment_name: str) -> Dict[str, Any]:
        """Analyze resource configuration for a specific deployment."""
        # Search for the deployment manifest
        results = self.vector_store.search(
            f"deployment {deployment_name}",
            k=3,
            filter_type="manifest"
        )
        
        analysis = {
            "deployment": deployment_name,
            "found": False,
            "resources": {},
            "recommendations": []
        }
        
        for doc in results:
            if doc.metadata.get("kind") == "Deployment" and deployment_name in doc.page_content:
                analysis["found"] = True
                analysis["resources"] = {
                    "cpu_requests": doc.metadata.get("cpu_requests", "Unknown"),
                    "memory_requests": doc.metadata.get("memory_requests", "Unknown"), 
                    "gpu_requests": doc.metadata.get("gpu_requests", "None"),
                    "replicas": doc.metadata.get("replicas", 0)
                }
                
                # Add basic recommendations
                if doc.metadata.get("replicas", 0) > 5:
                    analysis["recommendations"].append("Consider if high replica count is necessary")
                
                break
        
        return analysis
    
    def get_gpu_usage(self) -> List[Dict[str, Any]]:
        """Get GPU usage across all deployments."""
        # Search for GPU-related manifests
        results = self.vector_store.search(
            "GPU nvidia.com/gpu",
            k=10,
            filter_type="manifest"
        )
        
        gpu_deployments = []
        
        for doc in results:
            if "nvidia.com/gpu" in doc.page_content and doc.metadata.get("kind") == "Deployment":
                gpu_deployments.append({
                    "deployment": doc.metadata.get("name", "Unknown"),
                    "namespace": doc.metadata.get("namespace", "default"),
                    "gpu_requests": doc.metadata.get("gpu_requests", "Unknown"),
                    "replicas": doc.metadata.get("replicas", 0)
                })
        
        return gpu_deployments

class K8sCostAnalyzer:
    """Analyzes Kubernetes costs and provides cost optimization insights."""
    
    def __init__(self, vector_store: K8sVectorStore):
        self.vector_store = vector_store
    
    def get_deployment_costs(self, deployment_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get cost information for deployments."""
        query = f"cost {deployment_name}" if deployment_name else "deployment cost"
        
        results = self.vector_store.search(
            query,
            k=10,
            filter_type="cost_data"
        )
        
        costs = []
        for doc in results:
            if "Deployment:" in doc.page_content:
                # Parse cost information from the document
                lines = doc.page_content.split('\n')
                cost_info = {"deployment": doc.metadata.get("deployment", "Unknown")}
                
                for line in lines:
                    if "Total Cost" in line and "days" in line:
                        cost_info["total_cost"] = line.split('$')[1] if '$' in line else "Unknown"
                    elif "Average Daily Cost" in line:
                        cost_info["daily_cost"] = line.split('$')[1] if '$' in line else "Unknown"
                    elif "CPU Cost" in line:
                        cost_info["cpu_cost"] = line.split('$')[1] if '$' in line else "Unknown"
                    elif "Memory Cost" in line:
                        cost_info["memory_cost"] = line.split('$')[1] if '$' in line else "Unknown"
                
                costs.append(cost_info)
        
        return costs
    
    def get_highest_cost_deployments(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get the highest cost deployments."""
        results = self.vector_store.search(
            "deployment total cost expensive",
            k=15,
            filter_type="cost_data"
        )
        
        # Extract cost data and sort
        deployments_with_costs = []
        
        for doc in results:
            total_cost = doc.metadata.get("total_cost", 0)
            if total_cost > 0:
                deployments_with_costs.append({
                    "deployment": doc.metadata.get("deployment", "Unknown"),
                    "namespace": doc.metadata.get("namespace", "default"),
                    "total_cost": total_cost,
                    "daily_cost": doc.metadata.get("avg_daily_cost", 0)
                })
        
        # Sort by total cost and return top N
        deployments_with_costs.sort(key=lambda x: x["total_cost"], reverse=True)
        return deployments_with_costs[:top_n]

class K8sResourceOptimizer:
    """Provides resource optimization recommendations."""
    
    def __init__(self, vector_store: K8sVectorStore):
        self.vector_store = vector_store
    
    def get_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Get optimization opportunities from the resource analysis."""
        results = self.vector_store.search(
            "optimization opportunity savings",
            k=10,
            filter_type="optimization_opportunity"
        )
        
        opportunities = []
        
        for doc in results:
            opportunity = {
                "type": doc.metadata.get("optimization_type", "Unknown"),
                "potential_savings": doc.metadata.get("potential_savings", "Unknown"),
                "description": doc.page_content
            }
            opportunities.append(opportunity)
        
        return opportunities
    
    def analyze_resource_utilization(self) -> Dict[str, Any]:
        """Get cluster resource utilization analysis."""
        results = self.vector_store.search(
            "resource efficiency utilization CPU memory",
            k=5,
            filter_type="resource_efficiency"
        )
        
        utilization = {
            "cpu_utilization": "Unknown",
            "memory_utilization": "Unknown", 
            "storage_utilization": "Unknown",
            "network_utilization": "Unknown",
            "recommendations": []
        }
        
        for doc in results:
            lines = doc.page_content.split('\n')
            for line in lines:
                if "CPU Utilization:" in line:
                    utilization["cpu_utilization"] = line.split(':')[1].strip()
                elif "Memory Utilization:" in line:
                    utilization["memory_utilization"] = line.split(':')[1].strip()
                elif "Storage Utilization:" in line:
                    utilization["storage_utilization"] = line.split(':')[1].strip()
                elif "Network Utilization:" in line:
                    utilization["network_utilization"] = line.split(':')[1].strip()
        
        # Add recommendations based on utilization
        if utilization["cpu_utilization"] != "Unknown":
            cpu_pct = float(utilization["cpu_utilization"].replace('%', ''))
            if cpu_pct < 30:
                utilization["recommendations"].append("CPU utilization is low - consider reducing CPU requests")
            elif cpu_pct > 80:
                utilization["recommendations"].append("CPU utilization is high - consider increasing CPU limits or scaling")
        
        return utilization

class K8sQueryTool:
    """General-purpose tool for querying Kubernetes data."""
    
    def __init__(self, vector_store: K8sVectorStore):
        self.vector_store = vector_store
    
    def search_k8s_data(self, query: str, data_type: Optional[str] = None, k: int = 5) -> List[Dict[str, Any]]:
        """Search Kubernetes data with optional filtering."""
        results = self.vector_store.search(query, k=k, filter_type=data_type)
        
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "type": doc.metadata.get("type", "unknown")
            })
        
        return formatted_results

# Tool functions for LangGraph agents
def create_k8s_tools(vector_store: K8sVectorStore) -> List:
    """Create LangChain tools for Kubernetes operations."""
    
    manifest_analyzer = K8sManifestAnalyzer(vector_store)
    cost_analyzer = K8sCostAnalyzer(vector_store)
    optimizer = K8sResourceOptimizer(vector_store)
    query_tool = K8sQueryTool(vector_store)
    
    @tool
    def analyze_deployment_resources(
        deployment_name: Annotated[str, "Name of the Kubernetes deployment to analyze"]
    ) -> str:
        """Analyze resource configuration and usage for a specific Kubernetes deployment."""
        analysis = manifest_analyzer.analyze_deployment_resources(deployment_name)
        return json.dumps(analysis, indent=2)
    
    @tool
    def get_gpu_usage() -> str:
        """Get information about GPU usage across all Kubernetes deployments."""
        gpu_usage = manifest_analyzer.get_gpu_usage()
        return json.dumps(gpu_usage, indent=2)
    
    @tool
    def get_deployment_costs(
        deployment_name: Annotated[str, "Optional specific deployment name to get costs for"] = None
    ) -> str:
        """Get cost information for Kubernetes deployments."""
        costs = cost_analyzer.get_deployment_costs(deployment_name)
        return json.dumps(costs, indent=2)
    
    @tool
    def get_highest_cost_deployments(
        top_n: Annotated[int, "Number of top expensive deployments to return"] = 5
    ) -> str:
        """Get the most expensive Kubernetes deployments by cost."""
        high_cost = cost_analyzer.get_highest_cost_deployments(top_n)
        return json.dumps(high_cost, indent=2)
    
    @tool
    def get_optimization_opportunities() -> str:
        """Get resource optimization opportunities and cost-saving recommendations."""
        opportunities = optimizer.get_optimization_opportunities()
        return json.dumps(opportunities, indent=2)
    
    @tool
    def analyze_resource_utilization() -> str:
        """Analyze cluster resource utilization and get efficiency recommendations."""
        utilization = optimizer.analyze_resource_utilization()
        return json.dumps(utilization, indent=2)
    
    @tool
    def search_kubernetes_data(
        query: Annotated[str, "Search query for Kubernetes data"],
        data_type: Annotated[str, "Optional filter by data type (manifest, cost_data, kubectl_output, etc.)"] = None,
        k: Annotated[int, "Number of results to return"] = 5
    ) -> str:
        """Search through Kubernetes manifests, cost data, kubectl outputs, and resource information."""
        results = query_tool.search_k8s_data(query, data_type, k)
        return json.dumps(results, indent=2)
    
    @tool
    def generate_yaml_optimization(
        deployment_name: Annotated[str, "Name of the deployment to optimize"],
        optimization_type: Annotated[str, "Type of optimization (resource_limits, replicas, etc.)"]
    ) -> str:
        """Generate optimized YAML configuration for a Kubernetes deployment."""
        # Search for the current deployment
        current_results = query_tool.search_k8s_data(f"deployment {deployment_name}", "manifest", 1)
        
        if not current_results:
            return f"Deployment {deployment_name} not found"
        
        # Get optimization recommendations
        opportunities = optimizer.get_optimization_opportunities()
        
        # Find relevant optimization
        relevant_opt = None
        for opt in opportunities:
            if deployment_name.lower() in opt.get("description", "").lower():
                relevant_opt = opt
                break
        
        if not relevant_opt:
            return f"No specific optimization found for {deployment_name}"
        
        # Generate recommendation
        recommendation = {
            "deployment": deployment_name,
            "optimization_type": optimization_type,
            "current_config": "See search results above",
            "recommended_changes": relevant_opt.get("description", ""),
            "potential_savings": relevant_opt.get("potential_savings", "Unknown"),
            "yaml_snippet": f"""
# Optimized configuration for {deployment_name}
# Based on {optimization_type} optimization
# Potential savings: {relevant_opt.get('potential_savings', 'Unknown')}

# Apply these changes to your deployment manifest:
# 1. Review current resource requests and limits
# 2. Adjust based on actual usage patterns
# 3. Monitor after changes to ensure performance
"""
        }
        
        return json.dumps(recommendation, indent=2)
    
    return [
        analyze_deployment_resources,
        get_gpu_usage,
        get_deployment_costs,
        get_highest_cost_deployments,
        get_optimization_opportunities,
        analyze_resource_utilization,
        search_kubernetes_data,
        generate_yaml_optimization
    ]






