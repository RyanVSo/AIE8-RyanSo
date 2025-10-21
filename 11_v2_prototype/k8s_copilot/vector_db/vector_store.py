"""
Vector store implementation for Kubernetes data using Qdrant.
Based on patterns from the existing aimakerspace vectordatabase.py
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import json
import yaml
import pandas as pd

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

class K8sVectorStore:
    """Vector store specialized for Kubernetes data with semantic search capabilities."""
    
    def __init__(self, embedding_model: Optional[OpenAIEmbeddings] = None, collection_name: str = "k8s_data"):
        """Initialize the K8s vector store."""
        self.embedding_model = embedding_model or OpenAIEmbeddings(model="text-embedding-3-small")
        self.collection_name = collection_name
        
        # Initialize Qdrant client (in-memory for demo)
        self.client = QdrantClient(":memory:")
        
        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        
        # Initialize vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
        )
        
        self.documents = []
    
    def add_k8s_manifests(self, manifests_data: List[Dict[str, Any]]) -> None:
        """Add Kubernetes manifests to the vector store."""
        documents = []
        
        for manifest in manifests_data:
            # Create searchable text representation of the manifest
            manifest_text = self._manifest_to_searchable_text(manifest)
            
            # Create metadata
            metadata = {
                "type": "manifest",
                "kind": manifest.get("kind", "Unknown"),
                "name": manifest.get("metadata", {}).get("name", "Unknown"),
                "namespace": manifest.get("metadata", {}).get("namespace", "default"),
                "source": "kubernetes_manifest"
            }
            
            # Add resource-specific metadata
            if manifest.get("kind") == "Deployment":
                spec = manifest.get("spec", {})
                metadata.update({
                    "replicas": spec.get("replicas", 0),
                    "cpu_requests": self._extract_cpu_requests(manifest),
                    "memory_requests": self._extract_memory_requests(manifest),
                    "gpu_requests": self._extract_gpu_requests(manifest)
                })
            
            doc = Document(page_content=manifest_text, metadata=metadata)
            documents.append(doc)
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        self.documents.extend(documents)
        
        print(f"Added {len(documents)} manifest documents to vector store")
    
    def add_kubectl_outputs(self, kubectl_data: Dict[str, str]) -> None:
        """Add kubectl command outputs to the vector store."""
        documents = []
        
        for command, output in kubectl_data.items():
            # Create metadata
            metadata = {
                "type": "kubectl_output",
                "command": command,
                "source": "kubectl_command"
            }
            
            doc = Document(page_content=f"kubectl {command} output:\n{output}", metadata=metadata)
            documents.append(doc)
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        self.documents.extend(documents)
        
        print(f"Added {len(documents)} kubectl output documents to vector store")
    
    def add_cost_data(self, cost_data: Dict[str, Any]) -> None:
        """Add cost analysis data to the vector store."""
        documents = []
        
        # Process deployment costs
        if "deployments" in cost_data:
            df = pd.DataFrame(cost_data["deployments"])
            
            # Create summaries by deployment
            for deployment in df["deployment"].unique():
                deployment_data = df[df["deployment"] == deployment]
                
                total_cost = deployment_data["total_cost"].sum()
                avg_daily_cost = deployment_data["total_cost"].mean()
                cpu_cost = deployment_data["cpu_cost"].sum()
                memory_cost = deployment_data["memory_cost"].sum()
                
                cost_text = f"""
Deployment: {deployment}
Namespace: {deployment_data['namespace'].iloc[0]}
Total Cost (30 days): ${total_cost:.2f}
Average Daily Cost: ${avg_daily_cost:.2f}
CPU Cost: ${cpu_cost:.2f}
Memory Cost: ${memory_cost:.2f}
Storage Cost: ${deployment_data['storage_cost'].sum():.2f}
Network Cost: ${deployment_data['network_cost'].sum():.2f}
Total CPU Hours: {deployment_data['cpu_hours'].sum():.1f}
Total Memory GB Hours: {deployment_data['memory_gb_hours'].sum():.1f}
"""
                
                metadata = {
                    "type": "cost_data",
                    "deployment": deployment,
                    "namespace": deployment_data['namespace'].iloc[0],
                    "total_cost": total_cost,
                    "avg_daily_cost": avg_daily_cost,
                    "source": "cost_analysis"
                }
                
                doc = Document(page_content=cost_text.strip(), metadata=metadata)
                documents.append(doc)
        
        # Process node costs
        if "nodes" in cost_data:
            df_nodes = pd.DataFrame(cost_data["nodes"])
            
            for node in df_nodes["node"].unique():
                node_data = df_nodes[df_nodes["node"] == node]
                
                total_cost = node_data["total_cost"].sum()
                node_type = node_data["node_type"].iloc[0]
                hourly_rate = node_data["hourly_rate"].iloc[0]
                
                cost_text = f"""
Node: {node}
Node Type: {node_type}
Total Cost (30 days): ${total_cost:.2f}
Hourly Rate: ${hourly_rate:.2f}
Total Hours: {node_data['hours_used'].sum()}
"""
                
                metadata = {
                    "type": "node_cost",
                    "node": node,
                    "node_type": node_type,
                    "total_cost": total_cost,
                    "hourly_rate": hourly_rate,
                    "source": "cost_analysis"
                }
                
                doc = Document(page_content=cost_text.strip(), metadata=metadata)
                documents.append(doc)
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        self.documents.extend(documents)
        
        print(f"Added {len(documents)} cost data documents to vector store")
    
    def add_resource_summary(self, resource_data: Dict[str, Any]) -> None:
        """Add resource utilization and optimization data to the vector store."""
        documents = []
        
        # Cluster overview
        if "cluster_overview" in resource_data:
            overview = resource_data["cluster_overview"]
            overview_text = f"""
Cluster Overview:
Total Nodes: {overview.get('total_nodes', 0)}
Total Pods: {overview.get('total_pods', 0)}
Total Deployments: {overview.get('total_deployments', 0)}
Total Services: {overview.get('total_services', 0)}
Total CPU Requests: {overview.get('total_cpu_requests', 'Unknown')}
Total Memory Requests: {overview.get('total_memory_requests', 'Unknown')}
Total GPUs: {overview.get('total_gpus', 0)}
GPU Utilization: {overview.get('gpu_utilization', 'Unknown')}
"""
            
            metadata = {
                "type": "cluster_overview",
                "source": "resource_analysis"
            }
            
            doc = Document(page_content=overview_text.strip(), metadata=metadata)
            documents.append(doc)
        
        # Resource efficiency
        if "resource_efficiency" in resource_data:
            efficiency = resource_data["resource_efficiency"]
            efficiency_text = f"""
Resource Efficiency:
CPU Utilization: {efficiency.get('cpu_utilization', 'Unknown')}
Memory Utilization: {efficiency.get('memory_utilization', 'Unknown')}
Storage Utilization: {efficiency.get('storage_utilization', 'Unknown')}
Network Utilization: {efficiency.get('network_utilization', 'Unknown')}
"""
            
            metadata = {
                "type": "resource_efficiency",
                "source": "resource_analysis"
            }
            
            doc = Document(page_content=efficiency_text.strip(), metadata=metadata)
            documents.append(doc)
        
        # Optimization opportunities
        if "optimization_opportunities" in resource_data:
            for i, opportunity in enumerate(resource_data["optimization_opportunities"]):
                opt_text = f"""
Optimization Opportunity:
Type: {opportunity.get('type', 'Unknown')}
Target: {opportunity.get('deployment', opportunity.get('current_nodes', 'Unknown'))}
Current Configuration: {opportunity.get('current_limits', opportunity.get('current_replicas', opportunity.get('current_nodes', 'Unknown')))}
Recommended Configuration: {opportunity.get('recommended_limits', opportunity.get('recommended_replicas', opportunity.get('recommended_nodes', 'Unknown')))}
Potential Savings: {opportunity.get('potential_savings', 'Unknown')}
"""
                
                metadata = {
                    "type": "optimization_opportunity",
                    "optimization_type": opportunity.get('type', 'Unknown'),
                    "potential_savings": opportunity.get('potential_savings', 'Unknown'),
                    "source": "resource_analysis"
                }
                
                doc = Document(page_content=opt_text.strip(), metadata=metadata)
                documents.append(doc)
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        self.documents.extend(documents)
        
        print(f"Added {len(documents)} resource summary documents to vector store")
    
    def search(self, query: str, k: int = 5, filter_type: Optional[str] = None) -> List[Document]:
        """Search the vector store for relevant documents."""
        # Use direct similarity search instead of retriever to avoid filter issues
        if filter_type:
            # For now, we'll do post-filtering since Qdrant filters are complex
            docs = self.vector_store.similarity_search(query, k=k*2)  # Get more docs to filter
            # Filter by document type
            filtered_docs = [doc for doc in docs if doc.metadata.get("type") == filter_type]
            return filtered_docs[:k]
        else:
            return self.vector_store.similarity_search(query, k=k)
    
    def get_retriever(self, k: int = 5, filter_type: Optional[str] = None):
        """Get a retriever instance for use in chains."""
        # Return a simple retriever without filters to avoid Qdrant filter issues
        return self.vector_store.as_retriever(search_kwargs={"k": k})
    
    def _manifest_to_searchable_text(self, manifest: Dict[str, Any]) -> str:
        """Convert a Kubernetes manifest to searchable text."""
        kind = manifest.get("kind", "Unknown")
        metadata = manifest.get("metadata", {})
        name = metadata.get("name", "Unknown")
        namespace = metadata.get("namespace", "default")
        
        text_parts = [
            f"Kubernetes {kind}: {name}",
            f"Namespace: {namespace}",
        ]
        
        # Add labels
        labels = metadata.get("labels", {})
        if labels:
            label_text = ", ".join([f"{k}={v}" for k, v in labels.items()])
            text_parts.append(f"Labels: {label_text}")
        
        # Add spec details based on kind
        spec = manifest.get("spec", {})
        
        if kind == "Deployment":
            text_parts.append(f"Replicas: {spec.get('replicas', 0)}")
            
            # Extract container information
            containers = spec.get("template", {}).get("spec", {}).get("containers", [])
            for container in containers:
                text_parts.append(f"Container: {container.get('name', 'Unknown')}")
                text_parts.append(f"Image: {container.get('image', 'Unknown')}")
                
                # Resource information
                resources = container.get("resources", {})
                if "requests" in resources:
                    requests = resources["requests"]
                    text_parts.append(f"CPU Request: {requests.get('cpu', 'None')}")
                    text_parts.append(f"Memory Request: {requests.get('memory', 'None')}")
                    if "nvidia.com/gpu" in requests:
                        text_parts.append(f"GPU Request: {requests['nvidia.com/gpu']}")
                
                if "limits" in resources:
                    limits = resources["limits"]
                    text_parts.append(f"CPU Limit: {limits.get('cpu', 'None')}")
                    text_parts.append(f"Memory Limit: {limits.get('memory', 'None')}")
                    if "nvidia.com/gpu" in limits:
                        text_parts.append(f"GPU Limit: {limits['nvidia.com/gpu']}")
        
        elif kind == "Service":
            service_type = spec.get("type", "ClusterIP")
            text_parts.append(f"Service Type: {service_type}")
            
            ports = spec.get("ports", [])
            for port in ports:
                text_parts.append(f"Port: {port.get('port', 'Unknown')} -> {port.get('targetPort', 'Unknown')}")
        
        elif kind == "Ingress":
            rules = spec.get("rules", [])
            for rule in rules:
                host = rule.get("host", "Unknown")
                text_parts.append(f"Host: {host}")
        
        # Add the raw YAML as well for comprehensive search
        text_parts.append("Raw manifest:")
        text_parts.append(yaml.dump(manifest, default_flow_style=False))
        
        return "\n".join(text_parts)
    
    def _extract_cpu_requests(self, manifest: Dict[str, Any]) -> str:
        """Extract CPU requests from a deployment manifest."""
        containers = manifest.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
        cpu_requests = []
        
        for container in containers:
            cpu = container.get("resources", {}).get("requests", {}).get("cpu", "0")
            cpu_requests.append(cpu)
        
        return ", ".join(cpu_requests) if cpu_requests else "None"
    
    def _extract_memory_requests(self, manifest: Dict[str, Any]) -> str:
        """Extract memory requests from a deployment manifest."""
        containers = manifest.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
        memory_requests = []
        
        for container in containers:
            memory = container.get("resources", {}).get("requests", {}).get("memory", "0")
            memory_requests.append(memory)
        
        return ", ".join(memory_requests) if memory_requests else "None"
    
    def _extract_gpu_requests(self, manifest: Dict[str, Any]) -> str:
        """Extract GPU requests from a deployment manifest."""
        containers = manifest.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
        gpu_requests = []
        
        for container in containers:
            gpu = container.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", "0")
            if gpu != "0":
                gpu_requests.append(gpu)
        
        return ", ".join(gpu_requests) if gpu_requests else "None"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        type_counts = {}
        for doc in self.documents:
            doc_type = doc.metadata.get("type", "unknown")
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        return {
            "total_documents": len(self.documents),
            "document_types": type_counts,
            "collection_name": self.collection_name
        }
