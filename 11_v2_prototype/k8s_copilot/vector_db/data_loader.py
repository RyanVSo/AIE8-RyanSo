"""
Data loader for Kubernetes data from various sources.
Loads manifests, kubectl outputs, cost data, etc. into the vector store.
"""

import json
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional

from .vector_store import K8sVectorStore

class K8sDataLoader:
    """Loads Kubernetes data from files into the vector store."""
    
    def __init__(self, data_dir: Path):
        """Initialize the data loader with a data directory."""
        self.data_dir = Path(data_dir)
        
    def load_all_data(self, vector_store: K8sVectorStore) -> None:
        """Load all available data into the vector store."""
        print("Loading Kubernetes data into vector store...")
        
        # Load manifests
        self.load_manifests(vector_store)
        
        # Load kubectl outputs
        self.load_kubectl_outputs(vector_store)
        
        # Load cost data
        self.load_cost_data(vector_store)
        
        # Load resource summary
        self.load_resource_summary(vector_store)
        
        print("Data loading complete!")
        print(f"Vector store stats: {vector_store.get_stats()}")
    
    def load_manifests(self, vector_store: K8sVectorStore) -> None:
        """Load Kubernetes manifests from YAML files."""
        manifests_file = self.data_dir / "all_manifests.yaml"
        
        if not manifests_file.exists():
            print(f"Manifests file not found: {manifests_file}")
            return
        
        try:
            with open(manifests_file, 'r') as f:
                manifests = list(yaml.safe_load_all(f))
            
            # Filter out None values
            manifests = [m for m in manifests if m is not None]
            
            vector_store.add_k8s_manifests(manifests)
            print(f"Loaded {len(manifests)} manifests")
            
        except Exception as e:
            print(f"Error loading manifests: {e}")
    
    def load_kubectl_outputs(self, vector_store: K8sVectorStore) -> None:
        """Load kubectl command outputs."""
        kubectl_file = self.data_dir / "kubectl_outputs.json"
        
        if not kubectl_file.exists():
            print(f"Kubectl outputs file not found: {kubectl_file}")
            return
        
        try:
            with open(kubectl_file, 'r') as f:
                kubectl_data = json.load(f)
            
            vector_store.add_kubectl_outputs(kubectl_data)
            print(f"Loaded {len(kubectl_data)} kubectl outputs")
            
        except Exception as e:
            print(f"Error loading kubectl outputs: {e}")
    
    def load_cost_data(self, vector_store: K8sVectorStore) -> None:
        """Load cost analysis data."""
        cost_file = self.data_dir / "cost_data.json"
        
        if not cost_file.exists():
            print(f"Cost data file not found: {cost_file}")
            return
        
        try:
            with open(cost_file, 'r') as f:
                cost_data = json.load(f)
            
            vector_store.add_cost_data(cost_data)
            print("Loaded cost data")
            
        except Exception as e:
            print(f"Error loading cost data: {e}")
    
    def load_resource_summary(self, vector_store: K8sVectorStore) -> None:
        """Load resource utilization and optimization data."""
        resource_file = self.data_dir / "resource_summary.json"
        
        if not resource_file.exists():
            print(f"Resource summary file not found: {resource_file}")
            return
        
        try:
            with open(resource_file, 'r') as f:
                resource_data = json.load(f)
            
            vector_store.add_resource_summary(resource_data)
            print("Loaded resource summary")
            
        except Exception as e:
            print(f"Error loading resource summary: {e}")
    
    def load_individual_manifests(self, vector_store: K8sVectorStore) -> None:
        """Load individual manifest files from the manifests directory."""
        manifests_dir = self.data_dir / "manifests"
        
        if not manifests_dir.exists():
            print(f"Manifests directory not found: {manifests_dir}")
            return
        
        manifests = []
        
        for manifest_file in manifests_dir.glob("*.yaml"):
            try:
                with open(manifest_file, 'r') as f:
                    manifest = yaml.safe_load(f)
                    if manifest:
                        manifests.append(manifest)
            except Exception as e:
                print(f"Error loading manifest {manifest_file}: {e}")
        
        if manifests:
            vector_store.add_k8s_manifests(manifests)
            print(f"Loaded {len(manifests)} individual manifests")
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get a summary of available data files."""
        summary = {
            "data_directory": str(self.data_dir),
            "available_files": {},
            "manifests_count": 0
        }
        
        # Check for main data files
        files_to_check = [
            "all_manifests.yaml",
            "kubectl_outputs.json", 
            "cost_data.json",
            "resource_summary.json",
            "deployment_costs.csv",
            "node_costs.csv"
        ]
        
        for file_name in files_to_check:
            file_path = self.data_dir / file_name
            summary["available_files"][file_name] = file_path.exists()
        
        # Count manifests
        manifests_dir = self.data_dir / "manifests"
        if manifests_dir.exists():
            summary["manifests_count"] = len(list(manifests_dir.glob("*.yaml")))
        
        return summary






