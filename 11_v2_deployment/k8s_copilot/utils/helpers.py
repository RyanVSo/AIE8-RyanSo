"""Helper functions for the Kubernetes copilot system."""

import re
from typing import Dict, Any, List, Optional

def format_cost(cost: float) -> str:
    """Format cost values for display."""
    if cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 1:
        return f"${cost:.2f}"
    else:
        return f"${cost:,.2f}"

def parse_k8s_resources(resource_str: str) -> Dict[str, Any]:
    """Parse Kubernetes resource strings (e.g., '100m', '256Mi')."""
    if not resource_str or resource_str == "None":
        return {"value": 0, "unit": "", "normalized": 0}
    
    # CPU resources
    if resource_str.endswith('m'):
        return {
            "value": int(resource_str[:-1]),
            "unit": "millicores",
            "normalized": int(resource_str[:-1])  # millicores
        }
    elif resource_str.isdigit():
        return {
            "value": int(resource_str),
            "unit": "cores", 
            "normalized": int(resource_str) * 1000  # convert to millicores
        }
    
    # Memory resources
    memory_units = {
        'Ki': 1024,
        'Mi': 1024**2,
        'Gi': 1024**3,
        'Ti': 1024**4,
        'K': 1000,
        'M': 1000**2,
        'G': 1000**3,
        'T': 1000**4
    }
    
    for unit, multiplier in memory_units.items():
        if resource_str.endswith(unit):
            value = float(resource_str[:-len(unit)])
            return {
                "value": value,
                "unit": unit,
                "normalized": int(value * multiplier)  # bytes
            }
    
    # Default case
    try:
        return {
            "value": float(resource_str),
            "unit": "",
            "normalized": float(resource_str)
        }
    except ValueError:
        return {"value": 0, "unit": "", "normalized": 0}

def format_response(response: str, max_length: Optional[int] = None) -> str:
    """Format agent response for better readability."""
    if max_length and len(response) > max_length:
        response = response[:max_length] + "..."
    
    # Add some basic formatting
    response = response.replace("\\n", "\n")
    
    # Format JSON-like structures
    if response.strip().startswith('{') and response.strip().endswith('}'):
        try:
            import json
            parsed = json.loads(response)
            response = json.dumps(parsed, indent=2)
        except:
            pass
    
    return response

def extract_metrics_from_text(text: str) -> Dict[str, str]:
    """Extract key metrics from text responses."""
    metrics = {}
    
    # Cost patterns
    cost_pattern = r'\$(\d+(?:\.\d{2})?)'
    costs = re.findall(cost_pattern, text)
    if costs:
        metrics['costs'] = costs
    
    # Percentage patterns
    percentage_pattern = r'(\d+(?:\.\d+)?)%'
    percentages = re.findall(percentage_pattern, text)
    if percentages:
        metrics['percentages'] = percentages
    
    # Resource patterns
    cpu_pattern = r'(\d+(?:\.\d+)?m?)\s*(?:CPU|cpu|cores?)'
    memory_pattern = r'(\d+(?:\.\d+)?(?:Mi|Gi|M|G))\s*(?:memory|Memory|RAM)'
    
    cpu_matches = re.findall(cpu_pattern, text)
    memory_matches = re.findall(memory_pattern, text)
    
    if cpu_matches:
        metrics['cpu'] = cpu_matches
    if memory_matches:
        metrics['memory'] = memory_matches
    
    return metrics

def validate_k8s_manifest(manifest: Dict[str, Any]) -> List[str]:
    """Validate a Kubernetes manifest and return any issues."""
    issues = []
    
    # Check required fields
    if 'apiVersion' not in manifest:
        issues.append("Missing 'apiVersion' field")
    
    if 'kind' not in manifest:
        issues.append("Missing 'kind' field")
    
    if 'metadata' not in manifest:
        issues.append("Missing 'metadata' field")
    else:
        if 'name' not in manifest['metadata']:
            issues.append("Missing 'metadata.name' field")
    
    # Check deployment-specific requirements
    if manifest.get('kind') == 'Deployment':
        spec = manifest.get('spec', {})
        
        if 'selector' not in spec:
            issues.append("Deployment missing 'spec.selector'")
        
        if 'template' not in spec:
            issues.append("Deployment missing 'spec.template'")
        else:
            template_spec = spec['template'].get('spec', {})
            if 'containers' not in template_spec:
                issues.append("Deployment template missing 'containers'")
    
    return issues

def summarize_cluster_resources(manifests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize resource usage across all manifests."""
    summary = {
        "total_deployments": 0,
        "total_services": 0,
        "total_configmaps": 0,
        "total_ingresses": 0,
        "total_cpu_requests": 0,
        "total_memory_requests": 0,
        "total_gpu_requests": 0,
        "total_replicas": 0
    }
    
    for manifest in manifests:
        kind = manifest.get('kind', '')
        
        if kind == 'Deployment':
            summary["total_deployments"] += 1
            
            spec = manifest.get('spec', {})
            replicas = spec.get('replicas', 1)
            summary["total_replicas"] += replicas
            
            # Extract resource requests
            containers = spec.get('template', {}).get('spec', {}).get('containers', [])
            
            for container in containers:
                resources = container.get('resources', {}).get('requests', {})
                
                # CPU
                cpu = resources.get('cpu', '0')
                cpu_parsed = parse_k8s_resources(cpu)
                summary["total_cpu_requests"] += cpu_parsed["normalized"] * replicas
                
                # Memory
                memory = resources.get('memory', '0')
                memory_parsed = parse_k8s_resources(memory)
                summary["total_memory_requests"] += memory_parsed["normalized"] * replicas
                
                # GPU
                gpu = resources.get('nvidia.com/gpu', '0')
                if gpu != '0':
                    summary["total_gpu_requests"] += int(gpu) * replicas
        
        elif kind == 'Service':
            summary["total_services"] += 1
        elif kind == 'ConfigMap':
            summary["total_configmaps"] += 1
        elif kind == 'Ingress':
            summary["total_ingresses"] += 1
    
    return summary






