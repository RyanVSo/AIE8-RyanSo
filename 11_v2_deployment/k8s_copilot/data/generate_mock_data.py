"""
Generate mock Kubernetes data including manifests, cost data, and kubectl outputs.
This creates realistic K8s data for testing and demonstration purposes.
"""

import json
import yaml
import random
import datetime
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# Create data directory
data_dir = Path(__file__).parent
data_dir.mkdir(exist_ok=True)

def generate_deployment_manifest(name: str, namespace: str = "default", replicas: int = 3, 
                                cpu_request: str = "100m", memory_request: str = "128Mi",
                                cpu_limit: str = "200m", memory_limit: str = "256Mi",
                                gpu_count: int = 0) -> Dict[str, Any]:
    """Generate a realistic Kubernetes deployment manifest."""
    
    containers = [{
        "name": name,
        "image": f"{name}:latest",
        "ports": [{"containerPort": 8080}],
        "resources": {
            "requests": {
                "cpu": cpu_request,
                "memory": memory_request
            },
            "limits": {
                "cpu": cpu_limit,
                "memory": memory_limit
            }
        },
        "env": [
            {"name": "APP_ENV", "value": "production"},
            {"name": "LOG_LEVEL", "value": "info"}
        ]
    }]
    
    if gpu_count > 0:
        containers[0]["resources"]["limits"]["nvidia.com/gpu"] = str(gpu_count)
        containers[0]["resources"]["requests"]["nvidia.com/gpu"] = str(gpu_count)
    
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {
                "app": name,
                "version": "v1.0.0",
                "environment": "production"
            }
        },
        "spec": {
            "replicas": replicas,
            "selector": {
                "matchLabels": {
                    "app": name
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": name,
                        "version": "v1.0.0"
                    }
                },
                "spec": {
                    "containers": containers
                }
            }
        }
    }

def generate_service_manifest(name: str, namespace: str = "default", port: int = 80, target_port: int = 8080) -> Dict[str, Any]:
    """Generate a Kubernetes service manifest."""
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": f"{name}-service",
            "namespace": namespace,
            "labels": {
                "app": name
            }
        },
        "spec": {
            "selector": {
                "app": name
            },
            "ports": [{
                "protocol": "TCP",
                "port": port,
                "targetPort": target_port
            }],
            "type": "ClusterIP"
        }
    }

def generate_configmap_manifest(name: str, namespace: str = "default") -> Dict[str, Any]:
    """Generate a Kubernetes ConfigMap manifest."""
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": f"{name}-config",
            "namespace": namespace
        },
        "data": {
            "database_url": "postgresql://db:5432/app",
            "redis_url": "redis://redis:6379",
            "app_config.yaml": """
app:
  name: """ + name + """
  debug: false
  workers: 4
logging:
  level: info
  format: json
"""
        }
    }

def generate_ingress_manifest(name: str, namespace: str = "default", host: str = None) -> Dict[str, Any]:
    """Generate a Kubernetes Ingress manifest."""
    if not host:
        host = f"{name}.example.com"
    
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "Ingress",
        "metadata": {
            "name": f"{name}-ingress",
            "namespace": namespace,
            "annotations": {
                "kubernetes.io/ingress.class": "nginx",
                "cert-manager.io/cluster-issuer": "letsencrypt-prod"
            }
        },
        "spec": {
            "tls": [{
                "hosts": [host],
                "secretName": f"{name}-tls"
            }],
            "rules": [{
                "host": host,
                "http": {
                    "paths": [{
                        "path": "/",
                        "pathType": "Prefix",
                        "backend": {
                            "service": {
                                "name": f"{name}-service",
                                "port": {
                                    "number": 80
                                }
                            }
                        }
                    }]
                }
            }]
        }
    }

def generate_kubectl_outputs():
    """Generate mock kubectl command outputs."""
    
    # kubectl get pods output
    pods_output = """NAME                                READY   STATUS    RESTARTS   AGE
nginx-deployment-7d5c7d8c9f-2xk8p   1/1     Running   0          2d
nginx-deployment-7d5c7d8c9f-7h9m2   1/1     Running   0          2d
nginx-deployment-7d5c7d8c9f-k5n4x   1/1     Running   0          2d
api-server-6b8d7c9f5e-3j2k1         1/1     Running   1          5d
api-server-6b8d7c9f5e-8m7n4         1/1     Running   0          5d
ml-training-5f9e8d7c6b-9p2q1        1/1     Running   0          1d
ml-training-5f9e8d7c6b-4k7j8        0/1     Pending   0          1d
database-postgresql-0               1/1     Running   0          7d
redis-master-0                      1/1     Running   0          7d
frontend-react-7c8d9e5f4g-1h3j5     1/1     Running   0          3d"""

    # kubectl top pods output
    top_pods_output = """NAME                                CPU(cores)   MEMORY(bytes)
nginx-deployment-7d5c7d8c9f-2xk8p   15m          64Mi
nginx-deployment-7d5c7d8c9f-7h9m2   12m          58Mi
nginx-deployment-7d5c7d8c9f-k5n4x   18m          72Mi
api-server-6b8d7c9f5e-3j2k1         45m          256Mi
api-server-6b8d7c9f5e-8m7n4         38m          234Mi
ml-training-5f9e8d7c6b-9p2q1        1200m        8Gi
database-postgresql-0               85m          512Mi
redis-master-0                      25m          128Mi
frontend-react-7c8d9e5f4g-1h3j5     22m          156Mi"""

    # kubectl top nodes output
    top_nodes_output = """NAME           CPU(cores)   CPU%   MEMORY(bytes)   MEMORY%
k8s-node-1     1250m        31%    4Gi             65%
k8s-node-2     890m         22%    3.2Gi           52%
k8s-node-3     2100m        52%    7.8Gi           80%
k8s-gpu-node-1 3200m        80%    15.2Gi          95%"""

    # kubectl describe node output (sample)
    describe_node_output = """Name:               k8s-gpu-node-1
Roles:              <none>
Labels:             beta.kubernetes.io/arch=amd64
                    beta.kubernetes.io/os=linux
                    kubernetes.io/arch=amd64
                    kubernetes.io/hostname=k8s-gpu-node-1
                    kubernetes.io/os=linux
                    node-type=gpu
Annotations:        kubeadm.alpha.kubernetes.io/cri-socket: /var/run/containerd/containerd.sock
                    node.alpha.kubernetes.io/ttl: 0
                    projectcalico.org/IPv4Address: 10.244.3.1/24
                    projectcalico.org/IPv4IPIPTunnelAddr: 192.168.219.64
CreationTimestamp:  Mon, 01 Jan 2024 10:00:00 +0000
Taints:             <none>
Unschedulable:      false
Lease:
  HolderIdentity:  k8s-gpu-node-1
  AcquireTime:     <unset>
  RenewTime:       Wed, 15 Jan 2025 14:30:00 +0000
Conditions:
  Type             Status  LastHeartbeatTime                 LastTransitionTime                Reason                       Message
  ----             ------  -----------------                 ------------------                ------                       -------
  MemoryPressure   False   Wed, 15 Jan 2025 14:29:45 +0000   Mon, 01 Jan 2024 10:00:00 +0000   KubeletHasSufficientMemory   kubelet has sufficient memory available
  DiskPressure     False   Wed, 15 Jan 2025 14:29:45 +0000   Mon, 01 Jan 2024 10:00:00 +0000   KubeletHasNoDiskPressure     kubelet has no disk pressure
  PIDPressure      False   Wed, 15 Jan 2025 14:29:45 +0000   Mon, 01 Jan 2024 10:00:00 +0000   KubeletHasSufficientPID      kubelet has sufficient PID available
  Ready            True    Wed, 15 Jan 2025 14:29:45 +0000   Mon, 01 Jan 2024 10:00:00 +0000   KubeletReady                 kubelet is posting ready status
Addresses:
  InternalIP:  10.244.3.1
  Hostname:    k8s-gpu-node-1
Capacity:
  cpu:                4
  ephemeral-storage:  100Gi
  hugepages-1Gi:      0
  hugepages-2Mi:      0
  memory:             16Gi
  nvidia.com/gpu:     2
  pods:               110
Allocatable:
  cpu:                3800m
  ephemeral-storage:  92Gi
  hugepages-1Gi:      0
  hugepages-2Mi:      0
  memory:             15Gi
  nvidia.com/gpu:     2
  pods:               110"""

    return {
        "get_pods": pods_output,
        "top_pods": top_pods_output,
        "top_nodes": top_nodes_output,
        "describe_node_gpu": describe_node_output
    }

def generate_cost_data():
    """Generate mock cost data for Kubernetes resources."""
    
    # Generate cost data for the last 30 days
    dates = []
    base_date = datetime.datetime.now() - datetime.timedelta(days=30)
    for i in range(30):
        dates.append((base_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d"))
    
    # Cost data for different deployments
    deployments_cost_data = []
    
    deployments = [
        {"name": "nginx-deployment", "namespace": "default", "base_cost": 15.50},
        {"name": "api-server", "namespace": "default", "base_cost": 45.80},
        {"name": "ml-training", "namespace": "ml", "base_cost": 125.00},
        {"name": "database-postgresql", "namespace": "default", "base_cost": 35.20},
        {"name": "redis-master", "namespace": "default", "base_cost": 12.75},
        {"name": "frontend-react", "namespace": "default", "base_cost": 22.40}
    ]
    
    for deployment in deployments:
        for date in dates:
            # Add some variance to the cost
            variance = random.uniform(0.8, 1.2)
            daily_cost = deployment["base_cost"] * variance
            
            deployments_cost_data.append({
                "date": date,
                "deployment": deployment["name"],
                "namespace": deployment["namespace"],
                "cpu_cost": daily_cost * 0.4,
                "memory_cost": daily_cost * 0.3,
                "storage_cost": daily_cost * 0.2,
                "network_cost": daily_cost * 0.1,
                "total_cost": daily_cost,
                "cpu_hours": random.uniform(20, 80),
                "memory_gb_hours": random.uniform(50, 200),
                "storage_gb": random.uniform(10, 100)
            })
    
    # Node cost data
    nodes_cost_data = []
    nodes = [
        {"name": "k8s-node-1", "type": "standard", "hourly_rate": 0.15},
        {"name": "k8s-node-2", "type": "standard", "hourly_rate": 0.15},
        {"name": "k8s-node-3", "type": "memory-optimized", "hourly_rate": 0.25},
        {"name": "k8s-gpu-node-1", "type": "gpu", "hourly_rate": 1.20}
    ]
    
    for node in nodes:
        for date in dates:
            daily_cost = node["hourly_rate"] * 24 * random.uniform(0.9, 1.1)
            nodes_cost_data.append({
                "date": date,
                "node": node["name"],
                "node_type": node["type"],
                "hourly_rate": node["hourly_rate"],
                "hours_used": 24,
                "total_cost": daily_cost
            })
    
    return {
        "deployments": deployments_cost_data,
        "nodes": nodes_cost_data
    }

def main():
    """Generate all mock data."""
    
    print("Generating mock Kubernetes data...")
    
    # Generate deployment manifests
    deployments = [
        {"name": "nginx-deployment", "replicas": 3, "cpu_request": "100m", "memory_request": "128Mi"},
        {"name": "api-server", "replicas": 2, "cpu_request": "200m", "memory_request": "512Mi", "cpu_limit": "500m", "memory_limit": "1Gi"},
        {"name": "ml-training", "namespace": "ml", "replicas": 2, "cpu_request": "1000m", "memory_request": "4Gi", "cpu_limit": "2000m", "memory_limit": "8Gi", "gpu_count": 1},
        {"name": "database-postgresql", "replicas": 1, "cpu_request": "250m", "memory_request": "1Gi", "cpu_limit": "500m", "memory_limit": "2Gi"},
        {"name": "redis-master", "replicas": 1, "cpu_request": "100m", "memory_request": "256Mi"},
        {"name": "frontend-react", "replicas": 3, "cpu_request": "50m", "memory_request": "64Mi", "cpu_limit": "200m", "memory_limit": "256Mi"}
    ]
    
    # Create manifests directory
    manifests_dir = data_dir / "manifests"
    manifests_dir.mkdir(exist_ok=True)
    
    all_manifests = []
    
    for deployment in deployments:
        # Generate deployment manifest
        deploy_manifest = generate_deployment_manifest(**deployment)
        all_manifests.append(deploy_manifest)
        
        # Generate service manifest
        service_manifest = generate_service_manifest(deployment["name"], deployment.get("namespace", "default"))
        all_manifests.append(service_manifest)
        
        # Generate configmap manifest
        configmap_manifest = generate_configmap_manifest(deployment["name"], deployment.get("namespace", "default"))
        all_manifests.append(configmap_manifest)
        
        # Generate ingress for web services
        if deployment["name"] in ["nginx-deployment", "api-server", "frontend-react"]:
            ingress_manifest = generate_ingress_manifest(deployment["name"], deployment.get("namespace", "default"))
            all_manifests.append(ingress_manifest)
    
    # Save individual manifest files
    for i, manifest in enumerate(all_manifests):
        filename = f"{manifest['metadata']['name']}-{manifest['kind'].lower()}.yaml"
        with open(manifests_dir / filename, 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False)
    
    # Save combined manifests file
    with open(data_dir / "all_manifests.yaml", 'w') as f:
        yaml.dump_all(all_manifests, f, default_flow_style=False)
    
    print(f"Generated {len(all_manifests)} manifests")
    
    # Generate kubectl outputs
    kubectl_outputs = generate_kubectl_outputs()
    with open(data_dir / "kubectl_outputs.json", 'w') as f:
        json.dump(kubectl_outputs, f, indent=2)
    
    print("Generated kubectl outputs")
    
    # Generate cost data
    cost_data = generate_cost_data()
    
    # Save as CSV files
    pd.DataFrame(cost_data["deployments"]).to_csv(data_dir / "deployment_costs.csv", index=False)
    pd.DataFrame(cost_data["nodes"]).to_csv(data_dir / "node_costs.csv", index=False)
    
    # Save as JSON
    with open(data_dir / "cost_data.json", 'w') as f:
        json.dump(cost_data, f, indent=2)
    
    print("Generated cost data")
    
    # Generate resource utilization summary
    resource_summary = {
        "cluster_overview": {
            "total_nodes": 4,
            "total_pods": 15,
            "total_deployments": 6,
            "total_services": 6,
            "total_cpu_requests": "2.84 cores",
            "total_memory_requests": "15.2 GB",
            "total_gpus": 2,
            "gpu_utilization": "50%"
        },
        "resource_efficiency": {
            "cpu_utilization": "65%",
            "memory_utilization": "72%",
            "storage_utilization": "45%",
            "network_utilization": "30%"
        },
        "optimization_opportunities": [
            {
                "type": "resource_right_sizing",
                "deployment": "nginx-deployment",
                "current_limits": {"cpu": "200m", "memory": "256Mi"},
                "recommended_limits": {"cpu": "150m", "memory": "200Mi"},
                "potential_savings": "$5.20/month"
            },
            {
                "type": "replica_optimization",
                "deployment": "frontend-react",
                "current_replicas": 3,
                "recommended_replicas": 2,
                "potential_savings": "$7.50/month"
            },
            {
                "type": "node_consolidation",
                "current_nodes": 4,
                "recommended_nodes": 3,
                "potential_savings": "$108/month"
            }
        ]
    }
    
    with open(data_dir / "resource_summary.json", 'w') as f:
        json.dump(resource_summary, f, indent=2)
    
    print("Generated resource summary")
    
    print(f"\nMock data generated successfully in {data_dir}")
    print("Files created:")
    for file in data_dir.rglob("*"):
        if file.is_file():
            print(f"  - {file.relative_to(data_dir)}")

if __name__ == "__main__":
    main()






