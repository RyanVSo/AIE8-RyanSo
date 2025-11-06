"""
Kubernetes Copilot - Agentic RAG system for Kubernetes management.

This package provides an intelligent assistant for managing Kubernetes clusters
through natural language interactions, cost analysis, and optimization recommendations.
"""

__version__ = "0.1.0"
__author__ = "AI Engineering Course"

from .vector_db import K8sVectorStore, K8sDataLoader
from .agents import K8sCopilotAgent, K8sRAGAgent
from .tools import create_k8s_tools
from .evaluation import K8sEvaluator
from .retrieval import K8sRetrieverFactory, RetrieverType, RetrievalPerformanceEvaluator

__all__ = [
    "K8sVectorStore",
    "K8sDataLoader", 
    "K8sCopilotAgent",
    "K8sRAGAgent",
    "create_k8s_tools",
    "K8sEvaluator",
    "K8sRetrieverFactory",
    "RetrieverType",
    "RetrievalPerformanceEvaluator"
]






