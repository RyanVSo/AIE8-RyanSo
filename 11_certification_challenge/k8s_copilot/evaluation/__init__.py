"""RAGAS evaluation framework for the Kubernetes copilot system."""

from .evaluator import K8sEvaluator, create_evaluation_dataset

__all__ = ["K8sEvaluator", "create_evaluation_dataset"]






