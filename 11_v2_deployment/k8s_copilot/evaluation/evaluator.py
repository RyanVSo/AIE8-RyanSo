"""
RAGAS evaluation framework for the Kubernetes copilot system.
Based on patterns from the existing evaluation notebooks.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path

from ragas import EvaluationDataset, evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    LLMContextRecall, 
    Faithfulness, 
    FactualCorrectness, 
    ResponseRelevancy,
    ContextEntityRecall
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ..agents.k8s_agent import K8sCopilotAgent, K8sRAGAgent
from ..vector_db.vector_store import K8sVectorStore

class K8sEvaluator:
    """Evaluator for the Kubernetes copilot system using RAGAS."""
    
    def __init__(self, vector_store: K8sVectorStore):
        """Initialize the evaluator."""
        self.vector_store = vector_store
        
        # Initialize evaluation models
        self.evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
        self.evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
        
        # Initialize agents for evaluation
        self.copilot_agent = K8sCopilotAgent(vector_store)
        self.rag_agent = K8sRAGAgent(vector_store)
    
    def create_test_dataset(self) -> List[Dict[str, Any]]:
        """Create a test dataset for evaluating the K8s copilot system."""
        
        test_cases = [
            {
                "user_input": "What are the costs of my Kubernetes deployments?",
                "reference": "The system should provide specific cost information for each deployment including total costs, daily averages, and cost breakdowns by resource type (CPU, memory, storage, network).",
                "reference_contexts": [
                    "Cost data for deployments including ml-training ($125/day), api-server ($45.80/day), nginx-deployment ($15.50/day)",
                    "Deployment costs broken down by CPU, memory, storage, and network usage"
                ]
            },
            {
                "user_input": "How many GPUs does the ml-training deployment use?",
                "reference": "The ml-training deployment uses 1 GPU per replica, with 2 replicas, for a total of 2 GPUs requested.",
                "reference_contexts": [
                    "ml-training deployment manifest showing nvidia.com/gpu: 1 in resource requests",
                    "Deployment has 2 replicas configured"
                ]
            },
            {
                "user_input": "How can I improve resource utilization?",
                "reference": "The system should provide specific optimization opportunities such as right-sizing resources, adjusting replica counts, and node consolidation with potential cost savings.",
                "reference_contexts": [
                    "Resource utilization showing CPU at 65%, memory at 72%",
                    "Optimization opportunities for nginx-deployment, frontend-react, and node consolidation",
                    "Potential savings of $5.20/month for nginx, $7.50/month for frontend, $108/month for nodes"
                ]
            },
            {
                "user_input": "Which deployments are using the most memory?",
                "reference": "Based on resource requests and limits, the ml-training deployment uses the most memory with 4Gi requests and 8Gi limits per container.",
                "reference_contexts": [
                    "ml-training deployment: memory request 4Gi, limit 8Gi",
                    "database-postgresql: memory request 1Gi, limit 2Gi",
                    "api-server: memory request 512Mi, limit 1Gi"
                ]
            },
            {
                "user_input": "Show me the total cluster cost for the last 30 days",
                "reference": "The total cluster cost should include all deployment costs and node costs, providing a comprehensive view of spending.",
                "reference_contexts": [
                    "Total deployment costs across all services",
                    "Node costs for k8s-node-1, k8s-node-2, k8s-node-3, k8s-gpu-node-1",
                    "Cost breakdown by resource type and time period"
                ]
            },
            {
                "user_input": "Analyze the nginx-deployment resources and suggest optimizations",
                "reference": "The nginx-deployment uses 3 replicas with 100m CPU request, 128Mi memory request. Optimization suggests reducing to 150m CPU limit and 200Mi memory limit for $5.20/month savings.",
                "reference_contexts": [
                    "nginx-deployment manifest with current resource configuration",
                    "Optimization opportunity for resource right-sizing",
                    "Potential savings of $5.20/month"
                ]
            },
            {
                "user_input": "What's the GPU utilization across my cluster?",
                "reference": "The cluster has 2 total GPUs with 50% utilization. The ml-training deployment is using GPUs for machine learning workloads.",
                "reference_contexts": [
                    "Cluster overview showing 2 total GPUs",
                    "GPU utilization at 50%",
                    "ml-training deployment using nvidia.com/gpu resources"
                ]
            },
            {
                "user_input": "Generate YAML optimization for the api-server deployment",
                "reference": "The system should provide specific YAML configuration changes for the api-server deployment based on current usage patterns and optimization opportunities.",
                "reference_contexts": [
                    "api-server deployment current configuration",
                    "Resource usage patterns and optimization recommendations",
                    "YAML snippet with recommended changes"
                ]
            }
        ]
        
        return test_cases
    
    def run_evaluation(self, agent_type: str = "copilot") -> Dict[str, Any]:
        """Run evaluation on the specified agent."""
        
        print(f"Running evaluation on {agent_type} agent...")
        
        # Create test dataset
        test_cases = self.create_test_dataset()
        
        # Run queries through the agent and collect responses
        evaluation_data = []
        
        for test_case in test_cases:
            query = test_case["user_input"]
            
            print(f"Evaluating query: {query}")
            
            try:
                if agent_type == "copilot":
                    response = self.copilot_agent.invoke(query)
                    # Get retrieved contexts by searching vector store
                    retrieved_docs = self.vector_store.search(query, k=3)
                    retrieved_contexts = [doc.page_content for doc in retrieved_docs]
                else:
                    response = self.rag_agent.invoke(query)
                    # Get retrieved contexts by searching vector store
                    retrieved_docs = self.vector_store.search(query, k=3)
                    retrieved_contexts = [doc.page_content for doc in retrieved_docs]
                
                evaluation_data.append({
                    "user_input": query,
                    "response": response,
                    "reference": test_case["reference"],
                    "reference_contexts": test_case["reference_contexts"],
                    "retrieved_contexts": retrieved_contexts
                })
                
            except Exception as e:
                print(f"Error evaluating query '{query}': {e}")
                continue
        
        # Convert to DataFrame and then to EvaluationDataset
        df = pd.DataFrame(evaluation_data)
        evaluation_dataset = EvaluationDataset.from_pandas(df)
        
        # Define metrics to evaluate
        metrics = [
            Faithfulness(),
            FactualCorrectness(),
            ResponseRelevancy(),
            LLMContextRecall(),
            ContextEntityRecall()
        ]
        
        # Run evaluation
        try:
            results = evaluate(
                dataset=evaluation_dataset,
                metrics=metrics,
                llm=self.evaluator_llm,
                run_config=RunConfig(timeout=360)
            )
            
            return {
                "agent_type": agent_type,
                "results": results,
                "test_cases_count": len(evaluation_data),
                "metrics_evaluated": [metric.__class__.__name__ for metric in metrics]
            }
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return {
                "agent_type": agent_type,
                "error": str(e),
                "test_cases_count": len(evaluation_data)
            }
    
    def compare_agents(self) -> Dict[str, Any]:
        """Compare copilot agent vs RAG agent performance."""
        
        print("Running comparative evaluation...")
        
        copilot_results = self.run_evaluation("copilot")
        rag_results = self.run_evaluation("rag")
        
        comparison = {
            "copilot_agent": copilot_results,
            "rag_agent": rag_results,
            "comparison_summary": {}
        }
        
        # Add comparison summary if both evaluations succeeded
        if "results" in copilot_results and "results" in rag_results:
            copilot_scores = copilot_results["results"]
            rag_scores = rag_results["results"]
            
            comparison["comparison_summary"] = {
                "better_agent": "Analysis would go here based on metric scores",
                "notes": "Copilot agent has access to specialized tools while RAG agent uses simple retrieval"
            }
        
        return comparison
    
    def create_custom_evaluation_dataset(self, queries: List[str], expected_responses: List[str]) -> EvaluationDataset:
        """Create a custom evaluation dataset from provided queries and expected responses."""
        
        evaluation_data = []
        
        for query, expected in zip(queries, expected_responses):
            # Run through both agents
            copilot_response = self.copilot_agent.invoke(query)
            rag_response = self.rag_agent.invoke(query)
            
            # Get retrieved contexts
            retrieved_docs = self.vector_store.search(query, k=3)
            retrieved_contexts = [doc.page_content for doc in retrieved_docs]
            
            evaluation_data.append({
                "user_input": query,
                "response": copilot_response,  # Use copilot by default
                "reference": expected,
                "retrieved_contexts": retrieved_contexts
            })
        
        df = pd.DataFrame(evaluation_data)
        return EvaluationDataset.from_pandas(df)
    
    def save_evaluation_results(self, results: Dict[str, Any], output_path: Path):
        """Save evaluation results to a file."""
        import json
        
        # Convert results to JSON-serializable format
        serializable_results = {}
        
        for key, value in results.items():
            if hasattr(value, 'to_dict'):
                serializable_results[key] = value.to_dict()
            elif hasattr(value, '__dict__'):
                serializable_results[key] = str(value)
            else:
                serializable_results[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Evaluation results saved to {output_path}")

def create_evaluation_dataset(vector_store: K8sVectorStore, queries: List[str]) -> EvaluationDataset:
    """Create an evaluation dataset for given queries."""
    evaluator = K8sEvaluator(vector_store)
    
    # Generate expected responses based on the queries
    expected_responses = []
    for query in queries:
        if "cost" in query.lower():
            expected_responses.append("Should provide specific cost information with dollar amounts and breakdowns.")
        elif "gpu" in query.lower():
            expected_responses.append("Should provide GPU usage information including counts and utilization.")
        elif "optimize" in query.lower() or "improve" in query.lower():
            expected_responses.append("Should provide specific optimization recommendations with potential savings.")
        else:
            expected_responses.append("Should provide accurate information based on the Kubernetes data.")
    
    return evaluator.create_custom_evaluation_dataset(queries, expected_responses)






