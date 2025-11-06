"""
RAGAS evaluation framework for the Kubernetes RAG system.
"""

import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    LLMContextRecall, 
    Faithfulness, 
    ResponseRelevancy,
    ContextEntityRecall
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ..agents.base_agent import K8sBaseRAGAgent
from ..vector_db.vector_store import K8sDocVectorStore
from ..retrieval.advanced_retrievers import K8sAdvancedRetrieverFactory, RetrieverType


class K8sRAGEvaluator:
    """Evaluator for the Kubernetes RAG system using RAGAS."""
    
    def __init__(self, vector_store: K8sDocVectorStore):
        """Initialize the evaluator."""
        self.vector_store = vector_store
        
        # Initialize evaluation models
        self.evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
        self.evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
        
        # Initialize agent and retriever factory
        self.agent = K8sBaseRAGAgent(vector_store)
        self.retriever_factory = K8sAdvancedRetrieverFactory(vector_store)
        
        # Define evaluation metrics (matching the course notebook pattern)
        self.metrics = [
            LLMContextRecall(),
            Faithfulness(), 
            ResponseRelevancy(),
            ContextEntityRecall()
        ]
    
    def create_test_dataset(self) -> List[Dict[str, Any]]:
        """Create a test dataset for evaluating the K8s RAG system."""
        
        test_cases = [
            {
                "user_input": "What is a Kubernetes Pod?",
                "reference": "A Pod is the smallest deployable unit in Kubernetes that represents a single instance of a running process in a cluster. It can contain one or more containers that share storage and network resources.",
                "reference_contexts": [
                    "Pods are the smallest deployable units of computing that you can create and manage in Kubernetes",
                    "A Pod represents a single instance of a running process in your cluster"
                ]
            },
            {
                "user_input": "How do Services work in Kubernetes?",
                "reference": "Services provide stable network endpoints for accessing Pods. They use selectors to identify target Pods and provide load balancing, service discovery, and stable IP addresses even as Pods are created and destroyed.",
                "reference_contexts": [
                    "A Service is an abstract way to expose an application running on a set of Pods as a network service",
                    "Services use selectors to determine which Pods to target"
                ]
            },
            {
                "user_input": "What are the main components of the Kubernetes control plane?",
                "reference": "The control plane components include the API server (kube-apiserver), etcd (distributed key-value store), scheduler (kube-scheduler), and controller manager (kube-controller-manager). These components manage the cluster state and make decisions about the cluster.",
                "reference_contexts": [
                    "Control plane components make global decisions about the cluster",
                    "kube-apiserver exposes the Kubernetes API",
                    "etcd is a consistent and highly-available key value store",
                    "kube-scheduler watches for newly created Pods"
                ]
            },
            {
                "user_input": "How do I create a Deployment in Kubernetes?",
                "reference": "You can create a Deployment using kubectl with a YAML manifest file that defines the desired state, including the container image, number of replicas, and other specifications. Use 'kubectl apply -f deployment.yaml' to create it.",
                "reference_contexts": [
                    "A Deployment provides declarative updates for Pods and ReplicaSets",
                    "You can create a Deployment using kubectl apply",
                    "Deployments manage ReplicaSets and provide rollout capabilities"
                ]
            },
            {
                "user_input": "What are resource limits and requests in Kubernetes?",
                "reference": "Resource requests specify the minimum amount of CPU and memory that a container needs, while resource limits specify the maximum amount it can use. These help with scheduling decisions and prevent containers from consuming too many resources.",
                "reference_contexts": [
                    "Resource requests specify the minimum resources a container needs",
                    "Resource limits specify the maximum resources a container can use",
                    "The scheduler uses resource requests to decide which node to place the Pod on"
                ]
            },
            {
                "user_input": "How does Kubernetes networking work?",
                "reference": "Kubernetes networking follows a flat network model where every Pod gets its own IP address and can communicate with other Pods directly. Services provide stable endpoints and load balancing, while Ingress manages external access to services.",
                "reference_contexts": [
                    "Every Pod gets its own IP address",
                    "Pods can communicate with all other Pods without NAT",
                    "Services provide stable network endpoints",
                    "Ingress manages external access to services in a cluster"
                ]
            },
            {
                "user_input": "What are liveness and readiness probes?",
                "reference": "Liveness probes determine if a container is running and healthy - if they fail, Kubernetes restarts the container. Readiness probes determine if a container is ready to serve traffic - if they fail, the Pod is removed from service endpoints.",
                "reference_contexts": [
                    "Liveness probes indicate whether a container is running",
                    "Readiness probes indicate whether a container is ready to serve requests",
                    "Failed liveness probes result in container restart",
                    "Failed readiness probes remove the Pod from service endpoints"
                ]
            },
            {
                "user_input": "How do I manage secrets in Kubernetes?",
                "reference": "Secrets are created using kubectl or YAML manifests and store sensitive data like passwords, tokens, and keys in base64 encoded format. They can be mounted as volumes or exposed as environment variables in Pods.",
                "reference_contexts": [
                    "Secrets store sensitive data such as passwords, OAuth tokens, and SSH keys",
                    "Secrets can be mounted as data volumes or exposed as environment variables",
                    "Secret data is stored in base64 encoded format"
                ]
            }
        ]
        
        return test_cases
    
    def evaluate_retriever(self, 
                          retriever_type: RetrieverType, 
                          test_cases: List[Dict[str, Any]], 
                          k: int = 5,
                          **retriever_kwargs) -> Dict[str, Any]:
        """Evaluate a specific retriever type.
        
        Args:
            retriever_type: Type of retriever to evaluate
            test_cases: Test cases to evaluate on
            k: Number of documents to retrieve
            **retriever_kwargs: Additional arguments for retriever creation
            
        Returns:
            Evaluation results
        """
        print(f"\n🔍 Evaluating {retriever_type.value} retriever...")
        
        # Create the retriever
        retriever = self.retriever_factory.create_retriever(
            retriever_type, k=k, **retriever_kwargs
        )
        
        # Update agent with the new retriever
        self.agent.update_retriever(retriever)
        
        # Generate responses and contexts
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        
        for test_case in test_cases:
            question = test_case["user_input"]
            reference = test_case["reference"]
            
            try:
                # Get response from agent
                answer = self.agent.invoke(question)
                
                # Get contexts used for the answer
                retrieved_docs = self.agent.get_relevant_docs(question, k=k)
                context = [doc.page_content for doc in retrieved_docs]
                
                questions.append(question)
                answers.append(answer)
                contexts.append(context)
                ground_truths.append(reference)
                
            except Exception as e:
                print(f"❌ Error processing question '{question}': {e}")
                continue
        
        if not questions:
            return {"error": "No questions were successfully processed"}
        
        # Create RAGAS evaluation dataset
        eval_dataset = EvaluationDataset.from_list([
            {
                "user_input": q,
                "response": a, 
                "retrieved_contexts": c,
                "reference": gt
            }
            for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
        ])
        
        # Run evaluation (following the pattern from evaluation_example.ipynb)
        try:
            from ragas import evaluate, RunConfig
            
            # Set up run config with timeout (like in the course notebook)
            run_config = RunConfig(timeout=360)
            
            # Run evaluation - this returns a dictionary directly
            results = evaluate(
                dataset=eval_dataset,
                metrics=self.metrics,
                llm=self.evaluator_llm,
                run_config=run_config
            )
            
            # RAGAS evaluate() returns a dictionary directly, like in the course notebook
            # Example: {'context_recall': 0.1396, 'faithfulness': 0.5506, 'answer_relevancy': 0.5751}
            
            return {
                "retriever_type": retriever_type.value,
                "results": results,  # This is already a dictionary
                "num_questions": len(questions)
            }
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            return {"error": str(e)}
    
    def compare_retrievers(self, 
                          retriever_types: List[RetrieverType], 
                          test_cases: Optional[List[Dict[str, Any]]] = None,
                          k: int = 5) -> pd.DataFrame:
        """Compare multiple retriever types.
        
        Args:
            retriever_types: List of retriever types to compare
            test_cases: Test cases to use (if None, uses default test cases)
            k: Number of documents to retrieve
            
        Returns:
            DataFrame with comparison results
        """
        if test_cases is None:
            test_cases = self.create_test_dataset()
        
        print(f"🧪 Comparing {len(retriever_types)} retriever types on {len(test_cases)} test cases...")
        
        comparison_results = []
        
        for retriever_type in retriever_types:
            try:
                # Special handling for ensemble retriever
                kwargs = {}
                if retriever_type == RetrieverType.ENSEMBLE:
                    kwargs = {"retrievers": ["base", "bm25", "multi_query"]}
                
                eval_result = self.evaluate_retriever(
                    retriever_type, test_cases, k=k, **kwargs
                )
                
                if "error" not in eval_result:
                    results = eval_result["results"]
                    
                    # Extract metrics with fallback names
                    def get_metric_value(results, metric_names):
                        for name in metric_names:
                            if name in results:
                                return results[name]
                        return 0.0
                    
                    # Extract metrics using exact keys from RAGAS (as shown in course notebook)
                    comparison_results.append({
                        "Retriever": retriever_type.value.replace('_', ' ').title(),
                        "Faithfulness": results.get("faithfulness", 0.0),
                        "Response Relevancy": results.get("answer_relevancy", 0.0),  # RAGAS uses 'answer_relevancy'
                        "Context Recall": results.get("context_recall", 0.0),
                        "Context Precision": results.get("context_entity_recall", 0.0),  # Using context_entity_recall as proxy
                        "Questions Processed": eval_result["num_questions"]
                    })
                else:
                    print(f"❌ Failed to evaluate {retriever_type.value}: {eval_result['error']}")
                    
            except Exception as e:
                print(f"❌ Error evaluating {retriever_type.value}: {e}")
                continue
        
        if comparison_results:
            df = pd.DataFrame(comparison_results)
            return df
        else:
            print("❌ No successful evaluations to compare")
            return pd.DataFrame()
    
    def run_comprehensive_evaluation(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run a comprehensive evaluation of all retriever types.
        
        Returns:
            Tuple of (comparison DataFrame, detailed results)
        """
        print("🚀 Starting comprehensive evaluation...")
        
        # Create test dataset
        test_cases = self.create_test_dataset()
        print(f"📝 Created {len(test_cases)} test cases")
        
        # Define retrievers to evaluate
        retrievers_to_test = [
            RetrieverType.BASE,
            RetrieverType.BM25,
            RetrieverType.MULTI_QUERY,
            RetrieverType.CONTEXTUAL_COMPRESSION,
            RetrieverType.ENSEMBLE
        ]
        
        # Note: Parent document retriever can be memory intensive, so we skip it in comprehensive evaluation
        # You can add it back if needed: RetrieverType.PARENT_DOCUMENT
        
        # Run comparison
        comparison_df = self.compare_retrievers(retrievers_to_test, test_cases)
        
        # Calculate summary statistics
        summary = {}
        if not comparison_df.empty:
            numeric_columns = ["Faithfulness", "Response Relevancy", "Context Recall", "Context Entity Recall"]
            
            # Best performer for each metric
            for col in numeric_columns:
                if col in comparison_df.columns:
                    best_idx = comparison_df[col].idxmax()
                    summary[f"Best {col}"] = comparison_df.loc[best_idx, "Retriever"]
                    summary[f"Best {col} Score"] = comparison_df.loc[best_idx, col]
            
            # Overall average scores
            summary["Average Scores"] = comparison_df[numeric_columns].mean().to_dict()
        
        return comparison_df, summary
