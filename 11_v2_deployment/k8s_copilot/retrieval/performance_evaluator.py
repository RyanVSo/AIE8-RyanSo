"""
Performance evaluation framework for comparing different retrieval methods.

This module provides comprehensive evaluation of retrieval performance using
RAGAS metrics, timing analysis, and K8s-specific evaluation criteria.
"""

import time
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    ContextPrecision,
    ContextRecall, 
    Faithfulness,
    ResponseRelevancy,
    FactualCorrectness
)

from .retriever_factory import K8sRetrieverFactory, RetrieverType
from ..vector_db.vector_store import K8sVectorStore
from ..agents.k8s_agent import K8sRAGAgent


@dataclass
class RetrievalMetrics:
    """Container for retrieval performance metrics."""
    retriever_type: str
    avg_retrieval_time: float
    context_precision: float
    context_recall: float
    faithfulness: float
    response_relevancy: float
    factual_correctness: float
    total_queries: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
        return {
            "retriever_type": self.retriever_type,
            "avg_retrieval_time": self.avg_retrieval_time,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "faithfulness": self.faithfulness,
            "response_relevancy": self.response_relevancy,
            "factual_correctness": self.factual_correctness,
            "total_queries": self.total_queries
        }


class RetrievalPerformanceEvaluator:
    """Evaluator for comparing retrieval method performance."""
    
    def __init__(self, vector_store: K8sVectorStore):
        """Initialize the performance evaluator.
        
        Args:
            vector_store: The K8s vector store containing the data
        """
        self.vector_store = vector_store
        self.retriever_factory = K8sRetrieverFactory(vector_store)
        
        # Initialize evaluation models
        self.evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
        self.evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
        
        # Test queries for evaluation
        self.test_queries = [
            "What are the costs of my Kubernetes deployments?",
            "How many GPUs does the ml-training deployment use?", 
            "Which deployments are using the most memory?",
            "How can I improve resource utilization?",
            "Show me the total cluster cost for the last 30 days",
            "Analyze the nginx-deployment resources",
            "What's the GPU utilization across my cluster?",
            "Generate optimization recommendations for api-server",
            "What deployments are running in the default namespace?",
            "Which services have the highest CPU requests?"
        ]
        
        # Expected answers for evaluation
        self.expected_answers = [
            "Deployment costs include ml-training ($125/day), api-server ($45.80/day), nginx-deployment ($15.50/day) with detailed breakdowns by resource type.",
            "The ml-training deployment uses 1 GPU per replica with 2 replicas for a total of 2 GPUs requested.",
            "The ml-training deployment uses the most memory with 4Gi requests and 8Gi limits per container.",
            "Resource utilization can be improved through right-sizing CPU/memory requests, reducing replica counts, and node consolidation with potential savings.",
            "Total cluster cost includes all deployment and node costs across the specified time period with comprehensive breakdown.",
            "The nginx-deployment uses 3 replicas with 100m CPU and 128Mi memory requests, with optimization opportunities available.",
            "The cluster has 2 total GPUs with 50% utilization primarily from ml-training workloads.",
            "API-server optimization should focus on resource limits and replica configuration based on usage patterns.",
            "Multiple deployments run in the default namespace including nginx, api-server, and ml-training deployments.",
            "Services with highest CPU requests include ml-training and api-server based on their resource configurations."
        ]
    
    def evaluate_retriever(self, 
                          retriever_type: RetrieverType,
                          k: int = 5,
                          **retriever_kwargs) -> RetrievalMetrics:
        """Evaluate a specific retriever type.
        
        Args:
            retriever_type: Type of retriever to evaluate
            k: Number of documents to retrieve
            **retriever_kwargs: Additional configuration for the retriever
            
        Returns:
            Performance metrics for the retriever
        """
        print(f"Evaluating {retriever_type.value} retriever...")
        
        # Create the retriever
        retriever = self.retriever_factory.create_retriever(
            retriever_type, k=k, **retriever_kwargs
        )
        
        # Measure retrieval timing
        retrieval_times = []
        evaluation_data = []
        
        for i, query in enumerate(self.test_queries):
            print(f"  Processing query {i+1}/{len(self.test_queries)}: {query[:50]}...")
            
            # Time the retrieval
            start_time = time.time()
            try:
                retrieved_docs = retriever.invoke(query)
                retrieval_time = time.time() - start_time
                retrieval_times.append(retrieval_time)
                
                # Create RAG agent with this retriever
                rag_agent = self._create_rag_agent_with_retriever(retriever)
                response = rag_agent.invoke(query)
                
                # Prepare data for RAGAS evaluation
                evaluation_data.append({
                    "user_input": query,
                    "response": response,
                    "reference": self.expected_answers[i],
                    "retrieved_contexts": [doc.page_content for doc in retrieved_docs]
                })
                
            except Exception as e:
                print(f"    Error processing query: {e}")
                continue
        
        # Calculate average retrieval time
        avg_retrieval_time = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0
        
        # Run RAGAS evaluation
        if evaluation_data:
            df = pd.DataFrame(evaluation_data)
            evaluation_dataset = EvaluationDataset.from_pandas(df)
            
            # Define metrics
            metrics = [
                ContextPrecision(),
                ContextRecall(),
                Faithfulness(),
                ResponseRelevancy(),
                FactualCorrectness()
            ]
            
            try:
                results = evaluate(
                    dataset=evaluation_dataset,
                    metrics=metrics,
                    llm=self.evaluator_llm,
                    embeddings=self.evaluator_embeddings,
                    run_config=RunConfig(timeout=360)
                )
                
                # Handle different RAGAS result formats
                if hasattr(results, 'scores'):
                    # New RAGAS format - scores is a list of dicts, need to aggregate
                    raw_scores = results.scores
                    
                    # Calculate mean scores across all samples
                    if isinstance(raw_scores, list) and raw_scores:
                        scores = {}
                        # Get all metric names from first sample
                        metric_names = raw_scores[0].keys()
                        for metric_name in metric_names:
                            # Calculate mean for this metric across all samples
                            metric_values = [sample.get(metric_name, 0) for sample in raw_scores if sample.get(metric_name) is not None]
                            scores[metric_name] = sum(metric_values) / len(metric_values) if metric_values else 0.0
                    else:
                        scores = raw_scores if isinstance(raw_scores, dict) else {}
                        
                elif hasattr(results, 'to_pandas'):
                    # DataFrame format - get mean scores
                    df = results.to_pandas()
                    scores = df.mean().to_dict()
                elif isinstance(results, dict):
                    # Direct dictionary format
                    scores = results
                else:
                    # Fallback - try to extract from results object
                    scores = {}
                    for metric in metrics:
                        metric_name = metric.__class__.__name__.lower().replace('llm', '').replace('_', '')
                        if hasattr(results, metric_name):
                            scores[metric_name] = getattr(results, metric_name)
                    
                    # Additional fallback - try common attribute names
                    if not scores:
                        for attr in ['mean', 'scores', 'results', 'metrics']:
                            if hasattr(results, attr):
                                attr_value = getattr(results, attr)
                                if isinstance(attr_value, dict):
                                    scores = attr_value
                                    break
                
                return RetrievalMetrics(
                    retriever_type=retriever_type.value,
                    avg_retrieval_time=avg_retrieval_time,
                    context_precision=scores.get("context_precision", scores.get("contextprecision", 0.0)),
                    context_recall=scores.get("context_recall", scores.get("contextrecall", 0.0)),
                    faithfulness=scores.get("faithfulness", 0.0),
                    response_relevancy=scores.get("answer_relevancy", scores.get("response_relevancy", scores.get("responserelevancy", 0.0))),
                    factual_correctness=scores.get("factual_correctness(mode=f1)", scores.get("factual_correctness", scores.get("factualcorrectness", 0.0))),
                    total_queries=len(evaluation_data)
                )
                
            except Exception as e:
                print(f"  RAGAS evaluation failed: {e}")
                return RetrievalMetrics(
                    retriever_type=retriever_type.value,
                    avg_retrieval_time=avg_retrieval_time,
                    context_precision=0.0,
                    context_recall=0.0,
                    faithfulness=0.0,
                    response_relevancy=0.0,
                    factual_correctness=0.0,
                    total_queries=len(evaluation_data)
                )
        
        # Fallback if no evaluation data
        return RetrievalMetrics(
            retriever_type=retriever_type.value,
            avg_retrieval_time=avg_retrieval_time,
            context_precision=0.0,
            context_recall=0.0,
            faithfulness=0.0,
            response_relevancy=0.0,
            factual_correctness=0.0,
            total_queries=0
        )
    
    def compare_all_retrievers(self, 
                              k: int = 5,
                              retriever_configs: Optional[Dict[str, Dict]] = None) -> List[RetrievalMetrics]:
        """Compare all available retriever types.
        
        Args:
            k: Number of documents to retrieve
            retriever_configs: Optional configuration for each retriever type
            
        Returns:
            List of performance metrics for all retrievers
        """
        if retriever_configs is None:
            retriever_configs = {}
        
        results = []
        
        for retriever_type in RetrieverType:
            try:
                config = retriever_configs.get(retriever_type.value, {})
                metrics = self.evaluate_retriever(retriever_type, k=k, **config)
                results.append(metrics)
            except Exception as e:
                print(f"Failed to evaluate {retriever_type.value}: {e}")
                continue
        
        return results
    
    def _create_rag_agent_with_retriever(self, retriever: BaseRetriever) -> K8sRAGAgent:
        """Create a RAG agent with a specific retriever.
        
        Args:
            retriever: The retriever to use
            
        Returns:
            RAG agent configured with the retriever
        """
        # Create a modified RAG agent that uses the provided retriever
        class CustomK8sRAGAgent(K8sRAGAgent):
            def __init__(self, custom_retriever):
                self.vector_store = None  # Not used
                self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                self.retriever = custom_retriever
                self.rag_chain = self._create_rag_chain()
        
        return CustomK8sRAGAgent(retriever)
    
    def generate_comparison_report(self, 
                                  metrics_list: List[RetrievalMetrics],
                                  output_path: Optional[Path] = None) -> str:
        """Generate a detailed comparison report.
        
        Args:
            metrics_list: List of retrieval metrics to compare
            output_path: Optional path to save the report
            
        Returns:
            Formatted comparison report as string
        """
        if not metrics_list:
            return "No metrics available for comparison."
        
        # Create comparison DataFrame
        df = pd.DataFrame([metrics.to_dict() for metrics in metrics_list])
        
        # Sort by overall performance (weighted average of key metrics)
        df['overall_score'] = (
            df['context_precision'] * 0.25 +
            df['context_recall'] * 0.25 + 
            df['faithfulness'] * 0.25 +
            df['response_relevancy'] * 0.25
        )
        df = df.sort_values('overall_score', ascending=False)
        
        # Generate report
        report_lines = [
            "# Retrieval Method Performance Comparison",
            "=" * 50,
            "",
            "## Summary Rankings (by Overall Score)",
            ""
        ]
        
        for i, row in df.iterrows():
            report_lines.append(
                f"{row.name + 1}. {row['retriever_type'].upper()}: "
                f"Overall Score = {row['overall_score']:.3f}"
            )
        
        report_lines.extend([
            "",
            "## Detailed Metrics",
            ""
        ])
        
        for _, row in df.iterrows():
            report_lines.extend([
                f"### {row['retriever_type'].upper()}",
                f"- Average Retrieval Time: {row['avg_retrieval_time']:.3f}s",
                f"- Context Precision: {row['context_precision']:.3f}",
                f"- Context Recall: {row['context_recall']:.3f}",
                f"- Faithfulness: {row['faithfulness']:.3f}",
                f"- Response Relevancy: {row['response_relevancy']:.3f}",
                f"- Factual Correctness: {row['factual_correctness']:.3f}",
                f"- Total Queries Processed: {row['total_queries']}",
                ""
            ])
        
        # Add recommendations
        best_retriever = df.iloc[0]
        fastest_retriever = df.loc[df['avg_retrieval_time'].idxmin()]
        most_precise = df.loc[df['context_precision'].idxmax()]
        
        report_lines.extend([
            "## Recommendations",
            "",
            f"- **Best Overall**: {best_retriever['retriever_type']} "
            f"(Score: {best_retriever['overall_score']:.3f})",
            f"- **Fastest**: {fastest_retriever['retriever_type']} "
            f"({fastest_retriever['avg_retrieval_time']:.3f}s avg)",
            f"- **Most Precise**: {most_precise['retriever_type']} "
            f"(Precision: {most_precise['context_precision']:.3f})",
            ""
        ])
        
        report = "\n".join(report_lines)
        
        # Save to file if path provided
        if output_path:
            output_path.write_text(report)
            print(f"Report saved to {output_path}")
        
        return report
    
    def save_metrics_csv(self, 
                        metrics_list: List[RetrievalMetrics],
                        output_path: Path):
        """Save metrics to CSV file for further analysis.
        
        Args:
            metrics_list: List of retrieval metrics
            output_path: Path to save the CSV file
        """
        df = pd.DataFrame([metrics.to_dict() for metrics in metrics_list])
        df.to_csv(output_path, index=False)
        print(f"Metrics saved to {output_path}")
