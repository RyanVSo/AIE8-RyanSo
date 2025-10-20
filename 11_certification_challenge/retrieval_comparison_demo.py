#!/usr/bin/env python3
"""
Comprehensive demo script for comparing different retrieval methods.

This script demonstrates the performance differences between various
retrieval strategies on Kubernetes data.
"""

import os
import sys
from pathlib import Path
import time
import json

# Add the k8s_copilot module to the path
sys.path.append(str(Path(__file__).parent))

from k8s_copilot.vector_db.vector_store import K8sVectorStore
from k8s_copilot.vector_db.data_loader import K8sDataLoader
from k8s_copilot.retrieval import K8sRetrieverFactory, RetrieverType, RetrievalPerformanceEvaluator
from k8s_copilot.agents.enhanced_agents import EnhancedK8sRAGAgent
from k8s_copilot.utils.config import setup_environment, get_config


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_subheader(title: str):
    """Print a formatted subheader."""
    print(f"\n--- {title} ---")


def demo_retrieval_methods(vector_store: K8sVectorStore):
    """Demonstrate different retrieval methods with sample queries."""
    print_header("RETRIEVAL METHODS DEMO")
    
    # Sample queries for demonstration
    sample_queries = [
        "What are the costs of my Kubernetes deployments?",
        "How many GPUs does the ml-training deployment use?",
        "Which deployments are using the most memory?"
    ]
    
    # Configure retrievers to test
    retriever_configs = {
        RetrieverType.BASE: {},
        RetrieverType.BM25: {"bm25_params": {"k1": 1.5, "b": 0.75}},
        RetrieverType.MULTI_QUERY: {"num_queries": 3},
        RetrieverType.PARENT_DOCUMENT: {
            "parent_chunk_size": 2000,
            "child_chunk_size": 400,
            "chunk_overlap": 50
        },
        RetrieverType.ENSEMBLE: {
            "retrievers": ["base", "bm25", "multi_query"],
            "weights": [0.4, 0.3, 0.3]
        }
    }
    
    factory = K8sRetrieverFactory(vector_store)
    
    for retriever_type, config in retriever_configs.items():
        print_subheader(f"{retriever_type.value.upper()} Retriever")
        
        try:
            # Create enhanced agent with this retriever
            agent = EnhancedK8sRAGAgent(
                vector_store=vector_store,
                retriever_type=retriever_type,
                retriever_config=config
            )
            
            print(f"Retriever Info: {agent.get_retriever_info()}")
            
            # Test with sample queries
            for i, query in enumerate(sample_queries[:2]):  # Limit for demo
                print(f"\n  Query {i+1}: {query}")
                
                start_time = time.time()
                try:
                    response = agent.invoke(query)
                    response_time = time.time() - start_time
                    
                    print(f"  Response Time: {response_time:.3f}s")
                    print(f"  Response: {response[:200]}...")
                    
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    
        except Exception as e:
            print(f"❌ Failed to create {retriever_type.value} retriever: {e}")
        
        print("\n" + "-"*50)


def demo_performance_comparison(vector_store: K8sVectorStore):
    """Run comprehensive performance comparison."""
    print_header("PERFORMANCE COMPARISON")
    
    evaluator = RetrievalPerformanceEvaluator(vector_store)
    
    # Configure retrievers for evaluation
    retriever_configs = {
        "base": {},
        "bm25": {"bm25_params": {"k1": 1.5, "b": 0.75}},
        "multi_query": {"num_queries": 3},
        "ensemble": {
            "retrievers": ["base", "bm25"],
            "weights": [0.6, 0.4]
        }
    }
    
    print("Running comprehensive evaluation (this may take several minutes)...")
    print("Note: Contextual compression requires COHERE_API_KEY environment variable")
    
    # Run comparison
    results = evaluator.compare_all_retrievers(
        k=5,
        retriever_configs=retriever_configs
    )
    
    # Generate and display report
    report = evaluator.generate_comparison_report(results)
    print("\n" + report)
    
    # Save detailed results
    output_dir = Path(__file__).parent / "evaluation_results"
    output_dir.mkdir(exist_ok=True)
    
    # Save report
    report_path = output_dir / "retrieval_comparison_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Save metrics CSV
    csv_path = output_dir / "retrieval_metrics.csv"
    evaluator.save_metrics_csv(results, csv_path)
    
    print(f"\n📊 Detailed results saved to:")
    print(f"  - Report: {report_path}")
    print(f"  - Metrics: {csv_path}")
    
    return results


def demo_retriever_switching(vector_store: K8sVectorStore):
    """Demonstrate dynamic retriever switching."""
    print_header("DYNAMIC RETRIEVER SWITCHING DEMO")
    
    # Create agent with base retriever
    agent = EnhancedK8sRAGAgent(vector_store, RetrieverType.BASE)
    
    query = "What are the costs of my deployments?"
    retrievers_to_test = [
        (RetrieverType.BASE, {}),
        (RetrieverType.BM25, {}),
        (RetrieverType.MULTI_QUERY, {"num_queries": 2})
    ]
    
    for retriever_type, config in retrievers_to_test:
        print_subheader(f"Switching to {retriever_type.value} retriever")
        
        try:
            # Switch retriever
            agent.switch_retriever(retriever_type, config)
            
            print(f"Current retriever: {agent.get_retriever_info()}")
            
            # Test query
            start_time = time.time()
            response = agent.invoke(query)
            response_time = time.time() - start_time
            
            print(f"Response time: {response_time:.3f}s")
            print(f"Response: {response[:150]}...")
            
        except Exception as e:
            print(f"❌ Error with {retriever_type.value}: {e}")


def run_comprehensive_demo():
    """Run the complete retrieval comparison demo."""
    print("🔍 ADVANCED RETRIEVAL METHODS COMPARISON DEMO")
    print("=" * 70)
    
    # Set hardcoded API keys for demo purposes
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "INSERT API KEY"
    
    if not os.getenv("COHERE_API_KEY"):
        os.environ["COHERE_API_KEY"] = "INSERT COHERE API KEY "
    
    # Check environment
    if not setup_environment():
        print("❌ Environment setup failed. Please check your configuration.")
        return
    
    config = get_config()
    print(f"✅ Using LLM: {config['llm_model']}")
    print(f"✅ Using embedding model: {config['embedding_model']}")
    
    try:
        # Initialize vector store and load data
        print_header("INITIALIZING SYSTEM")
        print("📊 Loading Kubernetes data...")
        
        vector_store = K8sVectorStore()
        data_dir = Path(__file__).parent / "k8s_copilot" / "data"
        data_loader = K8sDataLoader(data_dir)
        data_loader.load_all_data(vector_store)
        
        stats = vector_store.get_stats()
        print(f"✅ Loaded {stats['total_documents']} documents")
        
        # Run demonstrations
        demo_retrieval_methods(vector_store)
        demo_retriever_switching(vector_store)
        
        # Ask user if they want to run full performance comparison
        print_header("PERFORMANCE EVALUATION")
        print("The full performance comparison includes RAGAS evaluation and may take 10-15 minutes.")
        
        if input("Run full performance comparison? (y/N): ").lower().startswith('y'):
            demo_performance_comparison(vector_store)
        else:
            print("Skipping full performance comparison.")
        
        print_header("DEMO COMPLETE")
        print("🎉 Advanced retrieval methods demonstration completed!")
        print("\nKey takeaways:")
        print("1. Different retrieval methods have different strengths")
        print("2. BM25 excels at lexical/keyword matching")
        print("3. Multi-query retrieval improves coverage")
        print("4. Parent-document retrieval balances precision and context")
        print("5. Ensemble methods can combine the best of multiple approaches")
        print("6. Performance varies by query type and data characteristics")
        
        print("\nNext steps:")
        print("1. Experiment with different retriever configurations")
        print("2. Test with your own queries and data")
        print("3. Use the performance evaluator for systematic comparison")
        print("4. Integrate the best-performing retriever into your application")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


def run_quick_test():
    """Run a quick test to verify the advanced retrieval system works."""
    print("🧪 QUICK ADVANCED RETRIEVAL TEST")
    print("=" * 40)
    
    # Set hardcoded API keys for demo purposes
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "INSERT API KEY"
    
    if not os.getenv("COHERE_API_KEY"):
        os.environ["COHERE_API_KEY"] = "INSERT COHERE API KEY "
    
    if not setup_environment():
        return False
    
    try:
        # Initialize system
        print("1. Loading data...")
        vector_store = K8sVectorStore()
        data_dir = Path(__file__).parent / "k8s_copilot" / "data"
        data_loader = K8sDataLoader(data_dir)
        data_loader.load_all_data(vector_store)
        
        print("2. Testing retriever factory...")
        factory = K8sRetrieverFactory(vector_store)
        base_retriever = factory.create_retriever(RetrieverType.BASE)
        
        print("3. Testing enhanced agent...")
        agent = EnhancedK8sRAGAgent(vector_store, RetrieverType.BASE)
        response = agent.invoke("How many deployments are in my cluster?")
        
        if response and len(response) > 10:
            print("✅ Quick test passed!")
            print(f"   Sample response: {response[:100]}...")
            return True
        else:
            print("❌ Quick test failed - empty or short response")
            return False
            
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = run_quick_test()
        sys.exit(0 if success else 1)
    else:
        run_comprehensive_demo()


if __name__ == "__main__":
    main()
