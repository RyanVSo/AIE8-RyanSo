#!/usr/bin/env python3
"""
Demo script for the K8s RAG Copilot system.
"""

import os
import sys
from pathlib import Path

# Add the k8s_rag module to path
sys.path.append(str(Path(__file__).parent))

from k8s_rag.vector_db.vector_store import K8sDocVectorStore
from k8s_rag.vector_db.data_loader import K8sDocumentationLoader
from k8s_rag.agents.base_agent import K8sBaseRAGAgent
from k8s_rag.retrieval.advanced_retrievers import K8sAdvancedRetrieverFactory, RetrieverType
from k8s_rag.utils.config import setup_environment, get_config


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def print_subheader(title: str):
    """Print a formatted subheader."""
    print(f"\n--- {title} ---")


def demo_data_loading():
    """Demonstrate data loading capabilities."""
    print_header("DATA LOADING DEMO")
    
    # Initialize vector store
    print("🔧 Initializing vector store...")
    vector_store = K8sDocVectorStore()
    
    # Load data
    data_dir = Path(__file__).parent / "data"
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Please ensure the Kubernetes documentation is available in the 'data' directory")
        return None
    
    loader = K8sDocumentationLoader(data_dir)
    
    print("📊 Loading Kubernetes documentation...")
    loader.load_all_data(vector_store)
    
    # Show statistics
    stats = vector_store.get_stats()
    print(f"\n✅ Data loaded successfully!")
    print(f"   Total documents: {stats['total_documents']}")
    print("   Document types:")
    for doc_type, count in stats['document_types'].items():
        print(f"     - {doc_type.replace('_', ' ').title()}: {count}")
    
    return vector_store


def demo_basic_queries(agent: K8sBaseRAGAgent):
    """Demonstrate basic query capabilities."""
    print_header("BASIC QUERY DEMO")
    
    basic_queries = [
        "What is a Kubernetes Pod?",
        "How do Services work in Kubernetes?",
        "What are the main components of the control plane?",
    ]
    
    for query in basic_queries:
        print_subheader(f"Query: {query}")
        
        try:
            response = agent.invoke(query)
            # Truncate response for demo
            if len(response) > 400:
                response = response[:400] + "..."
            print("Response:")
            print(response)
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "-"*40)


def demo_retrieval_comparison(vector_store: K8sDocVectorStore):
    """Demonstrate different retrieval methods."""
    print_header("RETRIEVAL METHODS COMPARISON")
    
    # Initialize retriever factory and agent
    factory = K8sAdvancedRetrieverFactory(vector_store)
    agent = K8sBaseRAGAgent(vector_store)
    
    # Test query
    test_query = "How do I create a Deployment in Kubernetes?"
    print(f"Test Query: {test_query}")
    
    # Test different retrieval methods
    retrieval_methods = [
        RetrieverType.BASE,
        RetrieverType.BM25,
        RetrieverType.MULTI_QUERY,
        RetrieverType.ENSEMBLE
    ]
    
    for method in retrieval_methods:
        print_subheader(f"{method.value.replace('_', ' ').title()} Retrieval")
        
        try:
            # Create retriever
            kwargs = {}
            if method == RetrieverType.ENSEMBLE:
                kwargs = {"retrievers": ["base", "bm25", "multi_query"]}
            
            retriever = factory.create_retriever(method, k=3, **kwargs)
            agent.update_retriever(retriever)
            
            # Get response
            response = agent.invoke(test_query)
            
            # Truncate for demo
            if len(response) > 300:
                response = response[:300] + "..."
            
            print("Response:")
            print(response)
            
        except Exception as e:
            print(f"❌ Error with {method.value}: {e}")
        
        print("\n" + "-"*50)


def demo_evaluation_preview():
    """Show a preview of what the evaluation notebook contains."""
    print_header("EVALUATION FRAMEWORK PREVIEW")
    
    print("📊 The evaluation notebook (k8s_rag_evaluation.ipynb) includes:")
    print("   • RAGAS framework integration")
    print("   • Faithfulness, Response Relevancy, Context Recall, and Context Entity Recall metrics")
    print("   • Comparison of all retrieval methods")
    print("   • Visualization of results with charts and graphs")
    print("   • Best performer analysis and recommendations")
    print("   • Exportable results to CSV")
    
    print("\n🚀 To run the full evaluation:")
    print("   1. jupyter notebook k8s_rag_evaluation.ipynb")
    print("   2. Run all cells to see comprehensive results")
    print("   3. View performance comparisons and insights")


def run_demo():
    """Run the complete demonstration."""
    print("🚀 KUBERNETES RAG COPILOT - DEMO")
    print("=================================")
    
    # Check environment
    if not setup_environment():
        print("❌ Environment setup failed. Please check your configuration.")
        return False
    
    config = get_config()
    print(f"✅ Using LLM: {config['llm_model']}")
    print(f"✅ Using embedding model: {config['embedding_model']}")
    
    try:
        # 1. Data Loading
        vector_store = demo_data_loading()
        if not vector_store:
            return False
        
        # 2. Initialize agent
        print_header("INITIALIZING AGENT")
        print("🤖 Creating Kubernetes RAG Agent...")
        agent = K8sBaseRAGAgent(vector_store)
        print("✅ Agent ready!")
        
        # 3. Basic queries
        demo_basic_queries(agent)
        
        # 4. Retrieval comparison
        demo_retrieval_comparison(vector_store)
        
        # 5. Evaluation preview
        demo_evaluation_preview()
        
        print_header("DEMO COMPLETE")
        print("🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Run the web UI: python run_app.py")
        print("2. Try the evaluation notebook: jupyter notebook k8s_rag_evaluation.ipynb")
        print("3. Explore different retrieval methods in the UI")
        print("4. Ask your own Kubernetes questions!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check your environment setup and try again.")
        import traceback
        traceback.print_exc()
        return False


def run_quick_test():
    """Run a quick test to verify the system works."""
    print("🧪 QUICK SYSTEM TEST")
    print("===================")
    
    if not setup_environment():
        return False
    
    try:
        # Initialize system
        print("1. Loading data...")
        vector_store = demo_data_loading()
        if not vector_store:
            return False
        
        print("2. Creating agent...")
        agent = K8sBaseRAGAgent(vector_store)
        
        print("3. Testing basic query...")
        response = agent.invoke("What is a Pod in Kubernetes?")
        
        if response and len(response) > 10:
            print("✅ Quick test passed!")
            print(f"   Sample response: {response[:100]}...")
            return True
        else:
            print("❌ Quick test failed - empty or short response")
            return False
            
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        if "api_key" in str(e).lower():
            print("\n💡 TIP: Make sure you've set your OpenAI API key:")
            print("   export OPENAI_API_KEY='your-api-key-here'")
        return False


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = run_quick_test()
        sys.exit(0 if success else 1)
    else:
        success = run_demo()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
