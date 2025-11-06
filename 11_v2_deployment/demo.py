#!/usr/bin/env python3
"""
Comprehensive demo and testing script for the Kubernetes Copilot system.
This script demonstrates all the key features and capabilities.
"""

import os
import sys
from pathlib import Path
import json
import time
from typing import List, Dict, Any

# Add the k8s_copilot module to the path
sys.path.append(str(Path(__file__).parent))

from k8s_copilot.vector_db.vector_store import K8sVectorStore
from k8s_copilot.vector_db.data_loader import K8sDataLoader
from k8s_copilot.agents.k8s_agent import K8sCopilotAgent, K8sRAGAgent
from k8s_copilot.evaluation.evaluator import K8sEvaluator
from k8s_copilot.utils.config import setup_environment, get_config
from k8s_copilot.utils.helpers import format_cost, format_response

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
    vector_store = K8sVectorStore()
    
    # Load data
    data_dir = Path(__file__).parent / "k8s_copilot" / "data"
    data_loader = K8sDataLoader(data_dir)
    
    print("📊 Loading Kubernetes data...")
    data_loader.load_all_data(vector_store)
    
    # Show statistics
    stats = vector_store.get_stats()
    print(f"\n✅ Data loaded successfully!")
    print(f"   Total documents: {stats['total_documents']}")
    print("   Document types:")
    for doc_type, count in stats['document_types'].items():
        print(f"     - {doc_type.replace('_', ' ').title()}: {count}")
    
    return vector_store

def demo_basic_queries(agent: K8sCopilotAgent):
    """Demonstrate basic query capabilities."""
    print_header("BASIC QUERY DEMO")
    
    basic_queries = [
        "What deployments are running in my cluster?",
        "How many total pods do I have?",
        "What's the resource utilization of my cluster?"
    ]
    
    for query in basic_queries:
        print_subheader(f"Query: {query}")
        
        try:
            response = agent.invoke(query)
            print("Response:")
            print(format_response(response, max_length=500))
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "-"*40)

def demo_cost_analysis(agent: K8sCopilotAgent):
    """Demonstrate cost analysis capabilities."""
    print_header("COST ANALYSIS DEMO")
    
    cost_queries = [
        "What are the costs of my Kubernetes deployments?",
        "Which deployment is the most expensive?",
        "Show me the total cluster cost for the last 30 days",
        "What's the cost breakdown by resource type?"
    ]
    
    for query in cost_queries:
        print_subheader(f"Query: {query}")
        
        try:
            response = agent.invoke(query)
            print("Response:")
            print(format_response(response, max_length=600))
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "-"*40)

def demo_resource_analysis(agent: K8sCopilotAgent):
    """Demonstrate resource analysis capabilities."""
    print_header("RESOURCE ANALYSIS DEMO")
    
    resource_queries = [
        "How many GPUs does the ml-training deployment use?",
        "Which deployments are using the most memory?",
        "Analyze the nginx-deployment resources",
        "What's the GPU utilization across my cluster?"
    ]
    
    for query in resource_queries:
        print_subheader(f"Query: {query}")
        
        try:
            response = agent.invoke(query)
            print("Response:")
            print(format_response(response, max_length=500))
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "-"*40)

def demo_optimization_recommendations(agent: K8sCopilotAgent):
    """Demonstrate optimization recommendation capabilities."""
    print_header("OPTIMIZATION RECOMMENDATIONS DEMO")
    
    optimization_queries = [
        "How can I improve resource utilization?",
        "Show me optimization opportunities",
        "Generate YAML optimization for the nginx-deployment",
        "What are the potential cost savings?"
    ]
    
    for query in optimization_queries:
        print_subheader(f"Query: {query}")
        
        try:
            response = agent.invoke(query)
            print("Response:")
            print(format_response(response, max_length=800))
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "-"*40)

def demo_agent_comparison(vector_store: K8sVectorStore):
    """Demonstrate the difference between copilot and RAG agents."""
    print_header("AGENT COMPARISON DEMO")
    
    copilot_agent = K8sCopilotAgent(vector_store)
    rag_agent = K8sRAGAgent(vector_store)
    
    comparison_queries = [
        "What are the costs of my deployments?",
        "How many GPUs does my cluster use?",
        "Suggest optimizations for my resources"
    ]
    
    for query in comparison_queries:
        print_subheader(f"Query: {query}")
        
        print("🤖 Copilot Agent (with tools):")
        try:
            copilot_response = copilot_agent.invoke(query)
            print(format_response(copilot_response, max_length=400))
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n📚 RAG Agent (simple Q&A):")
        try:
            rag_response = rag_agent.invoke(query)
            print(format_response(rag_response, max_length=400))
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "-"*50)

def demo_evaluation(vector_store: K8sVectorStore):
    """Demonstrate the evaluation framework."""
    print_header("EVALUATION FRAMEWORK DEMO")
    
    print("🧪 Setting up evaluator...")
    evaluator = K8sEvaluator(vector_store)
    
    print("📝 Creating test dataset...")
    test_cases = evaluator.create_test_dataset()
    print(f"   Created {len(test_cases)} test cases")
    
    print("\n📋 Sample test cases:")
    for i, test_case in enumerate(test_cases[:3]):
        print(f"   {i+1}. {test_case['user_input']}")
    
    print("\n⏱️  Running quick evaluation (first 3 test cases)...")
    
    # Run a limited evaluation for demo purposes
    sample_queries = [tc["user_input"] for tc in test_cases[:3]]
    sample_expected = [tc["reference"] for tc in test_cases[:3]]
    
    evaluation_results = []
    
    copilot_agent = K8sCopilotAgent(vector_store)
    
    for query, expected in zip(sample_queries, sample_expected):
        print(f"\n   Evaluating: {query[:50]}...")
        
        try:
            response = copilot_agent.invoke(query)
            
            # Simple evaluation - check if response contains key terms from expected
            expected_terms = expected.lower().split()
            response_terms = response.lower().split()
            
            overlap = len(set(expected_terms) & set(response_terms))
            relevance_score = overlap / len(expected_terms) if expected_terms else 0
            
            evaluation_results.append({
                "query": query,
                "response_length": len(response),
                "relevance_score": relevance_score,
                "has_numbers": any(char.isdigit() for char in response),
                "has_currency": '$' in response
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # Show evaluation summary
    if evaluation_results:
        print("\n📊 Evaluation Summary:")
        avg_relevance = sum(r["relevance_score"] for r in evaluation_results) / len(evaluation_results)
        avg_length = sum(r["response_length"] for r in evaluation_results) / len(evaluation_results)
        has_numbers_pct = sum(r["has_numbers"] for r in evaluation_results) / len(evaluation_results) * 100
        has_currency_pct = sum(r["has_currency"] for r in evaluation_results) / len(evaluation_results) * 100
        
        print(f"   Average relevance score: {avg_relevance:.2f}")
        print(f"   Average response length: {avg_length:.0f} characters")
        print(f"   Responses with numbers: {has_numbers_pct:.0f}%")
        print(f"   Responses with currency: {has_currency_pct:.0f}%")

def demo_vector_search(vector_store: K8sVectorStore):
    """Demonstrate vector search capabilities."""
    print_header("VECTOR SEARCH DEMO")
    
    search_queries = [
        ("cost expensive deployment", "cost_data"),
        ("GPU nvidia machine learning", "manifest"),
        ("optimization savings", "optimization_opportunity"),
        ("resource utilization CPU memory", None)
    ]
    
    for query, filter_type in search_queries:
        print_subheader(f"Search: '{query}'" + (f" (filter: {filter_type})" if filter_type else ""))
        
        results = vector_store.search(query, k=3, filter_type=filter_type)
        
        print(f"Found {len(results)} results:")
        for i, doc in enumerate(results):
            print(f"\n   Result {i+1}:")
            print(f"   Type: {doc.metadata.get('type', 'unknown')}")
            print(f"   Content: {doc.page_content[:200]}...")
            if doc.metadata.get('name'):
                print(f"   Name: {doc.metadata['name']}")

def run_comprehensive_demo():
    """Run the complete demonstration."""
    print("🚀 KUBERNETES COPILOT - COMPREHENSIVE DEMO")
    print("==========================================")
    
    # Set hardcoded API keys for demo purposes
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "INSERT OPENAI API KEY"
    
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
        # 1. Data Loading
        vector_store = demo_data_loading()
        
        # 2. Initialize main agent
        print_header("INITIALIZING AGENTS")
        print("🤖 Creating Kubernetes Copilot Agent...")
        copilot_agent = K8sCopilotAgent(vector_store)
        print("✅ Copilot agent ready!")
        
        # 3. Basic queries
        demo_basic_queries(copilot_agent)
        
        # 4. Cost analysis
        demo_cost_analysis(copilot_agent)
        
        # 5. Resource analysis
        demo_resource_analysis(copilot_agent)
        
        # 6. Optimization recommendations
        demo_optimization_recommendations(copilot_agent)
        
        # 7. Agent comparison
        demo_agent_comparison(vector_store)
        
        # 8. Vector search demo
        demo_vector_search(vector_store)
        
        # 9. Evaluation framework
        demo_evaluation(vector_store)
        
        print_header("DEMO COMPLETE")
        print("🎉 All demonstrations completed successfully!")
        print("\nNext steps:")
        print("1. Run the Streamlit UI: streamlit run k8s_copilot/ui/app.py")
        print("2. Try your own queries with the copilot")
        print("3. Explore the evaluation framework")
        print("4. Customize the system for your use case")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check your environment setup and try again.")
        import traceback
        traceback.print_exc()

def run_quick_test():
    """Run a quick test to verify the system works."""
    print("🧪 QUICK SYSTEM TEST")
    print("===================")
    
    # Set hardcoded API keys for demo purposes
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "INSERT OPENAI API KEY"
    
    if not os.getenv("COHERE_API_KEY"):
        os.environ["COHERE_API_KEY"] = "INSERT COHERE API KEY "
    
    if not setup_environment():
        return False
    
    try:
        # Initialize system
        print("1. Loading data...")
        vector_store = demo_data_loading()
        
        print("2. Creating agent...")
        agent = K8sCopilotAgent(vector_store)
        
        print("3. Testing basic query...")
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
        run_comprehensive_demo()

if __name__ == "__main__":
    main()
