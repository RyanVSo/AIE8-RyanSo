"""
Enhanced Streamlit UI with retrieval method comparison capabilities.

This module provides an interactive interface for comparing different
retrieval methods and their performance on Kubernetes queries.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
import json
from pathlib import Path
from typing import Dict, List, Any

# Add the parent directory to the path for imports
import sys
import os
sys.path.append(str(Path(__file__).parent.parent.parent))

# Set hardcoded API keys for demo purposes
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "INSERT API KEY"

if not os.getenv("COHERE_API_KEY"):
    os.environ["COHERE_API_KEY"] = "INSERT COHERE API KEY "

from k8s_copilot.vector_db.vector_store import K8sVectorStore
from k8s_copilot.vector_db.data_loader import K8sDataLoader
from k8s_copilot.retrieval import K8sRetrieverFactory, RetrieverType, RetrievalPerformanceEvaluator
from k8s_copilot.agents.enhanced_agents import EnhancedK8sRAGAgent


@st.cache_resource
def initialize_system():
    """Initialize the K8s copilot system with caching."""
    # Initialize vector store
    vector_store = K8sVectorStore()
    
    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    data_loader = K8sDataLoader(data_dir)
    data_loader.load_all_data(vector_store)
    
    return vector_store


def render_retrieval_method_selector():
    """Render the retrieval method selection interface."""
    st.sidebar.header("🔍 Retrieval Method")
    
    # Method selection
    method_options = {
        "Base Vector Search": RetrieverType.BASE,
        "BM25 (Lexical Search)": RetrieverType.BM25,
        "Multi-Query Retrieval": RetrieverType.MULTI_QUERY,
        "Parent-Document Retrieval": RetrieverType.PARENT_DOCUMENT,
        "Contextual Compression": RetrieverType.CONTEXTUAL_COMPRESSION,
        "Ensemble Retrieval": RetrieverType.ENSEMBLE
    }
    
    selected_method = st.sidebar.selectbox(
        "Choose Retrieval Method",
        options=list(method_options.keys()),
        index=0,
        help="Select the retrieval strategy to use for answering queries"
    )
    
    retriever_type = method_options[selected_method]
    
    # Method-specific configuration
    config = {}
    
    if retriever_type == RetrieverType.BM25:
        st.sidebar.subheader("BM25 Parameters")
        config["bm25_params"] = {
            "k1": st.sidebar.slider("k1 (term frequency saturation)", 0.5, 3.0, 1.5, 0.1),
            "b": st.sidebar.slider("b (field length normalization)", 0.0, 1.0, 0.75, 0.05)
        }
    
    elif retriever_type == RetrieverType.MULTI_QUERY:
        st.sidebar.subheader("Multi-Query Parameters")
        config["num_queries"] = st.sidebar.slider("Number of query variants", 2, 5, 3)
    
    elif retriever_type == RetrieverType.PARENT_DOCUMENT:
        st.sidebar.subheader("Parent-Document Parameters")
        config["parent_chunk_size"] = st.sidebar.slider("Parent chunk size", 1000, 4000, 2000, 100)
        config["child_chunk_size"] = st.sidebar.slider("Child chunk size", 200, 800, 400, 50)
        config["chunk_overlap"] = st.sidebar.slider("Chunk overlap", 0, 100, 50, 10)
    
    elif retriever_type == RetrieverType.CONTEXTUAL_COMPRESSION:
        st.sidebar.subheader("Compression Parameters")
        config["top_k"] = st.sidebar.slider("Documents after compression", 3, 10, 5)
        if not st.session_state.get("cohere_api_key"):
            st.sidebar.warning("⚠️ Cohere API key required for optimal compression")
    
    elif retriever_type == RetrieverType.ENSEMBLE:
        st.sidebar.subheader("Ensemble Parameters")
        available_methods = st.sidebar.multiselect(
            "Select methods to combine",
            ["base", "bm25", "multi_query"],
            default=["base", "bm25"]
        )
        config["retrievers"] = available_methods
        
        if len(available_methods) > 1:
            st.sidebar.write("Method weights:")
            weights = []
            for method in available_methods:
                weight = st.sidebar.slider(f"{method} weight", 0.1, 1.0, 1.0/len(available_methods), 0.1)
                weights.append(weight)
            config["weights"] = weights
    
    # Number of documents to retrieve
    k = st.sidebar.slider("Documents to retrieve (k)", 3, 10, 5)
    
    return retriever_type, config, k


def render_performance_comparison(vector_store: K8sVectorStore):
    """Render the performance comparison interface."""
    st.header("📊 Retrieval Method Performance Comparison")
    
    if st.button("Run Performance Evaluation", type="primary"):
        with st.spinner("Running comprehensive evaluation (this may take several minutes)..."):
            evaluator = RetrievalPerformanceEvaluator(vector_store)
            
            # Configure methods to compare
            retriever_configs = {
                "base": {},
                "bm25": {"bm25_params": {"k1": 1.5, "b": 0.75}},
                "multi_query": {"num_queries": 3},
                "ensemble": {
                    "retrievers": ["base", "bm25"],
                    "weights": [0.6, 0.4]
                }
            }
            
            try:
                results = evaluator.compare_all_retrievers(
                    k=5,
                    retriever_configs=retriever_configs
                )
                
                if results:
                    # Store results in session state
                    st.session_state.evaluation_results = results
                    st.success("Evaluation completed!")
                else:
                    st.error("No results obtained from evaluation")
                    
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
    
    # Display results if available
    if "evaluation_results" in st.session_state:
        results = st.session_state.evaluation_results
        
        # Create DataFrame for visualization
        df = pd.DataFrame([r.to_dict() for r in results])
        
        # Performance metrics visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Overall Performance")
            df['overall_score'] = (
                df['context_precision'] * 0.25 +
                df['context_recall'] * 0.25 + 
                df['faithfulness'] * 0.25 +
                df['response_relevancy'] * 0.25
            )
            
            fig = px.bar(
                df.sort_values('overall_score', ascending=True),
                x='overall_score',
                y='retriever_type',
                orientation='h',
                title="Overall Performance Score",
                color='overall_score',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Retrieval Speed")
            fig = px.bar(
                df.sort_values('avg_retrieval_time'),
                x='avg_retrieval_time',
                y='retriever_type',
                orientation='h',
                title="Average Retrieval Time (seconds)",
                color='avg_retrieval_time',
                color_continuous_scale='reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        st.subheader("Detailed Metrics")
        display_df = df[[
            'retriever_type', 'avg_retrieval_time', 'context_precision',
            'context_recall', 'faithfulness', 'response_relevancy',
            'factual_correctness', 'overall_score'
        ]].round(3)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "retriever_type": "Method",
                "avg_retrieval_time": "Avg Time (s)",
                "context_precision": "Context Precision",
                "context_recall": "Context Recall",
                "faithfulness": "Faithfulness",
                "response_relevancy": "Response Relevancy", 
                "factual_correctness": "Factual Correctness",
                "overall_score": "Overall Score"
            }
        )
        
        # Recommendations
        best_overall = df.loc[df['overall_score'].idxmax()]
        fastest = df.loc[df['avg_retrieval_time'].idxmin()]
        
        st.subheader("Recommendations")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Best Overall Performance",
                best_overall['retriever_type'],
                f"Score: {best_overall['overall_score']:.3f}"
            )
        
        with col2:
            st.metric(
                "Fastest Method",
                fastest['retriever_type'],
                f"{fastest['avg_retrieval_time']:.3f}s"
            )
        
        with col3:
            most_precise = df.loc[df['context_precision'].idxmax()]
            st.metric(
                "Most Precise",
                most_precise['retriever_type'],
                f"Precision: {most_precise['context_precision']:.3f}"
            )


def render_interactive_comparison(vector_store: K8sVectorStore):
    """Render interactive query comparison interface."""
    st.header("🔍 Interactive Query Comparison")
    
    # Query input
    query = st.text_input(
        "Enter your Kubernetes query:",
        placeholder="e.g., What are the costs of my deployments?",
        help="Ask questions about your Kubernetes cluster"
    )
    
    if query:
        # Method selection for comparison
        methods_to_compare = st.multiselect(
            "Select methods to compare:",
            ["Base", "BM25", "Multi-Query", "Ensemble"],
            default=["Base", "BM25"],
            help="Choose which retrieval methods to compare"
        )
        
        if st.button("Compare Methods", type="primary") and methods_to_compare:
            method_mapping = {
                "Base": (RetrieverType.BASE, {}),
                "BM25": (RetrieverType.BM25, {}),
                "Multi-Query": (RetrieverType.MULTI_QUERY, {"num_queries": 3}),
                "Ensemble": (RetrieverType.ENSEMBLE, {
                    "retrievers": ["base", "bm25"],
                    "weights": [0.6, 0.4]
                })
            }
            
            results = {}
            
            for method_name in methods_to_compare:
                retriever_type, config = method_mapping[method_name]
                
                with st.spinner(f"Processing with {method_name}..."):
                    try:
                        # Create agent with specific retriever
                        agent = EnhancedK8sRAGAgent(
                            vector_store=vector_store,
                            retriever_type=retriever_type,
                            retriever_config=config
                        )
                        
                        # Time the query
                        start_time = time.time()
                        response = agent.invoke(query)
                        response_time = time.time() - start_time
                        
                        results[method_name] = {
                            "response": response,
                            "time": response_time,
                            "retriever_info": agent.get_retriever_info()
                        }
                        
                    except Exception as e:
                        results[method_name] = {
                            "response": f"Error: {e}",
                            "time": 0,
                            "retriever_info": {}
                        }
            
            # Display results
            for method_name, result in results.items():
                with st.expander(f"{method_name} Results (⏱️ {result['time']:.3f}s)", expanded=True):
                    st.write("**Response:**")
                    st.write(result["response"])
                    
                    if result["retriever_info"]:
                        st.write("**Retriever Info:**")
                        st.json(result["retriever_info"])


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="K8s Copilot - Advanced Retrieval",
        page_icon="⚓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚓ Kubernetes Copilot - Advanced Retrieval Methods")
    st.markdown("Compare different retrieval strategies for Kubernetes data analysis")
    
    # Initialize system
    try:
        vector_store = initialize_system()
        
        # Sidebar configuration
        st.sidebar.title("🛠️ Configuration")
        
        # API key configuration
        st.sidebar.subheader("API Keys")
        if st.sidebar.checkbox("Use Cohere API (for compression)"):
            cohere_key = st.sidebar.text_input("Cohere API Key", type="password")
            st.session_state.cohere_api_key = cohere_key
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs([
            "🔍 Interactive Comparison", 
            "📊 Performance Analysis",
            "ℹ️ Method Information"
        ])
        
        with tab1:
            render_interactive_comparison(vector_store)
        
        with tab2:
            render_performance_comparison(vector_store)
        
        with tab3:
            st.header("📚 Retrieval Method Information")
            
            method_info = {
                "Base Vector Search": "Standard semantic similarity search using embeddings. Fast and effective for most queries.",
                "BM25 (Lexical Search)": "Keyword-based search using term frequency and inverse document frequency. Excellent for exact term matching.",
                "Multi-Query Retrieval": "Generates multiple query variants to improve coverage and recall. Better for complex or ambiguous queries.",
                "Parent-Document Retrieval": "Small-to-big strategy: searches small chunks but returns larger parent documents for better context.",
                "Contextual Compression": "Post-processes retrieved documents using reranking to improve relevance and reduce noise.",
                "Ensemble Retrieval": "Combines multiple retrieval methods using rank fusion for balanced performance across query types."
            }
            
            for method, description in method_info.items():
                with st.expander(method):
                    st.write(description)
        
        # System information in sidebar
        stats = vector_store.get_stats()
        st.sidebar.subheader("📊 System Stats")
        st.sidebar.metric("Total Documents", stats['total_documents'])
        
        for doc_type, count in stats['document_types'].items():
            st.sidebar.metric(
                doc_type.replace('_', ' ').title(), 
                count
            )
        
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        st.info("Please ensure all dependencies are installed and the data directory exists.")


if __name__ == "__main__":
    main()
