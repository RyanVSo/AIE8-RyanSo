"""
Streamlit web application for the K8s RAG Copilot.
"""

import os
import sys
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from k8s_rag.vector_db.vector_store import K8sDocVectorStore
from k8s_rag.vector_db.data_loader import K8sDocumentationLoader
from k8s_rag.agents.base_agent import K8sBaseRAGAgent
from k8s_rag.retrieval.advanced_retrievers import K8sAdvancedRetrieverFactory, RetrieverType


# Page configuration
st.set_page_config(
    page_title="K8s RAG Copilot",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def initialize_system():
    """Initialize the vector store and load data (cached for performance)."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please set your OPENAI_API_KEY environment variable")
        st.stop()
    
    # Initialize vector store
    vector_store = K8sDocVectorStore()
    
    # Load documentation
    data_dir = Path(__file__).parent.parent.parent / "data"
    if not data_dir.exists():
        st.error(f"Data directory not found: {data_dir}")
        st.stop()
    
    loader = K8sDocumentationLoader(data_dir)
    
    with st.spinner("Loading Kubernetes documentation..."):
        loader.load_all_data(vector_store)
    
    return vector_store


@st.cache_resource
def create_agent(_vector_store):
    """Create the RAG agent (cached for performance)."""
    return K8sBaseRAGAgent(_vector_store)


@st.cache_resource
def create_retriever_factory(_vector_store):
    """Create the advanced retriever factory (cached for performance)."""
    return K8sAdvancedRetrieverFactory(_vector_store)


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("⚓ Kubernetes RAG Copilot")
    st.markdown("Ask questions about Kubernetes and get answers from the official documentation!")
    
    # Initialize system
    try:
        vector_store = initialize_system()
        agent = create_agent(vector_store)
        retriever_factory = create_retriever_factory(vector_store)
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        st.stop()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Retriever selection
        st.subheader("Retrieval Method")
        retriever_type = st.selectbox(
            "Choose retrieval method:",
            options=[rt.value for rt in RetrieverType],
            index=0,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Number of documents to retrieve
        k_docs = st.slider("Documents to retrieve:", min_value=3, max_value=10, value=5)
        
        # Advanced options for specific retrievers
        if retriever_type == "ensemble":
            st.subheader("Ensemble Configuration")
            ensemble_methods = st.multiselect(
                "Select methods to combine:",
                options=["base", "bm25", "multi_query"],
                default=["base", "bm25", "multi_query"]
            )
        
        # System stats
        st.subheader("📊 System Stats")
        stats = vector_store.get_stats()
        st.metric("Total Documents", stats["total_documents"])
        
        # Document type breakdown
        st.write("**Document Types:**")
        for doc_type, count in stats["document_types"].items():
            st.write(f"- {doc_type.replace('_', ' ').title()}: {count}")
    
    # Main chat interface
    st.header("💬 Chat with K8s Documentation")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Hello! I'm your Kubernetes assistant. Ask me anything about Kubernetes concepts, tasks, or configurations!"
        })
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about Kubernetes..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documentation..."):
                try:
                    # Update agent with selected retriever if needed
                    if retriever_type != "base":
                        # Create the selected retriever
                        kwargs = {}
                        if retriever_type == "ensemble":
                            kwargs = {"retrievers": ensemble_methods}
                        
                        retriever = retriever_factory.create_retriever(
                            RetrieverType(retriever_type), 
                            k=k_docs,
                            **kwargs
                        )
                        agent.update_retriever(retriever)
                    
                    # Get response
                    response = agent.invoke(prompt)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Example queries
    st.header("💡 Example Queries")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Concepts")
        example_queries_concepts = [
            "What is a Kubernetes Pod?",
            "How do Services work in Kubernetes?",
            "What are the components of the control plane?",
            "Explain Kubernetes networking",
        ]
        
        for query in example_queries_concepts:
            st.markdown(f"• `{query}`")
    
    with col2:
        st.subheader("Tasks & Operations")
        example_queries_tasks = [
            "How do I create a Deployment?",
            "How to configure resource limits?",
            "What are liveness and readiness probes?",
            "How to manage secrets in Kubernetes?",
        ]
        
        for query in example_queries_tasks:
            st.markdown(f"• `{query}`")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, LangChain, and OpenAI")


if __name__ == "__main__":
    main()
