"""
Streamlit web application for the Kubernetes Copilot.
Provides an interactive interface for querying Kubernetes data.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import os
from pathlib import Path
from typing import Optional

# Add the parent directory to the path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from k8s_copilot.vector_db.vector_store import K8sVectorStore
from k8s_copilot.vector_db.data_loader import K8sDataLoader
from k8s_copilot.agents.k8s_agent import K8sCopilotAgent, K8sRAGAgent

def initialize_system() -> tuple[K8sVectorStore, K8sCopilotAgent, K8sRAGAgent]:
    """Initialize the K8s copilot system."""
    
    # Set hardcoded API keys for demo purposes
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "INSERT API KEY"
    
    if not os.getenv("COHERE_API_KEY"):
        os.environ["COHERE_API_KEY"] = "INSERT COHERE API KEY "
    
    # Initialize vector store
    vector_store = K8sVectorStore()
    
    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    data_loader = K8sDataLoader(data_dir)
    
    with st.spinner("Loading Kubernetes data..."):
        data_loader.load_all_data(vector_store)
    
    # Initialize agents
    copilot_agent = K8sCopilotAgent(vector_store)
    rag_agent = K8sRAGAgent(vector_store)
    
    return vector_store, copilot_agent, rag_agent

def render_sidebar(vector_store: K8sVectorStore):
    """Render the sidebar with system information."""
    st.sidebar.header("🚀 Kubernetes Copilot")
    
    # System stats
    stats = vector_store.get_stats()
    st.sidebar.subheader("System Status")
    st.sidebar.metric("Total Documents", stats["total_documents"])
    
    # Document type breakdown
    st.sidebar.subheader("Data Types")
    for doc_type, count in stats["document_types"].items():
        st.sidebar.write(f"• {doc_type.replace('_', ' ').title()}: {count}")
    
    # Example queries
    st.sidebar.subheader("💡 Example Queries")
    st.sidebar.markdown("*Copy and paste these into the query box:*")
    example_queries = [
        "What are the costs of my Kubernetes deployments?",
        "How many GPUs does the ml-training deployment use?",
        "How can I improve resource utilization?",
        "Which deployments are using the most memory?",
        "Show me optimization opportunities",
        "What's the total cluster cost?",
        "Analyze the nginx-deployment resources"
    ]
    
    for query in example_queries:
        st.sidebar.markdown(f"• {query}")

def render_cost_dashboard(vector_store: K8sVectorStore):
    """Render cost analysis dashboard."""
    st.subheader("💰 Cost Analysis Dashboard")
    
    # Load cost data for visualization
    data_dir = Path(__file__).parent.parent / "data"
    
    try:
        # Load deployment costs
        deployment_costs = pd.read_csv(data_dir / "deployment_costs.csv")
        
        # Create cost visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Total cost by deployment
            total_costs = deployment_costs.groupby('deployment')['total_cost'].sum().reset_index()
            fig_costs = px.bar(
                total_costs, 
                x='deployment', 
                y='total_cost',
                title='Total Cost by Deployment (30 days)',
                labels={'total_cost': 'Total Cost ($)', 'deployment': 'Deployment'}
            )
            fig_costs.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_costs, use_container_width=True)
        
        with col2:
            # Cost breakdown by type
            cost_breakdown = deployment_costs.groupby('deployment')[['cpu_cost', 'memory_cost', 'storage_cost', 'network_cost']].sum()
            
            # Create stacked bar chart
            fig_breakdown = go.Figure()
            
            for cost_type in ['cpu_cost', 'memory_cost', 'storage_cost', 'network_cost']:
                fig_breakdown.add_trace(go.Bar(
                    name=cost_type.replace('_', ' ').title(),
                    x=cost_breakdown.index,
                    y=cost_breakdown[cost_type]
                ))
            
            fig_breakdown.update_layout(
                barmode='stack',
                title='Cost Breakdown by Type',
                xaxis_title='Deployment',
                yaxis_title='Cost ($)',
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig_breakdown, use_container_width=True)
        
        # Cost trends over time
        st.subheader("Cost Trends")
        
        # Select deployment for trend analysis
        selected_deployment = st.selectbox(
            "Select deployment for trend analysis:",
            deployment_costs['deployment'].unique()
        )
        
        deployment_trend = deployment_costs[deployment_costs['deployment'] == selected_deployment].copy()
        deployment_trend['date'] = pd.to_datetime(deployment_trend['date'])
        
        fig_trend = px.line(
            deployment_trend,
            x='date',
            y='total_cost',
            title=f'Cost Trend for {selected_deployment}',
            labels={'total_cost': 'Daily Cost ($)', 'date': 'Date'}
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("Cost data not available. Please ensure the data has been generated.")

def render_resource_dashboard(vector_store: K8sVectorStore):
    """Render resource utilization dashboard."""
    st.subheader("📊 Resource Utilization")
    
    # Search for resource efficiency data
    efficiency_results = vector_store.search("resource efficiency utilization", k=1, filter_type="resource_efficiency")
    
    if efficiency_results:
        efficiency_doc = efficiency_results[0]
        lines = efficiency_doc.page_content.split('\n')
        
        metrics = {}
        for line in lines:
            if ':' in line and 'Utilization' in line:
                key = line.split(':')[0].strip()
                value = line.split(':')[1].strip()
                metrics[key] = value
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cpu_util = metrics.get('CPU Utilization', 'Unknown')
            st.metric("CPU Utilization", cpu_util)
        
        with col2:
            memory_util = metrics.get('Memory Utilization', 'Unknown')
            st.metric("Memory Utilization", memory_util)
        
        with col3:
            storage_util = metrics.get('Storage Utilization', 'Unknown')
            st.metric("Storage Utilization", storage_util)
        
        with col4:
            network_util = metrics.get('Network Utilization', 'Unknown')
            st.metric("Network Utilization", network_util)
    
    # Cluster overview
    overview_results = vector_store.search("cluster overview total nodes pods", k=1, filter_type="cluster_overview")
    
    if overview_results:
        st.subheader("Cluster Overview")
        overview_doc = overview_results[0]
        
        # Parse overview data
        lines = overview_doc.page_content.split('\n')
        overview_data = {}
        
        for line in lines:
            if ':' in line:
                key = line.split(':')[0].strip()
                value = line.split(':')[1].strip()
                overview_data[key] = value
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Nodes", overview_data.get('Total Nodes', 'Unknown'))
            st.metric("Total Pods", overview_data.get('Total Pods', 'Unknown'))
        
        with col2:
            st.metric("Total Deployments", overview_data.get('Total Deployments', 'Unknown'))
            st.metric("Total Services", overview_data.get('Total Services', 'Unknown'))
        
        with col3:
            st.metric("Total GPUs", overview_data.get('Total GPUs', 'Unknown'))
            st.metric("GPU Utilization", overview_data.get('GPU Utilization', 'Unknown'))

def render_chat_interface(copilot_agent: K8sCopilotAgent, rag_agent: K8sRAGAgent):
    """Render the main chat interface."""
    st.subheader("💬 Ask Your Kubernetes Copilot")
    
    # Agent selection
    agent_type = st.radio(
        "Select Agent Type:",
        ["🤖 Copilot Agent (with tools)", "📚 RAG Agent (simple Q&A)"],
        horizontal=True
    )
    
    # Query input - use key to properly handle session state
    query = st.text_input(
        "Ask a question about your Kubernetes cluster:",
        key="main_query_input",
        placeholder="e.g., What are the costs of my Kubernetes deployments?"
    )
    
    # No more session state handling needed for example queries
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Debug info
    if st.checkbox("Show Debug Info", key="debug_checkbox"):
        st.write(f"Query: '{query}'")
        st.write(f"Query length: {len(query) if query else 0}")
        st.write(f"Agent type: {agent_type}")
        st.write(f"Chat history length: {len(st.session_state.chat_history)}")
        
        # Manual test button
        if st.button("🧪 Test with Sample Query", key="test_button"):
            test_query = "What deployments are in my cluster?"
            st.info(f"🔄 Testing with: '{test_query}'")
            
            try:
                if "Copilot Agent" in agent_type:
                    response = copilot_agent.invoke(test_query)
                else:
                    response = rag_agent.invoke(test_query)
                
                st.session_state.chat_history.append({
                    "query": test_query,
                    "response": response,
                    "agent": agent_type
                })
                
                st.success("✅ Test query successful!")
                st.info(f"📝 Response: {response[:100]}...")
                
            except Exception as e:
                st.error(f"❌ Test failed: {e}")
    
    # Ask button with better handling
    ask_button_clicked = st.button("Ask", type="primary")
    
    if ask_button_clicked:
        if not query or query.strip() == "":
            st.error("⚠️ Please enter a question before clicking Ask!")
        else:
            # Debug: Show that button was clicked
            st.info(f"🔄 Processing query: '{query.strip()}' with {agent_type}")
            
            with st.spinner("Thinking..."):
                try:
                    query_to_process = query.strip()
                    
                    if "Copilot Agent" in agent_type:
                        response = copilot_agent.invoke(query_to_process)
                    else:
                        response = rag_agent.invoke(query_to_process)
                    
                    # Debug: Show response info
                    st.info(f"📝 Generated response ({len(response)} characters)")
                    
                    # Store in chat history
                    st.session_state.chat_history.append({
                        "query": query_to_process,
                        "response": response,
                        "agent": agent_type
                    })
                    
                    # Show success message
                    st.success("✅ Response generated successfully!")
                    
                except Exception as e:
                    error_msg = str(e)
                    if "invalid_api_key" in error_msg or "401" in error_msg:
                        st.error("🔑 **Invalid OpenAI API Key**")
                        st.markdown("""
                        The OpenAI API key appears to be invalid or expired. Please:
                        
                        1. Check your API key at https://platform.openai.com/account/api-keys
                        2. Update the hardcoded key in the code, or
                        3. Set a valid environment variable: `export OPENAI_API_KEY="your-valid-key"`
                        """)
                    elif "rate_limit" in error_msg.lower():
                        st.error("⏰ **Rate Limit Exceeded**")
                        st.write("Please wait a moment and try again.")
                    else:
                        st.error(f"**Error:** {error_msg}")
                        st.markdown("Please check your configuration and try again.")
                    
                    with st.expander("Full Error Details"):
                        st.code(error_msg)
    
    # Permanent response box
    st.subheader("💬 Latest Response")
    
    if st.session_state.chat_history:
        latest_chat = st.session_state.chat_history[-1]
        
        # Show response info
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Question:** {latest_chat['query']}")
        with col2:
            st.markdown(f"**Agent:** {latest_chat['agent']}")
        
        # Show the response in an expandable code block for better formatting
        with st.expander("📄 Full Response", expanded=True):
            st.write(latest_chat['response'])
        
        # Also show in a scrollable text area
        st.text_area(
            "Scrollable Response:",
            value=latest_chat['response'],
            height=200,
            key=f"response_{len(st.session_state.chat_history)}"  # Use length as key to ensure updates
        )
        
        st.divider()
    else:
        # Empty state
        st.info("💡 Ask a question above or use the example queries to get started!")
        st.empty()  # Placeholder for future responses
    
    # Chat history (show older conversations)
    if len(st.session_state.chat_history) > 1:
        st.subheader("📝 Previous Conversations")
        
        # Show all but the most recent (since we show that above)
        older_chats = st.session_state.chat_history[:-1]
        
        for i, chat in enumerate(reversed(older_chats[-4:])):  # Show last 4 older chats
            with st.expander(f"Q: {chat['query'][:50]}..." if len(chat['query']) > 50 else f"Q: {chat['query']}"):
                st.write(f"**Agent:** {chat['agent']}")
                st.write(f"**Question:** {chat['query']}")
                st.write(f"**Answer:** {chat['response']}")
    
    # Show helpful message if no chat history
    elif not st.session_state.chat_history:
        st.subheader("💡 Getting Started")
        st.markdown("""
        **Try asking questions like:**
        - What are the costs of my Kubernetes deployments?
        - How many GPUs does the ml-training deployment use?
        - How can I improve resource utilization?
        - Which deployments are using the most memory?
        - Show me optimization opportunities
        
        *Copy any of these questions into the text box above and click Ask!*
        """)

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Kubernetes Copilot",
        page_icon="⚓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚓ Kubernetes Copilot")
    st.markdown("*Your intelligent assistant for Kubernetes cluster management*")
    
    # Set hardcoded OpenAI API key for demo purposes
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "INSERT API KEY"
    
    # Initialize system (with caching)
    @st.cache_resource
    def get_system():
        return initialize_system()
    
    try:
        vector_store, copilot_agent, rag_agent = get_system()
        
        # Render sidebar
        render_sidebar(vector_store)
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "💰 Cost Analysis", "📊 Resources"])
        
        with tab1:
            render_chat_interface(copilot_agent, rag_agent)
        
        with tab2:
            render_cost_dashboard(vector_store)
        
        with tab3:
            render_resource_dashboard(vector_store)
        
        # Footer
        st.markdown("---")
        st.markdown("Built with ❤️ using LangChain, LangGraph, and Streamlit")
        
    except ValueError as e:
        if "OpenAI API key" in str(e):
            st.error("🔑 **OpenAI API Key Error**")
            st.markdown("""
            The OpenAI API key is missing or invalid. Please:
            
            1. Set your API key: `export OPENAI_API_KEY="your-key-here"`
            2. Restart the application: `python run_ui.py`
            """)
        else:
            st.error(f"Configuration Error: {str(e)}")
    except Exception as e:
        st.error(f"**Failed to initialize system:** {str(e)}")
        st.markdown("""
        **Troubleshooting steps:**
        1. Ensure OpenAI API key is set: `export OPENAI_API_KEY="your-key"`
        2. Verify data has been generated: `python k8s_copilot/data/generate_mock_data.py`
        3. Check dependencies are installed: `pip install -r requirements.txt`
        
        **Error details:**
        """)
        st.code(str(e))
        
        with st.expander("Full Error Traceback"):
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
