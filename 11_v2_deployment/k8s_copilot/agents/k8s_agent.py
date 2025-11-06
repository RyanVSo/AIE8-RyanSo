"""
Main LangGraph agent for the Kubernetes copilot system.
Based on patterns from the Multi-Agent RAG LangGraph notebook.
"""

import functools
import operator
from typing import Annotated, List, TypedDict, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

from ..vector_db.vector_store import K8sVectorStore
from ..tools.k8s_tools import create_k8s_tools

class K8sAgentState(TypedDict):
    """State for the Kubernetes copilot agent."""
    messages: Annotated[List[BaseMessage], operator.add]
    next: str

class K8sCopilotAgent:
    """Main Kubernetes copilot agent using LangGraph orchestration."""
    
    def __init__(self, vector_store: K8sVectorStore, llm: Optional[ChatOpenAI] = None):
        """Initialize the K8s copilot agent."""
        self.vector_store = vector_store
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Create K8s-specific tools
        self.k8s_tools = create_k8s_tools(vector_store)
        
        # Create the agent graph
        self.graph = self._create_agent_graph()
        self.compiled_graph = self.graph.compile()
    
    def _create_agent_graph(self) -> StateGraph:
        """Create the LangGraph for the K8s copilot agent."""
        
        # Create the main agent node
        def call_model(state: K8sAgentState):
            """Main agent node that calls the LLM with K8s context."""
            messages = state["messages"]
            
            # Create a specialized prompt for Kubernetes operations
            system_prompt = """You are a Kubernetes copilot assistant that helps engineers manage their Kubernetes clusters and deployments. 

You have access to specialized tools for:
- Analyzing Kubernetes manifests and resource configurations
- Retrieving cost information and optimization opportunities  
- Searching through kubectl outputs and cluster data
- Providing actionable recommendations for resource optimization

When users ask questions about their Kubernetes environment, use the appropriate tools to gather information and provide comprehensive, actionable answers. Always explain your reasoning and provide specific recommendations when possible.

For cost-related questions, provide specific dollar amounts and savings opportunities.
For resource questions, include specific CPU, memory, and GPU usage details.
For optimization questions, provide concrete YAML configuration suggestions.

Be concise but thorough in your responses."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ])
            
            # Bind tools to the model
            model_with_tools = self.llm.bind_tools(self.k8s_tools)
            
            # Create the chain
            chain = prompt | model_with_tools
            
            # Invoke with messages
            response = chain.invoke({"messages": messages})
            
            return {"messages": [response]}
        
        # Create tool node
        tool_node = ToolNode(self.k8s_tools)
        
        # Define conditional edge function
        def should_continue(state: K8sAgentState):
            """Determine if we should continue to tools or end."""
            last_message = state["messages"][-1]
            
            # If the LLM makes a tool call, continue to tools
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            
            # Otherwise, end the conversation
            return END
        
        # Create the graph
        graph = StateGraph(K8sAgentState)
        
        # Add nodes
        graph.add_node("agent", call_model)
        graph.add_node("tools", tool_node)
        
        # Add edges
        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                END: END,
            }
        )
        graph.add_edge("tools", "agent")
        
        return graph
    
    def invoke(self, query: str) -> str:
        """Invoke the agent with a user query."""
        initial_state = {
            "messages": [HumanMessage(content=query)]
        }
        
        # Run the graph
        final_state = self.compiled_graph.invoke(initial_state)
        
        # Return the last AI message
        last_message = final_state["messages"][-1]
        if isinstance(last_message, AIMessage):
            return last_message.content
        else:
            return str(last_message)
    
    async def ainvoke(self, query: str) -> str:
        """Async invoke the agent with a user query."""
        initial_state = {
            "messages": [HumanMessage(content=query)]
        }
        
        # Run the graph asynchronously
        final_state = await self.compiled_graph.ainvoke(initial_state)
        
        # Return the last AI message
        last_message = final_state["messages"][-1]
        if isinstance(last_message, AIMessage):
            return last_message.content
        else:
            return str(last_message)
    
    def stream(self, query: str):
        """Stream the agent's response."""
        initial_state = {
            "messages": [HumanMessage(content=query)]
        }
        
        # Stream the graph execution
        for chunk in self.compiled_graph.stream(initial_state, stream_mode="updates"):
            if "__end__" not in chunk:
                yield chunk
    
    def get_graph_visualization(self):
        """Get the graph visualization (if available)."""
        try:
            return self.compiled_graph
        except Exception as e:
            return f"Graph visualization not available: {e}"

class K8sRAGAgent:
    """Simplified RAG agent for basic question-answering."""
    
    def __init__(self, vector_store: K8sVectorStore, llm: Optional[ChatOpenAI] = None):
        """Initialize the RAG agent."""
        self.vector_store = vector_store
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.retriever = vector_store.get_retriever(k=5)
        
        # Create RAG chain
        self.rag_chain = self._create_rag_chain()
    
    def _create_rag_chain(self) -> Runnable:
        """Create a simple RAG chain for Q&A."""
        
        # RAG prompt template
        rag_prompt = ChatPromptTemplate.from_template("""
You are a Kubernetes expert assistant. Use the provided context to answer questions about Kubernetes clusters, deployments, costs, and optimizations.

Context:
{context}

Question: {question}

Provide a comprehensive answer based on the context. If you need more specific information, suggest what additional data might be helpful.

Answer:""")
        
        def format_docs(docs):
            """Format retrieved documents."""
            return "\n\n".join([
                f"Document Type: {doc.metadata.get('type', 'unknown')}\n{doc.page_content}"
                for doc in docs
            ])
        
        # Create the RAG chain
        def get_context(inputs):
            query = inputs["question"]
            docs = self.retriever.invoke(query)
            return format_docs(docs)
        
        chain = (
            {
                "context": get_context,
                "question": lambda x: x["question"]
            }
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def invoke(self, query: str) -> str:
        """Invoke the RAG agent."""
        return self.rag_chain.invoke({"question": query})
    
    async def ainvoke(self, query: str) -> str:
        """Async invoke the RAG agent."""
        return await self.rag_chain.ainvoke({"question": query})
