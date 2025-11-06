"""
Enhanced agent classes with configurable retrieval methods.

This module provides enhanced versions of the K8s agents that can use
different retrieval strategies for performance comparison and optimization.
"""

from typing import Optional
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from .k8s_agent import K8sRAGAgent
from ..vector_db.vector_store import K8sVectorStore
from ..retrieval.retriever_factory import K8sRetrieverFactory, RetrieverType


class EnhancedK8sRAGAgent(K8sRAGAgent):
    """Enhanced RAG agent with configurable retrieval methods."""
    
    def __init__(self, 
                 vector_store: K8sVectorStore,
                 retriever_type: RetrieverType = RetrieverType.BASE,
                 retriever_config: Optional[dict] = None,
                 llm: Optional[ChatOpenAI] = None):
        """Initialize the enhanced RAG agent.
        
        Args:
            vector_store: The K8s vector store
            retriever_type: Type of retriever to use
            retriever_config: Configuration for the retriever
            llm: Optional LLM instance
        """
        self.vector_store = vector_store
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.retriever_type = retriever_type
        self.retriever_config = retriever_config or {}
        
        # Create the retriever using the factory
        self.retriever_factory = K8sRetrieverFactory(vector_store)
        self.retriever = self.retriever_factory.create_retriever(
            retriever_type, **self.retriever_config
        )
        
        # Create RAG chain
        self.rag_chain = self._create_rag_chain()
    
    def get_retriever_info(self) -> dict:
        """Get information about the current retriever.
        
        Returns:
            Dictionary with retriever information
        """
        return {
            "type": self.retriever_type.value,
            "config": self.retriever_config,
            "class": self.retriever.__class__.__name__
        }
    
    def switch_retriever(self, 
                        retriever_type: RetrieverType,
                        retriever_config: Optional[dict] = None):
        """Switch to a different retriever type.
        
        Args:
            retriever_type: New retriever type
            retriever_config: Configuration for the new retriever
        """
        self.retriever_type = retriever_type
        self.retriever_config = retriever_config or {}
        
        # Create new retriever
        self.retriever = self.retriever_factory.create_retriever(
            retriever_type, **self.retriever_config
        )
        
        # Recreate RAG chain with new retriever
        self.rag_chain = self._create_rag_chain()


class ConfigurableK8sRAGAgent:
    """RAG agent that can be configured with any retriever instance."""
    
    def __init__(self, 
                 retriever: BaseRetriever,
                 llm: Optional[ChatOpenAI] = None):
        """Initialize with a specific retriever instance.
        
        Args:
            retriever: The retriever to use
            llm: Optional LLM instance
        """
        self.retriever = retriever
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.rag_chain = self._create_rag_chain()
    
    def _create_rag_chain(self) -> Runnable:
        """Create a RAG chain for Q&A."""
        
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
