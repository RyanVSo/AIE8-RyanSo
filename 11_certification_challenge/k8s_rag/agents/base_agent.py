"""
Base RAG agent for Kubernetes documentation Q&A.
"""

from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from ..vector_db.vector_store import K8sDocVectorStore


class K8sBaseRAGAgent:
    """Base RAG agent for answering Kubernetes-related questions."""
    
    def __init__(self, vector_store: K8sDocVectorStore, model_name: str = "gpt-4o-mini"):
        """Initialize the base RAG agent.
        
        Args:
            vector_store: The vector store containing K8s documentation
            model_name: The LLM model to use
        """
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
        # Create the RAG prompt
        self.prompt = ChatPromptTemplate.from_template("""
You are a helpful Kubernetes expert assistant. Use the provided context from the Kubernetes documentation to answer the user's question accurately and comprehensively.

Context:
{context}

Question: {question}

Instructions:
- Provide accurate information based on the context
- If the context doesn't contain enough information to fully answer the question, say so
- Include relevant examples or code snippets when helpful
- Structure your response clearly with headings or bullet points when appropriate
- Focus on practical, actionable information

Answer:""")
        
        # Create the retriever
        self.retriever = vector_store.get_retriever(k=5)
        
        # Build the RAG chain
        self.chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents for the prompt."""
        formatted_docs = []
        
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get('title', 'Unknown')
            doc_type = doc.metadata.get('doc_type', 'unknown')
            source = doc.metadata.get('source', 'unknown')
            
            formatted_doc = f"""
Document {i} ({doc_type}):
Title: {title}
Source: {source}
Content: {doc.page_content}
"""
            formatted_docs.append(formatted_doc)
        
        return "\n" + "="*50 + "\n".join(formatted_docs)
    
    def invoke(self, question: str) -> str:
        """Answer a question using the RAG system.
        
        Args:
            question: The user's question about Kubernetes
            
        Returns:
            The agent's response
        """
        try:
            response = self.chain.invoke(question)
            return response
        except Exception as e:
            return f"❌ Error processing question: {str(e)}"
    
    def get_relevant_docs(self, question: str, k: int = 5) -> List[Document]:
        """Get relevant documents for a question (useful for evaluation).
        
        Args:
            question: The user's question
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        return self.retriever.invoke(question)
    
    def update_retriever(self, retriever):
        """Update the retriever used by this agent.
        
        Args:
            retriever: New retriever to use
        """
        self.retriever = retriever
        
        # Rebuild the chain with the new retriever
        self.chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
