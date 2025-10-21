"""
Vector store implementation for Kubernetes documentation using Qdrant.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


class K8sDocVectorStore:
    """Vector store specialized for Kubernetes documentation with semantic search capabilities."""
    
    def __init__(self, embedding_model: Optional[OpenAIEmbeddings] = None, collection_name: str = "k8s_docs"):
        """Initialize the K8s documentation vector store."""
        self.embedding_model = embedding_model or OpenAIEmbeddings(model="text-embedding-3-small")
        self.collection_name = collection_name
        
        # Initialize Qdrant client (in-memory for demo)
        self.client = QdrantClient(":memory:")
        
        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        
        # Initialize vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
        )
        
        self.documents = []
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        if documents:
            # Store documents for later use (needed for BM25 and other retrievers)
            self.documents.extend(documents)
            
            # Add to vector store
            self.vector_store.add_documents(documents)
            
            print(f"✅ Added {len(documents)} documents to vector store")
    
    def search(self, query: str, k: int = 5, filter_type: Optional[str] = None) -> List[Document]:
        """Search for relevant documents."""
        # Build filter if specified
        search_filter = None
        if filter_type:
            search_filter = {"doc_type": filter_type}
        
        # Perform similarity search
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=search_filter
        )
        
        return results
    
    def get_retriever(self, k: int = 5, **kwargs):
        """Get a retriever for this vector store."""
        return self.vector_store.as_retriever(search_kwargs={"k": k, **kwargs})
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        total_docs = len(self.documents)
        
        # Count document types
        doc_types = {}
        for doc in self.documents:
            doc_type = doc.metadata.get('doc_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        return {
            "total_documents": total_docs,
            "document_types": doc_types,
            "collection_name": self.collection_name
        }
    
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        # Recreate collection
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass  # Collection might not exist
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        
        # Reinitialize vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
        )
        
        self.documents = []
        print("🗑️  Cleared vector store")
