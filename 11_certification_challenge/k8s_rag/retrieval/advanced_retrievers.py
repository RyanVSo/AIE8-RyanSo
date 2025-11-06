"""
Advanced retrieval methods for Kubernetes documentation.
Based on patterns from 09_Advanced_Retrieval.
"""

import os
from enum import Enum
from typing import List, Optional, Dict, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Advanced retrieval imports
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import (
    MultiQueryRetriever,
    ParentDocumentRetriever, 
    ContextualCompressionRetriever,
    EnsembleRetriever
)

# Try to import Cohere reranker
try:
    from langchain_cohere import CohereRerank
    COHERE_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.document_compressors import CohereRerank
        COHERE_AVAILABLE = True
    except ImportError:
        COHERE_AVAILABLE = False
        CohereRerank = None

from ..vector_db.vector_store import K8sDocVectorStore


class RetrieverType(Enum):
    """Available retriever types."""
    BASE = "base"
    BM25 = "bm25" 
    MULTI_QUERY = "multi_query"
    # PARENT_DOCUMENT = "parent_document"  # Removed for simplicity
    # CONTEXTUAL_COMPRESSION = "contextual_compression"  # Removed for simplicity
    ENSEMBLE = "ensemble"


class K8sAdvancedRetrieverFactory:
    """Factory for creating different types of retrievers for K8s documentation."""
    
    def __init__(self, vector_store: K8sDocVectorStore):
        """Initialize the retriever factory.
        
        Args:
            vector_store: The K8s vector store containing the documentation
        """
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Cache for created retrievers
        self._retriever_cache: Dict[str, BaseRetriever] = {}
    
    def create_retriever(self, 
                        retriever_type: RetrieverType, 
                        k: int = 5,
                        **kwargs) -> BaseRetriever:
        """Create a retriever of the specified type.
        
        Args:
            retriever_type: Type of retriever to create
            k: Number of documents to retrieve
            **kwargs: Additional configuration for specific retrievers
            
        Returns:
            Configured retriever instance
        """
        cache_key = f"{retriever_type.value}_{k}_{hash(str(sorted(kwargs.items())))}"
        
        if cache_key in self._retriever_cache:
            return self._retriever_cache[cache_key]
        
        if retriever_type == RetrieverType.BASE:
            retriever = self._create_base_retriever(k)
        elif retriever_type == RetrieverType.BM25:
            retriever = self._create_bm25_retriever(k, **kwargs)
        elif retriever_type == RetrieverType.MULTI_QUERY:
            retriever = self._create_multi_query_retriever(k, **kwargs)
        elif retriever_type == RetrieverType.PARENT_DOCUMENT:
            retriever = self._create_parent_document_retriever(k, **kwargs)
        elif retriever_type == RetrieverType.CONTEXTUAL_COMPRESSION:
            retriever = self._create_contextual_compression_retriever(k, **kwargs)
        elif retriever_type == RetrieverType.ENSEMBLE:
            retriever = self._create_ensemble_retriever(k, **kwargs)
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
        
        self._retriever_cache[cache_key] = retriever
        return retriever
    
    def _create_base_retriever(self, k: int) -> BaseRetriever:
        """Create a basic vector store retriever."""
        return self.vector_store.get_retriever(k=k)
    
    def _create_bm25_retriever(self, k: int, **kwargs) -> BaseRetriever:
        """Create a BM25 retriever for lexical search."""
        # Get all documents for BM25 indexing
        documents = self.vector_store.documents
        
        if not documents:
            raise ValueError("No documents available for BM25 indexing")
        
        # Configure BM25 parameters
        bm25_params = kwargs.get("bm25_params", {})
        
        retriever = BM25Retriever.from_documents(
            documents=documents,
            k=k,
            **bm25_params
        )
        
        return retriever
    
    def _create_multi_query_retriever(self, k: int, **kwargs) -> BaseRetriever:
        """Create a multi-query retriever that generates multiple query variants."""
        base_retriever = self.vector_store.get_retriever(k=k)
        
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm,
            include_original=True
        )
        
        return retriever
    
    def _create_parent_document_retriever(self, k: int, **kwargs) -> BaseRetriever:
        """Create a parent-document retriever (small-to-big strategy)."""
        # Configure text splitters
        parent_chunk_size = kwargs.get("parent_chunk_size", 2000)
        child_chunk_size = kwargs.get("child_chunk_size", 400)
        chunk_overlap = kwargs.get("chunk_overlap", 50)
        
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        
        # Create a new vector store for child documents
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams
        
        client = QdrantClient(":memory:")
        collection_name = f"k8s_parent_doc_{hash(str(kwargs))}"
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        
        child_vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )
        
        # Create document store for parents
        docstore = InMemoryStore()
        
        retriever = ParentDocumentRetriever(
            vectorstore=child_vectorstore,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            k=k
        )
        
        # Add documents to the retriever
        retriever.add_documents(self.vector_store.documents)
        
        return retriever
    
    def _create_contextual_compression_retriever(self, k: int, **kwargs) -> BaseRetriever:
        """Create a contextual compression retriever with reranking."""
        base_retriever = self.vector_store.get_retriever(k=k*2)  # Get more docs to rerank
        
        # Configure compression parameters
        top_k = kwargs.get("top_k", k)
        
        # Use Cohere reranker if available and API key is set
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if COHERE_AVAILABLE and cohere_api_key and cohere_api_key != "INSERT COHERE API KEY":
            try:
                compressor = CohereRerank(
                    cohere_api_key=cohere_api_key,
                    top_k=top_k
                )
            except Exception as e:
                print(f"Warning: Failed to initialize CohereRerank: {e}")
                compressor = self._create_llm_chain_filter()
        else:
            # Fallback to LLM-based relevance filter
            compressor = self._create_llm_chain_filter()
        
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        return retriever
    
    def _create_llm_chain_filter(self):
        """Create an LLM-based document filter as fallback for Cohere."""
        from langchain.retrievers.document_compressors import LLMChainFilter
        from langchain.chains import LLMChain
        from langchain_core.prompts import PromptTemplate
        
        prompt = PromptTemplate(
            template="""Given the following question about Kubernetes and a piece of documentation context, determine if the context is relevant to answering the question.
            
Question: {question}
Context: {context}

Is this context relevant for answering the question about Kubernetes? Answer 'YES' if relevant, 'NO' if not relevant.
Answer:""",
            input_variables=["question", "context"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return LLMChainFilter.from_llm(chain)
    
    def _create_ensemble_retriever(self, k: int, **kwargs) -> BaseRetriever:
        """Create an ensemble retriever combining multiple retrieval strategies."""
        # Configure ensemble parameters
        retrievers_config = kwargs.get("retrievers", ["base", "bm25", "multi_query"])
        weights = kwargs.get("weights", None)
        
        # Create individual retrievers
        retrievers = []
        for retriever_name in retrievers_config:
            if retriever_name == "base":
                retrievers.append(self._create_base_retriever(k))
            elif retriever_name == "bm25":
                retrievers.append(self._create_bm25_retriever(k))
            elif retriever_name == "multi_query":
                retrievers.append(self._create_multi_query_retriever(k))
            elif retriever_name == "parent_document":
                retrievers.append(self._create_parent_document_retriever(k))
            elif retriever_name == "contextual_compression":
                retrievers.append(self._create_contextual_compression_retriever(k))
        
        # Set default equal weights if not provided
        if weights is None:
            weights = [1.0 / len(retrievers)] * len(retrievers)
        
        retriever = EnsembleRetriever(
            retrievers=retrievers,
            weights=weights
        )
        
        return retriever
    
    def get_available_retrievers(self) -> List[str]:
        """Get list of available retriever types."""
        return [rt.value for rt in RetrieverType]
    
    def clear_cache(self):
        """Clear the retriever cache."""
        self._retriever_cache.clear()
