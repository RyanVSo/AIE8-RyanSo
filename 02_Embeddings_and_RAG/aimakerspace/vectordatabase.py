import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable, Dict, Any, Optional
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


def euclidean_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the Euclidean distance between two vectors (lower is more similar)."""
    distance = np.linalg.norm(vector_a - vector_b)
    # Convert to similarity (higher is more similar)
    return 1 / (1 + distance)


def manhattan_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the Manhattan distance between two vectors (lower is more similar)."""
    distance = np.sum(np.abs(vector_a - vector_b))
    # Convert to similarity (higher is more similar)
    return 1 / (1 + distance)


def dot_product_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the dot product similarity between two vectors."""
    return np.dot(vector_a, vector_b)


class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.vectors = defaultdict(np.array)
        self.metadata = defaultdict(dict)  # Store metadata for each document
        self.embedding_model = embedding_model or EmbeddingModel()

    def insert(self, key: str, vector: np.array, metadata: Dict[str, Any] = None) -> None:
        self.vectors[key] = vector
        if metadata:
            self.metadata[key] = metadata

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        # Filter by metadata if provided
        candidates = self.vectors.items()
        if filter_metadata:
            candidates = [
                (key, vector) for key, vector in candidates
                if self._matches_filter(key, filter_metadata)
            ]
        
        scores = [
            (key, distance_measure(query_vector, vector))
            for key, vector in candidates
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_metadata: bool = False,
    ) -> List[Tuple[str, float]]:
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure, filter_metadata)
        
        if include_metadata:
            # Return text, score, and metadata
            return [(result[0], result[1], self.metadata.get(result[0], {})) for result in results]
        elif return_as_text:
            return [result[0] for result in results]
        else:
            return results

    def _matches_filter(self, key: str, filter_metadata: Dict[str, Any]) -> bool:
        """Check if a document's metadata matches the filter criteria."""
        doc_metadata = self.metadata.get(key, {})
        for filter_key, filter_value in filter_metadata.items():
            if filter_key not in doc_metadata:
                return False
            if isinstance(filter_value, list):
                if doc_metadata[filter_key] not in filter_value:
                    return False
            elif doc_metadata[filter_key] != filter_value:
                return False
        return True

    def retrieve_from_key(self, key: str) -> np.array:
        return self.vectors.get(key, None)
    
    def get_metadata(self, key: str) -> Dict[str, Any]:
        return self.metadata.get(key, {})

    async def abuild_from_list(self, list_of_text: List[str], metadata_list: List[Dict[str, Any]] = None) -> "VectorDatabase":
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for i, (text, embedding) in enumerate(zip(list_of_text, embeddings)):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
            self.insert(text, np.array(embedding), metadata)
        return self
    
    async def abuild_from_documents_with_metadata(self, documents_with_metadata: List[Tuple[str, Dict[str, Any]]]) -> "VectorDatabase":
        """Build database from documents with metadata."""
        texts = [doc[0] for doc in documents_with_metadata]
        metadata_list = [doc[1] for doc in documents_with_metadata]
        
        embeddings = await self.embedding_model.async_get_embeddings(texts)
        for text, embedding, metadata in zip(texts, embeddings, metadata_list):
            self.insert(text, np.array(embedding), metadata)
        return self

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        if not self.vectors:
            return {"total_documents": 0}
        
        # Group by source type
        type_counts = defaultdict(int)
        source_counts = defaultdict(int)
        
        for key in self.vectors.keys():
            metadata = self.metadata.get(key, {})
            doc_type = metadata.get("type", "unknown")
            source = metadata.get("source", "unknown")
            
            type_counts[doc_type] += 1
            source_counts[source] += 1
        
        return {
            "total_documents": len(self.vectors),
            "documents_by_type": dict(type_counts),
            "documents_by_source": dict(source_counts),
            "embedding_dimension": len(next(iter(self.vectors.values()))),
        }


if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text))
    k = 2

    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s):", searched_vector)

    retrieved_vector = vector_db.retrieve_from_key(
        "I like to eat broccoli and bananas."
    )
    print("Retrieved vector:", retrieved_vector)

    relevant_texts = vector_db.search_by_text(
        "I think fruit is awesome!", k=k, return_as_text=True
    )
    print(f"Closest {k} text(s):", relevant_texts)
