"""ChromaDB implementation of TopicIndex interface for topic collections."""

from __future__ import annotations

from typing import List

from src.core.interfaces import TopicIndex, VectorDoc, ScoredDoc
from src.storage.vector_store import VectorStore


class ChromaTopicIndex:
    """ChromaDB adapter implementing TopicIndex protocol for topic collections."""

    def __init__(self, collection):
        """
        Initialize the adapter with a ChromaDB collection.

        Args:
            collection: ChromaDB Collection instance for topics (L1 or L2)
        """
        self.collection = collection

    def query(self, vector: List[float], top_k: int) -> List[ScoredDoc]:
        """
        Query topics by embedding vector.

        Args:
            vector: Query embedding vector
            top_k: Number of top topics to return

        Returns:
            List of ScoredDoc objects with topic IDs and similarity scores
        """
        if not vector:
            return []

        # Query ChromaDB collection
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            include=["metadatas", "distances"]
        )

        # Convert ChromaDB results to ScoredDoc objects
        scored_docs = []
        
        if results and "ids" in results and len(results["ids"]) > 0:
            ids_list = results["ids"][0] if results["ids"] else []
            distances_list = results["distances"][0] if results.get("distances") and results["distances"] else []
            metadatas_list = results["metadatas"][0] if results.get("metadatas") and results["metadatas"] else []

            for i, doc_id in enumerate(ids_list):
                distance = distances_list[i] if i < len(distances_list) else 1.0
                # Convert distance to similarity score
                score = max(0.0, 1.0 - distance) if distance <= 1.0 else 1.0 / (1.0 + distance)
                metadata = metadatas_list[i] if i < len(metadatas_list) else {}

                scored_docs.append(ScoredDoc(
                    id=doc_id,
                    score=score,
                    meta=metadata
                ))

        return scored_docs

    def get_all(self) -> List[VectorDoc]:
        """
        Get all topics with their embeddings.

        Returns:
            List of VectorDoc objects representing all topics
        """
        # Get all topics from collection
        result = self.collection.get(include=["embeddings", "metadatas"])

        if not result or not result.get("ids"):
            return []

        ids = result["ids"]
        embeddings = result.get("embeddings", [])
        metadatas = result.get("metadatas", [])

        vector_docs = []
        for i, topic_id in enumerate(ids):
            embedding = embeddings[i] if i < len(embeddings) else []
            metadata = metadatas[i] if i < len(metadatas) else {}

            vector_docs.append(VectorDoc(
                id=topic_id,
                vector=embedding,
                meta=metadata
            ))

        return vector_docs

    def count(self) -> int:
        """
        Get total number of topics in the index.

        Returns:
            Number of topics
        """
        return self.collection.count()

