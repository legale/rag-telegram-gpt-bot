"""ChromaDB implementation of VectorIndex interface."""

from __future__ import annotations

from typing import Dict, List, Optional, Any

from src.core.interfaces import VectorIndex, VectorDoc, ScoredDoc
from src.storage.vector_store import VectorStore


class ChromaVectorIndex:
    """ChromaDB adapter implementing VectorIndex protocol."""

    def __init__(self, vector_store: VectorStore):
        """
        Initialize the adapter with a VectorStore instance.

        Args:
            vector_store: VectorStore instance from src.storage.vector_store
        """
        self.vector_store = vector_store

    def upsert(self, items: List[VectorDoc]) -> None:
        """
        Upsert (insert or update) vector documents.

        Args:
            items: List of VectorDoc objects to upsert
        """
        if not items:
            return

        # Extract data from VectorDoc objects
        ids = [item.id for item in items]
        embeddings = [item.vector for item in items]
        metadatas = [item.meta for item in items]
        
        # ChromaDB expects documents (text), but we may not have them
        # Use empty strings or extract from metadata if available
        documents = []
        for item in items:
            # Try to get text from metadata, otherwise use empty string
            doc_text = item.meta.get("text", "") if isinstance(item.meta, dict) else ""
            documents.append(doc_text)

        # Use VectorStore's add_documents_with_embeddings method
        self.vector_store.add_documents_with_embeddings(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            show_progress=False  # Disable progress for programmatic use
        )

    def query(self, vector: List[float], top_k: int, filter: Optional[Dict] = None) -> List[ScoredDoc]:
        """
        Query the vector index with a pre-computed embedding vector.

        Args:
            vector: Query embedding vector
            top_k: Number of results to return
            filter: Optional metadata filter (ChromaDB where clause)

        Returns:
            List of ScoredDoc objects
        """
        if not vector:
            return []

        # Query ChromaDB collection directly with pre-computed embedding
        # VectorStore.query expects text, so we need to use the collection directly
        collection = self.vector_store.collection
        
        # Convert filter dict to ChromaDB where clause format if provided
        where = None
        if filter:
            where = filter  # ChromaDB uses where clause format

        # Query with the embedding vector
        results = collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances"]
        )

        # Convert ChromaDB results to ScoredDoc objects
        scored_docs = []
        
        # ChromaDB returns results in nested lists (one per query)
        if results and "ids" in results and len(results["ids"]) > 0:
            ids_list = results["ids"][0] if results["ids"] else []
            distances_list = results["distances"][0] if results.get("distances") and results["distances"] else []
            metadatas_list = results["metadatas"][0] if results.get("metadatas") and results["metadatas"] else []

            for i, doc_id in enumerate(ids_list):
                # ChromaDB returns distances (lower is better), convert to score (higher is better)
                # Distance is typically L2 or cosine distance
                distance = distances_list[i] if i < len(distances_list) else 1.0
                
                # Convert distance to similarity score (1.0 - distance for L2, or use distance directly for cosine similarity)
                # For cosine similarity, distance is already 1 - similarity, so score = 1 - distance
                # For L2, we might want to normalize differently
                # Assuming cosine similarity: score = 1 - distance
                score = max(0.0, 1.0 - distance) if distance <= 1.0 else 1.0 / (1.0 + distance)

                metadata = metadatas_list[i] if i < len(metadatas_list) else {}

                scored_docs.append(ScoredDoc(
                    id=doc_id,
                    score=score,
                    meta=metadata
                ))

        return scored_docs

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by their IDs.

        Args:
            ids: List of document IDs to delete
        """
        if not ids:
            return

        # Use ChromaDB collection's delete method
        collection = self.vector_store.collection
        collection.delete(ids=ids)

