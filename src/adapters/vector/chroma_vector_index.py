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
        where = self._convert_filter_to_where(filter)

        # Query with the embedding vector
        # Include documents to avoid reading from SQLite for each candidate
        results = collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances", "documents"]
        )

        # Convert ChromaDB results to ScoredDoc objects
        return self._convert_results_to_scored_docs(results)

    def _convert_filter_to_where(self, filter: Optional[Dict]) -> Optional[Dict]:
        """
        Convert filter dict to ChromaDB where clause format.

        Args:
            filter: Optional metadata filter dict

        Returns:
            ChromaDB where clause dict or None
        """
        if filter:
            return filter  # ChromaDB uses where clause format
        return None

    def _convert_results_to_scored_docs(self, results: Dict[str, Any]) -> List[ScoredDoc]:
        """
        Convert ChromaDB query results to ScoredDoc objects.

        Args:
            results: ChromaDB query results dict

        Returns:
            List of ScoredDoc objects with chunk_id, score, and metadata (including text)
            to avoid reading chunks from SQLite for each candidate
        """
        scored_docs = []
        
        # ChromaDB returns results in nested lists (one per query)
        if results and "ids" in results and len(results["ids"]) > 0:
            ids_list = results["ids"][0] if results["ids"] else []
            distances_list = results["distances"][0] if results.get("distances") and results["distances"] else []
            metadatas_list = results["metadatas"][0] if results.get("metadatas") and results["metadatas"] else []
            documents_list = results.get("documents", [[]])[0] if results.get("documents") and results["documents"] else []

            for i, doc_id in enumerate(ids_list):
                # ChromaDB returns distances (lower is better), convert to score (higher is better)
                # We use cosine similarity: distance = 1 - cosine_similarity
                # So: cosine_similarity = 1 - distance, and score = cosine_similarity
                distance = distances_list[i] if i < len(distances_list) else 1.0
                
                # For cosine similarity: distance = 1 - similarity, so similarity = 1 - distance
                # Clamp to [0, 1] range
                score = max(0.0, min(1.0, 1.0 - distance))

                # Get metadata and include text from documents to minimize SQLite roundtrips
                metadata = metadatas_list[i].copy() if i < len(metadatas_list) and metadatas_list[i] else {}
                
                # Ensure text is present in metadata to avoid SQLite roundtrips.
                # ChromaDB stores text in the "documents" field; for legacy data it can be empty.
                if "text" not in metadata:
                    document_text = documents_list[i] if i < len(documents_list) else ""
                    metadata["text"] = document_text or ""

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

    def count(self) -> int:
        """
        Get total number of documents in the index.

        Returns:
            Number of documents
        """
        return self.vector_store.count()

    def _fetch_embeddings(self, ids: List[str]) -> Dict[str, Any]:
        """
        Fetch embeddings from collection by document IDs.
        
        Args:
            ids: List of document IDs
            
        Returns:
            Dictionary with 'ids' and 'embeddings' keys
        """
        return self.vector_store.get_embeddings_by_ids(ids)
    
    def _convert_to_dict(self, result: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Convert embeddings result to dictionary format.
        
        Args:
            result: Dictionary with 'ids' and 'embeddings' keys
            
        Returns:
            Dictionary mapping document ID to embedding vector
        """
        embeddings_dict = {}
        result_ids = result.get("ids", [])
        result_embeddings = result.get("embeddings", [])
        
        for i, doc_id in enumerate(result_ids):
            if i < len(result_embeddings):
                embeddings_dict[doc_id] = result_embeddings[i]
        
        return embeddings_dict

    def get_embeddings_by_ids(self, ids: List[str]) -> Dict[str, List[float]]:
        """
        Get embeddings by document IDs.

        Args:
            ids: List of document IDs

        Returns:
            Dictionary mapping document ID to embedding vector
        """
        if not ids:
            return {}

        # Fetch embeddings from collection
        result = self._fetch_embeddings(ids)
        
        # Convert to dict format: {id: embedding}
        return self._convert_to_dict(result)
