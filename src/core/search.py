"""Hybrid search use case - clean search logic using core interfaces."""

from __future__ import annotations

from typing import List, Optional
from datetime import datetime, timedelta

from src.core.domain import SearchResult, Chunk, Message
from src.core.interfaces import Embedder, VectorIndex, ChunkStore, MessageStore, ScoredDoc


class HybridSearch:
    """
    Hybrid search use case that combines vector search with message context enrichment.
    
    This use case depends only on core interfaces, making it testable and framework-agnostic.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_index: VectorIndex,
        chunk_store: ChunkStore,
        message_store: MessageStore,
    ):
        """
        Initialize HybridSearch use case.

        Args:
            embedder: Interface for computing query embeddings
            vector_index: Interface for vector similarity search
            chunk_store: Interface for retrieving chunks by IDs
            message_store: Interface for retrieving message context
        """
        self.embedder = embedder
        self.vector_index = vector_index
        self.chunk_store = chunk_store
        self.message_store = message_store

    def _compute_query_embedding(self, query: str) -> List[float]:
        """
        Compute embedding vector for the query.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector
        """
        return self.embedder.embed_query(query)
    
    def _perform_vector_search(
        self,
        query_vector: List[float],
        top_k: int,
        chat_id: Optional[str] = None
    ) -> List[ScoredDoc]:
        """
        Perform vector similarity search.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of top results to return
            chat_id: Optional chat ID filter
            
        Returns:
            List of ScoredDoc objects with chunk IDs and similarity scores
        """
        vector_filter = None
        if chat_id:
            vector_filter = {"chat_id": chat_id}
        
        return self.vector_index.query(
            vector=query_vector,
            top_k=top_k,
            filter=vector_filter
        )
    
    def _enrich_with_messages(
        self,
        chunk: Chunk,
        enrich_with_messages: bool,
        message_window_sec: int
    ) -> List[Message]:
        """
        Enrich chunk with original messages from the time period.
        
        Args:
            chunk: Chunk to enrich
            enrich_with_messages: Whether to enrich with messages
            message_window_sec: Time window in seconds for message context retrieval
            
        Returns:
            List of Message objects from the time window
        """
        original_messages: List[Message] = []
        if enrich_with_messages and chunk.valid_period:
            # Get messages around the chunk's time period
            time_point = chunk.valid_period[0]
            chat_id = chunk.metadata.get("chat_id") if chunk.metadata else None
            
            if chat_id:
                # Retrieve messages in the time window
                messages = self.message_store.get_context(
                    chat_id=chat_id,
                    time_point=time_point,
                    window_sec=message_window_sec
                )
                original_messages = messages
        
        return original_messages
    
    def _filter_by_threshold(
        self,
        scored_docs: List[ScoredDoc],
        threshold: Optional[float]
    ) -> List[ScoredDoc]:
        """
        Filter scored documents by threshold if provided.
        
        Args:
            scored_docs: List of ScoredDoc objects
            threshold: Minimum similarity score (0.0-1.0), None to skip filtering
            
        Returns:
            Filtered list of ScoredDoc objects
        """
        if threshold is not None:
            return [doc for doc in scored_docs if doc.score >= threshold]
        return scored_docs
    
    def _get_chunks_by_ids(
        self,
        chunk_ids: List[str]
    ) -> dict[str, Chunk]:
        """
        Retrieve chunks by IDs and create a mapping for efficient lookup.
        
        Args:
            chunk_ids: List of chunk IDs to retrieve
            
        Returns:
            Dictionary mapping chunk_id to Chunk object
        """
        chunks = self.chunk_store.get_by_ids(chunk_ids)
        return {chunk.id: chunk for chunk in chunks}
    
    def _build_search_results(
        self,
        scored_docs: List[ScoredDoc],
        chunk_map: dict[str, Chunk],
        enrich_with_messages: bool,
        message_window_sec: int
    ) -> List[SearchResult]:
        """
        Build SearchResult objects from scored documents and chunks.
        
        Args:
            scored_docs: List of ScoredDoc objects with chunk IDs and scores
            chunk_map: Dictionary mapping chunk_id to Chunk object
            enrich_with_messages: Whether to enrich results with original messages
            message_window_sec: Time window in seconds for message context retrieval
            
        Returns:
            List of SearchResult objects, sorted by score (descending)
        """
        results = []
        for scored_doc in scored_docs:
            chunk = chunk_map.get(scored_doc.id)
            if not chunk:
                # Chunk not found in store, skip
                continue

            # Extract topics from chunk metadata
            topics = []
            if chunk.metadata:
                # Look for topic information in metadata
                # topic_l1 and topic_l2 removed - clustering is deprecated
                
                # Also check for topic_ids list
                topic_ids = chunk.metadata.get("topic_ids", [])
                topics.extend([str(tid) for tid in topic_ids if tid not in topics])

            # Enrich with original messages if requested
            original_messages = self._enrich_with_messages(
                chunk,
                enrich_with_messages,
                message_window_sec
            )

            # Create SearchResult
            result = SearchResult(
                chunk=chunk,
                score=scored_doc.score,
                original_messages=original_messages,
                topics=topics
            )
            results.append(result)

        # Sort by score (descending) - highest score first
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
        enrich_with_messages: bool = True,
        message_window_sec: int = 300,
        chat_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Perform hybrid search: vector search + message context enrichment.

        Args:
            query: Search query text
            top_k: Number of top results to return
            threshold: Minimum similarity score (0.0-1.0). Results below threshold are filtered out.
            enrich_with_messages: Whether to enrich results with original messages
            message_window_sec: Time window in seconds for message context retrieval

        Returns:
            List of SearchResult objects, sorted by score (descending)
        """
        if not query or not query.strip():
            return []

        # Step 1: Compute query embedding
        query_vector = self._compute_query_embedding(query)

        # Step 2: Vector search to get candidate chunk IDs with scores
        scored_docs = self._perform_vector_search(query_vector, top_k, chat_id)

        if not scored_docs:
            return []

        # Step 3: Filter by threshold if provided
        scored_docs = self._filter_by_threshold(scored_docs, threshold)

        if not scored_docs:
            return []

        # Step 4: Retrieve full chunk objects by IDs
        chunk_ids = [doc.id for doc in scored_docs]
        chunk_map = self._get_chunks_by_ids(chunk_ids)

        # Step 5: Build SearchResult objects with enrichment
        results = self._build_search_results(
            scored_docs,
            chunk_map,
            enrich_with_messages,
            message_window_sec
        )

        return results

