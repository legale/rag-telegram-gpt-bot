"""Hybrid search use case - clean search logic using core interfaces."""

from __future__ import annotations

from typing import List, Optional
from datetime import datetime, timedelta

from ..domain import SearchResult, Chunk, Message
from ..interfaces import Embedder, VectorIndex, ChunkStore, MessageStore


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
        query_vector = self.embedder.embed_query(query)

        # Step 2: Vector search to get candidate chunk IDs with scores
        # Build filter if chat_id is provided
        vector_filter = None
        if chat_id:
            vector_filter = {"chat_id": chat_id}
        
        scored_docs = self.vector_index.query(
            vector=query_vector,
            top_k=top_k,
            filter=vector_filter
        )

        if not scored_docs:
            return []

        # Step 3: Filter by threshold if provided
        if threshold is not None:
            scored_docs = [doc for doc in scored_docs if doc.score >= threshold]

        if not scored_docs:
            return []

        # Step 4: Retrieve full chunk objects by IDs
        chunk_ids = [doc.id for doc in scored_docs]
        chunks = self.chunk_store.get_by_ids(chunk_ids)

        # Create a mapping from chunk_id to chunk for efficient lookup
        chunk_map = {chunk.id: chunk for chunk in chunks}

        # Step 5: Build SearchResult objects with enrichment
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
                topic_l1 = chunk.metadata.get("topic_l1_title") or chunk.metadata.get("topic_l1")
                topic_l2 = chunk.metadata.get("topic_l2_title") or chunk.metadata.get("topic_l2")
                
                if topic_l2:
                    topics.append(str(topic_l2))
                if topic_l1:
                    topics.append(str(topic_l1))
                
                # Also check for topic_ids list
                topic_ids = chunk.metadata.get("topic_ids", [])
                topics.extend([str(tid) for tid in topic_ids if tid not in topics])

            # Enrich with original messages if requested
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

