"""Hybrid retrieval service combining FTS5 and vector search."""

from __future__ import annotations

from typing import List, Optional, Dict, Set
from datetime import datetime
import json

from ..domain import SearchResult, Chunk, Message
from ..interfaces import (
    FTSIndex, VectorIndex, Embedder, ChunkStore, MessageStore, SearchFilters, LLM
)
from ..distance_utils import similarity_to_distance
from src.lib.syslog2 import *


class HybridRetrievalService:
    """
    Hybrid retrieval service: FTS5 candidates → vector rerank → context packing.
    
    This service provides:
    1. FTS5 keyword search for initial candidates
    2. Vector reranking of candidates
    3. Context packing (dedup, neighbors, token budget)
    4. Multiple output modes (context for LLM vs evidence timeline)
    """

    def __init__(
        self,
        fts_index: FTSIndex,
        vector_index: VectorIndex,
        embedder: Embedder,
        chunk_store: ChunkStore,
        message_store: MessageStore,
        log_level: int = LOG_WARNING,
        fts_only: bool = False,
    ):
        """
        Initialize HybridRetrievalService.

        Args:
            fts_index: FTS5 index for keyword search
            vector_index: Vector index for semantic search
            embedder: Embedder for computing embeddings
            chunk_store: Chunk store for retrieving chunks
            message_store: Message store for retrieving messages
            log_level: Logging level
            fts_only: If True, skip vector reranking and return FTS results directly
        """
        self.fts_index = fts_index
        self.vector_index = vector_index
        self.embedder = embedder
        self.chunk_store = chunk_store
        self.message_store = message_store
        self.log_level = log_level
        self.fts_only = fts_only
        # Compatibility attributes
        self.rag_ntop = 0  # Not used in hybrid retrieval, kept for compatibility

    def search(
        self,
        query: str,
        top_k: int = 50,
        filters: Optional[SearchFilters] = None,
        rerank_top_k: int = 20,
        output_mode: str = "context",  # "context" | "evidence"
        message_window_sec: int = 300,
    ) -> List[SearchResult]:
        """
        Hybrid search: FTS5 candidates → vector rerank → context packing.

        Args:
            query: Search query text
            top_k: Number of FTS5 candidates to retrieve
            filters: Optional filters (author, time_range, chat_id)
            rerank_top_k: Number of results after reranking
            output_mode: "context" for LLM context, "evidence" for timeline list
            message_window_sec: Time window for message context

        Returns:
            List of SearchResult objects
        """
        if not query or not query.strip():
            return []

        # Step 1: FTS5 search for candidates
        if self.log_level <= LOG_DEBUG:
            syslog2(LOG_DEBUG, "hybrid_retrieval: FTS5 search", query=query, top_k=top_k, fts_only=self.fts_only)
        
        # Use search with table parameter for chunks
        # Note: FTSIndex.search() should support table parameter, but interface doesn't specify it
        # For now, we'll use a workaround - check if it's SqliteFTSIndex
        try:
            if hasattr(self.fts_index, 'search_chunks'):
                fts_results = self.fts_index.search_chunks(query, top_k=top_k, filters=filters)
            else:
                # Fallback: use generic search (assumes chunks)
                fts_results = self.fts_index.search(query, top_k=top_k, filters=filters)
        except Exception as e:
            syslog2(LOG_ERR, "hybrid_retrieval: FTS5 search failed", query=query, error=str(e))
            return []
        
        if not fts_results:
            if self.log_level <= LOG_DEBUG:
                syslog2(LOG_DEBUG, "hybrid_retrieval: no FTS5 results")
            return []

        if self.log_level <= LOG_DEBUG:
            syslog2(LOG_DEBUG, "hybrid_retrieval: FTS5 candidates", count=len(fts_results))

        # FTS-only mode: skip vector reranking and return FTS results directly
        if self.fts_only:
            if self.log_level <= LOG_DEBUG:
                syslog2(LOG_DEBUG, "hybrid_retrieval: FTS-only mode, skipping vector reranking")
            
            # Get chunks from FTS results
            candidate_ids = [doc.id for doc in fts_results]
            candidate_chunks = self.chunk_store.get_by_ids(candidate_ids)
            
            if not candidate_chunks:
                return []
            
            # Create scored candidates from FTS results only
            scored_candidates = []
            for doc in fts_results:
                chunk = next((c for c in candidate_chunks if c.id == doc.id), None)
                if chunk:
                    # Normalize FTS score to 0-1 range (assume max is around 10-20)
                    normalized_fts = min(doc.score / 20.0, 1.0)
                    scored_candidates.append((chunk, normalized_fts, 0.0))  # vector_score = 0.0 for FTS-only
            
            # Sort by FTS score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = scored_candidates[:rerank_top_k]
        else:
            # Step 2: Get embeddings for candidates
            candidate_ids = [doc.id for doc in fts_results]
            candidate_chunks = self.chunk_store.get_by_ids(candidate_ids)
            
            if not candidate_chunks:
                return []

            # Step 3: Compute query embedding and rerank by vector similarity
            try:
                query_vector = self.embedder.embed_query(query)
            except Exception as e:
                syslog2(LOG_ERR, "hybrid_retrieval: embedding computation failed", query=query, error=str(e))
                # Fallback to FTS-only if embedding fails
                scored_candidates = []
                for doc in fts_results:
                    chunk = next((c for c in candidate_chunks if c.id == doc.id), None)
                    if chunk:
                        normalized_fts = min(doc.score / 20.0, 1.0)
                        scored_candidates.append((chunk, normalized_fts, 0.0))
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                top_candidates = scored_candidates[:rerank_top_k]
            else:
                # Get embeddings for candidates
                candidate_embeddings = {}
                for chunk in candidate_chunks:
                    if chunk.embedding:
                        candidate_embeddings[chunk.id] = chunk.embedding

                # Compute similarities
                scored_candidates = []
                for chunk in candidate_chunks:
                    if chunk.id not in candidate_embeddings:
                        continue
                    
                    # Cosine similarity
                    embedding = candidate_embeddings[chunk.id]
                    similarity = self._cosine_similarity(query_vector, embedding)
                    
                    # Combine FTS5 score and vector similarity
                    # Weight: 0.3 FTS5 + 0.7 vector (can be tuned)
                    fts_score = next((doc.score for doc in fts_results if doc.id == chunk.id), 0.0)
                    # Normalize FTS5 score (assume max is around 10-20)
                    normalized_fts = min(fts_score / 20.0, 1.0)
                    combined_score = 0.3 * normalized_fts + 0.7 * similarity
                    
                    scored_candidates.append((chunk, combined_score, similarity))

                # Sort by combined score
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                
                # Take top rerank_top_k
                top_candidates = scored_candidates[:rerank_top_k]

        if self.log_level <= LOG_DEBUG:
            syslog2(LOG_DEBUG, "hybrid_retrieval: after rerank", count=len(top_candidates))

        # Step 4: Pack context (dedup by msg_id, neighbors, token budget)
        packed_results = self._pack_context(
            top_candidates,
            output_mode=output_mode,
            message_window_sec=message_window_sec
        )

        if self.log_level <= LOG_DEBUG:
            syslog2(LOG_DEBUG, "hybrid_retrieval: final results", count=len(packed_results))

        return packed_results

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def _pack_context(
        self,
        candidates: List[tuple],
        output_mode: str = "context",
        message_window_sec: int = 300,
        max_tokens: int = 4000,
    ) -> List[SearchResult]:
        """
        Pack context: dedup by msg_id, add neighbors, respect token budget.

        Args:
            candidates: List of (chunk, combined_score, vector_score) tuples
            output_mode: "context" or "evidence"
            message_window_sec: Time window for neighbors
            max_tokens: Maximum tokens for context

        Returns:
            List of SearchResult objects
        """
        results = []
        seen_msg_ids: Set[str] = set()
        current_tokens = 0
        
        # Estimate tokens (rough: 1 token ≈ 4 characters)
        def estimate_tokens(text: str) -> int:
            return len(text) // 4

        for chunk, combined_score, vector_score in candidates:
            # Dedup by msg_id
            if chunk.msg_ids:
                msg_id_key = f"{chunk.msg_ids[0]}_{chunk.msg_ids[1] if len(chunk.msg_ids) > 1 else chunk.msg_ids[0]}"
                if msg_id_key in seen_msg_ids:
                    continue
                seen_msg_ids.add(msg_id_key)

            # Check token budget
            chunk_tokens = estimate_tokens(chunk.text)
            if current_tokens + chunk_tokens > max_tokens:
                break

            current_tokens += chunk_tokens

            # Get topics from metadata
            topics = []
            if chunk.metadata:
                topic_l1 = chunk.metadata.get("topic_l1_title") or chunk.metadata.get("topic_l1")
                if topic_l1:
                    topics.append(str(topic_l1))

            # Enrich with messages based on output mode
            original_messages: List[Message] = []
            if output_mode == "context" and chunk.valid_period:
                # Get messages around chunk time
                time_point = chunk.valid_period[0]
                chat_id = chunk.metadata.get("chat_id") if chunk.metadata else None
                
                if chat_id:
                    messages = self.message_store.get_context(
                        chat_id=chat_id,
                        time_point=time_point,
                        window_sec=message_window_sec
                    )
                    # Filter out messages we've already seen
                    for msg in messages:
                        if msg.id not in seen_msg_ids:
                            msg_tokens = estimate_tokens(msg.text or "")
                            if current_tokens + msg_tokens <= max_tokens:
                                original_messages.append(msg)
                                seen_msg_ids.add(msg.id)
                                current_tokens += msg_tokens

            # Create SearchResult
            result = SearchResult(
                chunk=chunk,
                score=combined_score,
                original_messages=original_messages,
                topics=topics
            )
            results.append(result)

        return results

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        score_threshold: float = 0.5,
        use_topics: Optional[bool] = None,  # Ignored for compatibility
        llm: Optional[LLM] = None  # Ignored for compatibility
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a given query (compatibility method).

        This method provides compatibility with RetrievalService.retrieve() interface.

        Args:
            query: User query string
            n_results: Number of results to return
            score_threshold: Minimum similarity score (converted to threshold)
            use_topics: Ignored (kept for compatibility)
            llm: Ignored (kept for compatibility)

        Returns:
            List of chunk dictionaries compatible with RetrievalService format
        """
        # Convert score_threshold to similarity threshold
        # score_threshold is similarity (0.0-1.0), we use it directly
        threshold = score_threshold if score_threshold > 0 else None
        
        # Perform hybrid search
        search_results = self.search(
            query=query,
            top_k=n_results * 2,  # Get more candidates for filtering
            rerank_top_k=n_results,
            output_mode="context"
        )
        
        # Convert SearchResult to dict format
        from ..chunk_utils import build_chunk_dict_from_domain_chunk
        
        chunk_dicts = []
        for result in search_results:
            # Filter by threshold if provided
            if threshold and result.score < threshold:
                continue
            
            chunk_dict = build_chunk_dict_from_domain_chunk(
                result.chunk,
                similarity=result.score,
                source="hybrid_retrieval"
            )
            chunk_dicts.append(chunk_dict)
        
        return chunk_dicts[:n_results]

    def search_chunks_basic(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Simple chunk search without LLM processing (compatibility method).

        Args:
            query: Search query string
            n_results: Number of results to return

        Returns:
            List of dictionaries with keys: id, distance, metadata
            Sorted by distance (ascending)
        """
        # Perform hybrid search
        search_results = self.search(
            query=query,
            top_k=n_results * 2,
            rerank_top_k=n_results,
            output_mode="context"
        )
        
        # Convert to basic format
        from ..chunk_utils import build_chunk_dict_from_domain_chunk
        from ..distance_utils import similarity_to_distance
        
        results = []
        for result in search_results:
            chunk_dict = build_chunk_dict_from_domain_chunk(
                result.chunk,
                similarity=result.score,
                source="hybrid_retrieval"
            )
            # Add distance (convert from similarity)
            distance = similarity_to_distance(result.score)
            chunk_dict["distance"] = distance
            results.append(chunk_dict)
        
        # Sort by distance (ascending)
        results.sort(key=lambda x: x.get("distance", 1.0))
        
        return results[:n_results]

