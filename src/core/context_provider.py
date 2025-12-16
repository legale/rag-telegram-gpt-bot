"""Context provider for RAG retrieval and caching."""

from typing import List, Dict, Optional
from src.lib.syslog2 import *


class ContextProvider:
    """Provides RAG context with caching and quality evaluation."""
    
    def __init__(
        self,
        retrieval_service,
        config,
        conversation_state,
        log_level: int = LOG_WARNING
    ):
        """
        Initialize context provider.
        
        Args:
            retrieval_service: Retrieval service instance
            config: Bot config instance
            conversation_state: ConversationState instance
            log_level: Logging level
        """
        self.retrieval_service = retrieval_service
        self.config = config
        self.conversation_state = conversation_state
        self.log_level = log_level
    
    def _is_good_context(self, context_chunks: List[Dict]) -> tuple[bool, float]:
        """
        Evaluate if RAG context is good enough to cache.
        
        Args:
            context_chunks: List of retrieved chunk dictionaries
            
        Returns:
            Tuple of (is_good, max_score)
        """
        if not context_chunks:
            if self.log_level <= LOG_DEBUG:
                syslog2(LOG_DEBUG, "rag_context_evaluated", good=False, max_score=0.0, chunks=0)
            return (False, 0.0)
        
        # Extract scores from chunks
        # Chunks from HybridRetrievalService have 'score' field (similarity 0.0-1.0)
        scores = []
        for chunk in context_chunks:
            score = chunk.get("score")
            if score is not None:
                scores.append(float(score))
        
        if not scores:
            if self.log_level <= LOG_DEBUG:
                syslog2(LOG_DEBUG, "rag_context_evaluated", good=False, max_score=0.0, chunks=len(context_chunks), note="no_scores")
            return (False, 0.0)
        
        max_score = max(scores)
        
        # Use threshold from config
        # cosine_distance_thr in config is distance (higher = worse), but chunks have similarity (0.0-1.0, higher = better)
        # For similarity scores, we want at least some reasonable quality
        # Default threshold: 0.3 similarity (reasonable quality)
        threshold = 0.3
        
        # Check if we have at least one chunk with good similarity
        is_good = max_score >= threshold and len(context_chunks) > 0
        
        if self.log_level <= LOG_DEBUG:
            syslog2(LOG_DEBUG, "rag_context_evaluated", good=is_good, max_score=max_score, chunks=len(context_chunks), threshold=threshold)
        
        return (is_good, max_score)
    
    def _should_refresh_context(self) -> bool:
        """
        Determine if RAG context should be refreshed.
        
        Returns:
            True if context should be refreshed (no active context), False otherwise
        """
        return self.conversation_state.active_context_chunks is None
    
    def _build_new_context(self, user_input: str, n_results: int) -> List[Dict]:
        """
        Build new RAG context by retrieving chunks.
        
        Args:
            user_input: User query string
            n_results: Number of chunks to retrieve
            
        Returns:
            List of context chunk dictionaries
        """
        return self.retrieval_service.retrieve(
            user_input, n_results=n_results, score_threshold=self.config.fts5_score_thr
        )
    
    def _evaluate_context_quality(self, context_chunks: List[Dict]) -> tuple[bool, float]:
        """
        Evaluate context quality and return (is_good, max_score).
        
        Args:
            context_chunks: List of context chunk dictionaries
            
        Returns:
            Tuple of (is_good, max_score)
        """
        return self._is_good_context(context_chunks)
    
    def get_or_build_context(self, user_input: str, n_results: int) -> List[Dict]:
        """
        Get cached RAG context or build new one if needed.
        
        Args:
            user_input: User query string
            n_results: Number of chunks to retrieve
            
        Returns:
            List of context chunk dictionaries
        """
        if self._should_refresh_context():
            # Build new context
            context_chunks = self._build_new_context(user_input, n_results)
            
            # Evaluate context quality
            is_good, max_score = self._evaluate_context_quality(context_chunks)
            
            if is_good:
                # Cache the context
                self.conversation_state.active_context_chunks = context_chunks
                self.conversation_state.active_context_query = user_input
                self.conversation_state.active_context_score = max_score
                syslog2(LOG_NOTICE, "rag_context_new", query=user_input[:80], chunks=len(context_chunks), max_score=max_score)
            else:
                # Context not good enough, don't cache but still use it once
                self.conversation_state.clear_active_context(reason="context_not_good")
                if self.log_level <= LOG_DEBUG:
                    syslog2(LOG_DEBUG, "rag_context_not_cached", query=user_input[:80], chunks=len(context_chunks), max_score=max_score)
            
            return context_chunks
        else:
            # Reuse cached context
            syslog2(LOG_NOTICE, "rag_context_reused", source_query=self.conversation_state.active_context_query[:80] if self.conversation_state.active_context_query else None, chunks=len(self.conversation_state.active_context_chunks))
            return self.conversation_state.active_context_chunks

