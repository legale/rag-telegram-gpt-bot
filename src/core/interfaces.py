from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol
from datetime import datetime

from .domain import Message, Chunk, TopicUpdate


@dataclass
class VectorDoc:
    id: str
    vector: List[float]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredDoc:
    id: str
    score: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchFilters:
    """Filters for search queries."""
    author: Optional[str] = None  # from_id or user name
    time_from: Optional[datetime] = None
    time_to: Optional[datetime] = None
    chat_id: Optional[str] = None
    # topic_l1_id removed - clustering is deprecated


class MessageStore(Protocol):
    """Contract for persisting chat messages.

    Example:

        class SQLiteMessageStore(MessageStore):
            def save_batch(self, messages):
                ...
    """

    def save_batch(self, messages: List[Message]) -> int:
        ...

    def get_by_chat(self, chat_id: str, limit: int, offset: int) -> List[Message]:
        ...

    def get_context(self, chat_id: str, time_point, window_sec: int) -> List[Message]:
        ...

    def count(self) -> int:
        ...


class ChunkStore(Protocol):
    """Persistence boundary for chunks derived from messages.

    Example:

        class InMemoryChunkStore(ChunkStore):
            def get_by_ids(self, ids):
                ...
    """

    def save_batch(self, chunks: List[Chunk]) -> int:
        ...

    def get_by_ids(self, ids: List[str]) -> List[Chunk]:
        ...

    def update_topics(self, updates: Dict[str, TopicUpdate]) -> None:
        ...

    def clear(self) -> None:
        ...

    # get_by_topic_l1 and get_by_topic_l2 removed - clustering is deprecated


class VectorIndex(Protocol):
    """Vector index interface used by core search and retrieval layers.

    Example:

        class ChromaIndex(VectorIndex):
            def query(self, vector, top_k, filter=None):
                ...
    """

    def upsert(self, items: List[VectorDoc]) -> None:
        ...

    def query(self, vector: List[float], top_k: int, filter: Optional[Dict] = None) -> List[ScoredDoc]:
        ...

    def delete(self, ids: List[str]) -> None:
        ...

    def count(self) -> int:
        """
        Get total number of documents in the index.

        Returns:
            Number of documents
        """
        ...

    def get_embeddings_by_ids(self, ids: List[str]) -> Dict[str, List[float]]:
        """
        Get embeddings by document IDs.

        Args:
            ids: List of document IDs

        Returns:
            Dictionary mapping document ID to embedding vector
        """
        ...


class Embedder(Protocol):
    """Embedding client interface for documents and queries.

    Example:

        class OpenAIEmbedder(Embedder):
            def embed_query(self, text):
                return ...
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...


class FTSIndex(Protocol):
    """Interface for Full-Text Search index (FTS5).

    Example:

        class SQLiteFTSIndex(FTSIndex):
            def search(self, query, top_k, filters=None):
                ...
    """
    
    def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[SearchFilters] = None
    ) -> List[ScoredDoc]:
        """
        Search using FTS5.
        
        Args:
            query: Search query text
            top_k: Number of results
            filters: Optional filters (author, time_range, chat_id, etc.)
        
        Returns:
            List of ScoredDoc with document IDs and scores
        """
        ...
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for indexing/searching.
        
        Normalization includes:
        - Lowercase
        - ё -> е conversion
        - Punctuation removal
        
        Args:
            text: Text to normalize
        
        Returns:
            Normalized text
        """
        ...


class LLM(Protocol):
    """High-level interface for language model completions.

    Example:

        class OpenAILLM(LLM):
            def complete(self, prompt, system=None, **kwargs):
                ...
    """

    def complete(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        ...


class ConfigProvider(Protocol):
    """Protocol for configuration providers. Core uses this interface instead of direct BotConfig access.

    Example:

        class InMemoryConfig(ConfigProvider):
            @property
            def admin_password(self):
                return "secret"
    """
    
    @property
    def admin_password(self) -> str:
        """Get admin password."""
        ...
    
    def get_system_prompt(self) -> str:
        """Get system prompt (with default fallback)."""
        ...
    
    @property
    def embedding_model(self) -> str:
        """Get embedding model name."""
        ...
    
    @embedding_model.setter
    def embedding_model(self, value: str) -> None:
        """Set embedding model name."""
        ...
    
    @property
    def embedding_generator(self) -> str:
        """Get embedding generator type."""
        ...
    
    @embedding_generator.setter
    def embedding_generator(self, value: str) -> None:
        """Set embedding generator type."""
        ...
    
    @property
    def chunk_token_min(self) -> int:
        """Get minimum chunk token count."""
        ...
    
    @chunk_token_min.setter
    def chunk_token_min(self, value: int) -> None:
        """Set minimum chunk token count."""
        ...
    
    @property
    def chunk_token_max(self) -> int:
        """Get maximum chunk token count."""
        ...
    
    @chunk_token_max.setter
    def chunk_token_max(self, value: int) -> None:
        """Set maximum chunk token count."""
        ...
    
    @property
    def chunk_overlap_ratio(self) -> float:
        """Get chunk overlap ratio."""
        ...
    
    @chunk_overlap_ratio.setter
    def chunk_overlap_ratio(self, value: float) -> None:
        """Set chunk overlap ratio."""
        ...
    
    @property
    def cosine_distance_thr(self) -> float:
        """Get cosine distance threshold."""
        ...
    
    @cosine_distance_thr.setter
    def cosine_distance_thr(self, value: float) -> None:
        """Set cosine distance threshold."""
        ...
    
    @property
    def rag_ntop(self) -> int:
        """Get RAG top N value."""
        ...
    
    @rag_ntop.setter
    def rag_ntop(self, value: int) -> None:
        """Set RAG top N value."""
        ...
    
    @property
    def fts5_score_thr(self) -> float:
        """Get FTS5 score threshold."""
        ...
    
    @fts5_score_thr.setter
    def fts5_score_thr(self, value: float) -> None:
        """Set FTS5 score threshold."""
        ...
    
    def save(self) -> None:
        """Save configuration to file."""
        ...


# TopicIndex and TopicIndexProvider removed - clustering is deprecated
