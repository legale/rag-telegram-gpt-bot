from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol
from contextlib import AbstractContextManager
from datetime import datetime

from .domain import Message, Chunk, TopicUpdate, ProfileConfig


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
    def save_batch(self, messages: List[Message]) -> int:
        ...

    def get_by_chat(self, chat_id: str, limit: int, offset: int) -> List[Message]:
        ...

    def get_context(self, chat_id: str, time_point, window_sec: int) -> List[Message]:
        ...

    def count(self) -> int:
        ...


class ChunkStore(Protocol):
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
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...


class FTSIndex(Protocol):
    """Interface for Full-Text Search index (FTS5)."""
    
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
    def complete(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        ...


class ConfigProvider(Protocol):
    def get_profile_config(self, profile_name: str) -> ProfileConfig:
        ...


class TransactionManager(Protocol):
    def atomic(self) -> AbstractContextManager[None]:
        ...


class TopicIndex(Protocol):
    """Interface for topic vector index (L1 or L2 topics)."""

    def query(self, vector: List[float], top_k: int) -> List[ScoredDoc]:
        """
        Query topics by embedding vector.

        Args:
            vector: Query embedding vector
            top_k: Number of top topics to return

        Returns:
            List of ScoredDoc objects with topic IDs and similarity scores
        """
        ...

    def get_all(self) -> List[VectorDoc]:
        """
        Get all topics with their embeddings.

        Returns:
            List of VectorDoc objects representing all topics
        """
        ...

    def count(self) -> int:
        """
        Get total number of topics in the index.

        Returns:
            Number of topics
        """
        ...


class TopicIndexProvider(Protocol):
    """Provider for topic indices (L1 and L2)."""

    def get_l1_index(self) -> TopicIndex:
        """
        Get L1 topic index.

        Returns:
            TopicIndex instance for L1 topics
        """
        ...

    def get_l2_index(self) -> TopicIndex:
        """
        Get L2 topic index.

        Returns:
            TopicIndex instance for L2 topics
        """
        ...

