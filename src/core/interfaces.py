from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol
from contextlib import AbstractContextManager

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


class VectorIndex(Protocol):
    def upsert(self, items: List[VectorDoc]) -> None:
        ...

    def query(self, vector: List[float], top_k: int, filter: Optional[Dict] = None) -> List[ScoredDoc]:
        ...

    def delete(self, ids: List[str]) -> None:
        ...


class Embedder(Protocol):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
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

