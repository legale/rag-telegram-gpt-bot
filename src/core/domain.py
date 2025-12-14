from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Message:
    id: str
    chat_id: str
    from_id: str
    text: str
    timestamp: datetime
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    text: str
    msg_ids: Optional[Tuple[str, str]] = None  # (start_id, end_id)
    valid_period: Optional[Tuple[datetime, datetime]] = None  # (from, to)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    original_messages: List[Message] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)


@dataclass
class IngestionJob:
    status: str
    stage: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileConfig:
    model_name: str
    embedding_provider: str
    paths: Dict[str, str] = field(default_factory=dict)


@dataclass
class TopicUpdate:
    topic_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

