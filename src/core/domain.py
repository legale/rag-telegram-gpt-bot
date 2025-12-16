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
    
    def validate(self) -> None:
        """
        Validate Message object.
        
        Raises:
            ValueError: If message is invalid
        """
        if not self.id or not isinstance(self.id, str):
            raise ValueError("Message.id must be a non-empty string")
        
        if not self.chat_id or not isinstance(self.chat_id, str):
            raise ValueError("Message.chat_id must be a non-empty string")
        
        if not isinstance(self.from_id, str):
            raise ValueError("Message.from_id must be a string")
        
        if not isinstance(self.text, str):
            raise ValueError("Message.text must be a string")
        
        if not isinstance(self.timestamp, datetime):
            raise ValueError("Message.timestamp must be a datetime object")
        
        if not isinstance(self.meta, dict):
            raise ValueError("Message.meta must be a dictionary")


@dataclass
class Chunk:
    id: str
    text: str
    msg_ids: Optional[Tuple[str, str]] = None  # (start_id, end_id)
    valid_period: Optional[Tuple[datetime, datetime]] = None  # (from, to)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """
        Validate Chunk object.
        
        Raises:
            ValueError: If chunk is invalid
        """
        if not self.id or not isinstance(self.id, str):
            raise ValueError("Chunk.id must be a non-empty string")
        
        if not isinstance(self.text, str):
            raise ValueError("Chunk.text must be a string")
        
        if self.msg_ids is not None:
            if not isinstance(self.msg_ids, tuple) or len(self.msg_ids) != 2:
                raise ValueError("Chunk.msg_ids must be a tuple of 2 strings or None")
            if not all(isinstance(msg_id, str) for msg_id in self.msg_ids):
                raise ValueError("Chunk.msg_ids must contain only strings")
        
        if self.valid_period is not None:
            if not isinstance(self.valid_period, tuple) or len(self.valid_period) != 2:
                raise ValueError("Chunk.valid_period must be a tuple of 2 datetimes or None")
            if not all(isinstance(ts, datetime) for ts in self.valid_period):
                raise ValueError("Chunk.valid_period must contain only datetime objects")
        
        if self.embedding is not None:
            if not isinstance(self.embedding, list):
                raise ValueError("Chunk.embedding must be a list of floats or None")
            if not all(isinstance(val, (int, float)) for val in self.embedding):
                raise ValueError("Chunk.embedding must contain only numeric values")
        
        if not isinstance(self.metadata, dict):
            raise ValueError("Chunk.metadata must be a dictionary")


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

