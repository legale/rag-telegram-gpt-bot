"""Application request/response types for transport layer."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AppRequest:
    """Request from transport layer to application core."""
    user_id: Optional[str] = None
    chat_id: Optional[str] = None
    text: str = ""
    transport: str = "cli"  # "cli" or "telegram"
    meta: Optional[Dict[str, Any]] = None


@dataclass
class AppResponse:
    """Response from application core to transport layer."""
    text: str = ""
    actions: Optional[List[Dict[str, Any]]] = None  # For future use (e.g., send file, edit message)


@dataclass
class CommandRequest:
    """Parsed command request independent from transport/core implementation details."""
    user_id: Optional[str] = None
    chat_id: Optional[str] = None
    name: str = ""
    args: List[str] = field(default_factory=list)
    raw: str = ""
    meta: Optional[Dict[str, Any]] = None


@dataclass
class QueryRequest:
    """Non-command query request independent from transport/core implementation details."""
    user_id: Optional[str] = None
    chat_id: Optional[str] = None
    text: str = ""
    meta: Optional[Dict[str, Any]] = None
