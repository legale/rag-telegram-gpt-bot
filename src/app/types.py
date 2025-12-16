"""Application request/response types for transport layer."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List


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

