# src/bot/command_parser.py
"""
Common command parsing utilities for CLI and Telegram bot.
"""

from typing import Tuple, Optional
from src.bot.admin import AdminManager


def parse_find_command_args(
    text: str,
    admin_manager: Optional[AdminManager] = None,
    default_threshold: float = 1.5
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse find command arguments (rag_method and query or list command).
    
    Args:
        text: Command text (e.g., "/find hybrid query text" or "/find vector_only list")
        admin_manager: AdminManager instance for config access (optional, not used in new format)
        default_threshold: Default threshold if admin_manager is None (not used in new format)
        
    Returns:
        Tuple of (rag_method, action, query) or (None, None, error_message)
        rag_method: "hybrid", "vector_only", or "fts_only"
        action: "list" or None (for search)
        query: Search query text or error message
    """
    # Remove "/find" prefix if present
    if text.startswith("/find"):
        text = text[5:].strip()
    
    if not text or not text.strip():
        return None, None, (
            "Использование: /find <rag_method> <query>\n"
            "              /find <rag_method> list\n\n"
            "Доступные методы RAG:\n"
            "  • hybrid      - гибридный поиск (FTS5 + векторный)\n"
            "  • vector_only - только векторный поиск\n"
            "  • fts_only    - только FTS5 поиск\n\n"
            "Примеры:\n"
            "  /find hybrid vpn туннель\n"
            "  /find vector_only список\n"
            "  /find fts_only list"
        )
    
    parts = text.split(maxsplit=2)
    
    # First part must be rag_method
    rag_method = parts[0].strip().lower()
    valid_methods = {"hybrid", "vector_only", "fts_only"}
    
    if rag_method not in valid_methods:
        return None, None, (
            f"Неизвестный метод RAG: {rag_method}\n\n"
            "Доступные методы:\n"
            "  • hybrid\n"
            "  • vector_only\n"
            "  • fts_only\n\n"
            "Использование: /find <rag_method> <query> или /find <rag_method> list"
        )
    
    # Second part: either "list" or start of query
    if len(parts) < 2:
        return None, None, (
            "Необходимо указать действие или запрос.\n\n"
            "Использование:\n"
            "  /find <rag_method> <query>  - выполнить поиск\n"
            "  /find <rag_method> list     - показать список методов"
        )
    
    second_part = parts[1].strip().lower()
    
    if second_part == "list":
        # Show list of available methods
        return rag_method, "list", None
    
    # Otherwise, second part and rest is the query
    if len(parts) >= 3:
        query = f"{parts[1]} {parts[2]}".strip()
    else:
        query = parts[1].strip()
    
    if not query:
        return None, None, (
            "Необходимо указать запрос для поиска.\n\n"
            "Использование: /find <rag_method> <query>"
        )
    
    return rag_method, None, query

