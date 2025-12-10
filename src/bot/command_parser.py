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
) -> Tuple[Optional[float], Optional[str]]:
    """
    Parse find command arguments (threshold and query).
    
    Args:
        text: Command text (e.g., "/find 2.0 vpn туннель" or "2.0 vpn туннель" or "vpn туннель")
        admin_manager: AdminManager instance for config access (optional)
        default_threshold: Default threshold if admin_manager is None
        
    Returns:
        Tuple of (threshold, query) or (None, error_message)
        threshold uses config default if admin_manager is provided, else uses default_threshold
    """
    # Remove "/find" prefix if present
    if text.startswith("/find"):
        text = text[5:].strip()
    
    # Get default threshold from config or parameter
    if admin_manager:
        threshold_default = admin_manager.config.cosine_distance_thr
    else:
        threshold_default = default_threshold
    
    if not text or not text.strip():
        return None, (
            f"Использование: /find [thr] <запрос>\n\n"
            f"Примеры:\n"
            f"  /find vpn туннель          - поиск с threshold={threshold_default} (по умолчанию)\n"
            f"  /find 2.0 vpn туннель       - поиск с threshold=2.0\n"
            f"  /find 0.5 test              - поиск с threshold=0.5"
        )
    
    parts = text.split(maxsplit=1)
    threshold = threshold_default
    search_query = ""
    
    try:
        # Check if first argument is a number
        potential_threshold = float(parts[0].strip())
        threshold = potential_threshold
        # If threshold parsed successfully, query is the rest
        if len(parts) >= 2:
            search_query = parts[1].strip()
        else:
            return None, (
                "Использование: /find [thr] <запрос>\n\n"
                "Если указан threshold, необходимо также указать запрос.\n"
                "Пример: /find 2.0 vpn туннель"
            )
    except ValueError:
        # First argument is not a number, treat entire text as query
        search_query = text.strip()
    
    if not search_query:
        return None, (
            "Использование: /find [thr] <запрос>\n\n"
            "Необходимо указать запрос для поиска.\n"
            "Пример: /find vpn туннель"
        )
    
    return threshold, search_query

