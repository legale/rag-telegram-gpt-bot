"""
Telegram link generation utilities.
"""

from typing import Optional


def build_message_link(chat_id: int, msg_id: int, chat_username: Optional[str] = None) -> str:
    """
    Build a Telegram message link.
    
    Args:
        chat_id: Chat ID (Bot API format, can be negative for groups/channels)
        msg_id: Message ID
        chat_username: Optional chat username (e.g., "my_channel")
        
    Returns:
        Telegram message link (e.g., "https://t.me/c/1234567890/123" or "https://t.me/my_channel/123")
    """
    if chat_username and chat_username.strip():
        return f"https://t.me/{chat_username}/{msg_id}"
    
    # Handle Bot API chat_id format (can be negative for groups/channels)
    cid = int(chat_id)
    
    if cid < 0:
        # Convert negative chat_id to internal format
        internal = -cid
        s = str(internal)
        
        # Remove "100" prefix if present (Telegram internal format)
        if s.startswith("100"):
            internal = int(s[3:])
        
        return f"https://t.me/c/{internal}/{msg_id}"
    else:
        # For positive chat_id, use as-is (fallback)
        return f"https://t.me/c/{chat_id}/{msg_id}"

