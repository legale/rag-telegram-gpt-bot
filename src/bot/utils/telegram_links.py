"""
Telegram link generation utilities.
"""

from typing import Optional


def _build_link_with_username(chat_username: str, msg_id: int) -> str:
    """
    Build Telegram message link for public channel/chat with username.
    
    Args:
        chat_username: Chat username (e.g., "my_channel")
        msg_id: Message ID
        
    Returns:
        Telegram message link: https://t.me/{username}/{msg_id}
    """
    return f"https://t.me/{chat_username}/{msg_id}"


def _build_link_without_username(chat_id: int, msg_id: int) -> str:
    """
    Build Telegram message link without username.
    
    Args:
        chat_id: Chat ID (Bot API format, can be negative for groups/channels)
        msg_id: Message ID
        
    Returns:
        Telegram message link:
        - For channels/supergroups (-100xxxxxxxxxx): https://t.me/c/{internal}/{msg_id}
        - For regular groups: tg://openmessage?chat_id={cid}&message_id={msg_id}
        - For private chats: tg://openmessage?user_id={cid}&message_id={msg_id}
    """
    cid = int(chat_id)
    
    # канал/супергруппа (Bot API: -100xxxxxxxxxx) -> t.me/c/...
    if cid < 0:
        s = str(-cid)
        if s.startswith("100"):
            internal = int(s[3:])  # убираем префикс 100
            return f"https://t.me/c/{internal}/{msg_id}"
        # обычная группа без префикса 100 – http-ссылки нет, нужен tg://
        return f"tg://openmessage?chat_id={cid}&message_id={msg_id}"
    
    # приватный диалог (положительный id) – http-ссылки тоже нет, только tg://
    return f"tg://openmessage?user_id={cid}&message_id={msg_id}"


def build_message_link(chat_id: int, msg_id: int, chat_username: Optional[str] = None) -> str:
    """
    Build a Telegram message link.
    
    Args:
        chat_id: Chat ID (Bot API format, can be negative for groups/channels)
        msg_id: Message ID
        chat_username: Optional chat username (e.g., "my_channel")
        
    Returns:
        Telegram message link:
        - For public channels/chats with username: https://t.me/{username}/{msg_id}
        - For channels/supergroups (-100xxxxxxxxxx): https://t.me/c/{internal}/{msg_id}
        - For regular groups: tg://openmessage?chat_id={cid}&message_id={msg_id}
        - For private chats: tg://openmessage?user_id={cid}&message_id={msg_id}
    """
    # публичный канал/чат с username
    if chat_username and chat_username.strip():
        return _build_link_with_username(chat_username, msg_id)
    
    return _build_link_without_username(chat_id, msg_id)

