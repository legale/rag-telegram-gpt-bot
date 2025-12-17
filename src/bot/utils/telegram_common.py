"""
Common Telegram utilities for message formatting and splitting.
"""

import html
from typing import List, Dict, Optional

# Maximum Telegram message content length
MAX_TG_CONTENT_LEN = 4096


def format_message_html(msg_data: Dict, message_id: int) -> str:
    """
    Format message data as HTML with monospace formatting.
    
    Args:
        msg_data: Dictionary with keys: text, date, sender, sender_id (optional)
        message_id: Message ID to include in the output
        
    Returns:
        HTML formatted string with message content
    """
    text = msg_data.get("text", "")
    date = msg_data.get("date", "")
    sender = msg_data.get("sender", "Unknown")
    sender_id = msg_data.get("sender_id")
    
    # Escape HTML in text content
    escaped_text = html.escape(text)
    
    # Build HTML content
    parts = [
        f"<code>id: {message_id}</code>",
        f"<code>date: {date}</code>",
    ]
    
    if sender_id:
        parts.append(f"<code>sender: {html.escape(sender)} (user_id: {sender_id})</code>")
    else:
        parts.append(f"<code>sender: {html.escape(sender)}</code>")
    
    parts.append(f"<pre>{escaped_text}</pre>")
    
    return "\n".join(parts)


def _needs_splitting(full_content: str, max_len: int) -> bool:
    """
    Check if message needs to be split.
    
    Args:
        full_content: Full formatted message content
        max_len: Maximum content length
        
    Returns:
        True if message needs splitting, False otherwise
    """
    return len(full_content) > max_len


def _find_split_pos(text: str, max_len: int) -> int:
    """Find smart split position for text."""
    if len(text) <= max_len:
        return len(text)
    
    # Try to find newline near the split point
    newline_pos = text.rfind('\n', 0, max_len)
    if newline_pos > max_len * 0.8:  # If newline is in last 20%, use it
        return newline_pos + 1
    
    # Try to find space
    space_pos = text.rfind(' ', 0, max_len)
    if space_pos > max_len * 0.8:
        return space_pos + 1
        
    return max_len

def split_text_smartly(text: str, max_len: int = MAX_TG_CONTENT_LEN) -> List[str]:
    """
    Split plain text smartly into chunks respecting max length.
    
    Args:
        text: Text to split
        max_len: Maximum length of each chunk
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_len:
        return [text]
        
    parts = []
    remaining = text
    while remaining:
        if len(remaining) <= max_len:
            parts.append(remaining)
            break
            
        split_pos = _find_split_pos(remaining, max_len)
        parts.append(remaining[:split_pos])
        remaining = remaining[split_pos:]
        
    return parts

def _split_message(
    msg_data: Dict,
    message_id: int,
    escaped_text: str,
    base_prefix: str,
    part_prefix_template: str,
    available_content_len: int,
    max_len: int
) -> List[Dict]:
    """
    Split message text into multiple parts.
    """
    # Use shared splitting logic
    text_chunks = split_text_smartly(escaped_text, available_content_len)
    
    parts = []
    for i, chunk in enumerate(text_chunks, 1):
        # Build part HTML
        part_html = base_prefix + part_prefix_template.format(i) + f"<pre>{chunk}</pre>"
        
        parts.append({
            "id": message_id,
            "date": msg_data.get("date", ""),
            "sender": msg_data.get("sender", "Unknown"),
            "part": i,
            "content": part_html
        })
    
    return parts


def split_message_if_needed(msg_data: Dict, message_id: int, max_len: int = MAX_TG_CONTENT_LEN) -> List[Dict]:
    """
    Split message into parts if it exceeds maximum length.
    
    Args:
        msg_data: Dictionary with keys: text, date, sender, sender_id (optional)
        message_id: Message ID
        max_len: Maximum content length (default: MAX_TG_CONTENT_LEN)
        
    Returns:
        List of dictionaries, each representing a message part:
        - Simple message: [{"id": id, "date": date, "sender": sender, "content": content}]
        - Multipart: [{"id": id, "date": date, "sender": sender, "part": 1, "content": content}, ...]
    """
    # Format full message HTML
    full_content = format_message_html(msg_data, message_id)
    
    # Calculate prefix size (id, date, sender lines + part line if multipart)
    # Approximate: "id: X\n" + "date: Y\n" + "sender: Z\n" + "part: N\n" (if multipart)
    base_prefix = f"<code>id: {message_id}</code>\n<code>date: {msg_data.get('date', '')}</code>\n<code>sender: {html.escape(msg_data.get('sender', 'Unknown'))}</code>\n"
    part_prefix_template = f"<code>part: {{}}</code>\n"
    
    if not _needs_splitting(full_content, max_len):
        # Single message, no splitting needed
        return [{
            "id": message_id,
            "date": msg_data.get("date", ""),
            "sender": msg_data.get("sender", "Unknown"),
            "content": full_content
        }]
    
    # Need to split - calculate available space for content
    # We need to account for prefix in each part
    base_prefix_len = len(base_prefix)
    part_prefix_len = len(part_prefix_template.format(1))
    available_content_len = max_len - base_prefix_len - part_prefix_len - len("<pre></pre>")
    
    if available_content_len <= 0:
        # Even prefix is too long, return as-is (will be truncated by Telegram)
        return [{
            "id": message_id,
            "date": msg_data.get("date", ""),
            "sender": msg_data.get("sender", "Unknown"),
            "content": full_content[:max_len]
        }]
    
    # Split text content
    text = msg_data.get("text", "")
    escaped_text = html.escape(text)
    
    return _split_message(
        msg_data, message_id, escaped_text,
        base_prefix, part_prefix_template,
        available_content_len, max_len
    )

