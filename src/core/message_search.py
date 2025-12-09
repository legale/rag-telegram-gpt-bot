"""
High-level message search functionality.
"""

from typing import List, Dict, Optional
from src.core.retrieval import RetrievalService
from src.storage.db import Database, MessageModel
from src.bot.utils import build_message_link
from src.bot.utils.telegram_common import split_message_if_needed, MAX_TG_CONTENT_LEN
from src.core.syslog2 import *


def search_message_links(
    retrieval: RetrievalService,
    db: Database,
    query: str,
    top_k: int = 3,
) -> List[str]:
    """
    Search for message links by text query.
    
    Args:
        retrieval: RetrievalService instance
        db: Database instance
        query: Search query string
        top_k: Number of results to return
        
    Returns:
        List of Telegram message links
    """
    # Search for chunks
    results = retrieval.search_chunks_basic(query, n_results=top_k)
    
    links = []
    for item in results:
        chunk_id = item.get("id")
        if not chunk_id:
            continue
        
        # Get link info from database
        chat_id, msg_id, chat_username = db.get_chunk_link_info(chunk_id)
        
        if chat_id is None or msg_id is None:
            continue
        
        # Build message link
        link = build_message_link(chat_id, msg_id, chat_username)
        links.append(link)
    
    syslog2(LOG_INFO, "message_search", query=query, top_k=top_k, links=len(links))
    return links


def search_message_contents(
    retrieval: RetrievalService,
    db: Database,
    query: str,
    top_k: int = 3,
) -> List[List[Dict]]:
    """
    Search for message contents by text query and return formatted message parts.
    
    Args:
        retrieval: RetrievalService instance
        db: Database instance
        query: Search query string
        top_k: Number of results to return
        
    Returns:
        List of lists, where each inner list contains message parts (Dict with id, date, sender, content, part)
        Each message may be split into multiple parts if it exceeds MAX_TG_CONTENT_LEN
    """
    # Search for chunks
    results = retrieval.search_chunks_basic(query, n_results=top_k)
    
    all_message_parts = []
    
    for item in results:
        chunk_id = item.get("id")
        if not chunk_id:
            continue
        
        # Get messages from database
        messages = db.get_messages_by_chunk(chunk_id)
        
        if not messages:
            continue
        
        # Process each message
        for msg in messages:
            # Extract message ID from composite format if needed
            # msg_id format: "{chat_id}_{msg_id}" or just "{msg_id}"
            msg_id_str = msg.msg_id
            try:
                # Try to extract numeric part
                if '_' in msg_id_str:
                    msg_id = int(msg_id_str.split('_', 1)[1])
                else:
                    msg_id = int(msg_id_str)
            except (ValueError, IndexError):
                # Fallback: use hash of string as ID
                msg_id = hash(msg_id_str) % (10 ** 9)  # 9-digit number
            
            # Prepare message data
            msg_data = {
                "text": msg.text or "",
                "date": msg.ts.isoformat() if msg.ts else "",
                "sender": msg.from_id or "Unknown",
                "sender_id": None  # from_id is string, not numeric user_id
            }
            
            # Split message into parts if needed
            parts = split_message_if_needed(msg_data, msg_id, MAX_TG_CONTENT_LEN)
            all_message_parts.append(parts)
    
    syslog2(LOG_INFO, "message_search", query=query, top_k=top_k, messages=len(all_message_parts))
    return all_message_parts

