"""
High-level message search functionality.
"""

from typing import List
from src.core.retrieval import RetrievalService
from src.storage.db import Database
from src.bot.utils import build_message_link
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

