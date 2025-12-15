#./src/core/message_search.py
"""
High-level message search functionality.
"""

from typing import List, Dict, Optional
from src.core.hybrid_retrieval import HybridRetrievalService
from src.core.domain import SearchResult
from src.storage.db import Database, MessageModel
from src.bot.utils import build_message_link
from src.bot.utils.telegram_common import split_message_if_needed, MAX_TG_CONTENT_LEN
from src.lib.syslog2 import *


def convert_search_results_to_dict(results: List[SearchResult]) -> List[Dict]:
    """
    Convert SearchResult domain objects to dict format expected by message formatting.

    Args:
        results: List of SearchResult objects from HybridSearch

    Returns:
        List of dictionaries with keys: id, distance, metadata
    """
    dict_results = []
    for result in results:
        # Convert similarity score to distance (1.0 - score)
        distance = 1.0 - result.score if result.score <= 1.0 else 0.0

        # Extract metadata from chunk
        metadata = result.chunk.metadata.copy() if result.chunk.metadata else {}
        
        # Add topics to metadata if present
        if result.topics:
            metadata["topics"] = result.topics

        dict_results.append({
            "id": result.chunk.id,
            "distance": distance,
            "metadata": metadata
        })

    return dict_results


def _convert_similarity_to_distance(score: float) -> float:
    """
    Convert similarity score to distance.
    
    Args:
        score: Similarity score (0.0-1.0)
        
    Returns:
        Distance value
    """
    return 1.0 - float(score)


def _log_retrieval_distances(
    results: List[Dict],
    query: str,
    context: str,
    log_level: int = LOG_ALERT,
    debug_rag: bool = False,
    threshold: Optional[float] = None
) -> None:
    """
    Log distances for retrieval results.
    
    Args:
        results: List of chunk dictionaries with id and distance
        query: Search query string
        context: Context string for logging (e.g., "links", "contents", "two-stage", "basic")
        log_level: Log level for distance logging (default: LOG_ALERT)
        debug_rag: Enable detailed debug logging
        threshold: Optional threshold value to include in logs
    """
    if not results:
        return
    
    syslog2(
        log_level,
        f"msg_search {context} results",
        query=query,
        result_count=len(results),
    )
    
    for idx, item in enumerate(results):
        chunk_id = item.get("id", "unknown")
        distance = item.get("distance")
        source = item.get("source")
        
        log_data = {
            "idx": idx,
            "chunk_id": chunk_id,
        }
        if distance is not None:
            log_data["distance"] = distance
        if source:
            log_data["source"] = source
        if threshold is not None:
            log_data["thr"] = threshold
        
        syslog2(
            log_level,
            f"msg_search {context} result distance",
            **log_data
        )
        
        if debug_rag:
            metadata = item.get("metadata") or {}
            syslog2(
                LOG_DEBUG,
                f"msg_search {context} result details",
                idx=idx,
                chunk_id=chunk_id,
                metadata=metadata,
            )


def search_message_links(
    retrieval: HybridRetrievalService,
    db: Database,
    query: str,
    top_k: int = 3,
) -> List[str]:
    """
    Search for message links by text query.
    
    Args:
        retrieval: HybridRetrievalService instance
        db: Database instance
        query: Search query string
        top_k: Number of results to return
        
    Returns:
        List of message link strings
    """
    results = retrieval.search_chunks_basic(query, n_results=top_k)
    
    if not results:
        return []
    
    links = []
    for item in results:
        chunk_id = item.get("id")
        if not chunk_id:
            continue
        
        link_info = db.get_chunk_link_info(chunk_id)
        chat_id, msg_id, chat_username = link_info
        
        if chat_id and msg_id:
            link = build_message_link(chat_id, msg_id, chat_username)
            if link:
                links.append(link)
    
    return links


def search_message_contents(
    retrieval: HybridRetrievalService,
    db: Database,
    query: str,
    top_k: int = 3,
    threshold: Optional[float] = None,
    debug_rag: bool = False,
) -> List[List[Dict]]:
    """
    Search for message contents by text query.
    
    Args:
        retrieval: HybridRetrievalService instance
        db: Database instance
        query: Search query string
        top_k: Number of results to return
        threshold: Optional distance threshold for filtering
        debug_rag: Enable detailed RAG debug logging
        
    Returns:
        List of lists, where each inner list contains message parts
    """
    results = retrieval.search_chunks_basic(query, n_results=top_k)
    
    if threshold is not None:
        results = _filter_by_threshold(results, threshold, debug_rag)
    
    _log_retrieval_distances(results, query, "contents", LOG_ALERT, debug_rag, threshold)
    
    return _prepare_message_parts(db, results, debug_rag)


def _filter_by_threshold(
    results: List[Dict],
    threshold: float,
    debug_rag: bool = False
) -> List[Dict]:
    """
    Filter results by distance threshold.
    
    Args:
        results: List of chunk dictionaries with id and distance
        threshold: Maximum distance threshold
        debug_rag: Enable detailed debug logging
        
    Returns:
        Filtered list of results
    """
    original_count = len(results)
    if debug_rag:
        # Log items that will be filtered out before filtering
        for idx, item in enumerate(results):
            item_distance = float(item.get("distance", float('inf')))
            if item_distance > threshold:
                syslog2(
                    LOG_DEBUG,
                    "msg_search filtering out",
                    idx=idx,
                    chunk_id=item.get("id"),
                    distance=item_distance,
                    thr=threshold,
                )
    
    filtered = [item for item in results if float(item.get("distance", float('inf'))) <= threshold]
    filtered_count = original_count - len(filtered)
    
    if debug_rag:
        syslog2(
            LOG_DEBUG,
            "msg_search threshold filtering",
            thr=threshold,
            original_count=original_count,
            filtered_count=filtered_count,
            remaining_count=len(filtered),
        )
    
    return filtered


def _parse_msg_id(msg_id_str: str) -> int:
    """
    Parse message ID from composite format string.
    
    Args:
        msg_id_str: Message ID string in format "{chat_id}_{msg_id}" or just "{msg_id}"
        
    Returns:
        Numeric message ID (or hash-based fallback if parsing fails)
    """
    try:
        # try to extract numeric part
        if "_" in msg_id_str:
            return int(msg_id_str.split("_", 1)[1])
        else:
            return int(msg_id_str)
    except (ValueError, IndexError):
        # fallback: use hash of string as id
        return hash(msg_id_str) % (10 ** 9)  # 9-digit number


def _format_message_parts(msg: MessageModel, msg_id: int, distance: float, chunk_id: str, 
                         msg_idx: int, debug_rag: bool) -> List[Dict]:
    """
    Format message data into HTML parts.
    
    Args:
        msg: MessageModel instance
        msg_id: Parsed numeric message ID
        distance: Distance value for this message
        chunk_id: Chunk ID for logging
        msg_idx: Message index in chunk for logging
        debug_rag: Enable detailed RAG debug logging
        
    Returns:
        List of message part dictionaries
    """
    snippet = (msg.text or "")[:64]
    if debug_rag:
        syslog2(
            LOG_DEBUG,
            "msg_search contents message",
            chunk_id=chunk_id,
            msg_idx=msg_idx,
            msg_id_str=msg.msg_id,
            msg_id=msg_id,
            sender=msg.from_id or "Unknown",
            ts=msg.ts.isoformat() if msg.ts else "",
            text_snippet=snippet,
            distance=distance,
        )
    
    # prepare message data
    msg_data = {
        "text": msg.text or "",
        "date": msg.ts.isoformat() if msg.ts else "",
        "sender": msg.from_id or "Unknown",
        "sender_id": None,  # from_id is string, not numeric user_id
        "distance": distance,
    }
    
    # split message into parts if needed
    parts = split_message_if_needed(msg_data, msg_id, MAX_TG_CONTENT_LEN)
    # propagate distance into each part for caller
    for part in parts:
        part["distance"] = distance
    return parts


def _prepare_message_parts(
    db: Database,
    results: List[Dict],
    debug_rag: bool
) -> List[List[Dict]]:
    """
    Prepare message parts from search results.
    
    Args:
        db: Database instance
        results: List of chunk dictionaries with id and distance
        debug_rag: Enable detailed RAG debug logging
        
    Returns:
        List of lists, where each inner list contains message parts
    """
    all_message_parts: List[List[Dict]] = []
    
    for idx, item in enumerate(results):
        chunk_id = item.get("id")
        if chunk_id is None:
            if debug_rag:
                syslog2(
                    LOG_DEBUG,
                    "msg_search contents skip result without chunk_id",
                    idx=idx,
                )
            continue

        distance = float(item.get("distance", 0.0))
        
        # get messages from database
        messages = db.get_messages_by_chunk(chunk_id)
        
        if debug_rag:
            syslog2(
                LOG_DEBUG,
                "msg_search contents chunk messages",
                chunk_id=chunk_id,
                msg_count=len(messages or []),
                distance=distance,
            )

        if not messages:
            continue
        
        # process each message
        for msg_idx, msg in enumerate(messages):
            msg_id = _parse_msg_id(msg.msg_id)
            parts = _format_message_parts(msg, msg_id, distance, chunk_id, msg_idx, debug_rag)
            all_message_parts.append(parts)
    
    return all_message_parts
