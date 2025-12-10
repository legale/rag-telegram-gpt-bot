#./src/core/message_search.py
"""
High-level message search functionality.
"""

from typing import List, Dict, Optional
from src.core.retrieval import RetrievalService
from src.storage.db import Database, MessageModel
from src.bot.utils import build_message_link
from src.bot.utils.telegram_common import split_message_if_needed, MAX_TG_CONTENT_LEN
from src.core.syslog2 import *


def _convert_similarity_to_distance(score: float) -> float:
    """
    Convert similarity score to distance.
    
    Args:
        score: Similarity score (0.0-1.0)
        
    Returns:
        Distance value
    """
    return 1.0 - float(score)


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
    debug_rag = getattr(retrieval, "debug_rag", False)

    if debug_rag:
        syslog2(LOG_DEBUG, "msg_search links start", query=query, top_k=top_k)

    # Search for chunks
    results = retrieval.search_chunks_basic(query, n_results=top_k)

    if debug_rag:
        syslog2(LOG_DEBUG, "msg_search links basic results",
                query=query,
                result_count=len(results))
        for idx, item in enumerate(results):
            syslog2(
                LOG_ALERT,
                "msg_search links result",
                idx=idx,
                chunk_id=item.get("id"),
                distance=item.get("distance"),
                metadata=item.get("metadata") or {},
            )

    links = []
    for idx, item in enumerate(results):
        chunk_id = item.get("id")
        if not chunk_id:
            if debug_rag:
                syslog2(LOG_DEBUG, "msg_search links skip result without chunk_id", idx=idx)
            continue
        
        # Get link info from database
        chat_id, msg_id, chat_username = db.get_chunk_link_info(chunk_id)
        
        if chat_id is None or msg_id is None:
            if debug_rag:
                syslog2(
                    LOG_DEBUG,
                    "msg_search links no_link_info",
                    idx=idx,
                    chunk_id=chunk_id,
                    chat_id=str(chat_id),
                    msg_id=str(msg_id),
                )
            continue
        
        # Build message link
        link = build_message_link(chat_id, msg_id, chat_username)
        links.append(link)

        if debug_rag:
            syslog2(
                LOG_DEBUG,
                "msg_search link built",
                idx=idx,
                chunk_id=chunk_id,
                chat_id=chat_id,
                msg_id=msg_id,
                chat_username=chat_username or "",
                link=link,
            )
    
    syslog2(LOG_INFO, "message_search links", query=query, top_k=top_k, links=len(links))

    if debug_rag:
        syslog2(
            LOG_DEBUG,
            "msg_search links done",
            query=query,
            requested=top_k,
            returned=len(links),
        )

    return links


def _search_chunks(
    retrieval: RetrievalService,
    query: str,
    top_k: int,
    use_two_stage: bool,
    debug_rag: bool
) -> List[Dict]:
    """
    Search for chunks using two-stage or direct search.
    
    Args:
        retrieval: RetrievalService instance
        query: Search query string
        top_k: Number of results to return
        use_two_stage: If True, use two-stage search, else use direct search
        debug_rag: Enable detailed RAG debug logging
        
    Returns:
        List of chunk dictionaries with id, distance, metadata, source
    """
    if use_two_stage:
        if debug_rag:
            syslog2(LOG_DEBUG, "msg_search using two-stage search")
        # retrieve() returns List[Dict] with keys: id, text, metadata, score, source
        # score is similarity (0-1), need to convert to distance
        retrieve_results = retrieval.retrieve(query, n_results=top_k * 2)  # Get more results for filtering
        
        # Convert retrieve results to format compatible with search_chunks_basic
        # retrieve returns similarity (score) and may include original distance
        # Use original distance if available, otherwise convert: distance = 1 - similarity
        results = []
        for item in retrieve_results:
            score = item.get("score", 0.0)
            # Use original distance if available (from ChromaDB or computed distance)
            if "distance" in item:
                distance = float(item["distance"])
            else:
                # Fallback: convert similarity to distance for backward compatibility
                distance = _convert_similarity_to_distance(score)
            results.append({
                "id": item.get("id"),
                "distance": distance,
                "metadata": item.get("metadata", {}),
                "source": item.get("source", "unknown")
            })
        
        # Log distance for all two-stage results (always visible)
        syslog2(
            LOG_ALERT,
            "msg_search contents two-stage results",
            query=query,
            result_count=len(results),
        )
        for idx, item in enumerate(results):
            syslog2(
                LOG_ALERT,
                "msg_search two-stage result distance",
                idx=idx,
                chunk_id=item.get("id"),
                distance=item.get("distance"),
                source=item.get("source"),
            )
        if debug_rag:
            for idx, item in enumerate(results):
                syslog2(
                    LOG_DEBUG,
                    "msg_search contents two-stage result details",
                    idx=idx,
                    chunk_id=item.get("id"),
                    metadata=item.get("metadata") or {},
                )
    else:
        if debug_rag:
            syslog2(LOG_DEBUG, "msg_search using direct search")
        # Direct search using search_chunks_basic
        results = retrieval.search_chunks_basic(query, n_results=top_k * 2)  # Get more results for filtering
        
        # Log distance for all direct search results (always visible)
        syslog2(
            LOG_ALERT,
            "msg_search contents basic results",
            query=query,
            result_count=len(results),
        )
        for idx, item in enumerate(results):
            syslog2(
                LOG_ALERT,
                "msg_search basic result distance",
                idx=idx,
                chunk_id=item.get("id"),
                distance=item.get("distance"),
            )
        if debug_rag:
            for idx, item in enumerate(results):
                syslog2(
                    LOG_DEBUG,
                    "msg_search contents basic result details",
                    idx=idx,
                    chunk_id=item.get("id"),
                    metadata=item.get("metadata") or {},
                )
    
    return results


def _apply_threshold_filter(
    results: List[Dict],
    threshold: Optional[float],
    debug_rag: bool
) -> List[Dict]:
    """
    Apply threshold filtering to search results.
    
    Args:
        results: List of chunk dictionaries
        threshold: Maximum distance threshold (None = no filtering)
        debug_rag: Enable detailed RAG debug logging
        
    Returns:
        Filtered list of chunk dictionaries
    """
    if threshold is None:
        return results
    
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
            # extract message id from composite format if needed
            # msg_id format: "{chat_id}_{msg_id}" or just "{msg_id}"
            msg_id_str = msg.msg_id
            try:
                # try to extract numeric part
                if "_" in msg_id_str:
                    msg_id = int(msg_id_str.split("_", 1)[1])
                else:
                    msg_id = int(msg_id_str)
            except (ValueError, IndexError):
                # fallback: use hash of string as id
                msg_id = hash(msg_id_str) % (10 ** 9)  # 9-digit number
            
            snippet = (msg.text or "")[:64]
            if debug_rag:
                syslog2(
                    LOG_DEBUG,
                    "msg_search contents message",
                    chunk_id=chunk_id,
                    msg_idx=msg_idx,
                    msg_id_str=msg_id_str,
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
            all_message_parts.append(parts)
    
    return all_message_parts


def search_message_contents(
    retrieval: RetrievalService,
    db: Database,
    query: str,
    top_k: int = 3,
    debug_rag: bool = False,
    thr: Optional[float] = None,
    use_two_stage: bool = False,
) -> List[List[Dict]]:
    """
    Search for message contents by text query and return formatted message parts.
    
    Args:
        retrieval: RetrievalService instance
        db: Database instance
        query: Search query string
        top_k: Number of results to return
        debug_rag: Enable detailed RAG debug logging
        thr: Maximum distance threshold for filtering results (None = no filtering)
        use_two_stage: If True, use two-stage search (L2 topics -> chunks), else use direct search
        
    Returns:
        List of lists, where each inner list contains message parts (Dict with id, date, sender, content, part, distance)
        Each message may be split into multiple parts if it exceeds MAX_TG_CONTENT_LEN
    """
    if debug_rag:
        syslog2(LOG_DEBUG, "msg_search contents start", query=query, top_k=top_k, use_two_stage=use_two_stage, thr=thr)

    # Step 1: Search for chunks
    results = _search_chunks(retrieval, query, top_k * 2, use_two_stage, debug_rag)
    
    # Step 2: Apply threshold filtering
    results = _apply_threshold_filter(results, thr, debug_rag)
    
    # Step 3: Limit to top_k results after filtering
    results = results[:top_k]
    
    # Log distance for all results (always visible, not just in debug mode)
    for idx, item in enumerate(results):
        distance = float(item.get("distance", 0.0))
        chunk_id = item.get("id", "unknown")
        syslog2(
            LOG_ALERT,
            "msg_search result distance",
            idx=idx,
            chunk_id=chunk_id,
            distance=distance,
            thr=thr,
        )
    
    # Step 4: Prepare message parts
    all_message_parts = _prepare_message_parts(db, results, debug_rag)
    
    syslog2(
        LOG_INFO,
        "message_search contents",
        query=query,
        top_k=top_k,
        messages=len(all_message_parts),
    )

    if debug_rag:
        syslog2(
            LOG_ALERT,
            "msg_search contents done",
            query=query,
            requested=top_k,
            returned=len(all_message_parts),
        )

    return all_message_parts
