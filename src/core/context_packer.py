"""Context packer for deduplication, neighbor enrichment, and token budget management."""

from typing import List, Set, Tuple, Optional, Callable
from src.core.domain import SearchResult, Chunk, Message
from src.core.interfaces import MessageStore
from src.lib.syslog2 import *


def pack(
    candidates: List[tuple],
    message_store: MessageStore,
    output_mode: str = "context",
    message_window_sec: int = 300,
    max_tokens: int = 4000,
    log_level: int = LOG_WARNING,
) -> List[SearchResult]:
    """
    Pack context: dedup by msg_id, add neighbors, respect token budget.

    Args:
        candidates: List of (chunk, combined_score, vector_score) tuples
        message_store: Message store for retrieving messages
        output_mode: "context" or "evidence"
        message_window_sec: Time window for neighbors
        max_tokens: Maximum tokens for context
        log_level: Logging level

    Returns:
        List of SearchResult objects
    """
    results = []
    seen_msg_ids: Set[str] = set()
    current_tokens = 0
    
    # Estimate tokens (rough: 1 token ≈ 4 characters)
    def estimate_tokens(text: str) -> int:
        return len(text) // 4

    for chunk, combined_score, vector_score in candidates:
        # Dedup by msg_id
        if not _deduplicate_by_msg_id(chunk, seen_msg_ids):
            continue

        # Check token budget
        chunk_tokens = estimate_tokens(chunk.text)
        if not _check_token_budget(chunk_tokens, current_tokens, max_tokens):
            break
        current_tokens += chunk_tokens

        # Get topics from metadata (removed - clustering is deprecated)
        topics = []

        # Enrich with messages based on output mode
        original_messages, tokens_added = _enrich_with_messages(
            chunk, message_store, output_mode, message_window_sec, seen_msg_ids, current_tokens, max_tokens, estimate_tokens, log_level
        )
        current_tokens += tokens_added

        # Create SearchResult
        result = SearchResult(
            chunk=chunk,
            score=combined_score,
            original_messages=original_messages,
            topics=topics
        )
        results.append(result)

    return results


def _deduplicate_by_msg_id(chunk: Chunk, seen_msg_ids: Set[str]) -> bool:
    """
    Check if chunk should be included based on msg_id deduplication.
    
    Args:
        chunk: Chunk to check
        seen_msg_ids: Set of already seen msg_id keys
        
    Returns:
        True if chunk should be included (not duplicate), False otherwise
    """
    if chunk.msg_ids:
        msg_id_key = f"{chunk.msg_ids[0]}_{chunk.msg_ids[1] if len(chunk.msg_ids) > 1 else chunk.msg_ids[0]}"
        if msg_id_key in seen_msg_ids:
            return False
        seen_msg_ids.add(msg_id_key)
    return True


def _check_token_budget(chunk_tokens: int, current_tokens: int, max_tokens: int) -> bool:
    """
    Check if adding chunk tokens would exceed token budget.
    
    Args:
        chunk_tokens: Number of tokens in the chunk
        current_tokens: Current token count
        max_tokens: Maximum allowed tokens
        
    Returns:
        True if chunk can be added without exceeding budget, False otherwise
    """
    return current_tokens + chunk_tokens <= max_tokens


def _enrich_with_messages(
    chunk: Chunk,
    message_store: MessageStore,
    output_mode: str,
    message_window_sec: int,
    seen_msg_ids: Set[str],
    current_tokens: int,
    max_tokens: int,
    estimate_tokens: Callable[[str], int],
    log_level: int = LOG_WARNING,
) -> Tuple[List[Message], int]:
    """
    Enrich chunk with surrounding messages based on output mode.
    
    Args:
        chunk: Chunk to enrich
        message_store: Message store for retrieving messages
        output_mode: "context" or "evidence"
        message_window_sec: Time window for neighbors
        seen_msg_ids: Set of already seen msg_ids
        current_tokens: Current token count
        max_tokens: Maximum allowed tokens
        estimate_tokens: Function to estimate tokens in text
        log_level: Logging level
        
    Returns:
        Tuple of (list of messages, total tokens added)
    """
    original_messages: List[Message] = []
    tokens_added = 0
    
    if output_mode == "context" and chunk.valid_period:
        # Get messages around chunk time
        time_point = chunk.valid_period[0]
        chat_id = chunk.metadata.get("chat_id") if chunk.metadata else None
        
        if chat_id:
            messages = message_store.get_context(
                chat_id=chat_id,
                time_point=time_point,
                window_sec=message_window_sec
            )
            # Filter out messages we've already seen
            for msg in messages:
                if msg.id not in seen_msg_ids:
                    msg_tokens = estimate_tokens(msg.text or "")
                    if current_tokens + tokens_added + msg_tokens <= max_tokens:
                        original_messages.append(msg)
                        seen_msg_ids.add(msg.id)
                        tokens_added += msg_tokens
    
    return original_messages, tokens_added

