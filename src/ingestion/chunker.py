from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime
import tiktoken
from src.ingestion.parser import ChatMessage

@dataclass
class ChunkMetadata:
    msg_id_start: str
    msg_id_end: str
    ts_from: datetime
    ts_to: datetime
    message_count: int

@dataclass
class EnhancedTextChunk:
    text: str
    metadata: ChunkMetadata
    original_messages: List[ChatMessage]

class MessageChunker:
    """
    Splits chat messages into chunks based on token limits with overlap.
    Uses token-based chunking instead of message count.
    """
    
    def __init__(self, chunk_token_min: int = 50, chunk_token_max: int = 400, chunk_overlap_ratio: float = 0.3):
        """
        Initialize MessageChunker with token-based parameters.
        
        Args:
            chunk_token_min: Minimum chunk size in tokens (for post-processing)
            chunk_token_max: Maximum chunk size in tokens
            chunk_overlap_ratio: Ratio of overlap tokens (0.0-1.0)
        """
        self.chunk_token_min = chunk_token_min
        self.chunk_token_max = chunk_token_max
        self.chunk_overlap_ratio = chunk_overlap_ratio
        
        # Initialize tiktoken encoder
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback if tiktoken fails
            self.encoding = None
        
        # Token count cache: message_id -> token_count
        self._token_cache: Dict[str, int] = {}
    
    def _count_tokens(self, text: str, cache_key: Optional[str] = None) -> int:
        """
        Count tokens in text using tiktoken.
        
        Args:
            text: Text to count tokens for
            cache_key: Optional cache key (message ID) to cache the result
        """
        # Check cache first if key provided
        if cache_key and cache_key in self._token_cache:
            return self._token_cache[cache_key]
        
        # Count tokens
        if self.encoding is None:
            # Fallback: approximate token count (1 token ≈ 4 characters)
            count = len(text) // 4
        else:
            count = len(self.encoding.encode(text))
        
        # Cache result if key provided
        if cache_key:
            self._token_cache[cache_key] = count
        
        return count
    
    def _format_message(self, msg: ChatMessage) -> str:
        """Format a single message with metadata."""
        date_str = msg.timestamp.strftime("%Y-%m-%d %H:%M")
        # Using sender as both user_id and user_name since we don't have numeric IDs
        return f"[date: {date_str}] [user: {msg.sender}] {msg.content}"
    
    def _create_prefix(self, start_date: datetime, end_date: datetime, unique_senders: set) -> str:
        """Create prefix for chunk with date range and participants."""
        start_str = start_date.strftime("%Y-%m-%d %H:%M")
        end_str = end_date.strftime("%Y-%m-%d %H:%M")
        participants = ", ".join(sorted(unique_senders))
        return f"snippet: {start_str}–{end_str}. participants: {participants}.\n"
    
    def chunk_messages(self, messages: List[ChatMessage]) -> List[EnhancedTextChunk]:
        """
        Splits a list of messages into token-based chunks with overlap.
        Assumes messages are sorted by timestamp (will sort if not).
        """
        if not messages:
            return []
        
        # Sort messages by timestamp to ensure chronological order
        sorted_messages = sorted(messages, key=lambda m: m.timestamp)
        
        # Pre-compute and cache token counts for all messages
        self._token_cache.clear()  # Clear cache for new batch
        for msg in sorted_messages:
            formatted_msg = self._format_message(msg)
            # Cache token count using message ID as key
            self._count_tokens(formatted_msg, cache_key=msg.id)
        
        # Step 1: Create initial chunks based on token limits
        raw_chunks = []
        current_chunk_messages = []
        current_chunk_tokens = 0
        
        for msg in sorted_messages:
            formatted_msg = self._format_message(msg)
            msg_tokens = self._count_tokens(formatted_msg, cache_key=msg.id)
            
            # Check if adding this message would exceed the limit
            # Account for newline character
            newline_tokens = self._count_tokens("\n") if current_chunk_messages else 0
            total_tokens_if_added = current_chunk_tokens + newline_tokens + msg_tokens
            
            if total_tokens_if_added > self.chunk_token_max and current_chunk_messages:
                # Current chunk is full, save it and start a new one
                raw_chunks.append(current_chunk_messages)
                current_chunk_messages = [msg]
                current_chunk_tokens = msg_tokens
            else:
                # Add message to current chunk
                if current_chunk_messages:
                    current_chunk_tokens += newline_tokens
                current_chunk_messages.append(msg)
                current_chunk_tokens += msg_tokens
        
        # Add the last chunk if it has messages
        if current_chunk_messages:
            raw_chunks.append(current_chunk_messages)
        
        # Step 2: Add overlap between chunks
        chunks_with_overlap = []
        previous_overlap_messages = []
        
        for i, chunk_messages in enumerate(raw_chunks):
            # Calculate overlap tokens needed
            overlap_tokens_target = int(self.chunk_token_max * self.chunk_overlap_ratio)
            
            # If we have previous overlap, add it to the beginning
            if previous_overlap_messages:
                # Add overlap messages to the beginning of current chunk
                chunk_messages = previous_overlap_messages + chunk_messages
            
            # Calculate overlap for next chunk (last N messages that fit in overlap_tokens_target)
            if i < len(raw_chunks) - 1:  # Not the last chunk
                overlap_messages = []
                overlap_tokens = 0
                
                # Start from the end and work backwards
                for msg in reversed(chunk_messages):
                    formatted_msg = self._format_message(msg)
                    msg_tokens = self._count_tokens(formatted_msg, cache_key=msg.id)
                    newline_tokens = self._count_tokens("\n") if overlap_messages else 0
                    
                    if overlap_tokens + newline_tokens + msg_tokens <= overlap_tokens_target:
                        overlap_messages.insert(0, msg)
                        overlap_tokens += newline_tokens + msg_tokens
                    else:
                        break
                
                previous_overlap_messages = overlap_messages
            else:
                previous_overlap_messages = []
            
            chunks_with_overlap.append(chunk_messages)
        
        # Step 3: Create EnhancedTextChunk objects with prefixes and suffixes
        enhanced_chunks = []
        total_messages = len(sorted_messages)
        
        for i, chunk_messages in enumerate(chunks_with_overlap):
            if not chunk_messages:
                continue
            
            # Get unique senders
            unique_senders = set(msg.sender for msg in chunk_messages)
            
            # Create prefix
            start_date = chunk_messages[0].timestamp
            end_date = chunk_messages[-1].timestamp
            prefix = self._create_prefix(start_date, end_date, unique_senders)
            
            # Format messages
            formatted_messages = [self._format_message(msg) for msg in chunk_messages]
            messages_text = "\n".join(formatted_messages)
            
            # Determine if this is the last chunk
            is_last_chunk = (i == len(chunks_with_overlap) - 1)
            last_msg_id = chunk_messages[-1].id
            is_actually_last = (last_msg_id == sorted_messages[-1].id)
            
            # Add suffix if chunk doesn't end with the last message
            suffix = ""
            if not is_actually_last:
                suffix = "\n[продолжение следует]"
            
            # Combine prefix, messages, and suffix
            full_text = prefix + messages_text + suffix
            
            # Create metadata
            meta = ChunkMetadata(
                msg_id_start=chunk_messages[0].id,
                msg_id_end=chunk_messages[-1].id,
                ts_from=start_date,
                ts_to=end_date,
                message_count=len(chunk_messages)
            )
            
            enhanced_chunks.append(EnhancedTextChunk(
                text=full_text,
                metadata=meta,
                original_messages=chunk_messages
            ))
        
        # Step 4: Post-processing - merge small chunks with previous ones
        # Use structural message storage instead of string manipulation
        final_chunks = []
        for i, chunk in enumerate(enhanced_chunks):
            # Calculate token count using cached values
            chunk_tokens = sum(
                self._count_tokens(self._format_message(msg), cache_key=msg.id)
                for msg in chunk.original_messages
            )
            # Add prefix and suffix tokens (approximate)
            prefix_tokens = self._count_tokens(self._create_prefix(
                chunk.metadata.ts_from, chunk.metadata.ts_to,
                set(msg.sender for msg in chunk.original_messages)
            ))
            suffix_tokens = self._count_tokens("\n[продолжение следует]") if i < len(enhanced_chunks) - 1 else 0
            chunk_tokens += prefix_tokens + suffix_tokens
            
            if chunk_tokens < self.chunk_token_min and final_chunks:
                # Merge with previous chunk using structural approach
                prev_chunk = final_chunks[-1]
                
                # Combine messages structurally
                combined_messages = prev_chunk.original_messages + chunk.original_messages
                
                # Update metadata
                combined_meta = ChunkMetadata(
                    msg_id_start=prev_chunk.metadata.msg_id_start,
                    msg_id_end=chunk.metadata.msg_id_end,
                    ts_from=prev_chunk.metadata.ts_from,
                    ts_to=chunk.metadata.ts_to,
                    message_count=len(combined_messages)
                )
                
                # Re-render text from messages (no string manipulation needed)
                unique_senders = set(msg.sender for msg in combined_messages)
                prefix = self._create_prefix(combined_meta.ts_from, combined_meta.ts_to, unique_senders)
                formatted_messages = [self._format_message(msg) for msg in combined_messages]
                messages_text = "\n".join(formatted_messages)
                
                # Determine if this is the last chunk
                is_last = (chunk.metadata.msg_id_end == sorted_messages[-1].id)
                suffix = "" if is_last else "\n[продолжение следует]"
                
                combined_text = prefix + messages_text + suffix
                
                # Update previous chunk
                final_chunks[-1] = EnhancedTextChunk(
                    text=combined_text,
                    metadata=combined_meta,
                    original_messages=combined_messages
                )
            else:
                # Keep chunk as is
                final_chunks.append(chunk)
        
        return final_chunks
