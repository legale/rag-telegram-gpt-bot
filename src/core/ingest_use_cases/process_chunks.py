"""Use case for processing messages into chunks."""

from __future__ import annotations

import uuid
from typing import List, Optional
from datetime import datetime

from src.core.domain import Chunk, Message
from src.core.interfaces import MessageStore, ChunkStore
from src.ingestion.chunker import MessageChunker, EnhancedTextChunk
from src.ingestion.parser import ChatMessage
from src.lib.syslog2 import *


class ProcessChunks:
    """
    Use case for creating chunks from messages and storing them.
    
    This use case depends only on MessageStore and ChunkStore interfaces.
    """

    def __init__(
        self,
        message_store: MessageStore,
        chunk_store: ChunkStore,
        chunker: MessageChunker,
    ):
        """
        Initialize use case.

        Args:
            message_store: MessageStore interface implementation
            chunk_store: ChunkStore interface implementation
            chunker: MessageChunker instance with configured parameters
        """
        self.message_store = message_store
        self.chunk_store = chunk_store
        self.chunker = chunker

    def execute(self, chat_id: Optional[str] = None, limit: int = 0, offset: int = 0) -> int:
        """
        Process messages into chunks and store them.

        Args:
            chat_id: Optional chat ID to filter messages (if None, processes all)
            limit: Maximum number of messages to process (0 = all)
            offset: Offset for pagination

        Returns:
            Number of chunks created
        """
        syslog2(LOG_NOTICE, "starting process chunks", chat_id=chat_id, limit=limit, offset=offset)

        # Get messages from store
        if chat_id:
            # Get messages for specific chat
            messages = self.message_store.get_by_chat(chat_id, limit=limit if limit > 0 else 1000000, offset=offset)
        else:
            # Get all messages (using count and pagination)
            # For simplicity, get in batches
            messages = []
            batch_size = 10000
            current_offset = offset
            while True:
                batch = self.message_store.get_by_chat("", limit=batch_size, offset=current_offset)
                if not batch:
                    break
                messages.extend(batch)
                if limit > 0 and len(messages) >= limit:
                    messages = messages[:limit]
                    break
                current_offset += batch_size

        if not messages:
            syslog2(LOG_ERR, "no messages found in store")
            raise ValueError("no messages found in store")

        syslog2(LOG_NOTICE, "found messages in store", count=len(messages))

        # Convert domain Message to ChatMessage for chunker
        chat_messages: List[ChatMessage] = []
        chat_id_from_messages = None
        for msg in messages:
            # Extract original msg_id from composite_id (remove chat_id prefix)
            original_id = msg.id.split('_', 1)[1] if '_' in msg.id else msg.id
            chat_messages.append(ChatMessage(
                id=original_id,
                timestamp=msg.timestamp,
                sender=msg.from_id or "Unknown",
                content=msg.text
            ))
            if chat_id_from_messages is None:
                chat_id_from_messages = msg.chat_id

        # Use chunker to create chunks
        syslog2(LOG_NOTICE, "creating chunks",
                chunk_token_min=self.chunker.chunk_token_min,
                chunk_token_max=self.chunker.chunk_token_max,
                chunk_overlap_ratio=self.chunker.chunk_overlap_ratio)
        enhanced_chunks = self.chunker.chunk_messages(chat_messages)
        syslog2(LOG_NOTICE, "chunks created", count=len(enhanced_chunks))

        # Convert EnhancedTextChunk to domain Chunk objects
        domain_chunks: List[Chunk] = []
        final_chat_id = chat_id or chat_id_from_messages or "unknown_chat"

        for i, enhanced_chunk in enumerate(enhanced_chunks, 1):
            print(f"\rProcessing chunks: {i}/{len(enhanced_chunks)}", end="", flush=True)

            chunk_id = str(uuid.uuid4())

            # Prepare metadata
            meta_dict = {
                "message_count": enhanced_chunk.metadata.message_count,
                "start_date": enhanced_chunk.metadata.ts_from.isoformat(),
                "end_date": enhanced_chunk.metadata.ts_to.isoformat(),
                "chat_id": final_chat_id,
            }

            # Construct composite msg_ids for backward compatibility
            msg_id_start = f"{final_chat_id}_{enhanced_chunk.metadata.msg_id_start}"
            msg_id_end = f"{final_chat_id}_{enhanced_chunk.metadata.msg_id_end}"

            domain_chunk = Chunk(
                id=chunk_id,
                text=enhanced_chunk.text,
                msg_ids=(msg_id_start, msg_id_end),
                valid_period=(enhanced_chunk.metadata.ts_from, enhanced_chunk.metadata.ts_to),
                embedding=None,  # Will be set in GenerateEmbeddings use case
                metadata=meta_dict
            )
            domain_chunks.append(domain_chunk)

        print()  # Newline after progress
        syslog2(LOG_NOTICE, "chunks prepared for storage", count=len(domain_chunks))

        # Save chunks using ChunkStore
        syslog2(LOG_NOTICE, "saving chunks to store")
        try:
            saved_count = self.chunk_store.save_batch(domain_chunks)
            syslog2(LOG_NOTICE, "chunks saved to store", count=saved_count)
        except Exception as e:
            syslog2(LOG_ERR, "error saving chunks", error=str(e))
            raise

        syslog2(LOG_NOTICE, "process chunks complete", chunks_saved=saved_count)
        return saved_count

