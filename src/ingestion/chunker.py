from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
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
    Splits chat messages into logical chunks of fixed size (N messages).
    Designed for Phase 14 hierarchical clustering.
    """
    
    def __init__(self, max_messages_per_chunk: int = 10, overlap: int = 0):
        self.max_messages_per_chunk = max_messages_per_chunk
        self.overlap = overlap
        
    def chunk_messages(self, messages: List[ChatMessage]) -> List[EnhancedTextChunk]:
        """
        Splits a list of messages into chunks.
        Assumes messages are sorted by timestamp.
        """
        chunks = []
        step = max(1, self.max_messages_per_chunk - self.overlap)
        
        for i in range(0, len(messages), step):
            msg_batch = messages[i:i + self.max_messages_per_chunk]
            if not msg_batch:
                continue
                
            text = self._chunk_to_text(msg_batch)
            
            meta = ChunkMetadata(
                msg_id_start=msg_batch[0].id,
                msg_id_end=msg_batch[-1].id,
                ts_from=msg_batch[0].timestamp,
                ts_to=msg_batch[-1].timestamp,
                message_count=len(msg_batch)
            )
            
            chunks.append(EnhancedTextChunk(
                text=text,
                metadata=meta,
                original_messages=msg_batch
            ))
            
        return chunks

    def _chunk_to_text(self, chunk: List[ChatMessage]) -> str:
        """Converts a chunk of messages into a single string."""
        return "\n".join([f"{msg.sender}: {msg.content}" for msg in chunk])
