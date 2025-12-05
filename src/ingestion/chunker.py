from dataclasses import dataclass
from typing import List, Dict, Any
from src.ingestion.parser import ChatMessage

@dataclass
class TextChunk:
    text: str
    metadata: Dict[str, Any]
    original_messages: List[ChatMessage]

class TextChunker:
    """Splits chat messages into logical chunks."""
    
    def __init__(self, max_messages_per_chunk: int = 10, overlap: int = 2):
        self.max_messages_per_chunk = max_messages_per_chunk
        self.overlap = overlap
        
    def chunk_messages(self, messages: List[ChatMessage]) -> List[TextChunk]:
        """
        Splits a list of messages into chunks with overlap.
        
        Args:
            messages: List of ChatMessage objects.
            
        Returns:
            List of TextChunk objects.
        """
        chunks = []
        step = self.max_messages_per_chunk - self.overlap
        if step <= 0:
            step = 1 # Prevent infinite loop if overlap >= max_messages
            
        for i in range(0, len(messages), step):
            msg_batch = messages[i:i + self.max_messages_per_chunk]
            
            # If the last chunk is too small (e.g. just the overlap), we might want to skip it 
            # or merge it, but for now we'll keep it unless it's empty.
            if not msg_batch:
                break
                
            text = self.chunk_to_text(msg_batch)
            
            # Create metadata
            metadata = {
                "message_count": len(msg_batch),
                "start_date": msg_batch[0].timestamp.isoformat() if msg_batch else None,
                "end_date": msg_batch[-1].timestamp.isoformat() if msg_batch else None
            }
            
            chunks.append(TextChunk(
                text=text,
                metadata=metadata,
                original_messages=msg_batch
            ))
            
            # Stop if we've reached the end of the list
            if i + self.max_messages_per_chunk >= len(messages):
                break
                
        return chunks

    def chunk_to_text(self, chunk: List[ChatMessage]) -> str:
        """Converts a chunk of messages into a single string for embedding."""
        return "\n".join([f"{msg.sender}: {msg.content}" for msg in chunk])
