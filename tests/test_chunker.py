import pytest
from datetime import datetime
from src.ingestion.parser import ChatMessage
from src.ingestion.chunker import TextChunker

def test_chunk_messages():
    chunker = TextChunker(max_messages_per_chunk=10, overlap=2)
    
    # Create 25 dummy messages
    messages = [
        ChatMessage(
            id=str(i),
            timestamp=datetime.now(),
            sender=f"User {i}",
            content=f"Message {i}"
        ) for i in range(25)
    ]
    
    chunks = chunker.chunk_messages(messages)
    
    # Expected chunks with overlap=2, max=10, step=8:
    # 1. 0-9 (10 msgs)
    # 2. 8-17 (10 msgs)
    # 3. 16-25 (9 msgs) -> 16-24 actually (indices 0-24)
    
    assert len(chunks) == 3
    assert len(chunks[0].original_messages) == 10
    assert len(chunks[1].original_messages) == 10
    assert len(chunks[2].original_messages) == 9
    
    # Check overlap
    # Chunk 1 starts at 8 (Message 8)
    assert chunks[1].original_messages[0].content == "Message 8"
    # Chunk 0 ends at 9 (Message 9)
    assert chunks[0].original_messages[-1].content == "Message 9"
    
    # Overlap check: Message 8 and 9 should be in both chunk 0 and 1
    chunk0_ids = [m.id for m in chunks[0].original_messages]
    chunk1_ids = [m.id for m in chunks[1].original_messages]
    
    assert "8" in chunk0_ids and "8" in chunk1_ids
    assert "9" in chunk0_ids and "9" in chunk1_ids
    
    # Check text content
    assert "User 0: Message 0" in chunks[0].text
    assert "User 9: Message 9" in chunks[0].text
    
    # Check metadata
    assert chunks[0].metadata["message_count"] == 10
    assert chunks[1].metadata["message_count"] == 10
    assert chunks[2].metadata["message_count"] == 9

def test_chunk_to_text():
    chunker = TextChunker()
    messages = [
        ChatMessage(id="1", timestamp=datetime.now(), sender="Alice", content="Hi"),
        ChatMessage(id="2", timestamp=datetime.now(), sender="Bob", content="Hello"),
    ]
    text = chunker.chunk_to_text(messages)
    assert "Alice: Hi" in text
    assert "Bob: Hello" in text
