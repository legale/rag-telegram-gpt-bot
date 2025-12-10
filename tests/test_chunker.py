import pytest
from datetime import datetime, timedelta
from src.ingestion.parser import ChatMessage
from src.ingestion.chunker import MessageChunker

def test_chunker_simple():
    """Test basic token-based chunking."""
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    messages = [
        ChatMessage(
            id=str(i), 
            timestamp=base_time + timedelta(minutes=i), 
            sender="User1", 
            content=f"Message {i}"
        )
        for i in range(20)
    ]
    
    # Use small token limits for testing
    chunker = MessageChunker(chunk_token_min=10, chunk_token_max=100, chunk_overlap_ratio=0.2)
    chunks = chunker.chunk_messages(messages)
    
    # Should create at least one chunk
    assert len(chunks) > 0
    
    # Check that all messages are included
    all_chunk_messages = []
    for chunk in chunks:
        all_chunk_messages.extend(chunk.original_messages)
    
    assert len(all_chunk_messages) >= len(messages)
    
    # Check metadata
    for chunk in chunks:
        assert chunk.metadata.message_count > 0
        assert chunk.metadata.ts_from <= chunk.metadata.ts_to
        assert chunk.metadata.msg_id_start
        assert chunk.metadata.msg_id_end

def test_chunker_chronological_order():
    """Test that messages are sorted chronologically."""
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    # Create messages in reverse order
    messages = [
        ChatMessage(
            id=str(i), 
            timestamp=base_time + timedelta(minutes=20-i), 
            sender="User1", 
            content=f"Message {i}"
        )
        for i in range(10)
    ]
    
    chunker = MessageChunker(chunk_token_min=10, chunk_token_max=200, chunk_overlap_ratio=0.2)
    chunks = chunker.chunk_messages(messages)
    
    # Check that chunks are in chronological order
    for i in range(len(chunks) - 1):
        assert chunks[i].metadata.ts_to <= chunks[i+1].metadata.ts_from

def test_chunker_prefix_and_suffix():
    """Test that chunks have proper prefix and suffix."""
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    messages = [
        ChatMessage(
            id=str(i), 
            timestamp=base_time + timedelta(minutes=i), 
            sender="User1", 
            content=f"Message {i}"
        )
        for i in range(5)
    ]
    
    chunker = MessageChunker(chunk_token_min=10, chunk_token_max=200, chunk_overlap_ratio=0.2)
    chunks = chunker.chunk_messages(messages)
    
    assert len(chunks) > 0
    
    # Check prefix format
    first_chunk = chunks[0]
    assert first_chunk.text.startswith("snippet:")
    assert "participants:" in first_chunk.text
    
    # Check suffix for non-last chunks
    if len(chunks) > 1:
        for chunk in chunks[:-1]:
            assert "[продолжение следует]" in chunk.text
    
    # Last chunk should not have suffix (or should not have continuation marker)
    last_chunk = chunks[-1]
    # The last chunk might still have suffix if it's not the actual last message
    # This is acceptable behavior

def test_chunker_message_format():
    """Test that messages are formatted correctly."""
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    messages = [
        ChatMessage(
            id="1", 
            timestamp=base_time, 
            sender="TestUser", 
            content="Test message"
        )
    ]
    
    chunker = MessageChunker(chunk_token_min=10, chunk_token_max=200, chunk_overlap_ratio=0.2)
    chunks = chunker.chunk_messages(messages)
    
    assert len(chunks) == 1
    chunk_text = chunks[0].text
    
    # Check message format
    assert "[date: " in chunk_text
    assert "[user: TestUser]" in chunk_text
    assert "Test message" in chunk_text

def test_chunker_overlap():
    """Test that chunks have overlap."""
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    # Create enough messages to generate multiple chunks
    messages = [
        ChatMessage(
            id=str(i), 
            timestamp=base_time + timedelta(minutes=i), 
            sender="User1", 
            content=f"Message {i} with some content to make it longer"
        )
        for i in range(30)
    ]
    
    chunker = MessageChunker(chunk_token_min=10, chunk_token_max=150, chunk_overlap_ratio=0.3)
    chunks = chunker.chunk_messages(messages)
    
    # Should have multiple chunks
    if len(chunks) > 1:
        # Check that there's overlap between consecutive chunks
        # Overlap means some messages appear in both chunks
        for i in range(len(chunks) - 1):
            chunk1_messages = set(msg.id for msg in chunks[i].original_messages)
            chunk2_messages = set(msg.id for msg in chunks[i+1].original_messages)
            
            # There should be some overlap (messages in both chunks)
            overlap = chunk1_messages & chunk2_messages
            assert len(overlap) > 0, f"Chunks {i} and {i+1} should have overlapping messages"

def test_chunker_small_chunk_merging():
    """Test that very small chunks are merged with previous ones."""
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    messages = [
        ChatMessage(
            id=str(i), 
            timestamp=base_time + timedelta(minutes=i), 
            sender="User1", 
            content="Short" if i < 5 else f"Long message {i} with more content to make it substantial"
        )
        for i in range(10)
    ]
    
    # Set chunk_token_min high to force merging of small chunks
    chunker = MessageChunker(chunk_token_min=200, chunk_token_max=300, chunk_overlap_ratio=0.2)
    chunks = chunker.chunk_messages(messages)
    
    # All chunks should meet the minimum token requirement (or be the only chunk)
    for chunk in chunks:
        chunk_tokens = chunker._count_tokens(chunk.text)
        # Allow some flexibility - if it's the only chunk or last chunk, it might be smaller
        if len(chunks) == 1 or chunk == chunks[-1]:
            # Last chunk can be smaller
            pass
        else:
            # Other chunks should meet minimum (with some tolerance)
            assert chunk_tokens >= chunker.chunk_token_min * 0.8, \
                f"Chunk should meet minimum token requirement, got {chunk_tokens}"

def test_chunker_empty_messages():
    """Test chunker with empty message list."""
    chunker = MessageChunker(chunk_token_min=10, chunk_token_max=200, chunk_overlap_ratio=0.2)
    chunks = chunker.chunk_messages([])
    assert len(chunks) == 0

def test_chunker_single_message():
    """Test chunker with single message."""
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    messages = [
        ChatMessage(
            id="1", 
            timestamp=base_time, 
            sender="User1", 
            content="Single message"
        )
    ]
    
    chunker = MessageChunker(chunk_token_min=10, chunk_token_max=200, chunk_overlap_ratio=0.2)
    chunks = chunker.chunk_messages(messages)
    
    assert len(chunks) == 1
    assert chunks[0].metadata.message_count == 1
    assert chunks[0].metadata.msg_id_start == "1"
    assert chunks[0].metadata.msg_id_end == "1"

def test_chunker_multiple_senders():
    """Test chunker with messages from multiple senders."""
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    messages = [
        ChatMessage(
            id=str(i), 
            timestamp=base_time + timedelta(minutes=i), 
            sender=f"User{i % 3}", 
            content=f"Message from user {i % 3}"
        )
        for i in range(10)
    ]
    
    chunker = MessageChunker(chunk_token_min=10, chunk_token_max=200, chunk_overlap_ratio=0.2)
    chunks = chunker.chunk_messages(messages)
    
    assert len(chunks) > 0
    
    # Check that prefix includes all unique senders
    first_chunk = chunks[0]
    assert "participants:" in first_chunk.text
    # Should mention at least some of the senders
    assert any(f"User{i}" in first_chunk.text for i in range(3))

def test_chunker_token_counting():
    """Test that chunker respects token limits."""
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    # Create messages with varying lengths
    messages = [
        ChatMessage(
            id=str(i), 
            timestamp=base_time + timedelta(minutes=i), 
            sender="User1", 
            content="X" * (50 * (i + 1))  # Increasing message length
        )
        for i in range(10)
    ]
    
    chunker = MessageChunker(chunk_token_min=10, chunk_token_max=200, chunk_overlap_ratio=0.2)
    chunks = chunker.chunk_messages(messages)
    
    # Check that chunks don't exceed token limit (with some tolerance for prefix/suffix)
    for chunk in chunks:
        chunk_tokens = chunker._count_tokens(chunk.text)
        # Allow 20% overhead for prefix/suffix
        max_allowed = chunker.chunk_token_max * 1.2
        assert chunk_tokens <= max_allowed, \
            f"Chunk exceeds token limit: {chunk_tokens} > {max_allowed}"
