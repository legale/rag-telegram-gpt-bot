
import pytest
from datetime import datetime
from src.ingestion.parser import ChatMessage
from src.ingestion.chunker import MessageChunker

def test_chunker_simple():
    messages = [
        ChatMessage(id=str(i), timestamp=datetime.now(), sender="Me", content=f"msg {i}")
        for i in range(25)
    ]
    
    chunker = MessageChunker(max_messages_per_chunk=10, overlap=0)
    chunks = chunker.chunk_messages(messages)
    
    # 0-9, 10-19, 20-24
    assert len(chunks) == 3
    assert chunks[0].metadata.message_count == 10
    assert chunks[1].metadata.message_count == 10
    assert chunks[2].metadata.message_count == 5
    
    assert chunks[0].metadata.msg_id_start == "0"
    assert chunks[0].metadata.msg_id_end == "9"
    
    assert chunks[1].metadata.msg_id_start == "10"
    assert chunks[2].metadata.msg_id_start == "20"
