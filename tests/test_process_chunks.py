"""
Tests for process_chunks use case.
"""

import pytest
from src.core.ingest_use_cases.process_chunks import ProcessChunks
from src.core.domain import Message
from src.ingestion.chunker import MessageChunker
from datetime import datetime


class TestProcessChunks:
    """Tests for ProcessChunks class."""
    
    def test_init(self):
        """Test initialization."""
        class MockMessageStore:
            pass
        
        class MockChunkStore:
            pass
        
        chunker = MessageChunker()
        use_case = ProcessChunks(MockMessageStore(), MockChunkStore(), chunker)
        assert use_case.message_store is not None
        assert use_case.chunk_store is not None
        assert use_case.chunker == chunker
    
    def test_execute_with_no_messages(self):
        """Test execute with no messages."""
        class MockMessageStore:
            def get_by_chat(self, chat_id, limit, offset):
                return []
        
        class MockChunkStore:
            pass
        
        chunker = MessageChunker()
        use_case = ProcessChunks(MockMessageStore(), MockChunkStore(), chunker)
        
        with pytest.raises(ValueError):
            use_case.execute()

