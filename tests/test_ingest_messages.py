"""
Tests for ingest_messages use case.
"""

import pytest
from src.core.ingest_use_cases.ingest_messages import IngestMessages
from src.core.domain import Message
from src.ingestion.parser import ChatParser, ChatMessage
from datetime import datetime


class TestIngestMessages:
    """Tests for IngestMessages class."""
    
    def test_init(self):
        """Test initialization."""
        class MockMessageStore:
            def save_batch(self, messages):
                return len(messages)
        
        use_case = IngestMessages(MockMessageStore())
        assert use_case.message_store is not None
        assert use_case.parser is not None
    
    def test_init_with_parser(self):
        """Test initialization with custom parser."""
        class MockMessageStore:
            pass
        
        parser = ChatParser()
        use_case = IngestMessages(MockMessageStore(), parser=parser)
        assert use_case.parser == parser
    
    def test_execute_empty_file_path(self):
        """Test execute with empty file path."""
        class MockMessageStore:
            pass
        
        use_case = IngestMessages(MockMessageStore())
        with pytest.raises(ValueError):
            use_case.execute("")

