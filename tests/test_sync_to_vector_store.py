"""
Tests for sync_to_vector_store use case.
"""

import pytest
from src.core.ingest_use_cases.sync_to_vector_store import SyncToVectorStore
from src.core.domain import Chunk


class TestSyncToVectorStore:
    """Tests for SyncToVectorStore class."""
    
    def test_init(self):
        """Test initialization."""
        class MockChunkStore:
            pass
        
        class MockVectorIndex:
            pass
        
        use_case = SyncToVectorStore(MockChunkStore(), MockVectorIndex())
        assert use_case.chunk_store is not None
        assert use_case.vector_index is not None
    
    def test_execute_with_chunk_ids(self):
        """Test execute with chunk IDs."""
        class MockChunkStore:
            def get_by_ids(self, ids):
                return [
                    Chunk(id="chunk1", text="Test", metadata={}, embedding=[0.1] * 384)
                ]
        
        class MockVectorIndex:
            def upsert(self, items):
                pass
        
        use_case = SyncToVectorStore(MockChunkStore(), MockVectorIndex())
        result = use_case.execute(chunk_ids=["chunk1"])
        assert result == 1
    
    def test_execute_with_chunks_without_embeddings(self):
        """Test execute with chunks without embeddings."""
        class MockChunkStore:
            def get_by_ids(self, ids):
                return [
                    Chunk(id="chunk1", text="Test", metadata={}, embedding=None)
                ]
        
        class MockVectorIndex:
            pass
        
        use_case = SyncToVectorStore(MockChunkStore(), MockVectorIndex())
        result = use_case.execute(chunk_ids=["chunk1"])
        assert result == 0
    
    def test_execute_without_chunk_ids(self):
        """Test execute without chunk IDs."""
        class MockChunkStore:
            pass
        
        class MockVectorIndex:
            pass
        
        use_case = SyncToVectorStore(MockChunkStore(), MockVectorIndex())
        with pytest.raises(NotImplementedError):
            use_case.execute(chunk_ids=None)

