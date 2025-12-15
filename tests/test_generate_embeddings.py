"""
Tests for generate_embeddings use case.
"""

import pytest
from src.core.ingest_use_cases.generate_embeddings import GenerateEmbeddings
from src.core.domain import Chunk


class TestGenerateEmbeddings:
    """Tests for GenerateEmbeddings class."""
    
    def test_init(self):
        """Test initialization."""
        class MockChunkStore:
            def get_by_ids(self, ids):
                return []
        
        class MockEmbedder:
            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]
        
        use_case = GenerateEmbeddings(MockChunkStore(), MockEmbedder())
        assert use_case.chunk_store is not None
        assert use_case.embedder is not None
    
    def test_execute_with_chunk_ids(self):
        """Test execute with chunk IDs."""
        class MockChunkStore:
            def get_by_ids(self, ids):
                return [
                    Chunk(id="chunk1", text="Test", metadata={}, embedding=None)
                ]
            
            def save_batch(self, chunks):
                return len(chunks)
        
        class MockEmbedder:
            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]
        
        use_case = GenerateEmbeddings(MockChunkStore(), MockEmbedder())
        result = use_case.execute(chunk_ids=["chunk1"])
        assert result == 1
    
    def test_execute_with_chunks_already_embedded(self):
        """Test execute with chunks that already have embeddings."""
        class MockChunkStore:
            def get_by_ids(self, ids):
                return [
                    Chunk(id="chunk1", text="Test", metadata={}, embedding=[0.1] * 384)
                ]
        
        class MockEmbedder:
            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]
        
        use_case = GenerateEmbeddings(MockChunkStore(), MockEmbedder())
        result = use_case.execute(chunk_ids=["chunk1"])
        assert result == 0
    
    def test_execute_without_chunk_ids(self):
        """Test execute without chunk IDs."""
        class MockChunkStore:
            pass
        
        class MockEmbedder:
            pass
        
        use_case = GenerateEmbeddings(MockChunkStore(), MockEmbedder())
        with pytest.raises(NotImplementedError):
            use_case.execute(chunk_ids=None)

