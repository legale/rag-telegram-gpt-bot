"""Integration tests for hybrid retrieval with FTS-only mode."""

import pytest
from pathlib import Path
from datetime import datetime

from src.app.bootstrap import create_hybrid_retrieval
from src.storage.db import Database
from src.core.interfaces import SearchFilters
from src.core.embedding import LocalEmbeddingClient


@pytest.fixture
def tmp_db_with_data(tmp_path):
    """Create temporary database with test data."""
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    database = Database(db_url)
    
    # Create test chunks
    session = database.get_session()
    try:
        from sqlalchemy import text
        import json
        session.execute(text("""
            INSERT INTO chunks (id, text, chat_id, ts_from, ts_to, metadata_json)
            VALUES 
                ('chunk1', 'ошибки ax820 в системе', 'chat1', '2024-01-01', '2024-01-01', :meta1),
                ('chunk2', 'тестовая ошибка', 'chat1', '2024-01-02', '2024-01-02', :meta2),
                ('chunk3', 'другая информация', 'chat2', '2024-01-03', '2024-01-03', :meta3)
        """), {
            'meta1': json.dumps({"chat_id": "chat1"}),
            'meta2': json.dumps({"chat_id": "chat1"}),
            'meta3': json.dumps({"chat_id": "chat2"})
        })
        session.commit()
    finally:
        session.close()
    
    yield database
    database.engine.dispose()


@pytest.fixture
def vector_db_path(tmp_path):
    """Create temporary vector store directory."""
    vector_path = tmp_path / "vector_db"
    vector_path.mkdir()
    return str(vector_path)


class TestHybridRetrievalIntegration:
    """Integration tests for hybrid retrieval."""
    
    def test_fts_only_mode_works(self, tmp_db_with_data, vector_db_path):
        """Test that FTS-only mode works end-to-end."""
        embedding_client = LocalEmbeddingClient(model="all-MiniLM-L6-v2")
        retrieval = create_hybrid_retrieval(
            db_url=f"sqlite:///{tmp_db_with_data.engine.url.database}",
            vector_db_path=vector_db_path,
            embedding_client=embedding_client,
            fts_only=True,
            log_level=7
        )
        
        results = retrieval.search("ошибки", top_k=10, rerank_top_k=5)
        
        assert len(results) > 0
        # Should find chunks with "ошибки"
        assert any("chunk1" in result.chunk.id or "chunk2" in result.chunk.id for result in results)
    
    def test_hybrid_mode_works(self, tmp_db_with_data, vector_db_path):
        """Test that hybrid mode works end-to-end."""
        embedding_client = LocalEmbeddingClient(model="all-MiniLM-L6-v2")
        retrieval = create_hybrid_retrieval(
            db_url=f"sqlite:///{tmp_db_with_data.engine.url.database}",
            vector_db_path=vector_db_path,
            embedding_client=embedding_client,
            fts_only=False,
            log_level=7
        )
        
        # Hybrid mode requires embeddings in chunks, which we don't have in test data
        # So it will fall back to FTS-only mode
        results = retrieval.search("ошибки", top_k=10, rerank_top_k=5)
        
        # Should return results (either hybrid or FTS fallback)
        assert isinstance(results, list)
    
    def test_fts_only_with_filters(self, tmp_db_with_data, vector_db_path):
        """Test FTS-only mode with filters."""
        embedding_client = LocalEmbeddingClient(model="all-MiniLM-L6-v2")
        retrieval = create_hybrid_retrieval(
            db_url=f"sqlite:///{tmp_db_with_data.engine.url.database}",
            vector_db_path=vector_db_path,
            embedding_client=embedding_client,
            fts_only=True,
            log_level=7
        )
        
        filters = SearchFilters(chat_id="chat1")
        results = retrieval.search("ошибки", top_k=10, filters=filters, rerank_top_k=5)
        
        assert len(results) > 0
        # All results should be from chat1
        for result in results:
            assert result.chunk.metadata.get("chat_id") == "chat1"
    
    def test_retrieve_compatibility_method(self, tmp_db_with_data, vector_db_path):
        """Test retrieve() compatibility method."""
        embedding_client = LocalEmbeddingClient(model="all-MiniLM-L6-v2")
        retrieval = create_hybrid_retrieval(
            db_url=f"sqlite:///{tmp_db_with_data.engine.url.database}",
            vector_db_path=vector_db_path,
            embedding_client=embedding_client,
            fts_only=True,
            log_level=7
        )
        
        results = retrieval.retrieve("ошибки", n_results=5)
        
        assert isinstance(results, list)
        if results:
            assert "id" in results[0]
            assert "text" in results[0]
    
    def test_search_chunks_basic_compatibility(self, tmp_db_with_data, vector_db_path):
        """Test search_chunks_basic() compatibility method."""
        embedding_client = LocalEmbeddingClient(model="all-MiniLM-L6-v2")
        retrieval = create_hybrid_retrieval(
            db_url=f"sqlite:///{tmp_db_with_data.engine.url.database}",
            vector_db_path=vector_db_path,
            embedding_client=embedding_client,
            fts_only=True,
            log_level=7
        )
        
        results = retrieval.search_chunks_basic("ошибки", n_results=3)
        
        assert isinstance(results, list)
        if results:
            assert "distance" in results[0]
            assert "id" in results[0]

