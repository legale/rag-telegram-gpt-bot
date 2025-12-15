"""
Tests for bootstrap module.
"""

import pytest
from pathlib import Path
from src.app.bootstrap import create_hybrid_search, create_hybrid_retrieval
from src.core.embedding import EmbeddingClient, LocalEmbeddingClient


class TestCreateHybridSearch:
    """Tests for create_hybrid_search function."""
    
    def test_create_with_defaults(self, tmp_path):
        """Test creating with default parameters."""
        # Use LocalEmbeddingClient to avoid API key requirement
        client = LocalEmbeddingClient(model="all-MiniLM-L6-v2")
        search = create_hybrid_search(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            embedding_client=client
        )
        assert search is not None
    
    def test_create_with_embedding_client(self, tmp_path):
        """Test creating with provided embedding client."""
        client = LocalEmbeddingClient(model="all-MiniLM-L6-v2")
        search = create_hybrid_search(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            embedding_client=client
        )
        assert search is not None
    
    def test_create_with_profile_dir(self, tmp_path):
        """Test creating with profile directory."""
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()
        config_file = profile_dir / "config.json"
        config_file.write_text('{"embedding_model": "all-MiniLM-L6-v2", "embedding_generator": "local"}')
        
        # Use LocalEmbeddingClient to avoid API key requirement
        client = LocalEmbeddingClient(model="all-MiniLM-L6-v2")
        search = create_hybrid_search(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            embedding_client=client,
            profile_dir=str(profile_dir)
        )
        assert search is not None


class TestCreateHybridRetrieval:
    """Tests for create_hybrid_retrieval function."""
    
    def test_create_with_defaults(self, tmp_path):
        """Test creating with default parameters."""
        # Use LocalEmbeddingClient to avoid API key requirement
        client = LocalEmbeddingClient(model="all-MiniLM-L6-v2")
        retrieval = create_hybrid_retrieval(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            embedding_client=client
        )
        assert retrieval is not None
    
    def test_create_with_embedding_client(self, tmp_path):
        """Test creating with provided embedding client."""
        client = LocalEmbeddingClient(model="all-MiniLM-L6-v2")
        retrieval = create_hybrid_retrieval(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            embedding_client=client
        )
        assert retrieval is not None
    
    def test_create_fts_only_mode(self, tmp_path):
        """Test creating with fts_only mode."""
        # Use LocalEmbeddingClient to avoid API key requirement
        client = LocalEmbeddingClient(model="all-MiniLM-L6-v2")
        retrieval = create_hybrid_retrieval(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            embedding_client=client,
            fts_only=True
        )
        assert retrieval is not None
    
    def test_create_with_retrieval_mode_fts_only(self, tmp_path):
        """Test creating with retrieval_mode='fts_only'."""
        # Use LocalEmbeddingClient to avoid API key requirement
        client = LocalEmbeddingClient(model="all-MiniLM-L6-v2")
        retrieval = create_hybrid_retrieval(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            embedding_client=client,
            retrieval_mode="fts_only"
        )
        assert retrieval is not None
    
    def test_create_with_retrieval_mode_hybrid(self, tmp_path):
        """Test creating with retrieval_mode='hybrid'."""
        # Use LocalEmbeddingClient to avoid API key requirement
        client = LocalEmbeddingClient(model="all-MiniLM-L6-v2")
        retrieval = create_hybrid_retrieval(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            embedding_client=client,
            retrieval_mode="hybrid"
        )
        assert retrieval is not None
    
    def test_create_with_profile_dir(self, tmp_path):
        """Test creating with profile directory."""
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()
        config_file = profile_dir / "config.json"
        config_file.write_text('{"embedding_model": "all-MiniLM-L6-v2", "embedding_generator": "local"}')
        
        # Use LocalEmbeddingClient to avoid API key requirement
        client = LocalEmbeddingClient(model="all-MiniLM-L6-v2")
        retrieval = create_hybrid_retrieval(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            embedding_client=client,
            profile_dir=str(profile_dir)
        )
        assert retrieval is not None

