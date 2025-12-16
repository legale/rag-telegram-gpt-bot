"""
Tests for bootstrap module.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.app.bootstrap import create_app, create_hybrid_retrieval, create_hybrid_search
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


class TestCreateApp:
    def test_create_without_profile_dir(self):
        with patch("src.bot.core.LegaleBot") as mock_bot_cls, \
             patch("src.core.command_service.CommandService") as mock_command_service_cls, \
             patch("src.app.main_cli.register_sync_handlers") as mock_register_sync, \
             patch("src.app.main_cli.register_async_handlers") as mock_register_async, \
             patch("src.app.app.App") as mock_app_cls:
            bot = Mock()
            command_service = Mock()
            app_instance = Mock()

            mock_bot_cls.return_value = bot
            mock_command_service_cls.return_value = command_service
            mock_app_cls.return_value = app_instance

            app = create_app(
                db_url="sqlite:///test.db",
                vector_db_path="vector",
                profile_dir=None,
            )

            assert app is app_instance
            assert callable(getattr(app, "ingest"))
            mock_register_sync.assert_called_once_with(command_service, bot, admin_manager=None, debug_rag=False)
            mock_register_async.assert_not_called()

    def test_create_with_profile_dir(self, tmp_path):
        with patch("src.bot.core.LegaleBot") as mock_bot_cls, \
             patch("src.bot.admin.AdminManager") as mock_admin_manager_cls, \
             patch("src.bot.admin_router.AdminCommandRouter") as mock_admin_router_cls, \
             patch("src.core.command_service.CommandService") as mock_command_service_cls, \
             patch("src.app.main_cli.register_sync_handlers") as mock_register_sync, \
             patch("src.app.main_cli.register_async_handlers") as mock_register_async, \
             patch("src.app.app.App") as mock_app_cls:
            bot = Mock()
            admin_manager = Mock()
            admin_router = Mock()
            command_service = Mock()
            app_instance = Mock()

            mock_bot_cls.return_value = bot
            mock_admin_manager_cls.return_value = admin_manager
            mock_admin_router_cls.return_value = admin_router
            mock_command_service_cls.return_value = command_service
            mock_app_cls.return_value = app_instance

            app = create_app(
                db_url="sqlite:///test.db",
                vector_db_path="vector",
                profile_dir=str(tmp_path),
            )

            assert app is app_instance
            assert callable(getattr(app, "ingest"))
            mock_register_sync.assert_called_once_with(command_service, bot, admin_manager=admin_manager, debug_rag=False)
            mock_register_async.assert_called_once_with(command_service, admin_manager=admin_manager, admin_router=admin_router)
    
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
