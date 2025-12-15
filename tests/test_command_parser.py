"""
Tests for command_parser module.
"""

import pytest
from src.bot.command_parser import parse_find_command_args
from src.bot.admin import AdminManager
from pathlib import Path
import tempfile
import json


@pytest.fixture
def temp_profile_dir(tmp_path):
    """Create a temporary profile directory with config."""
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    config_file = profile_dir / "config.json"
    config_data = {
        "cosine_distance_thr": 1.5,
        "embedding_model": "test-model",
        "embedding_generator": "local"
    }
    config_file.write_text(json.dumps(config_data))
    return str(profile_dir)


@pytest.fixture
def admin_manager(temp_profile_dir):
    """Create AdminManager instance."""
    return AdminManager(temp_profile_dir)


class TestParseFindCommandArgs:
    """Tests for parse_find_command_args function."""
    
    def test_parse_with_rag_method_and_query(self):
        """Test parsing with rag_method and query."""
        rag_method, action, query = parse_find_command_args("hybrid vpn туннель")
        assert rag_method == "hybrid"
        assert action is None
        assert query == "vpn туннель"
    
    def test_parse_with_list_action(self):
        """Test parsing with list action."""
        rag_method, action, query = parse_find_command_args("hybrid list")
        assert rag_method == "hybrid"
        assert action == "list"
        assert query is None
    
    def test_parse_with_slash_find_prefix(self):
        """Test parsing with /find prefix."""
        rag_method, action, query = parse_find_command_args("/find vector_only test query")
        assert rag_method == "vector_only"
        assert action is None
        assert query == "test query"
    
    def test_parse_with_fts_only(self):
        """Test parsing with fts_only method."""
        rag_method, action, query = parse_find_command_args("/find fts_only поиск")
        assert rag_method == "fts_only"
        assert action is None
        assert query == "поиск"
    
    def test_parse_empty_string(self):
        """Test parsing empty string."""
        rag_method, action, error = parse_find_command_args("")
        assert rag_method is None
        assert action is None
        assert "Использование" in error
    
    def test_parse_invalid_rag_method(self):
        """Test parsing with invalid rag_method."""
        rag_method, action, error = parse_find_command_args("invalid_method query")
        assert rag_method is None
        assert action is None
        assert "Неизвестный метод RAG" in error
    
    def test_parse_only_rag_method(self):
        """Test parsing with only rag_method (no query or list)."""
        rag_method, action, error = parse_find_command_args("hybrid")
        assert rag_method is None
        assert action is None
        assert "Необходимо указать действие или запрос" in error
    
    def test_parse_with_multi_word_query(self):
        """Test parsing with multi-word query."""
        rag_method, action, query = parse_find_command_args("hybrid vpn туннель настройка")
        assert rag_method == "hybrid"
        assert action is None
        assert query == "vpn туннель настройка"
    
    def test_parse_case_insensitive_rag_method(self):
        """Test parsing with case-insensitive rag_method."""
        rag_method, action, query = parse_find_command_args("HYBRID test")
        assert rag_method == "hybrid"
        assert action is None
        assert query == "test"
    
    def test_parse_list_case_insensitive(self):
        """Test parsing with case-insensitive list action."""
        rag_method, action, query = parse_find_command_args("vector_only LIST")
        assert rag_method == "vector_only"
        assert action == "list"
        assert query is None

