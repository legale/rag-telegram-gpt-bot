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
    
    def test_parse_with_threshold_and_query(self):
        """Test parsing with threshold and query."""
        threshold, query = parse_find_command_args("2.0 vpn туннель")
        assert threshold == 2.0
        assert query == "vpn туннель"
    
    def test_parse_with_query_only(self):
        """Test parsing with query only."""
        threshold, query = parse_find_command_args("vpn туннель")
        assert threshold == 1.5  # default
        assert query == "vpn туннель"
    
    def test_parse_with_slash_find_prefix(self):
        """Test parsing with /find prefix."""
        threshold, query = parse_find_command_args("/find 2.0 vpn туннель")
        assert threshold == 2.0
        assert query == "vpn туннель"
    
    def test_parse_with_slash_find_and_query_only(self):
        """Test parsing /find with query only."""
        threshold, query = parse_find_command_args("/find vpn туннель")
        assert threshold == 1.5
        assert query == "vpn туннель"
    
    def test_parse_empty_string(self):
        """Test parsing empty string."""
        threshold, error = parse_find_command_args("")
        assert threshold is None
        assert "Использование" in error
    
    def test_parse_only_threshold(self):
        """Test parsing with only threshold (no query)."""
        threshold, error = parse_find_command_args("2.0")
        assert threshold is None
        assert "Использование" in error
    
    def test_parse_with_admin_manager(self, admin_manager):
        """Test parsing with AdminManager."""
        threshold, query = parse_find_command_args("vpn туннель", admin_manager=admin_manager)
        assert threshold == 1.5  # from config
        assert query == "vpn туннель"
    
    def test_parse_with_admin_manager_custom_threshold(self, admin_manager):
        """Test parsing with AdminManager and custom threshold."""
        threshold, query = parse_find_command_args("2.5 vpn туннель", admin_manager=admin_manager)
        assert threshold == 2.5
        assert query == "vpn туннель"
    
    def test_parse_whitespace_handling(self):
        """Test parsing with extra whitespace."""
        threshold, query = parse_find_command_args("  2.0   vpn   туннель  ")
        assert threshold == 2.0
        assert query == "vpn   туннель"
    
    def test_parse_float_threshold(self):
        """Test parsing with float threshold."""
        threshold, query = parse_find_command_args("0.5 test query")
        assert threshold == 0.5
        assert query == "test query"

