"""
Tests for BotConfig class.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from src.bot.config import BotConfig

@pytest.fixture
def temp_profile():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

class TestBotConfig:
    def test_load_defaults(self, temp_profile):
        config = BotConfig(temp_profile)
        assert config.admin_password == ""
        assert config.allowed_chats == []
        assert config.response_frequency == 1
        
    def test_save_and_load(self, temp_profile):
        config = BotConfig(temp_profile)
        config.admin_password = "secret"
        config.allowed_chats = [123, 456]
        config.response_frequency = 5
        
        # Create new instance to test loading
        new_config = BotConfig(temp_profile)
        assert new_config.admin_password == "secret"
        assert new_config.allowed_chats == [123, 456]
        assert new_config.response_frequency == 5

    def test_add_remove_chat(self, temp_profile):
        config = BotConfig(temp_profile)
        
        config.add_allowed_chat(111)
        assert config.allowed_chats == [111]
        
        # Add duplicate - shouldn't duplicate
        config.add_allowed_chat(111)
        assert config.allowed_chats == [111]
        
        config.add_allowed_chat(222)
        assert config.allowed_chats == [111, 222]
        
        config.remove_allowed_chat(111)
        assert config.allowed_chats == [222]
        
        # Remove non-existent
        config.remove_allowed_chat(999)
        assert config.allowed_chats == [222]

    def test_response_frequency_validation(self, temp_profile):
        config = BotConfig(temp_profile)
        
        config.response_frequency = 0
        assert config.response_frequency == 1
        
        config.response_frequency = -5
        assert config.response_frequency == 1
        
        config.response_frequency = 10
        assert config.response_frequency == 10

    def test_permissions(self, temp_profile):
        config = BotConfig(temp_profile)
        config.save()
        
        # Check if file mode is strict (on unix)
        if os.name == 'posix':
            st = os.stat(config.config_file)
            # 0o600 means only owner can read/write
            assert (st.st_mode & 0o777) == 0o600
