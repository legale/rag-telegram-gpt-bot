"""
Tests for AdminManager.
"""
import pytest
import os
from unittest.mock import patch, Mock
from pathlib import Path
from src.bot.admin import AdminManager

@pytest.fixture
def mock_profile(tmp_path):
    return tmp_path

class TestAdminManager:
    def test_init_migrates_password(self, mock_profile):
        # Ensure config doesn't exist yet
        with patch.dict(os.environ, {"ADMIN_PASSWORD": "env_password"}):
            manager = AdminManager(mock_profile)
            assert manager.config.admin_password == "env_password"
            assert manager.verify_password("env_password")
            
            # Check persistence
            assert (mock_profile / "config.json").exists()

    def test_verify_password(self, mock_profile):
        manager = AdminManager(mock_profile)
        manager.config.admin_password = "testpass"
        
        assert manager.verify_password("testpass")
        assert not manager.verify_password("wrong")

    def test_set_and_get_admin(self, mock_profile):
        manager = AdminManager(mock_profile)
        
        manager.set_admin(123, "username", "John", "Doe")
        
        admin = manager.get_admin()
        assert admin['user_id'] == 123
        assert admin['username'] == "username"
        assert admin['full_name'] == "John Doe"
        
        assert manager.is_admin(123)
        assert not manager.is_admin(456)

    def test_remove_admin(self, mock_profile):
        manager = AdminManager(mock_profile)
        manager.set_admin(123, "test", "Test")
        
        assert manager.is_admin(123)
        
        manager.remove_admin()
        assert not manager.is_admin(123)
        assert manager.get_admin() is None
