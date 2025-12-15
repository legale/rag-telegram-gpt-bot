"""Tests for src/bot/admin.py"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from src.bot.admin import AdminManager


class TestAdminManager:
    """Tests for AdminManager"""
    
    def test_init(self, tmp_path):
        """Test AdminManager initialization"""
        with patch('src.bot.admin.BotConfig') as mock_config:
            mock_config.return_value.data = {}
            mock_config.return_value.admin_password = None
            
            manager = AdminManager(tmp_path)
            
            assert manager.profile_dir == Path(tmp_path)
            assert manager.admin_file == tmp_path / "admin.json"
    
    def test_init_migrates_password_from_env(self, tmp_path):
        """Test AdminManager migrates password from environment"""
        with patch('src.bot.admin.BotConfig') as mock_config, \
             patch.dict('os.environ', {'ADMIN_PASSWORD': 'test_password'}):
            mock_config.return_value.data = {}
            mock_config.return_value.admin_password = None
            
            manager = AdminManager(tmp_path)
            
            assert manager.password == "test_password"
            mock_config.return_value.admin_password = "test_password"
    
    def test_load_admin_data_file_not_exists(self, tmp_path):
        """Test loading admin data when file doesn't exist"""
        with patch('src.bot.admin.BotConfig') as mock_config:
            mock_config.return_value.data = {}
            mock_config.return_value.admin_password = None
            
            manager = AdminManager(tmp_path)
            data = manager._load_admin_data()
            
            assert data == {}
    
    def test_load_admin_data_file_exists(self, tmp_path):
        """Test loading admin data from existing file"""
        admin_file = tmp_path / "admin.json"
        admin_file.write_text(json.dumps({"user_id": 123, "username": "test"}))
        
        with patch('src.bot.admin.BotConfig') as mock_config:
            mock_config.return_value.data = {}
            mock_config.return_value.admin_password = None
            
            manager = AdminManager(tmp_path)
            data = manager._load_admin_data()
            
            assert data["user_id"] == 123
            assert data["username"] == "test"
    
    def test_load_admin_data_invalid_json(self, tmp_path):
        """Test loading admin data with invalid JSON"""
        admin_file = tmp_path / "admin.json"
        admin_file.write_text("invalid json")
        
        with patch('src.bot.admin.BotConfig') as mock_config:
            mock_config.return_value.data = {}
            mock_config.return_value.admin_password = None
            
            manager = AdminManager(tmp_path)
            data = manager._load_admin_data()
            
            assert data == {}
    
    def test_save_admin_data(self, tmp_path):
        """Test saving admin data"""
        with patch('src.bot.admin.BotConfig') as mock_config:
            mock_config.return_value.data = {}
            mock_config.return_value.admin_password = None
            
            manager = AdminManager(tmp_path)
            manager._save_admin_data({"user_id": 123, "username": "test"})
            
            assert (tmp_path / "admin.json").exists()
            data = json.loads((tmp_path / "admin.json").read_text())
            assert data["user_id"] == 123
    
    def test_set_admin(self, tmp_path):
        """Test setting admin"""
        with patch('src.bot.admin.BotConfig') as mock_config:
            mock_config.return_value.data = {}
            mock_config.return_value.admin_password = None
            
            manager = AdminManager(tmp_path)
            result = manager.set_admin(123, "testuser", "Test", "User")
            
            assert result is True
            data = manager._load_admin_data()
            assert data["user_id"] == 123
            assert data["username"] == "testuser"
            assert data["full_name"] == "Test User"
    
    def test_set_admin_without_last_name(self, tmp_path):
        """Test setting admin without last name"""
        with patch('src.bot.admin.BotConfig') as mock_config:
            mock_config.return_value.data = {}
            mock_config.return_value.admin_password = None
            
            manager = AdminManager(tmp_path)
            manager.set_admin(123, "testuser", "Test")
            
            data = manager._load_admin_data()
            assert data["full_name"] == "Test"
    
    def test_get_admin_exists(self, tmp_path):
        """Test getting admin when admin exists"""
        admin_file = tmp_path / "admin.json"
        admin_file.write_text(json.dumps({"user_id": 123, "username": "test"}))
        
        with patch('src.bot.admin.BotConfig') as mock_config:
            mock_config.return_value.data = {}
            mock_config.return_value.admin_password = None
            
            manager = AdminManager(tmp_path)
            admin = manager.get_admin()
            
            assert admin is not None
            assert admin["user_id"] == 123
    
    def test_get_admin_not_exists(self, tmp_path):
        """Test getting admin when no admin set"""
        with patch('src.bot.admin.BotConfig') as mock_config:
            mock_config.return_value.data = {}
            mock_config.return_value.admin_password = None
            
            manager = AdminManager(tmp_path)
            admin = manager.get_admin()
            
            assert admin is None
    
    def test_is_admin_true(self, tmp_path):
        """Test checking if user is admin (true)"""
        admin_file = tmp_path / "admin.json"
        admin_file.write_text(json.dumps({"user_id": 123}))
        
        with patch('src.bot.admin.BotConfig') as mock_config:
            mock_config.return_value.data = {}
            mock_config.return_value.admin_password = None
            
            manager = AdminManager(tmp_path)
            assert manager.is_admin(123) is True
    
    def test_is_admin_false(self, tmp_path):
        """Test checking if user is admin (false)"""
        admin_file = tmp_path / "admin.json"
        admin_file.write_text(json.dumps({"user_id": 123}))
        
        with patch('src.bot.admin.BotConfig') as mock_config:
            mock_config.return_value.data = {}
            mock_config.return_value.admin_password = None
            
            manager = AdminManager(tmp_path)
            assert manager.is_admin(456) is False
    
    def test_verify_password_correct(self, tmp_path):
        """Test verifying correct password"""
        with patch('src.bot.admin.BotConfig') as mock_config:
            mock_config.return_value.data = {}
            mock_config.return_value.admin_password = "correct_password"
            
            manager = AdminManager(tmp_path)
            assert manager.verify_password("correct_password") is True
    
    def test_verify_password_incorrect(self, tmp_path):
        """Test verifying incorrect password"""
        with patch('src.bot.admin.BotConfig') as mock_config:
            mock_config.return_value.data = {}
            mock_config.return_value.admin_password = "correct_password"
            
            manager = AdminManager(tmp_path)
            assert manager.verify_password("wrong_password") is False
    
    def test_verify_password_not_set(self, tmp_path):
        """Test verifying password when not set"""
        with patch('src.bot.admin.BotConfig') as mock_config:
            mock_config.return_value.data = {}
            mock_config.return_value.admin_password = None
            
            manager = AdminManager(tmp_path)
            assert manager.verify_password("any_password") is False
    
    def test_remove_admin(self, tmp_path):
        """Test removing admin"""
        admin_file = tmp_path / "admin.json"
        admin_file.write_text(json.dumps({"user_id": 123}))
        
        with patch('src.bot.admin.BotConfig') as mock_config:
            mock_config.return_value.data = {}
            mock_config.return_value.admin_password = None
            
            manager = AdminManager(tmp_path)
            manager.remove_admin()
            
            assert not admin_file.exists()
            assert manager.get_admin() is None

