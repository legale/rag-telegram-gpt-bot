"""Tests for src/ingestion/telegram.py"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.ingestion.telegram import TelegramFetcher, json_serial
from datetime import datetime


class TestJsonSerial:
    """Tests for json_serial function"""
    
    def test_json_serial_datetime(self):
        """Test json_serial with datetime"""
        dt = datetime.now()
        result = json_serial(dt)
        
        assert isinstance(result, str)
        assert dt.isoformat() in result
    
    def test_json_serial_other_type(self):
        """Test json_serial with non-serializable type"""
        with pytest.raises(TypeError):
            json_serial("not datetime")


class TestTelegramFetcher:
    """Tests for TelegramFetcher"""
    
    def test_init(self):
        """Test TelegramFetcher initialization"""
        with patch('src.ingestion.telegram.TelegramClient') as mock_client:
            fetcher = TelegramFetcher(api_id=123, api_hash="hash", session_name="test")
            
            assert fetcher.api_id == 123
            assert fetcher.api_hash == "hash"
            assert fetcher.session_name == "test"
            mock_client.assert_called_once_with("test", 123, "hash", timeout=30)
    
    def test_find_chat_by_id(self):
        """Test finding chat by ID"""
        with patch('src.ingestion.telegram.TelegramClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_dialog = Mock()
            mock_dialog.id = 123
            mock_dialog.name = "Test Chat"
            mock_client.iter_dialogs.return_value = [mock_dialog]
            
            fetcher = TelegramFetcher(api_id=123, api_hash="hash")
            result = fetcher._find_chat("123")
            
            assert result == mock_dialog
    
    def test_find_chat_by_name(self):
        """Test finding chat by name"""
        with patch('src.ingestion.telegram.TelegramClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_dialog = Mock()
            mock_dialog.id = 123
            mock_dialog.name = "Test Chat"
            mock_client.iter_dialogs.return_value = [mock_dialog]
            
            fetcher = TelegramFetcher(api_id=123, api_hash="hash")
            result = fetcher._find_chat("Test Chat")
            
            assert result == mock_dialog
    
    def test_find_chat_not_found(self):
        """Test finding chat that doesn't exist"""
        with patch('src.ingestion.telegram.TelegramClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.iter_dialogs.return_value = []
            
            fetcher = TelegramFetcher(api_id=123, api_hash="hash")
            result = fetcher._find_chat("Nonexistent")
            
            assert result is None
    
    def test_search_chats_by_name(self):
        """Test searching chats by name substring"""
        with patch('src.ingestion.telegram.TelegramClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_dialog1 = Mock()
            mock_dialog1.id = 123
            mock_dialog1.name = "Test Chat"
            mock_dialog2 = Mock()
            mock_dialog2.id = 456
            mock_dialog2.name = "Another Chat"
            mock_client.iter_dialogs.return_value = [mock_dialog1, mock_dialog2]
            
            fetcher = TelegramFetcher(api_id=123, api_hash="hash")
            results = fetcher.search_chats_by_name("Test")
            
            assert len(results) == 1
            assert results[0][0] == 123
            assert results[0][1] == "Test Chat"
    
    def test_list_channels(self):
        """Test listing channels"""
        with patch('src.ingestion.telegram.TelegramClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_dialog = Mock()
            mock_dialog.id = 123
            mock_dialog.name = "Test"
            mock_client.iter_dialogs.return_value = [mock_dialog]
            
            fetcher = TelegramFetcher(api_id=123, api_hash="hash")
            
            with patch('builtins.print') as mock_print:
                fetcher.list_channels()
                mock_print.assert_called()
    
    def test_list_members_chat_not_found(self):
        """Test listing members when chat not found"""
        with patch('src.ingestion.telegram.TelegramClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.iter_dialogs.return_value = []
            
            fetcher = TelegramFetcher(api_id=123, api_hash="hash")
            
            with patch('src.ingestion.telegram.syslog2') as mock_syslog:
                fetcher.list_members("Nonexistent")
                mock_syslog.assert_called()
    
    def test_list_members_success(self):
        """Test listing members successfully"""
        with patch('src.ingestion.telegram.TelegramClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_dialog = Mock()
            mock_dialog.id = 123
            mock_dialog.name = "Test"
            mock_client.iter_dialogs.return_value = [mock_dialog]
            
            mock_user = Mock()
            mock_user.id = 1
            mock_user.first_name = "Test"
            mock_user.last_name = "User"
            mock_user.username = "testuser"
            mock_client.iter_participants.return_value = [mock_user]
            
            fetcher = TelegramFetcher(api_id=123, api_hash="hash")
            
            with patch('builtins.print') as mock_print:
                fetcher.list_members("Test")
                mock_print.assert_called()
    
    def test_dump_chat_chat_not_found(self):
        """Test dumping chat when chat not found"""
        with patch('src.ingestion.telegram.TelegramClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.iter_dialogs.return_value = []
            
            fetcher = TelegramFetcher(api_id=123, api_hash="hash")
            
            with patch('src.ingestion.telegram.syslog2') as mock_syslog:
                fetcher.dump_chat("Nonexistent")
                mock_syslog.assert_called()
    
    def test_dump_chat_success(self, tmp_path):
        """Test dumping chat successfully"""
        with patch('src.ingestion.telegram.TelegramClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_dialog = Mock()
            mock_dialog.id = 123
            mock_dialog.name = "Test"
            mock_client.iter_dialogs.return_value = [mock_dialog]
            
            mock_message = Mock()
            mock_message.id = 1
            mock_message.date = datetime.now()
            mock_message.text = "Test message"
            mock_message.sender = Mock()
            mock_message.sender.first_name = "Test"
            mock_message.sender.last_name = "User"
            mock_client.iter_messages.return_value = [mock_message]
            
            fetcher = TelegramFetcher(api_id=123, api_hash="hash")
            output_file = tmp_path / "output.json"
            
            with patch('builtins.print'):
                fetcher.dump_chat("Test", limit=10, output_file=str(output_file))
            
            assert output_file.exists()
            import json
            data = json.loads(output_file.read_text())
            assert len(data) == 1
            assert data[0]["content"] == "Test message"

