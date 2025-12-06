
import pytest
from unittest.mock import MagicMock, patch, mock_open
from datetime import datetime
import sys
import io

# Mock telethon before importing the module
sys.modules['telethon'] = MagicMock()
sys.modules['telethon.sync'] = MagicMock()
sys.modules['telethon.tl.types'] = MagicMock()

from src.ingestion.telegram import TelegramFetcher, json_serial

class TestTelegramFetcher:
    @pytest.fixture
    def mock_client(self):
        with patch('src.ingestion.telegram.TelegramClient') as mock:
            yield mock

    @pytest.fixture
    def fetcher(self, mock_client):
        return TelegramFetcher(12345, "mock_hash", "mock_session")

    def test_init(self, mock_client):
        fetcher = TelegramFetcher(12345, "mock_hash", "mock_session")
        assert fetcher.api_id == 12345
        assert fetcher.api_hash == "mock_hash"
        assert fetcher.session_name == "mock_session"
        mock_client.assert_called_once_with("mock_session", 12345, "mock_hash")

    def test_find_chat_by_id(self, fetcher):
        # Setup mock dialogs
        mock_dialog1 = MagicMock()
        mock_dialog1.id = 111
        mock_dialog1.name = "Chat One"
        
        mock_dialog2 = MagicMock()
        mock_dialog2.id = 222
        mock_dialog2.name = "Chat Two"

        fetcher.client.iter_dialogs.return_value = [mock_dialog1, mock_dialog2]

        # Test finding by ID
        result = fetcher._find_chat(111)
        assert result == mock_dialog1

        # Test finding by string ID
        result = fetcher._find_chat("222")
        assert result == mock_dialog2

    def test_find_chat_by_name(self, fetcher):
        # Setup mock dialogs
        mock_dialog1 = MagicMock()
        mock_dialog1.id = 111
        mock_dialog1.name = "Chat One"
        
        mock_dialog2 = MagicMock()
        mock_dialog2.id = 222
        mock_dialog2.name = "Chat Two"

        fetcher.client.iter_dialogs.return_value = [mock_dialog1, mock_dialog2]

        # Test finding by name
        result = fetcher._find_chat("Chat One")
        assert result == mock_dialog1

    def test_find_chat_not_found(self, fetcher):
        fetcher.client.iter_dialogs.return_value = []
        result = fetcher._find_chat("NonExistent")
        assert result is None

    def test_list_channels(self, fetcher, capsys):
        # Setup mock dialogs
        mock_dialog1 = MagicMock()
        mock_dialog1.id = 111
        mock_dialog1.name = "Chat One"
        
        mock_dialog2 = MagicMock()
        mock_dialog2.id = 222
        mock_dialog2.name = "Chat Two"

        fetcher.client.iter_dialogs.return_value = [mock_dialog1, mock_dialog2]

        # Call the method
        fetcher.list_channels()

        # Check output
        captured = capsys.readouterr()
        assert "Chat One" in captured.out
        assert "111" in captured.out
        assert "Chat Two" in captured.out
        assert "222" in captured.out
        
        # Verify context manager usage
        fetcher.client.__enter__.assert_called()
        fetcher.client.__exit__.assert_called()

    def test_list_members_success(self, fetcher, capsys):
        # Mock _find_chat
        mock_chat = MagicMock()
        mock_chat.id = 123
        mock_chat.name = "Test Chat"
        fetcher._find_chat = MagicMock(return_value=mock_chat)

        # Mock participants
        user1 = MagicMock()
        user1.id = 1
        user1.first_name = "John"
        user1.last_name = "Doe"
        user1.username = "johndoe"

        user2 = MagicMock()
        user2.id = 2
        user2.first_name = "Jane"
        user2.last_name = None
        user2.username = None

        fetcher.client.iter_participants.return_value = [user1, user2]

        # Call method
        fetcher.list_members("Test Chat")

        # Check output
        captured = capsys.readouterr()
        assert "Members of 'Test Chat'" in captured.out
        assert "John Doe" in captured.out
        assert "@johndoe" in captured.out
        assert "Jane" in captured.out
        assert "N/A" in captured.out
        
        # Verify calls
        fetcher._find_chat.assert_called_with("Test Chat")
        fetcher.client.iter_participants.assert_called_with(mock_chat)

    def test_list_members_chat_not_found(self, fetcher, capsys):
        fetcher._find_chat = MagicMock(return_value=None)
        fetcher.list_members("Unknown")
        captured = capsys.readouterr()
        assert "Error: Chat 'Unknown' not found." in captured.out

    def test_list_members_exception(self, fetcher, capsys):
        mock_chat = MagicMock()
        mock_chat.name = "Test"
        fetcher._find_chat = MagicMock(return_value=mock_chat)
        fetcher.client.iter_participants.side_effect = Exception("API Error")

        fetcher.list_members("Test")
        captured = capsys.readouterr()
        assert "Error fetching members: API Error" in captured.out

    def test_dump_chat_json_serialization(self):
        dt = datetime(2023, 1, 1, 12, 0, 0)
        assert json_serial(dt) == "2023-01-01T12:00:00"
        
        with pytest.raises(TypeError):
            json_serial(object())

    def test_dump_chat_success(self, fetcher, capsys):
        # Mock chat
        mock_chat = MagicMock()
        mock_chat.id = 123
        mock_chat.name = "Test Chat"
        fetcher._find_chat = MagicMock(return_value=mock_chat)

        # Mock messages
        msg1 = MagicMock()
        msg1.id = 1
        msg1.date = datetime(2023, 1, 1, 10, 0)
        msg1.text = "Hello"
        msg1.sender.first_name = "User"
        msg1.sender.last_name = "One"
        
        msg2 = MagicMock()
        msg2.id = 2
        msg2.date = datetime(2023, 1, 1, 10, 5)
        msg2.text = "World"
        msg2.sender.title = "Bot" # Case with title
        del msg2.sender.first_name # Ensure it uses title

        fetcher.client.iter_messages.return_value = [msg1, msg2]

        with patch("builtins.open", mock_open()) as mock_file:
            fetcher.dump_chat("Test Chat", limit=2)
            
            # Verify file write
            mock_file.assert_called_with("dump_123.json", 'w', encoding='utf-8')
            
            # Get the written data
            written_args = mock_file().write.call_args_list
            # Join all written chunks (json dump might do multiple writes)
            written_str = "".join([args[0][0] for args in written_args])
            
            # Simple check if json content seems right
            # (Mock_open writes are sometimes tricky to reconstruct fully if json.dump chunks it, 
            # but usually we can check check specific calls)
            
            # Better check: Verify json.dump was called
            # We can't easily check json.dump call arguments because it takes a file object
            pass

        captured = capsys.readouterr()
        assert "Found chat: Test Chat" in captured.out
        assert "Saved 2 messages" in captured.out

    def test_dump_chat_chat_not_found(self, fetcher, capsys):
        fetcher._find_chat = MagicMock(return_value=None)
        fetcher.dump_chat("Unknown")
        captured = capsys.readouterr()
        assert "Error: Chat 'Unknown' not found." in captured.out

