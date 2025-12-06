
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os
from src.bot.tgbot import register_webhook, delete_webhook, main, init_runtime_for_current_profile, run_server, run_daemon

class TestTgBotCLI:
    def test_register_webhook_success(self, capsys):
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"ok": True, "description": "Success"}
            
            register_webhook("https://example.com", "token123")
            
            captured = capsys.readouterr()
            assert "✓ Webhook registered successfully" in captured.out
            mock_post.assert_called_with(
                "https://api.telegram.org/bottoken123/setWebhook", 
                json={"url": "https://example.com"}
            )

    def test_register_webhook_failure_api(self, capsys):
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"ok": False, "description": "Error info"}
            
            with pytest.raises(SystemExit):
                register_webhook("https://example.com", "token123")
            
            captured = capsys.readouterr()
            assert "✗ Failed to register webhook: Error info" in captured.out

    def test_register_webhook_failure_http(self, capsys):
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 404
            
            with pytest.raises(SystemExit):
                register_webhook("https://example.com", "token123")
            
            captured = capsys.readouterr()
            assert "✗ HTTP error: 404" in captured.out

    def test_delete_webhook_success(self, capsys):
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"ok": True}
            
            delete_webhook("token123")
            
            captured = capsys.readouterr()
            assert "✓ Webhook deleted successfully" in captured.out

    def test_delete_webhook_failure(self, capsys):
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"ok": False, "description": "Error"}
            
            with pytest.raises(SystemExit):
                delete_webhook("token123")
            
            captured = capsys.readouterr()
            assert "✗ Failed to delete webhook" in captured.out

    def test_main_register(self):
        with patch('sys.argv', ['tgbot.py', 'register', '--url', 'http://url', '--token', 'tok']), \
             patch('src.bot.tgbot.register_webhook') as mock_reg:
            main()
            mock_reg.assert_called_with('http://url', 'tok')

    def test_main_delete(self):
        with patch('sys.argv', ['tgbot.py', 'delete', '--token', 'tok']), \
             patch('src.bot.tgbot.delete_webhook') as mock_del:
            main()
            mock_del.assert_called_with('tok')

    def test_main_run(self):
        with patch('sys.argv', ['tgbot.py', 'run', '--token', 'tok']), \
             patch('src.bot.tgbot.run_server') as mock_run:
            main()
            mock_run.assert_called_with('127.0.0.1', 8000, 0)
            assert os.environ["TELEGRAM_BOT_TOKEN"] == 'tok'

    def test_main_daemon(self):
        with patch('sys.argv', ['tgbot.py', 'daemon', '--token', 'tok']), \
             patch('src.bot.tgbot.run_daemon') as mock_run:
            main()
            mock_run.assert_called_with('127.0.0.1', 8000)

    def test_main_no_token(self):
        with patch('sys.argv', ['tgbot.py', 'run']), \
             patch.dict(os.environ, {}, clear=True):
             with pytest.raises(SystemExit):
                 main()

    def test_run_server(self):
        with patch('uvicorn.run') as mock_uvicorn:
            run_server(verbosity=2)
            mock_uvicorn.assert_called()
            # Check log level mapping
            assert mock_uvicorn.call_args[1]['log_level'] == 'debug'
            assert mock_uvicorn.call_args[1]['access_log'] is True

    def test_run_daemon(self):
        # Mock sys.exit to prevent actual exit during signal handling in daemon context
        # Mock daemon module
        with patch.dict(sys.modules, {'daemon': MagicMock(), 'daemon.pidfile': MagicMock()}), \
             patch('uvicorn.run') as mock_uvicorn:
            
            run_daemon()
            
            # Since we mock daemon context, it should just run uvicorn
            mock_uvicorn.assert_called()


@pytest.mark.asyncio
async def test_init_runtime_for_current_profile():
    # Setup mocks
    mock_pm = MagicMock()
    mock_pm.get_profile_paths.return_value = {
        "db_url": "sqlite:///test.db",
        "vector_db_path": "vec_path",
        "profile_dir": "prof_dir",
        "db_path": "db_path"
    }
    
    with patch("src.bot.tgbot.profile_manager", mock_pm), \
         patch("src.bot.tgbot.LegaleBot") as MockBot, \
         patch("src.bot.tgbot.AdminManager") as MockAdmin, \
         patch("src.bot.tgbot.AdminCommandRouter") as MockRouter, \
         patch("src.bot.tgbot.TaskManager"), \
         patch("src.bot.tgbot.ProfileCommands"), \
         patch("src.bot.tgbot.IngestCommands"), \
         patch("src.bot.tgbot.StatsCommands"), \
         patch("src.bot.tgbot.ControlCommands"), \
         patch("src.bot.tgbot.SettingsCommands"), \
         patch("src.bot.tgbot.HelpCommands"):
         
         # Mock AdminManager config
         mock_admin_instance = MockAdmin.return_value
         mock_admin_instance.config.current_model = "gpt-4"
         
         paths = await init_runtime_for_current_profile()
         
         assert paths == mock_pm.get_profile_paths.return_value
         MockAdmin.assert_called_with("prof_dir")
         MockBot.assert_called_with(
             db_url="sqlite:///test.db", 
             vector_db_path="vec_path",
             model_name="gpt-4"
         )

@pytest.mark.asyncio
async def test_init_runtime_no_profile_manager():
    with patch("src.bot.tgbot.profile_manager", None):
        with pytest.raises(RuntimeError, match="profile_manager is not initialized"):
            await init_runtime_for_current_profile()
