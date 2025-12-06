"""
Tests for Admin Commands.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.bot.admin_commands import SettingsCommands, ControlCommands, HelpCommands, ProfileCommands, StatsCommands, IngestCommands

@pytest.fixture
def mock_context():
    # Helper to create update/context/manager mocks
    update = MagicMock()
    update.message.chat_id = 123
    update.message.from_user.id = 1
    update.message.reply_text = AsyncMock()
    
    context = MagicMock()
    
    admin_manager = MagicMock()
    admin_manager.config = MagicMock()
    admin_manager.config.allowed_chats = []
    admin_manager.config.response_frequency = 1
    
    profile_manager = MagicMock()
    
    return update, context, admin_manager, profile_manager

class TestSettingsCommands:
    @pytest.mark.asyncio
    async def test_manage_chats_list(self, mock_context):
        update, context, admin_manager, pm = mock_context
        settings = SettingsCommands(pm)
        
        # Empty list
        response = await settings.manage_chats(update, context, admin_manager, ['list'])
        assert "список разрешенных чатов пуст" in response.lower()
        
        # With chats
        admin_manager.config.allowed_chats = [123, 456]
        response = await settings.manage_chats(update, context, admin_manager, ['list'])
        assert "123" in response
        assert "456" in response

    @pytest.mark.asyncio
    async def test_manage_chats_add(self, mock_context):
        update, context, admin_manager, pm = mock_context
        settings = SettingsCommands(pm)
        
        # Simulate 'in' operator on mock list by setting it to real list initially
        # admin_manager.config.allowed_chats is a Mock, but we can set it to list
        
        # Test 1: Add current chat (123)
        admin_manager.config.allowed_chats = []
        response = await settings.manage_chats(update, context, admin_manager, ['add'])
        assert "добавлен" in response
        admin_manager.config.add_allowed_chat.assert_called_with(123)
        
        # Test 2: Add explicit chat (999)
        response = await settings.manage_chats(update, context, admin_manager, ['add', '999'])
        assert "добавлен" in response
        admin_manager.config.add_allowed_chat.assert_called_with(999)
        
        # Test 3: Add duplicate
        admin_manager.config.allowed_chats = [123]
        response = await settings.manage_chats(update, context, admin_manager, ['add'])
        assert "уже в списке" in response.lower()

    @pytest.mark.asyncio
    async def test_manage_frequency(self, mock_context):
        update, context, admin_manager, pm = mock_context
        settings = SettingsCommands(pm)
        
        # Get
        response = await settings.manage_frequency(update, context, admin_manager, [])
        assert "1 ответ" in response
        
        # Set valid
        response = await settings.manage_frequency(update, context, admin_manager, ['5'])
        assert "5" in response
        assert admin_manager.config.response_frequency == 5
        
        # Set invalid
        response = await settings.manage_frequency(update, context, admin_manager, ['abc'])
        assert "используйте число" in response.lower()

class TestControlCommands:
    @pytest.mark.asyncio
    async def test_restart_bot(self, mock_context):
        update, context, admin_manager, pm = mock_context
        # Force no job queue to test asyncio fallback
        context.job_queue = None
        
        control = ControlCommands(pm)
        
        # We need to mock asyncio.create_task
        with patch('asyncio.create_task') as mock_task:
            response = await control.restart_bot(update, context, admin_manager, [])
            assert "перезапуск" in response.lower()
            mock_task.assert_called()
            
            # Close the coroutine to avoid RuntimeWarning
            # The coroutine _delayed_restart is created but never awaited by mocked create_task
            coro = mock_task.call_args[0][0]
            coro.close()

class TestHelpCommands:
    @pytest.mark.asyncio
    async def test_show_help(self, mock_context):
        update, context, admin_manager, pm = mock_context
        helper = HelpCommands()
        
        response = await helper.show_help(update, context, admin_manager, [])
        assert "справка по админ-командам" in response.lower()

class TestProfileCommands:
    @pytest.mark.asyncio
    async def test_list_profiles_empty(self, mock_context):
        update, context, admin_manager, pm = mock_context
        
        # Setup pm.profiles_dir as a temp path usually, but here mock
        # Mock iterdir to return empty list
        pm.profiles_dir.iterdir.return_value = []
        pm.profiles_dir.exists.return_value = True
        
        profiles = ProfileCommands(pm)
        response = await profiles.list_profiles(update, context, admin_manager, [])
        assert "Профили не найдены" in response
        
    @pytest.mark.asyncio
    async def test_list_profiles_with_data(self, mock_context):
        update, context, admin_manager, pm = mock_context
        
        # Mock profiles directory structure
        p1 = MagicMock()
        p1.name = "default"
        p1.is_dir.return_value = True
        
        # Mock p1 / "legale_bot.db"
        db_mock1 = MagicMock()
        db_mock1.exists.return_value = True
        db_mock1.stat.return_value.st_size = 5000000 # 5 MB
        p1.__truediv__.return_value = db_mock1
        
        p2 = MagicMock()
        p2.name = "test_profile"
        p2.is_dir.return_value = True
        
        # Mock p2 / "legale_bot.db" (not exists)
        db_mock2 = MagicMock()
        db_mock2.exists.return_value = False
        p2.__truediv__.return_value = db_mock2
        
        pm.profiles_dir.iterdir.return_value = [p1, p2]
        pm.profiles_dir.exists.return_value = True
        pm.get_current_profile.return_value = "default"
        
        profiles = ProfileCommands(pm)
        response = await profiles.list_profiles(update, context, admin_manager, [])
        
        assert "default" in response
        assert "test_profile" in response
        assert "✅" in response # Active marker

    @pytest.mark.asyncio
    async def test_create_profile(self, mock_context):
        update, context, admin_manager, pm = mock_context
        profiles = ProfileCommands(pm)
        
        # Success case
        pm.get_profile_dir.return_value.exists.return_value = False
        pm.get_profile_paths.return_value = {
            'profile_dir': '/tmp/p', 'db_path': '/tmp/d', 'vector_db_path': '/tmp/v'
        }
        
        response = await profiles.create_profile(update, context, admin_manager, ['new_prof'])
        assert "создан" in response.lower()
        pm.create_profile.assert_called_with('new_prof', set_active=False)
        
        # Already exists
        pm.get_profile_dir.return_value.exists.return_value = True
        response = await profiles.create_profile(update, context, admin_manager, ['existing'])
        assert "уже существует" in response.lower()
        
        # Invalid name
        response = await profiles.create_profile(update, context, admin_manager, ['bad name!'])
        assert "буквы, цифры" in response.lower()

    @pytest.mark.asyncio
    async def test_switch_profile(self, mock_context):
        update, context, admin_manager, pm = mock_context
        profiles = ProfileCommands(pm)
        
        pm.get_current_profile.return_value = "old"
        pm.get_profile_dir.return_value.exists.return_value = True
        
        # Success
        response = await profiles.switch_profile(update, context, admin_manager, ['new'])
        assert "Переключено" in response
        pm.set_current_profile.assert_called_with('new')
        
        # Already active
        pm.get_current_profile.return_value = "new"
        response = await profiles.switch_profile(update, context, admin_manager, ['new'])
        assert "уже активен" in response.lower()
        
        # Not found
        pm.get_profile_dir.return_value.exists.return_value = False
        response = await profiles.switch_profile(update, context, admin_manager, ['missing'])
        assert "не существует" in response.lower()

    @pytest.mark.asyncio
    async def test_delete_profile(self, mock_context):
        update, context, admin_manager, pm = mock_context
        profiles = ProfileCommands(pm)
        
        pm.get_current_profile.return_value = "active"
        
        # Mock profile dir and db
        profile_dir = MagicMock()
        profile_dir.exists.return_value = True
        
        db_mock = MagicMock()
        db_mock.exists.return_value = True
        db_mock.stat.return_value.st_size = 1024 * 1024 * 10 # 10 MB
        profile_dir.__truediv__.return_value = db_mock
        
        pm.get_profile_dir.return_value = profile_dir
        
        # Try delete active
        response = await profiles.delete_profile(update, context, admin_manager, ['active'])
        assert "невозможно удалить активный" in response.lower()
        
        # Try delete other (ask confirmation)
        response = await profiles.delete_profile(update, context, admin_manager, ['other'])
        assert "подтверждение удаления" in response.lower()
        assert "10.00 MB" in response
        
        # Confirm delete
        with patch('shutil.rmtree') as mock_rm:
            response = await profiles.delete_profile(update, context, admin_manager, ['other', 'confirm'])
            assert "удалён" in response.lower()
            mock_rm.assert_called()

class TestStatsCommands:
    @pytest.mark.asyncio
    async def test_show_stats(self, mock_context):
        update, context, admin_manager, pm = mock_context
        
        # Mock paths
        pm.get_current_profile.return_value = "test_prof"
        db_path = MagicMock()
        db_path.exists.return_value = False # Simple case without DB
        
        profile_dir = MagicMock()
        profile_dir.__str__.return_value = '/tmp/profile' 
        
        pm.get_profile_paths.return_value = {
            'profile_dir': profile_dir,
            'db_path': db_path,
            'vector_db_path': MagicMock()
        }
        
        # Mock psutil
        with patch('psutil.Process') as mock_proc, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.cpu_percent', return_value=10.0):
             
            mock_proc.return_value.memory_info.return_value.rss = 1024 * 1024 * 100 # 100MB
            
            mock_disk.return_value.free = 1024 * 1024 * 1024 * 10 # 10GB
            mock_disk.return_value.total = 1024 * 1024 * 1024 * 100 # 100GB
            mock_disk.return_value.percent = 90
            
            stats = StatsCommands(pm)
            response = await stats.show_stats(update, context, admin_manager, [])
            
            assert "Статистика бота" in response
            assert "100.0 MB" in response
            assert "10.0%" in response

    @pytest.mark.asyncio
    async def test_health_check(self, mock_context):
        update, context, admin_manager, pm = mock_context
        
        pm.get_profile_paths.return_value = {
            'db_path': MagicMock(),
            'vector_db_path': MagicMock(),
            'profile_dir': MagicMock()
        }
        pm.get_profile_paths.return_value['db_path'].exists.return_value = True
        pm.get_profile_paths.return_value['vector_db_path'].exists.return_value = True

        stats = StatsCommands(pm)
        
        with patch('sqlite3.connect') as mock_sql, \
             patch('os.getenv', return_value="key"), \
             patch('psutil.virtual_memory') as mock_mem, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_mem.return_value.percent = 50.0
            mock_disk.return_value.percent = 50.0
            
            response = await stats.health_check(update, context, admin_manager, [])
            assert "Проверка здоровья" in response
            assert "✅" in response

class TestIngestCommands:
    @pytest.mark.asyncio
    async def test_ingest_status_no_task(self, mock_context):
        update, context, admin_manager, pm = mock_context
        
        tm = MagicMock()
        tm.get_current_task.return_value = None
        
        ingest = IngestCommands(pm, tm)
        
        response = await ingest.ingest_status(update, context, admin_manager, [])
        assert "Нет активных задач" in response
        
    @pytest.mark.asyncio
    async def test_ingest_status_running(self, mock_context):
        update, context, admin_manager, pm = mock_context
        
        task = MagicMock()
        task.status = "running"
        task.progress = 50
        task.total = 100
        
        tm = MagicMock()
        tm.get_current_task.return_value = task
        
        ingest = IngestCommands(pm, tm)
        
        response = await ingest.ingest_status(update, context, admin_manager, [])
        assert "50/100" in response
        assert "50.0%" in response

    @pytest.mark.asyncio
    async def test_handle_file_upload_success(self, mock_context):
        update, context, admin_manager, pm = mock_context
        # Setup update with document
        update.message.document = MagicMock()
        update.message.document.file_name = "chat_export.json"
        update.message.document.file_size = 1024
        update.message.document.file_id = "file123"
        
        tm = MagicMock()
        task = MagicMock()
        tm.start_ingestion.return_value = task
        
        ingest = IngestCommands(pm, tm)
        # Set user waiting
        ingest.waiting_for_file[1] = True
        
        # Mock file download
        file_mock = MagicMock()
        file_mock.download_to_drive = AsyncMock()
        context.bot.get_file = AsyncMock(return_value=file_mock)
        
        # Mock temp dir and Path
        with patch('src.bot.admin_commands.Path') as mock_path_cls, \
             patch('tempfile.gettempdir', return_value='/tmp'), \
             patch('asyncio.create_task') as mock_create_task:
             
            # Setup path mock behavior
            mock_path = mock_path_cls.return_value
            mock_path.__truediv__.return_value = mock_path # Support / operator
            
            await ingest.handle_file_upload(update, context, admin_manager)
            
            # Assertions
            assert ingest.waiting_for_file[1] == False
            context.bot.get_file.assert_called_with("file123")
            tm.start_ingestion.assert_called()
            mock_create_task.assert_called()
            
            # Close task coroutine
            coro = mock_create_task.call_args[0][0]
            coro.close()

    @pytest.mark.asyncio
    async def test_handle_file_upload_not_waiting(self, mock_context):
        update, context, admin_manager, pm = mock_context
        tm = MagicMock()
        ingest = IngestCommands(pm, tm)
        
        # User not in dict
        res = await ingest.handle_file_upload(update, context, admin_manager)
        assert res is None
