"""
Admin command handlers for Legale Bot.
Implements handlers for /admin commands.
"""

from typing import List
from telegram import Update
from telegram.ext import ContextTypes
from pathlib import Path
import os
import logging

from src.bot.utils import (
    ResponseFormatter,
    DatabaseStatsService,
    CommandValidator,
    CommandValidator,
    HealthChecker,
)
from src.core.syslog2 import *

logger = logging.getLogger("legale_admin_commands")


class BaseAdminCommand:
    """Base class for admin command handlers with common utilities."""
    
    def __init__(self, profile_manager):
        """
        Initialize BaseAdminCommand.
        
        Args:
            profile_manager: ProfileManager instance
        """
        self.profile_manager = profile_manager
        self.formatter = ResponseFormatter()
        self.validator = CommandValidator()
        self.db_stats = DatabaseStatsService()
    
    async def handle_error(self, error: Exception, context: str) -> str:
        """
        Unified error handling for admin commands.
        
        Args:
            error: Exception that occurred
            context: Context description (e.g., operation name)
            
        Returns:
            Formatted error message
        """
        syslog2(LOG_ERR, "admin command error", context=context, error=str(error))
        return self.formatter.format_error_message(str(error), context)
    
    def get_profile_paths(self, profile_name: str = None):
        """
        Get paths for a profile with validation.
        
        Args:
            profile_name: Profile name (None = current profile)
            
        Returns:
            Dictionary with profile paths
        """
        if profile_name:
            return self.profile_manager.get_profile_paths(profile_name)
        return self.profile_manager.get_profile_paths()
    
    def validate_profile_exists(self, profile_name: str) -> tuple[bool, str]:
        """
        Validate that a profile exists.
        
        Args:
            profile_name: Profile name to check
            
        Returns:
            Tuple of (exists, error_message)
        """
        profile_dir = self.profile_manager.get_profile_dir(profile_name)
        if not profile_dir.exists():
            return False, f"Профиль `{profile_name}` не существует."
        return True, ""



class ProfileCommands(BaseAdminCommand):
    """Handlers for profile management commands."""
    
    def _get_profile_stats(self, profile_name: str) -> Dict:
        """
        Get statistics for a profile.
        
        Args:
            profile_name: Profile name
            
        Returns:
            Dictionary with profile statistics
        """
        paths = self.profile_manager.get_profile_paths(profile_name)
        profile_dir = self.profile_manager.get_profile_dir(profile_name)
        current = self.profile_manager.get_current_profile()
        is_active = profile_name == current
        
        db_path = paths['db_path']
        vector_path = paths['vector_db_path']
        
        db_stats = self.db_stats.get_database_stats(db_path)
        vector_stats = self.db_stats.get_vector_store_stats(vector_path)
        
        return {
            'profile_name': profile_name,
            'profile_dir': profile_dir,
            'paths': paths,
            'is_active': is_active,
            'db_stats': db_stats,
            'vector_stats': vector_stats,
        }
    
    def _format_profile_info(self, profile_name: str, stats: Dict, is_active: bool) -> str:
        """
        Format profile information in Markdown.
        
        Args:
            profile_name: Profile name
            stats: Statistics dictionary from _get_profile_stats
            is_active: Whether profile is active
            
        Returns:
            Formatted Markdown string
        """
        paths = stats['paths']
        db_stats = stats['db_stats']
        vector_stats = stats['vector_stats']
        
        response = f"**Информация о профиле `{profile_name}`**\n\n"
        response += "**Статус:** Активный\n\n" if is_active else "**Статус:** Неактивный\n\n"
        response += f"**Директория:** `{paths['profile_dir']}`\n\n"
        
        # Database info
        if db_stats['exists']:
            response += f"**База данных:**\n"
            response += f"   Путь: `{paths['db_path']}`\n"
            response += f"   Размер: {db_stats['size_mb']:.2f} MB\n"
            response += f"   Чанков: {self.formatter.format_number(db_stats['chunk_count'])}\n"
        else:
            response += f"**База данных:** Не создана\n"
        
        response += "\n"
        
        # Vector store info
        if vector_stats['exists']:
            response += f"**Векторное хранилище:**\n"
            response += f"   Путь: `{paths['vector_db_path']}`\n"
            response += f"   Размер: {vector_stats['size_mb']:.2f} MB\n"
        else:
            response += f"**Векторное хранилище:** Не создано\n"
        
        response += "\n"
        
        # Session file
        session_path = paths['session_file']
        if session_path.exists():
            response += f"**Telegram сессия:** Создана\n"
        else:
            response += f"**Telegram сессия:** Не создана\n"
        
        # Admin file
        admin_file = stats['profile_dir'] / "admin.json"
        if admin_file.exists():
            response += f"**Администратор:** Назначен\n"
        else:
            response += f"**Администратор:** Не назначен\n"
        
        return response
    
    async def list_profiles(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                           admin_manager, args: List[str]) -> str:
        """Handle /admin profile list command."""
        profiles_dir = self.profile_manager.profiles_dir
        
        if not profiles_dir.exists() or not list(profiles_dir.iterdir()):
            return "Профили не найдены.\n\nСоздайте первый профиль:\n`/admin profile create <name>`"
        
        profiles = [p for p in profiles_dir.iterdir() if p.is_dir()]
        if not profiles:
            return "Профили не найдены.\n\nСоздайте первый профиль:\n`/admin profile create <name>`"
        
        current = self.profile_manager.get_current_profile()
        response = "**Доступные профили:**\n\n"
        
        for profile_dir in sorted(profiles, key=lambda p: p.name):
            profile_name = profile_dir.name
            stats = self._get_profile_stats(profile_name)
            is_active = stats['is_active']
            
            # Get database info using utility
            db_path = profile_dir / "legale_bot.db"
            db_size = self.formatter.format_file_size(db_path.stat().st_size) if db_path.exists() else "0B"
            chunk_count = self.db_stats.get_chunk_count(db_path) if db_path.exists() else 0
            
            marker = "" if is_active else ""
            active_text = " **(активный)**" if is_active else ""
            db_text = f"БД: {db_size}, чанков: {self.formatter.format_number(chunk_count)}" if db_path.exists() else "БД не создана"
            
            response += f"{marker} `{profile_name}`{active_text}\n"
            response += f"   {db_text}\n\n"
        
        response += f"\n**Активный профиль:** `{current}`"
        return response
    
    async def create_profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                            admin_manager, args: List[str]) -> str:
        """Handle /admin profile create <name> command."""
        # Validate arguments
        is_valid, error = self.validator.validate_args_count(
            args, 1, 1, 
            usage="/admin profile create <name>"
        )
        if not is_valid:
            return error
        
        profile_name = args[0]
        
        # Validate profile name
        is_valid, error = self.validator.validate_profile_name(profile_name)
        if not is_valid:
            return f"{error}"
        
        profile_dir = self.profile_manager.get_profile_dir(profile_name)
        
        if profile_dir.exists():
            return f"Профиль `{profile_name}` уже существует.\n\nПуть: `{profile_dir}`"
        
        try:
            # Create profile
            self.profile_manager.create_profile(profile_name, set_active=False)
            paths = self.profile_manager.get_profile_paths(profile_name)
            
            response = self.formatter.format_success_message(
                f"Профиль `{profile_name}` создан!",
                {
                    "Директория": f"`{paths['profile_dir']}`",
                    "База данных": f"`{paths['db_path']}`",
                    "Векторное хранилище": f"`{paths['vector_db_path']}`",
                }
            )
            response += f"\nДля переключения на этот профиль:\n`/admin profile set {profile_name}`"
            
            syslog2(LOG_NOTICE, "profile created", profile=profile_name, admin_id=update.message.from_user.id)
            return response
            
        except Exception as e:
            return await self.handle_error(e, f"создании профиля '{profile_name}'")
    
    async def get_profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                            admin_manager, args: List[str]) -> str:
        """Handle /admin profile get command - returns current profile name."""
        # No arguments needed
        current_profile = self.profile_manager.get_current_profile()
        return self.formatter.format_info_message(f"Текущий активный профиль: `{current_profile}`")
    
    async def set_profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                            admin_manager, args: List[str]) -> str:
        """Handle /admin profile set <name> command."""
        # Validate arguments
        is_valid, error = self.validator.validate_args_count(
            args, 1, 1,
            usage="/admin profile set <name>"
        )
        if not is_valid:
            return error
        
        profile_name = args[0]
        
        # Validate profile exists
        exists, error = self.validate_profile_exists(profile_name)
        if not exists:
            return f"{error}\n\nСоздайте его:\n`/admin profile create {profile_name}`"
        
        current = self.profile_manager.get_current_profile()
        
        if profile_name == current:
            return self.formatter.format_info_message(f"Профиль `{profile_name}` уже активен.")
        
        try:
            # set profile
            self.profile_manager.set_current_profile(profile_name)
            
            response = (
                f"Переключено на профиль `{profile_name}`\n\n"
                f"**Внимание:** Для применения изменений необходимо перезапустить бота:\n"
                f"`/admin restart`"
            )
            
            syslog2(LOG_NOTICE, "profile set", profile=profile_name, admin_id=update.message.from_user.id)
            return response
            
        except Exception as e:
            return await self.handle_error(e, f"переключении на профиль '{profile_name}'")
    
    async def delete_profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                            admin_manager, args: List[str]) -> str:
        """Handle /admin profile delete <name> command."""
        # Validate arguments
        is_valid, error = self.validator.validate_args_count(
            args, 1, 2,
            usage="/admin profile delete <name> [confirm]"
        )
        if not is_valid:
            return error
        
        profile_name = args[0]
        current = self.profile_manager.get_current_profile()
        
        if profile_name == current:
            return (
                f"Невозможно удалить активный профиль `{profile_name}`.\n\n"
                f"Сначала переключитесь на другой профиль:\n"
                f"`/admin profile get <другой_профиль>`"
            )
        
        # Validate profile exists
        exists, error = self.validate_profile_exists(profile_name)
        if not exists:
            return error
        
        profile_dir = self.profile_manager.get_profile_dir(profile_name)
        
        # Get profile info before deletion
        db_path = profile_dir / "legale_bot.db"
        info_text = f"Профиль: `{profile_name}`\n"
        info_text += f"Путь: `{profile_dir}`\n"
        
        if db_path.exists():
            size_mb = self.db_stats.get_database_size(db_path)
            info_text += f"Размер БД: {size_mb:.2f} MB\n"
        
        # Check for confirmation
        if len(args) > 1 and args[1] == "confirm":
            try:
                import shutil
                shutil.rmtree(profile_dir)
                
                syslog2(LOG_WARNING, "profile deleted", profile=profile_name, admin_id=update.message.from_user.id)
                
                return (
                    f"Профиль `{profile_name}` удалён.\n\n"
                    f"Все данные профиля были удалены безвозвратно."
                )
                
            except Exception as e:
                return await self.handle_error(e, f"удалении профиля '{profile_name}'")
        
        # Show confirmation request
        return (
            f"**Подтверждение удаления**\n\n"
            f"{info_text}\n"
            f"**Это действие необратимо!**\n\n"
            f"Для подтверждения удаления используйте:\n"
            f"`/admin profile delete {profile_name} confirm`"
        )
    
    
    async def profile_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                          admin_manager, args: List[str]) -> str:
        """Handle /admin profile info [name] command."""
        profile_name = args[0] if args else self.profile_manager.get_current_profile()
        
        # Validate profile exists
        exists, error = self.validate_profile_exists(profile_name)
        if not exists:
            return error
        
        try:
            stats = self._get_profile_stats(profile_name)
            return self._format_profile_info(profile_name, stats, stats['is_active'])
            
        except Exception as e:
            return await self.handle_error(e, f"получении информации о профиле '{profile_name}'")



class HelpCommands:
    """Handlers for help commands."""
    
    async def show_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                       admin_manager, args: List[str]) -> str:
        """Handle /admin help [command] command."""
        if not args:
            # General help
            return (
                "**Справка по админ-командам**\n\n"
                "**Управление профилями:**\n"
                "• `profile list` - список всех профилей\n"
                "• `profile create <name>` - создать новый профиль\n"
                "• `profile get <name>` - переключиться на профиль\n"
                "• `profile delete <name>` - удалить профиль\n"
                "• `profile info [name]` - информация о профиле\n\n"
                "**Загрузка данных:**\n"
                "• `ingest` - загрузить данные (отправьте JSON)\n"
                "• `ingest clear` - очистить данные\n"
                "• `ingest status` - статус загрузки\n\n"
                "**Доступ:**\n"
                "• `allowed list` - список разрешенных чатов\n"
                "• `allowed add <id>` - разрешить чат\n"
                "• `allowed remove <id>` - запретить чат\n\n"
                "**Мониторинг:**\n"
                "• `stats` - статистика бота\n"
                "• `health` - проверка здоровья\n"
                "• `logs [N]` - последние N строк логов\n\n"
                "**Управление:**\n"
                "• `restart` - перезапустить бота\n"
                "• `reload` - перезагрузить конфигурацию\n\n"
                "Для подробной справки по команде:\n"
                "`/admin help <команда>`"
            )
        
        command = args[0]
        
        # Command-specific help
        help_texts = {
            "profile": (
                "**Управление профилями**\n\n"
                "Профили позволяют управлять несколькими ботами с отдельными базами данных.\n\n"
                "**Команды:**\n"
                "• `/admin profile list` - показать все профили\n"
                "• `/admin profile create <name>` - создать профиль\n"
                "• `/admin profile get <name>` - переключить профиль\n"
                "• `/admin profile delete <name>` - удалить профиль\n"
                "• `/admin profile info [name]` - информация о профиле\n\n"
                "**Примеры:**\n"
                "`/admin profile create production`\n"
                "`/admin profile get production`\n"
                "`/admin profile info production`"
            ),
            "ingest": (
                "**Загрузка данных**\n\n"
                "Загрузка данных из Telegram дампов в базу данных.\n\n"
                "**Команды:**\n"
                "• `/admin ingest` - начать загрузку (отправьте JSON файл)\n"
                "• `/admin ingest clear` - очистить данные профиля\n"
                "• `/admin ingest status` - статус текущей загрузки\n\n"
                "**Процесс:**\n"
                "1. Отправьте команду `/admin ingest`\n"
                "2. Загрузите JSON файл с дампом\n"
                "3. Дождитесь завершения обработки"
            ),
            "allowed": (
                "**Управление доступом**\n\n"
                "Управление списком разрешенных чатов и пользователей.\n\n"
                "**Команды:**\n"
                "• `/admin allowed list` - показать разрешенные ID\n"
                "• `/admin allowed add <id>` - добавить ID в белый список\n"
                "• `/admin allowed remove <id>` - удалить ID из белого списка\n\n"
                "**Логика:**\n"
                "- Администраторы всегда имеют доступ (везде).\n"
                "- Групповые чаты: Должны быть в списке allowed.\n"
                "- Личные сообщения: Если ID не в списке, работают только команды (текст игнорируется)."
            ),
            "chat": (
                 "**Управление чатами**\n\n"
                 "Алиас для `/admin allowed`.\n"
                 "Используйте `/admin help allowed` для подробностей."
            ),
        }
        if command in help_texts:
            return help_texts[command]
        else:
            return f"Справка по команде `{command}` не найдена.\n\nИспользуйте `/admin help` для списка команд."


class IngestCommands(BaseAdminCommand):
    """Handlers for data ingestion commands."""
    
    def __init__(self, profile_manager, task_manager):
        """
        Initialize IngestCommands.
        
        Args:
            profile_manager: ProfileManager instance
            task_manager: TaskManager instance
        """
        super().__init__(profile_manager)
        self.task_manager = task_manager
        self.waiting_for_file = {}  # user_id -> bool
    
    async def start_ingest(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                          admin_manager, args: List[str]) -> str:
        """Handle /admin ingest command."""
        user_id = update.message.from_user.id
        
        # Mark user as waiting for file
        self.waiting_for_file[user_id] = True
        
        return (
            "**Загрузка данных**\n\n"
            "Отправьте JSON файл с дампом Telegram чата.\n\n"
            "Файл должен быть в формате, созданном командой:\n"
            "`legale telegram dump \"Chat Name\"`\n\n"
            "После получения файла начнётся автоматическая обработка."
        )
    
    async def clear_data(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                        admin_manager, args: List[str]) -> str:
        """Handle /admin ingest clear command."""
        try:
            from src.ingestion.pipeline import IngestionPipeline
            
            # Get profile paths
            paths = self.get_profile_paths()
            
            # Get current stats before clearing using utility
            db_stats = self.db_stats.get_database_stats(paths['db_path'])
            chunk_count = db_stats['chunk_count']
            db_size = db_stats['size_mb']
            
            # Create pipeline and clear data
            pipeline = IngestionPipeline(
                db_url=paths['db_url'],
                vector_db_path=str(paths['vector_db_path'])
            )
            
            pipeline._clear_data()
            
            syslog2(LOG_NOTICE, "data cleared", admin_id=update.message.from_user.id)
            
            return (
                f"**Данные очищены**\n\n"
                f"Удалено чанков: {self.formatter.format_number(chunk_count)}\n"
                f"Освобождено места: {db_size:.2f} MB\n\n"
                f"База данных и векторное хранилище очищены."
            )
            
        except Exception as e:
            return await self.handle_error(e, "очистке данных")
    
    async def ingest_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                           admin_manager, args: List[str]) -> str:
        """Handle /admin ingest status command."""
        task = self.task_manager.get_current_task()
        
        if not task:
            return self.formatter.format_info_message("Нет активных задач загрузки.")
        
        if task.status == "pending":
            return "Задача загрузки ожидает запуска..."
        elif task.status == "running":
            progress_pct = (task.progress / task.total * 100) if task.total > 0 else 0
            progress_bar = self.formatter.create_progress_bar(task.progress, task.total, width=20)
            return (
                f"**Загрузка данных в процессе**\n\n"
                f"Прогресс: {self.formatter.format_number(task.progress)}/"
                f"{self.formatter.format_number(task.total)} "
                f"({progress_pct:.1f}%)\n\n"
                f"{progress_bar}"
            )
        elif task.status == "completed":
            return (
                f"**Загрузка завершена**\n\n"
                f"Обработано сообщений: {self.formatter.format_number(task.result['messages'])}\n"
                f"Создано чанков: {self.formatter.format_number(task.result['chunks'])}"
            )
        elif task.status == "failed":
            return f"**Загрузка завершилась с ошибкой:**\n\n`{task.error}`"
        else:
            return f"Неизвестный статус: {task.status}"
    
    async def handle_file_upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                                 admin_manager) -> str:
        """Handle file upload for ingestion."""
        user_id = update.message.from_user.id
        
        # Check if user is waiting for file
        if user_id not in self.waiting_for_file or not self.waiting_for_file[user_id]:
            return None  # Not waiting for file, ignore
        
        # Clear waiting flag
        self.waiting_for_file[user_id] = False
        
        document = update.message.document
        
        # Validate file
        if not document.file_name.endswith('.json'):
            return "Файл должен быть в формате JSON.\n\nОтправьте JSON файл с дампом чата."
        
        # Check file size (max 20MB)
        max_size_mb = 20
        if document.file_size > max_size_mb * 1024 * 1024:
            return (
                f"Файл слишком большой (макс. {max_size_mb}MB).\n\n"
                f"Используйте CLI для загрузки больших файлов."
            )
        
        try:
            # Download file
            file = await context.bot.get_file(document.file_id)
            
            # Create temp directory
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / "legale_bot"
            temp_dir.mkdir(exist_ok=True)
            
            # Save file - convert Path to string to avoid MagicMock issues in tests
            temp_file = temp_dir / f"{user_id}_{document.file_name}"
            await file.download_to_drive(str(temp_file))
            
            file_size_str = self.formatter.format_file_size(document.file_size)
            syslog2(LOG_NOTICE, "file downloaded for ingestion", path=str(temp_file), size=file_size_str)
            
            # Send initial message
            status_message = await update.message.reply_text(
                "Файл получен!\n\nПодготовка к загрузке..."
            )
            
            # Start ingestion task - convert Path to string to avoid MagicMock issues in tests
            task = self.task_manager.start_ingestion(str(temp_file), self.profile_manager)
            
            # Run task in background
            import asyncio
            asyncio.create_task(
                task.run(context.bot, update.message.chat_id, status_message.message_id)
            )
            
            return None  # Message already sent
            
        except Exception as e:
            return await self.handle_error(e, "обработке файла")
    
    def is_waiting_for_file(self, user_id: int) -> bool:
        """Check if user is waiting for file upload."""
        return user_id in self.waiting_for_file and self.waiting_for_file[user_id]


class StatsCommands(BaseAdminCommand):
    """Handlers for statistics and monitoring commands."""
    
    async def show_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                        admin_manager, args: List[str]) -> str:
        """Handle /admin stats command."""
        try:
            # Get profile paths
            paths = self.get_profile_paths()
            profile_name = self.profile_manager.get_current_profile()
            
            response = f"**Статистика бота**\n\n"
            response += f"**Профиль:** `{profile_name}`\n\n"
            
            # Get database stats using utility
            db_stats = self.db_stats.get_database_stats(paths['db_path'])
            response += self._format_database_stats(db_stats, paths['db_path'])
            
            response += "\n"
            
            # Get vector store stats using utility
            vector_stats = self.db_stats.get_vector_store_stats(paths['vector_db_path'])
            response += self._format_vector_stats(vector_stats)
            
            response += "\n"
            
            # System stats
            response += self._format_system_stats(paths['profile_dir'])
            
            return response
            
        except Exception as e:
            return await self.handle_error(e, "получении статистики")
    
    def _format_database_stats(self, stats: dict, db_path: Path) -> str:
        """Format database statistics section."""
        if not stats['exists']:
            return "**База данных:** Не создана\n"
        
        response = "**База данных:**\n"
        response += f"   Размер: {stats['size_mb']:.2f} MB\n"
        response += f"   Чанков: {self.formatter.format_number(stats['chunk_count'])}\n"
        
        if stats['date_range']:
            min_date, max_date = stats['date_range']
            response += f"   Период: {min_date} - {max_date}\n"
        
        return response
    
    def _format_vector_stats(self, stats: dict) -> str:
        """Format vector store statistics section."""
        if not stats['exists']:
            return "**Векторное хранилище:** Не создано\n"
        
        return (
            f"**Векторное хранилище:**\n"
            f"   Размер: {stats['size_mb']:.2f} MB\n"
        )
    
    def _format_system_stats(self, profile_dir: Path) -> str:
        """Format system statistics section."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            response = "**Система:**\n"
            response += f"   Память: {memory_mb:.1f} MB\n"
            response += f"   CPU: {psutil.cpu_percent()}%\n"
            
            # Disk space
            disk = psutil.disk_usage(str(profile_dir))
            disk_free_gb = disk.free / (1024 * 1024 * 1024)
            disk_total_gb = disk.total / (1024 * 1024 * 1024)
            disk_percent = disk.percent
            
            response += f"   Диск: {disk_free_gb:.1f}/{disk_total_gb:.1f} GB свободно ({100-disk_percent:.1f}%)\n"
            
            return response
        except Exception as e:
            syslog2(LOG_ERR, "system stats failed", error=str(e))
            return "**Система:** Ошибка получения данных\n"
    
    async def health_check(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                          admin_manager, args: List[str]) -> str:
        """Handle /admin health command."""
        try:
            paths = self.get_profile_paths()
            
            # Run all health checks using HealthChecker utility
            checks = HealthChecker.run_all_checks(
                db_path=paths['db_path'],
                vector_path=paths['vector_db_path'],
                profile_dir=paths['profile_dir']
            )
            
            # Format and return report
            return HealthChecker.format_health_report(checks)
            
        except Exception as e:
            return await self.handle_error(e, "проверке здоровья")
    
    async def show_logs(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                       admin_manager, args: List[str]) -> str:
        """Handle /admin logs [lines] command."""
        try:
            # Validate lines argument
            lines = 50
            if args:
                is_valid, parsed_lines, error = self.validator.validate_log_lines(args[0])
                if not is_valid:
                    return f"{error}"
                lines = parsed_lines
            
            # Get profile directory
            paths = self.get_profile_paths()
            log_file = paths['profile_dir'] / "bot.log"
            
            if not log_file.exists():
                return self.formatter.format_info_message(
                    "Лог-файл не найден.\n\nЛоги появятся после запуска бота в daemon режиме."
                )
            
            # Read last N lines
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                last_lines = all_lines[-lines:]
            
            if not last_lines:
                return self.formatter.format_info_message("Лог-файл пуст.")
            
            # Format response
            response = f"**Последние {len(last_lines)} строк логов**\n\n"
            response += "```\n"
            response += "".join(last_lines)
            response += "```"
            
            # Truncate if too long (Telegram limit is 4096 chars)
            if len(response) > 4000:
                response = response[:3900] + "\n...\n```\n\nЛоги обрезаны. Используйте меньшее количество строк."
            
            return response
            
        except Exception as e:
            return await self.handle_error(e, "чтении логов")


class ControlCommands(BaseAdminCommand):
    """Handlers for bot control commands."""
    
    def __init__(self, profile_manager, reload_callback=None):
        super().__init__(profile_manager)
        # reload_callback – async-функция, которая пересоздает bot_instance/admin_router/etc
        self.reload_callback = reload_callback

    async def restart_bot(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                         admin_manager, args: List[str]) -> str:
        """Handle /admin restart command."""
        # горячий reload без exit
        if self.reload_callback is None:
            # на всякий случай fallback, чтобы не молчать, если что-то не инициализировали
            syslog2(LOG_WARNING, "restart callback missing")
            return "Перезапуск недоступен: не настроен обработчик перезагрузки."
        
        try:
            paths = await self.reload_callback()
            profile_name = self.profile_manager.get_current_profile()
            return (
                "Перезапуск выполнен.\n\n"
                f"Профиль: `{profile_name}`\n"
                f"База данных: `{paths['db_path']}`\n"
                f"Векторное хранилище: `{paths['vector_db_path']}`"
            )
        except Exception as e:
            syslog2(LOG_ERR, "hot restart failed", error=str(e))
            return await self.handle_error(e, "перезапуске бота")

class ModelCommands(BaseAdminCommand):
    """Handlers for model management commands."""
    
    def __init__(self, profile_manager, bot_instance):
        super().__init__(profile_manager)
        self.bot_instance = bot_instance

    async def list_models(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                         admin_manager, args: List[str]) -> str:
        """Handle /admin model list command."""
        if not self.bot_instance:
             return "Бот не инициализирован."
        
        available = self.bot_instance.available_models
        current = self.bot_instance.current_model_name
        
        response = "**Доступные модели:**\n\n"
        for model in available:
            marker = "" if model == current else ""
            response += f"{marker} `{model}`\n"
            
        return response

    async def get_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                       admin_manager, args: List[str]) -> str:
        """Handle /admin model get command."""
        if not self.bot_instance:
             return "Бот не инициализирован."
            
        return self.bot_instance.get_current_model()
    
    async def set_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                       admin_manager, args: List[str]) -> str:
        """Handle /admin model set <name> command."""
        if not self.bot_instance:
             return "Бот не инициализирован."
            
        if not args:
            return "Укажите имя модели.\nПример: `/admin model set openai/gpt-4-turbo`"
        
        target_model = args[0]
        available = self.bot_instance.available_models
        
        if target_model not in available:
            return (
                f"Модель `{target_model}` не найдена в списке доступных.\n\n"
                f"Используйте `/admin model list` для просмотра списка."
            )
        
        try:
             # Set model in bot instance
             result_msg = self.bot_instance.set_model(target_model)
             
             # Persist to config
             admin_manager.config.current_model = target_model
             
             logger.info(f"Model changed to {target_model} by admin {update.message.from_user.id}")
             return result_msg
             
        except Exception as e:
            return await self.handle_error(e, f"установке модели {target_model}")


class SystemPromptCommands(BaseAdminCommand):
    """Handlers for system prompt management commands."""
    
    def __init__(self, profile_manager):
        super().__init__(profile_manager)

    async def get_prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                        admin_manager, args: List[str]) -> str:
        """Handle /admin system_prompt get command."""
        current_prompt = admin_manager.config.system_prompt
        
        if not current_prompt:
            from src.core.prompt import PromptEngine
            current_prompt = PromptEngine.SYSTEM_PROMPT_TEMPLATE
            return f"**Текущий системный промпт (по умолчанию):**\n\n```\n{current_prompt}\n```"
            
        return f"**Текущий системный промпт (пользовательский):**\n\n```\n{current_prompt}\n```"

    async def set_prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                        admin_manager, args: List[str]) -> str:
        """Handle /admin system_prompt set <prompt> command."""
        if not args:
            return "Укажите новый промпт.\nПример: `/admin system_prompt set Ты - полезный ассистент.`"
        
        # Join all args to form the prompt string, preserving spaces
        new_prompt = " ".join(args)
        
        # Or better, check if there is text in the message after the command
        # args comes from split() so it works for basic cases, but losing newlines if not careful.
        # But telegram update object has the full text.
        
        # Extract full text after "set"
        message_text = update.message.text
        # /admin system_prompt set ...
        # Ensure we find the right split point
        parts = message_text.split("set", 1)
        if len(parts) > 1:
            new_prompt = parts[1].strip()
        else:
             return "Не удалось прочитать промпт."

        if not new_prompt:
             return "Промпт не может быть пустым."

        try:
            admin_manager.config.system_prompt = new_prompt
            logger.info(f"System prompt updated by admin {update.message.from_user.id}")
            return "Системный промпт успешно обновлен!"
        except Exception as e:
            return await self.handle_error(e, "обновлении системного промпта")
            
    async def reset_prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                          admin_manager, args: List[str]) -> str:
        """Handle /admin system_prompt reset command."""
        try:
            admin_manager.config.system_prompt = ""
            logger.info(f"System prompt reset to default by admin {update.message.from_user.id}")
            
            from src.core.prompt import PromptEngine
            default_prompt = PromptEngine.SYSTEM_PROMPT_TEMPLATE
            
            return f"Системный промпт сброшен на значение по умолчанию:\n\n```\n{default_prompt}\n```"
        except Exception as e:
            return await self.handle_error(e, "сбросе системного промпта")

class SettingsCommands(BaseAdminCommand):
    """Handlers for bot configuration settings."""

    async def manage_chats(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                          admin_manager, args: List[str]) -> str:
        """Handle /admin chat commands."""
        if not args:
            return "Не указана подкоманда. Используйте: list, add, remove"
            
        subcommand = args[0].lower()
        config = admin_manager.config
        
        if subcommand == 'list':
            chats = config.allowed_chats
            if not chats:
                return self.formatter.format_info_message(
                    "Список разрешенных чатов пуст.\n"
                    "**Внимание**: Бот игнорирует ВСЕ сообщения в чатах, "
                    "которых нет в списке (кроме админ-команд)."
                )
            
            response = "**Разрешенные чаты**:\n\n"
            for chat_id in chats:
                response += f"- `{chat_id}`\n"
            return response
            
        elif subcommand == 'add':
            # Determine chat ID
            if len(args) < 2:
                chat_id = update.message.chat_id
            else:
                is_valid, parsed_id, error = self.validator.validate_chat_id(args[1])
                if not is_valid:
                    return error
                chat_id = parsed_id
            
            if chat_id in config.allowed_chats:
                return self.formatter.format_warning_message(f"Чат `{chat_id}` уже в списке.")
            
            config.add_allowed_chat(chat_id)
            return self.formatter.format_success_message(
                f"Чат `{chat_id}` добавлен в список разрешенных."
            )
            
        elif subcommand == 'remove':
            # Determine chat ID
            if len(args) < 2:
                chat_id = update.message.chat_id
            else:
                is_valid, parsed_id, error = self.validator.validate_chat_id(args[1])
                if not is_valid:
                    return error
                chat_id = parsed_id
            
            if chat_id not in config.allowed_chats:
                return self.formatter.format_warning_message(f"Чат `{chat_id}` отсутствует в списке.")
            
            config.remove_allowed_chat(chat_id)
            return self.formatter.format_success_message(
                f"Чат `{chat_id}` удален из списка разрешенных."
            )
            
        else:
            return f"Неизвестная команда: {subcommand}"

    async def manage_frequency(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                             admin_manager, args: List[str]) -> str:
        """Handle /admin frequency command."""
        config = admin_manager.config
        
        if not args:
            return self.formatter.format_info_message(
                f"Текущая частота ответов: **1 ответ на {config.response_frequency} сообщений**"
            )
        
        # Validate frequency value
        is_valid, freq, error = self.validator.validate_frequency(args[0])
        if not is_valid:
            return error
        
        config.response_frequency = freq
        
        if freq == 0:
            return self.formatter.format_success_message(
                "Частота установлена 0 сообщений, ответы будут отправляться "
                "только если бот упомянут в сообщении."
            )
        
        return self.formatter.format_success_message(
            f"Частота ответов установлена: **1 ответ на {freq} сообщений**"
        )
