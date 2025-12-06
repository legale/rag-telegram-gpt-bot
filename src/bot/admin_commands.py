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

logger = logging.getLogger("legale_admin_commands")


class ProfileCommands:
    """Handlers for profile management commands."""
    
    def __init__(self, profile_manager):
        """
        Initialize ProfileCommands.
        
        Args:
            profile_manager: ProfileManager instance
        """
        self.profile_manager = profile_manager
    
    async def list_profiles(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                           admin_manager, args: List[str]) -> str:
        """Handle /admin profile list command."""
        profiles_dir = self.profile_manager.profiles_dir
        
        if not profiles_dir.exists():
            return "📁 Профили не найдены.\n\nСоздайте первый профиль:\n`/admin profile create <name>`"
        
        profiles = [p for p in profiles_dir.iterdir() if p.is_dir()]
        
        if not profiles:
            return "📁 Профили не найдены.\n\nСоздайте первый профиль:\n`/admin profile create <name>`"
        
        current = self.profile_manager.get_current_profile()
        
        response = "📁 **Доступные профили:**\n\n"
        
        for profile_dir in sorted(profiles, key=lambda p: p.name):
            profile_name = profile_dir.name
            is_active = profile_name == current
            
            # Get database info
            db_path = profile_dir / "legale_bot.db"
            db_exists = db_path.exists()
            db_size = ""
            
            if db_exists:
                size_bytes = db_path.stat().st_size
                if size_bytes < 1024:
                    db_size = f"{size_bytes}B"
                elif size_bytes < 1024 * 1024:
                    db_size = f"{size_bytes / 1024:.1f}KB"
                else:
                    db_size = f"{size_bytes / (1024 * 1024):.1f}MB"
            
            # Get chunk count
            chunk_count = "?"
            if db_exists:
                try:
                    import sqlite3
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM chunks")
                    chunk_count = cursor.fetchone()[0]
                    conn.close()
                except Exception:
                    pass
            
            marker = "✅" if is_active else "📂"
            active_text = " **(активный)**" if is_active else ""
            db_text = f"БД: {db_size}, чанков: {chunk_count}" if db_exists else "БД не создана"
            
            response += f"{marker} `{profile_name}`{active_text}\n"
            response += f"   {db_text}\n\n"
        
        response += f"\n**Активный профиль:** `{current}`"
        
        return response
    
    async def create_profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                            admin_manager, args: List[str]) -> str:
        """Handle /admin profile create <name> command."""
        if not args:
            return "❌ Укажите имя профиля.\n\nИспользование: `/admin profile create <name>`"
        
        profile_name = args[0]
        
        # Validate profile name
        if not profile_name.replace('_', '').replace('-', '').isalnum():
            return "❌ Имя профиля может содержать только буквы, цифры, дефисы и подчёркивания."
        
        profile_dir = self.profile_manager.get_profile_dir(profile_name)
        
        if profile_dir.exists():
            return f"⚠️ Профиль `{profile_name}` уже существует.\n\nПуть: `{profile_dir}`"
        
        try:
            # Create profile
            self.profile_manager.create_profile(profile_name, set_active=False)
            
            paths = self.profile_manager.get_profile_paths(profile_name)
            
            response = (
                f"✅ Профиль `{profile_name}` создан!\n\n"
                f"📁 Директория: `{paths['profile_dir']}`\n"
                f"💾 База данных: `{paths['db_path']}`\n"
                f"🔍 Векторное хранилище: `{paths['vector_db_path']}`\n\n"
                f"Для переключения на этот профиль:\n"
                f"`/admin profile switch {profile_name}`"
            )
            
            logger.info(f"Profile '{profile_name}' created by admin {update.message.from_user.id}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error creating profile '{profile_name}': {e}", exc_info=True)
            return f"❌ Ошибка при создании профиля: {e}"
    
    async def switch_profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                            admin_manager, args: List[str]) -> str:
        """Handle /admin profile switch <name> command."""
        if not args:
            return "❌ Укажите имя профиля.\n\nИспользование: `/admin profile switch <name>`"
        
        profile_name = args[0]
        profile_dir = self.profile_manager.get_profile_dir(profile_name)
        
        if not profile_dir.exists():
            return (
                f"❌ Профиль `{profile_name}` не существует.\n\n"
                f"Создайте его:\n`/admin profile create {profile_name}`"
            )
        
        current = self.profile_manager.get_current_profile()
        
        if profile_name == current:
            return f"ℹ️ Профиль `{profile_name}` уже активен."
        
        try:
            # Switch profile
            self.profile_manager.set_current_profile(profile_name)
            
            response = (
                f"✅ Переключено на профиль `{profile_name}`\n\n"
                f"⚠️ **Внимание:** Для применения изменений необходимо перезапустить бота:\n"
                f"`/admin restart`"
            )
            
            logger.info(f"Profile switched to '{profile_name}' by admin {update.message.from_user.id}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error switching to profile '{profile_name}': {e}", exc_info=True)
            return f"❌ Ошибка при переключении профиля: {e}"
    
    async def delete_profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                            admin_manager, args: List[str]) -> str:
        """Handle /admin profile delete <name> command."""
        if not args:
            return "❌ Укажите имя профиля.\n\nИспользование: `/admin profile delete <name>`"
        
        profile_name = args[0]
        current = self.profile_manager.get_current_profile()
        
        if profile_name == current:
            return (
                f"❌ Невозможно удалить активный профиль `{profile_name}`.\n\n"
                f"Сначала переключитесь на другой профиль:\n"
                f"`/admin profile switch <другой_профиль>`"
            )
        
        profile_dir = self.profile_manager.get_profile_dir(profile_name)
        
        if not profile_dir.exists():
            return f"❌ Профиль `{profile_name}` не существует."
        
        # Get profile info before deletion
        db_path = profile_dir / "legale_bot.db"
        db_exists = db_path.exists()
        
        info_text = f"📁 Профиль: `{profile_name}`\n"
        info_text += f"📂 Путь: `{profile_dir}`\n"
        
        if db_exists:
            size_bytes = db_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            info_text += f"💾 Размер БД: {size_mb:.2f} MB\n"
        
        # For now, just show info and ask for confirmation
        # TODO: Implement confirmation with inline keyboard
        response = (
            f"⚠️ **Подтверждение удаления**\n\n"
            f"{info_text}\n"
            f"❌ **Это действие необратимо!**\n\n"
            f"Для подтверждения удаления используйте:\n"
            f"`/admin profile delete {profile_name} confirm`"
        )
        
        # Check for confirmation
        if len(args) > 1 and args[1] == "confirm":
            try:
                import shutil
                shutil.rmtree(profile_dir)
                
                logger.warning(f"Profile '{profile_name}' deleted by admin {update.message.from_user.id}")
                
                return (
                    f"✅ Профиль `{profile_name}` удалён.\n\n"
                    f"Все данные профиля были удалены безвозвратно."
                )
                
            except Exception as e:
                logger.error(f"Error deleting profile '{profile_name}': {e}", exc_info=True)
                return f"❌ Ошибка при удалении профиля: {e}"
        
        return response
    
    async def profile_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                          admin_manager, args: List[str]) -> str:
        """Handle /admin profile info [name] command."""
        profile_name = args[0] if args else self.profile_manager.get_current_profile()
        
        profile_dir = self.profile_manager.get_profile_dir(profile_name)
        
        if not profile_dir.exists():
            return f"❌ Профиль `{profile_name}` не существует."
        
        paths = self.profile_manager.get_profile_paths(profile_name)
        current = self.profile_manager.get_current_profile()
        is_active = profile_name == current
        
        response = f"📊 **Информация о профиле `{profile_name}`**\n\n"
        
        if is_active:
            response += "✅ **Статус:** Активный\n\n"
        else:
            response += "📂 **Статус:** Неактивный\n\n"
        
        response += f"📁 **Директория:** `{paths['profile_dir']}`\n\n"
        
        # Database info
        db_path = paths['db_path']
        if db_path.exists():
            size_bytes = db_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            response += f"💾 **База данных:**\n"
            response += f"   Путь: `{db_path}`\n"
            response += f"   Размер: {size_mb:.2f} MB\n"
            
            # Get chunk count
            try:
                import sqlite3
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM chunks")
                chunk_count = cursor.fetchone()[0]
                conn.close()
                response += f"   Чанков: {chunk_count:,}\n"
            except Exception as e:
                response += f"   Чанков: ошибка чтения ({e})\n"
        else:
            response += f"💾 **База данных:** Не создана\n"
        
        response += "\n"
        
        # Vector store info
        vector_path = paths['vector_db_path']
        if vector_path.exists():
            # Calculate directory size
            total_size = sum(f.stat().st_size for f in vector_path.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            response += f"🔍 **Векторное хранилище:**\n"
            response += f"   Путь: `{vector_path}`\n"
            response += f"   Размер: {size_mb:.2f} MB\n"
        else:
            response += f"🔍 **Векторное хранилище:** Не создано\n"
        
        response += "\n"
        
        # Session file
        session_path = paths['session_file']
        if session_path.exists():
            response += f"📱 **Telegram сессия:** Создана\n"
        else:
            response += f"📱 **Telegram сессия:** Не создана\n"
        
        # Admin file
        admin_file = profile_dir / "admin.json"
        if admin_file.exists():
            response += f"👤 **Администратор:** Назначен\n"
        else:
            response += f"👤 **Администратор:** Не назначен\n"
        
        return response


class HelpCommands:
    """Handlers for help commands."""
    
    async def show_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                       admin_manager, args: List[str]) -> str:
        """Handle /admin help [command] command."""
        if not args:
            # General help
            return (
                "📚 **Справка по админ-командам**\n\n"
                "**Управление профилями:**\n"
                "• `profile list` - список всех профилей\n"
                "• `profile create <name>` - создать новый профиль\n"
                "• `profile switch <name>` - переключиться на профиль\n"
                "• `profile delete <name>` - удалить профиль\n"
                "• `profile info [name]` - информация о профиле\n\n"
                "**Загрузка данных:**\n"
                "• `ingest` - загрузить данные (отправьте JSON)\n"
                "• `ingest clear` - очистить данные\n"
                "• `ingest status` - статус загрузки\n\n"
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
                "📁 **Управление профилями**\n\n"
                "Профили позволяют управлять несколькими ботами с отдельными базами данных.\n\n"
                "**Команды:**\n"
                "• `/admin profile list` - показать все профили\n"
                "• `/admin profile create <name>` - создать профиль\n"
                "• `/admin profile switch <name>` - переключить профиль\n"
                "• `/admin profile delete <name>` - удалить профиль\n"
                "• `/admin profile info [name]` - информация о профиле\n\n"
                "**Примеры:**\n"
                "`/admin profile create production`\n"
                "`/admin profile switch production`\n"
                "`/admin profile info production`"
            ),
            "ingest": (
                "📥 **Загрузка данных**\n\n"
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
        }
        if command in help_texts:
            return help_texts[command]
        else:
            return f"❌ Справка по команде `{command}` не найдена.\n\nИспользуйте `/admin help` для списка команд."


class IngestCommands:
    """Handlers for data ingestion commands."""
    
    def __init__(self, profile_manager, task_manager):
        """
        Initialize IngestCommands.
        
        Args:
            profile_manager: ProfileManager instance
            task_manager: TaskManager instance
        """
        self.profile_manager = profile_manager
        self.task_manager = task_manager
        self.waiting_for_file = {}  # user_id -> bool
    
    async def start_ingest(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                          admin_manager, args: List[str]) -> str:
        """Handle /admin ingest command."""
        user_id = update.message.from_user.id
        
        # Mark user as waiting for file
        self.waiting_for_file[user_id] = True
        
        return (
            "📤 **Загрузка данных**\n\n"
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
            paths = self.profile_manager.get_profile_paths()
            
            # Get current stats before clearing
            db_path = paths['db_path']
            chunk_count = 0
            db_size = 0
            
            if db_path.exists():
                db_size = db_path.stat().st_size / (1024 * 1024)
                try:
                    import sqlite3
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM chunks")
                    chunk_count = cursor.fetchone()[0]
                    conn.close()
                except Exception:
                    pass
            
            # Create pipeline and clear data
            pipeline = IngestionPipeline(
                db_url=paths['db_url'],
                vector_db_path=str(paths['vector_db_path'])
            )
            
            pipeline._clear_data()
            
            logger.info(f"Data cleared by admin {update.message.from_user.id}")
            
            return (
                f"✅ **Данные очищены**\n\n"
                f"Удалено чанков: {chunk_count:,}\n"
                f"Освобождено места: {db_size:.2f} MB\n\n"
                f"База данных и векторное хранилище очищены."
            )
            
        except Exception as e:
            logger.error(f"Error clearing data: {e}", exc_info=True)
            return f"❌ Ошибка при очистке данных: {e}"
    
    async def ingest_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                           admin_manager, args: List[str]) -> str:
        """Handle /admin ingest status command."""
        task = self.task_manager.get_current_task()
        
        if not task:
            return "ℹ️ Нет активных задач загрузки."
        
        if task.status == "pending":
            return "⏳ Задача загрузки ожидает запуска..."
        elif task.status == "running":
            progress_pct = (task.progress / task.total * 100) if task.total > 0 else 0
            return (
                f"⏳ **Загрузка данных в процессе**\n\n"
                f"Прогресс: {task.progress:,}/{task.total:,} ({progress_pct:.1f}%)\n\n"
                f"{'▓' * int(progress_pct / 5)}{'░' * (20 - int(progress_pct / 5))}"
            )
        elif task.status == "completed":
            return (
                f"✅ **Загрузка завершена**\n\n"
                f"Обработано сообщений: {task.result['messages']:,}\n"
                f"Создано чанков: {task.result['chunks']:,}"
            )
        elif task.status == "failed":
            return f"❌ **Загрузка завершилась с ошибкой:**\n\n`{task.error}`"
        else:
            return f"❓ Неизвестный статус: {task.status}"
    
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
            return "❌ Файл должен быть в формате JSON.\n\nОтправьте JSON файл с дампом чата."
        
        # Check file size (max 20MB)
        if document.file_size > 20 * 1024 * 1024:
            return "❌ Файл слишком большой (макс. 20MB).\n\nИспользуйте CLI для загрузки больших файлов."
        
        try:
            # Download file
            file = await context.bot.get_file(document.file_id)
            
            # Create temp directory
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / "legale_bot"
            temp_dir.mkdir(exist_ok=True)
            
            # Save file
            temp_file = temp_dir / f"{user_id}_{document.file_name}"
            await file.download_to_drive(temp_file)
            
            logger.info(f"File downloaded: {temp_file} ({document.file_size} bytes)")
            
            # Send initial message
            status_message = await update.message.reply_text(
                "📥 Файл получен!\n\nПодготовка к загрузке..."
            )
            
            # Start ingestion task
            task = self.task_manager.start_ingestion(temp_file, self.profile_manager)
            
            # Run task in background
            import asyncio
            asyncio.create_task(
                task.run(context.bot, update.message.chat_id, status_message.message_id)
            )
            
            return None  # Message already sent
            
        except Exception as e:
            logger.error(f"Error handling file upload: {e}", exc_info=True)
            return f"❌ Ошибка при обработке файла: {e}"
    
    def is_waiting_for_file(self, user_id: int) -> bool:
        """Check if user is waiting for file upload."""
        return user_id in self.waiting_for_file and self.waiting_for_file[user_id]


class StatsCommands:
    """Handlers for statistics and monitoring commands."""
    
    def __init__(self, profile_manager):
        """
        Initialize StatsCommands.
        
        Args:
            profile_manager: ProfileManager instance
        """
        self.profile_manager = profile_manager
    
    async def show_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                        admin_manager, args: List[str]) -> str:
        """Handle /admin stats command."""
        try:
            import sqlite3
            from datetime import datetime
            
            # Get profile paths
            paths = self.profile_manager.get_profile_paths()
            profile_name = self.profile_manager.get_current_profile()
            
            response = f"📊 **Статистика бота**\n\n"
            response += f"**Профиль:** `{profile_name}`\n\n"
            
            # Database stats
            db_path = paths['db_path']
            if db_path.exists():
                # Database size
                db_size = db_path.stat().st_size / (1024 * 1024)
                response += f"💾 **База данных:**\n"
                response += f"   Размер: {db_size:.2f} MB\n"
                
                # Chunk count
                try:
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()
                    
                    cursor.execute("SELECT COUNT(*) FROM chunks")
                    chunk_count = cursor.fetchone()[0]
                    response += f"   Чанков: {chunk_count:,}\n"
                    
                    # Get date range of chunks
                    cursor.execute("""
                        SELECT 
                            MIN(json_extract(metadata_json, '$.date')) as min_date,
                            MAX(json_extract(metadata_json, '$.date')) as max_date
                        FROM chunks
                        WHERE json_extract(metadata_json, '$.date') IS NOT NULL
                    """)
                    dates = cursor.fetchone()
                    if dates[0] and dates[1]:
                        response += f"   Период: {dates[0][:10]} - {dates[1][:10]}\n"
                    
                    conn.close()
                except Exception as e:
                    response += f"   Ошибка чтения: {e}\n"
            else:
                response += f"💾 **База данных:** Не создана\n"
            
            response += "\n"
            
            # Vector store stats
            vector_path = paths['vector_db_path']
            if vector_path.exists():
                total_size = sum(f.stat().st_size for f in vector_path.rglob('*') if f.is_file())
                size_mb = total_size / (1024 * 1024)
                response += f"🔍 **Векторное хранилище:**\n"
                response += f"   Размер: {size_mb:.2f} MB\n"
            else:
                response += f"🔍 **Векторное хранилище:** Не создано\n"
            
            response += "\n"
            
            # System info
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            response += f"⚙️ **Система:**\n"
            response += f"   Память: {memory_mb:.1f} MB\n"
            response += f"   CPU: {psutil.cpu_percent()}%\n"
            
            # Disk space
            disk = psutil.disk_usage(str(paths['profile_dir']))
            disk_free_gb = disk.free / (1024 * 1024 * 1024)
            disk_total_gb = disk.total / (1024 * 1024 * 1024)
            disk_percent = disk.percent
            
            response += f"   Диск: {disk_free_gb:.1f}/{disk_total_gb:.1f} GB свободно ({100-disk_percent:.1f}%)\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}", exc_info=True)
            return f"❌ Ошибка при получении статистики: {e}"
    
    async def health_check(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                          admin_manager, args: List[str]) -> str:
        """Handle /admin health command."""
        try:
            import sqlite3
            
            paths = self.profile_manager.get_profile_paths()
            
            response = "🏥 **Проверка здоровья системы**\n\n"
            
            checks = []
            
            # Database check
            db_path = paths['db_path']
            if db_path.exists():
                try:
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    conn.close()
                    checks.append(("💾 База данных", "✅ OK"))
                except Exception as e:
                    checks.append(("💾 База данных", f"❌ Ошибка: {e}"))
            else:
                checks.append(("💾 База данных", "⚠️ Не создана"))
            
            # Vector store check
            vector_path = paths['vector_db_path']
            if vector_path.exists():
                checks.append(("🔍 Векторное хранилище", "✅ OK"))
            else:
                checks.append(("🔍 Векторное хранилище", "⚠️ Не создано"))
            
            # LLM API check
            try:
                import os
                api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
                if api_key:
                    checks.append(("🤖 LLM API ключ", "✅ Установлен"))
                else:
                    checks.append(("🤖 LLM API ключ", "❌ Не установлен"))
            except Exception as e:
                checks.append(("🤖 LLM API ключ", f"❌ Ошибка: {e}"))
            
            # Embedding API check
            try:
                import os
                voyage_key = os.getenv("VOYAGE_API_KEY")
                if voyage_key:
                    checks.append(("🔤 Embedding API ключ", "✅ Установлен"))
                else:
                    checks.append(("🔤 Embedding API ключ", "❌ Не установлен"))
            except Exception as e:
                checks.append(("🔤 Embedding API ключ", f"❌ Ошибка: {e}"))
            
            # Memory check
            try:
                import psutil
                memory = psutil.virtual_memory()
                if memory.percent < 90:
                    checks.append(("💾 Память", f"✅ {memory.percent:.1f}% использовано"))
                else:
                    checks.append(("💾 Память", f"⚠️ {memory.percent:.1f}% использовано"))
            except Exception as e:
                checks.append(("💾 Память", f"❌ Ошибка: {e}"))
            
            # Disk check
            try:
                import psutil
                disk = psutil.disk_usage(str(paths['profile_dir']))
                if disk.percent < 90:
                    checks.append(("💿 Диск", f"✅ {disk.percent:.1f}% использовано"))
                else:
                    checks.append(("💿 Диск", f"⚠️ {disk.percent:.1f}% использовано"))
            except Exception as e:
                checks.append(("💿 Диск", f"❌ Ошибка: {e}"))
            
            # Format results
            for name, status in checks:
                response += f"{name}: {status}\n"
            
            # Overall status
            failed = sum(1 for _, status in checks if "❌" in status)
            warnings = sum(1 for _, status in checks if "⚠️" in status)
            
            response += "\n"
            if failed == 0 and warnings == 0:
                response += "✅ **Все системы работают нормально**"
            elif failed == 0:
                response += f"⚠️ **Обнаружено {warnings} предупреждений**"
            else:
                response += f"❌ **Обнаружено {failed} ошибок, {warnings} предупреждений**"
            
            return response
            
        except Exception as e:
            logger.error(f"Error in health check: {e}", exc_info=True)
            return f"❌ Ошибка при проверке здоровья: {e}"
    
    async def show_logs(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                       admin_manager, args: List[str]) -> str:
        """Handle /admin logs [lines] command."""
        try:
            # Get number of lines to show
            lines = 50
            if args:
                try:
                    lines = int(args[0])
                    if lines < 1 or lines > 200:
                        return "❌ Количество строк должно быть от 1 до 200."
                except ValueError:
                    return "❌ Неверное количество строк. Используйте число."
            
            # Get profile directory
            paths = self.profile_manager.get_profile_paths()
            log_file = paths['profile_dir'] / "bot.log"
            
            if not log_file.exists():
                return "ℹ️ Лог-файл не найден.\n\nЛоги появятся после запуска бота в daemon режиме."
            
            # Read last N lines
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                last_lines = all_lines[-lines:]
            
            if not last_lines:
                return "ℹ️ Лог-файл пуст."
            
            # Format response
            response = f"📋 **Последние {len(last_lines)} строк логов**\n\n"
            response += "```\n"
            response += "".join(last_lines)
            response += "```"
            
            # Truncate if too long (Telegram limit is 4096 chars)
            if len(response) > 4000:
                response = response[:3900] + "\n...\n```\n\n⚠️ Логи обрезаны. Используйте меньшее количество строк."
            
            return response
            
        except Exception as e:
            logger.error(f"Error reading logs: {e}", exc_info=True)
            return f"❌ Ошибка при чтении логов: {e}"


class ControlCommands:
    """Handlers for bot control commands."""
    
    def __init__(self, profile_manager):
        self.profile_manager = profile_manager
    
    async def restart_bot(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                         admin_manager, args: List[str]) -> str:
        """Handle /admin restart command."""
        # Use job queue to exit after sending response
        if context.job_queue:
            context.job_queue.run_once(self._perform_restart, 2)
            return "🔄 **Перезапуск бота...**\n\nБот перезагрузится через 2 секунды."
        else:
            # Fallback if no job queue
            import asyncio
            asyncio.create_task(self._delayed_restart())
            return "🔄 **Перезапуск бота...**\n\nБот перезагрузится через 2 секунды."

    async def _delayed_restart(self):
        import asyncio
        await asyncio.sleep(2)
        self._perform_restart(None)

    async def _perform_restart(self, context):
        """Internal method to stop the process."""
        import sys
        logger.info("Restarting bot via admin command (sys.exit(1))...")
        sys.exit(1)


class SettingsCommands:
    """Handlers for bot configuration settings."""
    
    def __init__(self, profile_manager):
        self.profile_manager = profile_manager

    async def manage_chats(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                          admin_manager, args: List[str]) -> str:
        """Handle /admin chat commands."""
        if not args:
            return "❌ Не указана подкоманда. Используйте: list, add, remove"
            
        subcommand = args[0].lower()
        config = admin_manager.config
        
        if subcommand == 'list':
            chats = config.allowed_chats
            if not chats:
                return "ℹ️ Список разрешенных чатов пуст.\n⚠️ **Внимание**: Бот игнорирует ВСЕ сообщения в чатах, которых нет в списке (кроме админ-команд)."
            
            response = "📋 **Разрешенные чаты**:\n\n"
            for chat_id in chats:
                response += f"- `{chat_id}`\n"
            return response
            
        elif subcommand == 'add':
            if len(args) < 2:
                # Try to use current chat ID
                chat_id = update.message.chat_id
            else:
                try:
                    chat_id = int(args[1])
                except ValueError:
                    return "❌ Неверный ID чата. Используйте число."
            
            if chat_id in config.allowed_chats:
                return f"⚠️ Чат `{chat_id}` уже в списке."
            
            config.add_allowed_chat(chat_id)
            return f"✅ Чат `{chat_id}` добавлен в список разрешенных."
            
        elif subcommand == 'remove':
            if len(args) < 2:
                # Try to use current chat ID
                chat_id = update.message.chat_id
            else:
                try:
                    chat_id = int(args[1])
                except ValueError:
                    return "❌ Неверный ID чата. Используйте число."
            
            if chat_id not in config.allowed_chats:
                return f"⚠️ Чат `{chat_id}` отсутствует в списке."
            
            config.remove_allowed_chat(chat_id)
            return f"✅ Чат `{chat_id}` удален из списка разрешенных."
            
        else:
            return f"❌ Неизвестная команда: {subcommand}"

    async def manage_frequency(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                             admin_manager, args: List[str]) -> str:
        """Handle /admin frequency command."""
        config = admin_manager.config
        
        if not args:
            return f"ℹ️ Текущая частота ответов: **1 ответ на {config.response_frequency} сообщений**"
            
        try:
            freq = int(args[0])
            if freq < 1:
                return "❌ Частота должна быть >= 1."
                
            config.response_frequency = freq
            return f"✅ Частота ответов установлена: **1 ответ на {freq} сообщений**"
        except ValueError:
            return "❌ Используйте число."
