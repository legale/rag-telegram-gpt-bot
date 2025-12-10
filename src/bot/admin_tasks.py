"""
Background task processing for admin commands.
Handles long-running operations like data ingestion.
"""

import asyncio
import logging
from pathlib import Path
from src.core.syslog2 import *
from typing import Optional, Callable, Tuple, List, Dict
from telegram import Bot

logger = logging.getLogger("legale_admin_tasks")


class IngestionTask:
    """Background task for data ingestion."""
    
    def __init__(self, file_path: Path, profile_manager, clear_existing: bool = False):
        """
        Initialize ingestion task.
        
        Args:
            file_path: Path to JSON file
            profile_manager: ProfileManager instance
            clear_existing: Whether to clear existing data
        """
        self.file_path = file_path
        self.profile_manager = profile_manager
        self.clear_existing = clear_existing
        self.status = "pending"
        self.progress = 0
        self.total = 0
        self.error = None
        self.result = None
    
    async def _update_progress(self, bot: Bot, chat_id: int, message_id: int, text: str) -> None:
        """Update progress indicator."""
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text
        )
    
    async def _parse_file(self, pipeline, bot: Bot, chat_id: int, message_id: int) -> Tuple[List, int]:
        """Parse file and return messages."""
        await self._update_progress(bot, chat_id, message_id, "Начинаю загрузку данных...\n\nЧтение файла...")
        
        syslog2(LOG_NOTICE, "parsing file", path=str(self.file_path))
        messages = pipeline.parser.parse_file(str(self.file_path))
        self.total = len(messages)
        
        await self._update_progress(
            bot, chat_id, message_id,
            f"Загрузка данных...\n\nНайдено сообщений: {self.total:,}\nСоздание чанков..."
        )
        
        return messages, self.total
    
    async def _chunk_messages(self, pipeline, messages: List, bot: Bot, chat_id: int, message_id: int) -> Tuple[List, int]:
        """Chunk messages and return chunks."""
        chunks = pipeline.chunker.chunk_messages(messages)
        chunk_count = len(chunks)
        
        await self._update_progress(
            bot, chat_id, message_id,
            f"Загрузка данных...\n\nСообщений: {self.total:,}\nЧанков: {chunk_count:,}\n\nСохранение в базу данных..."
        )
        
        return chunks, chunk_count
    
    async def _persist_sql(
        self, 
        pipeline, 
        chunks: List, 
        bot: Bot, 
        chat_id: int, 
        message_id: int
    ) -> Tuple[List[str], List[str], List[Dict]]:
        """Persist chunks to SQLite database."""
        import uuid
        import json
        from src.storage.db import ChunkModel
        
        session = pipeline.db.get_session()
        try:
            chunk_models = []
            ids = []
            documents = []
            metadatas = []
            
            chunk_count = len(chunks)
            for i, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                
                # SQL Model
                model = ChunkModel(
                    id=chunk_id,
                    text=chunk.text,
                    metadata_json=json.dumps(chunk.metadata)
                )
                chunk_models.append(model)
                
                # Vector Store Data
                ids.append(chunk_id)
                documents.append(chunk.text)
                metadatas.append(chunk.metadata)
                
                # Update progress every 100 chunks
                if (i + 1) % 100 == 0:
                    self.progress = i + 1
                    progress_pct = (self.progress / chunk_count) * 100
                    await self._update_progress(
                        bot, chat_id, message_id,
                        f"Загрузка данных...\n\nПрогресс: {self.progress:,}/{chunk_count:,} ({progress_pct:.1f}%)\n\n{'▓' * int(progress_pct / 5)}{'░' * (20 - int(progress_pct / 5))}"
                    )
            
            session.add_all(chunk_models)
            session.commit()
            syslog2(LOG_NOTICE, "saved chunks to database", count=len(chunk_models))
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
        
        await self._update_progress(
            bot, chat_id, message_id,
            f"Загрузка данных...\n\nДанные сохранены в БД\n\nСоздание векторных эмбеддингов..."
        )
        
        return ids, documents, metadatas
    
    async def _persist_vectors(
        self, 
        pipeline, 
        ids: List[str], 
        documents: List[str], 
        metadatas: List[Dict],
        bot: Bot,
        chat_id: int,
        message_id: int
    ) -> None:
        """Persist embeddings to vector store."""
        if ids:
            pipeline.vector_store.add_documents(ids=ids, documents=documents, metadatas=metadatas)
            syslog2(LOG_NOTICE, "saved embeddings to vector store", count=len(ids))
    
    async def run(self, bot: Bot, chat_id: int, message_id: int):
        """
        Run ingestion task with progress updates.
        
        Args:
            bot: Telegram bot instance
            chat_id: Chat ID for updates
            message_id: Message ID to edit
        """
        try:
            self.status = "running"
            
            # Import here to avoid circular dependencies
            from src.ingestion.pipeline import IngestionPipeline
            
            # Get profile paths
            paths = self.profile_manager.get_profile_paths()
            
            # Create pipeline
            pipeline = IngestionPipeline(
                db_url=paths['db_url'],
                vector_db_path=str(paths['vector_db_path'])
            )
            
            # Step 1: Parse file
            messages, _ = await self._parse_file(pipeline, bot, chat_id, message_id)
            
            # Step 2: Chunk messages
            chunks, chunk_count = await self._chunk_messages(pipeline, messages, bot, chat_id, message_id)
            
            # Step 3: Clear if requested
            if self.clear_existing:
                pipeline._clear_data()
                await self._update_progress(
                    bot, chat_id, message_id,
                    f"Загрузка данных...\n\nСтарые данные очищены\n\nСохранение новых данных..."
                )
            
            # Step 4: Persist to SQLite
            ids, documents, metadatas = await self._persist_sql(pipeline, chunks, bot, chat_id, message_id)
            
            # Step 5: Persist to vector store
            await self._persist_vectors(pipeline, ids, documents, metadatas, bot, chat_id, message_id)
            
            self.status = "completed"
            self.result = {
                'messages': self.total,
                'chunks': chunk_count
            }
            
            # Final success message
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=(
                    f"**Загрузка завершена!**\n\n"
                    f"Обработано сообщений: {self.total:,}\n"
                    f"Создано чанков: {chunk_count:,}\n"
                    f"Сохранено в БД: {chunk_count:,}\n"
                    f"Создано эмбеддингов: {len(ids):,}\n\n"
                    f"Данные готовы к использованию!"
                ),
                parse_mode='Markdown'
            )
            
        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            syslog2(LOG_ERR, "ingestion task failed", error=str(e))
            
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=f"**Ошибка при загрузке данных:**\n\n`{e}`",
                parse_mode='Markdown'
            )
        
        finally:
            # Clean up temp file
            try:
                if self.file_path.exists():
                    self.file_path.unlink()
                    syslog2(LOG_NOTICE, "cleaned up temp file", path=str(self.file_path))
            except Exception as e:
                syslog2(LOG_WARNING, "temp file cleanup failed", error=str(e))


class TaskManager:
    """Manages background tasks."""
    
    def __init__(self):
        self.tasks = {}
        self.current_task = None
    
    def start_ingestion(self, file_path, profile_manager, clear_existing: bool = False) -> IngestionTask:
        """
        Start ingestion task.
        
        Args:
            file_path: Path to JSON file (str or Path)
            profile_manager: ProfileManager instance
            clear_existing: Whether to clear existing data
        
        Returns:
            IngestionTask instance
        """
        # Convert to Path if string to avoid MagicMock issues in tests
        if isinstance(file_path, str):
            file_path = Path(file_path)
        task = IngestionTask(file_path, profile_manager, clear_existing)
        self.current_task = task
        return task
    
    def get_current_task(self) -> Optional[IngestionTask]:
        """Get current ingestion task."""
        return self.current_task
    
    def clear_current_task(self):
        """Clear current task."""
        self.current_task = None
