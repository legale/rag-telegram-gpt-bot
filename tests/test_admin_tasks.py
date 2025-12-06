"""
Tests for Admin Tasks.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
from src.bot.admin_tasks import TaskManager, IngestionTask

@pytest.fixture
def mock_deps():
    profile_manager = MagicMock()
    profile_manager.get_profile_paths.return_value = {
        'db_url': 'sqlite:///test.db',
        'vector_db_path': '/tmp/vec'
    }
    return profile_manager

def test_task_manager_start(mock_deps):
    tm = TaskManager()
    
    file_path = MagicMock()
    task = tm.start_ingestion(file_path, mock_deps)
    
    assert task.status == "pending"
    assert tm.get_current_task() == task

@pytest.mark.asyncio
async def test_ingestion_task_run_success(mock_deps):
    file_path = MagicMock()
    file_path.exists.return_value = True # For cleanup
    
    task = IngestionTask(file_path, mock_deps)
    
    bot = MagicMock()
    bot.edit_message_text = AsyncMock()
    
    chat_id = 123
    message_id = 456
    
    # Mock pipeline and its components
    with patch('src.ingestion.pipeline.IngestionPipeline') as MockPipeline, \
         patch('src.storage.db.ChunkModel') as MockChunkModel:
        
        pipeline = MockPipeline.return_value
        
        # Mock parser
        pipeline.parser.parse_file.return_value = [{"text": "msg1"}, {"text": "msg2"}]
        
        # Mock chunker
        chunk1 = MagicMock()
        chunk1.text = "chunk1"
        chunk1.metadata = {"meta": "data"}
        pipeline.chunker.chunk_messages.return_value = [chunk1]
        
        # Mock session
        session = MagicMock()
        pipeline.db.get_session.return_value = session
        
        # Run task
        await task.run(bot, chat_id, message_id)
        
        # Assertions
        assert task.status == "completed"
        assert task.total == 2
        assert task.result['chunks'] == 1
        
        # Status messages sent
        assert bot.edit_message_text.call_count >= 3
        
        # Check success message in last call
        # call_args.kwargs might be empty if positional args were used, but edit_message_text usually uses keywords in my code
        # Code: chat_id=chat_id, message_id=message_id, text="..."
        args, kwargs = bot.edit_message_text.call_args
        assert "Загрузка завершена" in kwargs['text']
               
        # DB interactions
        session.add_all.assert_called()
        session.commit.assert_called()
        
        # Vector store interactions
        pipeline.vector_store.add_documents.assert_called()
        
        # Cleanup
        file_path.unlink.assert_called()

@pytest.mark.asyncio
async def test_ingestion_task_run_failure(mock_deps):
    file_path = MagicMock()
    file_path.exists.return_value = True
    
    task = IngestionTask(file_path, mock_deps)
    
    bot = MagicMock()
    bot.edit_message_text = AsyncMock()
    
    with patch('src.ingestion.pipeline.IngestionPipeline', side_effect=Exception("Pipeline Error")):
        await task.run(bot, 123, 456)
        
        assert task.status == "failed"
        assert "Pipeline Error" in task.error
        
        # Verify error message sent
        args, kwargs = bot.edit_message_text.call_args
        assert "Ошибка" in kwargs['text']
