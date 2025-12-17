"""
Tests for ingestion pipeline.
"""
import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import json
import tempfile
import os
from src.ingestion.pipeline import IngestionPipeline, IngestionPipelineError, ConfigurationError


class TestIngestionPipelineError:
    def test_ingestion_pipeline_error(self):
        error = IngestionPipelineError("test error")
        assert str(error) == "test error"
        assert isinstance(error, Exception)

    def test_configuration_error(self):
        error = ConfigurationError("config error")
        assert str(error) == "config error"
        assert isinstance(error, IngestionPipelineError)
        assert isinstance(error, Exception)


class TestIngestionPipeline:
    @pytest.fixture
    def tmp_profile_dir(self, tmp_path):
        """Create a temporary profile directory with config."""
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()
        
        config_file = profile_dir / "config.json"
        config_data = {
            "embedding_model": "test-model",
            "embedding_generator": "local",
            "current_model": "openai/gpt-4",
            "chunk_token_min": 50,
            "chunk_token_max": 200,
            "chunk_overlap_ratio": 0.1
        }
        config_file.write_text(json.dumps(config_data))
        
        return profile_dir

    @pytest.fixture
    def mock_db(self):
        """Mock database."""
        db = Mock()
        db.get_session.return_value.__enter__ = Mock(return_value=Mock())
        db.get_session.return_value.__exit__ = Mock(return_value=False)
        db.count_chunks.return_value = 0
        db.clear.return_value = 0
        db.clear_messages.return_value = 0
        db.add_messages_batch.return_value = 0
        db.get_all_topics_l1.return_value = []
        db.get_all_topics_l2.return_value = []
        return db

    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store."""
        vs = Mock()
        vs.count.return_value = 0
        vs.clear.return_value = 0
        vs.collection = Mock()
        vs.collection.count.return_value = 0
        vs.collection.get.return_value = {"ids": []}
        vs.collection.delete.return_value = None
        vs.persist_directory = "/tmp/vec"
        vs.collection_name = "test-collection"
        vs.expected_dimension = 384
        vs.get_all_embeddings.return_value = {"ids": [], "embeddings": []}
        vs.add_documents_with_embeddings.return_value = None
        return vs

    @pytest.fixture
    def mock_parser(self):
        """Mock parser."""
        parser = Mock()
        parser.parse_file.return_value = []
        return parser

    @pytest.fixture
    def mock_chunker(self):
        """Mock chunker."""
        chunker = Mock()
        chunker.chunk_messages.return_value = []
        chunker.chunk_token_min = 50
        chunker.chunk_token_max = 200
        chunker.chunk_overlap_ratio = 0.1
        return chunker

    @pytest.fixture
    def mock_embedding_client(self):
        """Mock embedding client."""
        client = Mock()
        client.get_dimension.return_value = 384
        client.get_embeddings.return_value = [[0.1] * 384]
        return client

    @pytest.fixture
    def pipeline(self, tmp_profile_dir, mock_db, mock_vector_store, mock_parser, mock_chunker, mock_embedding_client, tmp_path):
        """Create pipeline instance with mocks."""
        with patch('src.ingestion.pipeline.ChatParser', return_value=mock_parser), \
             patch('src.ingestion.pipeline.Database', return_value=mock_db), \
             patch('src.ingestion.pipeline.VectorStore', return_value=mock_vector_store), \
             patch('src.ingestion.pipeline.create_embedding_client', return_value=mock_embedding_client), \
             patch('src.ingestion.pipeline.MessageChunker', return_value=mock_chunker), \
             patch('src.ingestion.pipeline.BotConfig') as mock_config_class:
            
            mock_config = Mock()
            mock_config.data = {
                "embedding_model": "test-model",
                "current_model": "openai/gpt-4"
            }
            mock_config.embedding_generator = "local"
            mock_config.chunk_token_min = 50
            mock_config.chunk_token_max = 200
            mock_config.chunk_overlap_ratio = 0.1
            mock_config.config_file = str(tmp_profile_dir / "config.json")
            mock_config_class.return_value = mock_config
            
            pipeline = IngestionPipeline(
                db_url="sqlite:///test.db",
                vector_db_path=str(tmp_path / "vector_db"),
                collection_name="test-collection",
                profile_dir=str(tmp_profile_dir)
            )
            return pipeline

    def test_init_success(self, tmp_profile_dir, mock_db, mock_vector_store, mock_parser, mock_chunker, mock_embedding_client, tmp_path):
        """Test successful pipeline initialization."""
        with patch('src.ingestion.pipeline.ChatParser', return_value=mock_parser), \
             patch('src.ingestion.pipeline.Database', return_value=mock_db), \
             patch('src.ingestion.pipeline.VectorStore', return_value=mock_vector_store), \
             patch('src.ingestion.pipeline.create_embedding_client', return_value=mock_embedding_client), \
             patch('src.ingestion.pipeline.MessageChunker', return_value=mock_chunker), \
             patch('src.ingestion.pipeline.BotConfig') as mock_config_class:
            
            mock_config = Mock()
            mock_config.data = {
                "embedding_model": "test-model",
                "current_model": "openai/gpt-4"
            }
            mock_config.embedding_generator = "local"
            mock_config.chunk_token_min = 50
            mock_config.chunk_token_max = 200
            mock_config.chunk_overlap_ratio = 0.1
            mock_config.config_file = str(tmp_profile_dir / "config.json")
            mock_config_class.return_value = mock_config
            
            pipeline = IngestionPipeline(
                db_url="sqlite:///test.db",
                vector_db_path=str(tmp_path / "vector_db"),
                collection_name="test-collection",
                profile_dir=str(tmp_profile_dir)
            )
            
            assert pipeline.db_url == "sqlite:///test.db"
            assert pipeline.profile_dir == Path(tmp_profile_dir)

    def test_init_no_profile_dir(self, tmp_path):
        """Test initialization without profile directory."""
        with pytest.raises(ConfigurationError, match="profile directory not found"):
            IngestionPipeline(
                db_url="sqlite:///test.db",
                vector_db_path=str(tmp_path / "vector_db"),
                collection_name="test-collection",
                profile_dir=str(tmp_path / "nonexistent")
            )

    def test_init_no_embedding_model(self, tmp_profile_dir, tmp_path):
        """Test initialization without embedding_model in config."""
        config_file = tmp_profile_dir / "config.json"
        config_data = {
            "current_model": "openai/gpt-4"
        }
        config_file.write_text(json.dumps(config_data))
        
        with patch('src.ingestion.pipeline.ChatParser'), \
             patch('src.ingestion.pipeline.Database'), \
             patch('src.ingestion.pipeline.VectorStore'), \
             patch('src.ingestion.pipeline.BotConfig') as mock_config_class:
            
            mock_config = Mock()
            mock_config.data = {}
            mock_config.config_file = str(config_file)
            mock_config_class.return_value = mock_config
            
            with pytest.raises(ConfigurationError, match="embedding_model is not set"):
                IngestionPipeline(
                    db_url="sqlite:///test.db",
                    vector_db_path=str(tmp_path / "vector_db"),
                    collection_name="test-collection",
                    profile_dir=str(tmp_profile_dir)
                )

    def test_clear_data(self, pipeline, mock_db, mock_vector_store):
        """Test clearing all data."""
        mock_db.count_chunks.return_value = 10
        mock_db.clear.return_value = 10
        mock_vector_store.count.return_value = 5
        mock_vector_store.clear.return_value = 5
        
        pipeline._clear_data()
        
        mock_db.clear.assert_called_once()
        mock_vector_store.clear.assert_called_once()

    def test_clear_stage0(self, pipeline, mock_db):
        """Test clearing stage 0 (messages)."""
        mock_db.clear_messages.return_value = 5
        
        result = pipeline.clear_stage0()
        
        assert result == 5
        mock_db.clear_messages.assert_called_once()

    def test_clear_stage1(self, pipeline, mock_db):
        """Test clearing stage 1 (chunks)."""
        mock_db.clear.return_value = 10
        
        result = pipeline.clear_stage1()
        
        assert result == 10
        mock_db.clear.assert_called_once()

    def test_clear_stage2(self, pipeline, mock_db):
        """Test clearing stage 2 (embeddings)."""
        session = Mock()
        session.query.return_value.update.return_value = 5
        session.commit.return_value = None
        mock_db.get_session.return_value = session
        
        result = pipeline.clear_stage2()
        
        assert result == 5
        session.query.return_value.update.assert_called_once()
        session.commit.assert_called_once()

    def test_clear_stage3(self, pipeline, mock_vector_store):
        """Test clearing stage 3 (vector DB)."""
        mock_vector_store.collection.count.return_value = 10
        mock_vector_store.collection.get.return_value = {"ids": ["id1", "id2"]}
        
        result = pipeline.clear_stage3()
        
        assert result == 10
        mock_vector_store.collection.delete.assert_called_once()

    def test_clear_all(self, pipeline):
        """Test clearing all stages."""
        pipeline.clear_stage3 = Mock(return_value=5)
        pipeline.clear_stage2 = Mock(return_value=3)
        pipeline.clear_stage1 = Mock(return_value=2)
        pipeline.clear_stage0 = Mock(return_value=1)
        
        pipeline.clear_all()
        
        pipeline.clear_stage3.assert_called_once()
        pipeline.clear_stage2.assert_called_once()
        pipeline.clear_stage1.assert_called_once()
        pipeline.clear_stage0.assert_called_once()

    def test_run_stage0(self, pipeline, mock_parser):
        """Test running stage 0."""
        test_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        test_file.write("test content")
        test_file.close()
        
        try:
            pipeline.parse_and_store_messages = Mock()
            pipeline.run_stage0(test_file.name)
            pipeline.parse_and_store_messages.assert_called_once_with(test_file.name)
        finally:
            os.unlink(test_file.name)

    def test_run_stage0_no_file(self, pipeline):
        """Test running stage 0 without file."""
        with pytest.raises(ValueError, match="file_path is required"):
            pipeline.run_stage0("")

    def test_run_stage1(self, pipeline, mock_db):
        """Test running stage 1."""
        pipeline.parse_and_store_chunks = Mock()
        pipeline.run_stage1()
        pipeline.parse_and_store_chunks.assert_called_once()

    def test_run_stage2(self, pipeline):
        """Test running stage 2."""
        pipeline.generate_embeddings = Mock()
        pipeline.run_stage2(model="test-model", batch_size=64)
        pipeline.generate_embeddings.assert_called_once_with(model="test-model", batch_size=64)

    def test_run_stage3(self, pipeline, mock_db, mock_vector_store):
        """Test running stage 3."""
        from src.storage.db import ChunkModel
        
        chunk1 = Mock()
        chunk1.id = "chunk1"
        chunk1.text = "text1"
        chunk1.embedding_json = json.dumps([0.1] * 384)
        chunk1.metadata_json = json.dumps({"key": "value"})
        
        session = Mock()
        session.query.return_value.filter.return_value.all.return_value = [chunk1]
        mock_db.get_session.return_value = session
        
        pipeline.run_stage3()
        
        mock_vector_store.add_documents_with_embeddings.assert_called_once()

    def test_run_stage3_no_chunks(self, pipeline, mock_db, mock_vector_store):
        """Test running stage 3 with no chunks."""
        session = Mock()
        session.query.return_value.filter.return_value.all.return_value = []
        mock_db.get_session.return_value = session
        
        pipeline.run_stage3()
        
        mock_vector_store.add_documents_with_embeddings.assert_not_called()

    def test_run_all(self, pipeline):
        """Test running all stages."""
        pipeline.run_stage0 = Mock()
        pipeline.run_stage1 = Mock()
        pipeline.run_stage2 = Mock()
        pipeline.run_stage3 = Mock()
        
        test_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        test_file.write("test")
        test_file.close()
        
        try:
            pipeline.run_all(test_file.name, model="test-model", batch_size=64)
            
            pipeline.run_stage0.assert_called_once_with(test_file.name)
            pipeline.run_stage1.assert_called_once()
            pipeline.run_stage2.assert_called_once_with(model="test-model", batch_size=64)
            pipeline.run_stage3.assert_called_once()
        finally:
            os.unlink(test_file.name)

    def test_parse_and_store_messages(self, pipeline, mock_parser, mock_db):
        """Test parsing and storing messages."""
        from src.ingestion.parser import ChatMessage
        
        msg1 = ChatMessage(id="1", timestamp="2023-01-01", sender="user1", content="msg1")
        msg2 = ChatMessage(id="2", timestamp="2023-01-02", sender="user2", content="msg2")
        mock_parser.parse_file.return_value = [msg1, msg2]
        mock_db.add_messages_batch.return_value = 2
        
        test_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_telegram_dump_123.txt')
        test_file.write("test")
        test_file.close()
        
        try:
            pipeline.parse_and_store_messages(test_file.name)
            
            mock_parser.parse_file.assert_called_once_with(test_file.name)
            mock_db.add_messages_batch.assert_called_once()
        finally:
            os.unlink(test_file.name)

    def test_parse_and_store_messages_no_file(self, pipeline):
        """Test parsing without file."""
        with pytest.raises(ValueError, match="file_path must be provided"):
            pipeline.parse_and_store_messages("")

    def test_parse_and_store_chunks(self, pipeline, mock_db, mock_chunker):
        """Test parsing and storing chunks."""
        from src.storage.db import MessageModel
        from src.ingestion.parser import ChatMessage
        
        msg1 = MessageModel(msg_id="chat1_1", chat_id="chat1", ts="2023-01-01", from_id="user1", text="msg1")
        msg2 = MessageModel(msg_id="chat1_2", chat_id="chat1", ts="2023-01-02", from_id="user2", text="msg2")
        
        chunk1 = Mock()
        chunk1.text = "chunk1"
        chunk1.metadata = Mock()
        chunk1.metadata.message_count = 1
        chunk1.metadata.ts_from = "2023-01-01"
        chunk1.metadata.ts_to = "2023-01-01"
        chunk1.metadata.msg_id_start = "1"
        chunk1.metadata.msg_id_end = "1"
        
        session = Mock()
        session.query.return_value.order_by.return_value.all.return_value = [msg1, msg2]
        mock_db.get_session.return_value = session
        mock_chunker.chunk_messages.return_value = [chunk1]
        
        pipeline.parse_and_store_chunks()
        
        mock_chunker.chunk_messages.assert_called_once()
        session.add_all.assert_called_once()
        session.commit.assert_called_once()

    def test_parse_and_store_chunks_no_messages(self, pipeline, mock_db):
        """Test parsing chunks with no messages."""
        session = Mock()
        session.query.return_value.order_by.return_value.all.return_value = []
        mock_db.get_session.return_value = session
        
        with pytest.raises(IngestionPipelineError, match="no messages found"):
            pipeline.parse_and_store_chunks()

    def test_generate_embeddings(self, pipeline, mock_db, mock_embedding_client):
        """Test generating embeddings."""
        from src.storage.db import ChunkModel
        
        chunk1 = Mock()
        chunk1.id = "chunk1"
        chunk1.text = "text1"
        chunk1.embedding_json = None
        chunk1.embedding_dim = None
        
        session = Mock()
        query = Mock()
        query.filter.return_value.count.return_value = 1
        query.filter.return_value.yield_per.return_value = [chunk1]
        session.query.return_value = query
        mock_db.get_session.return_value = session
        
        mock_embedding_client.get_embeddings.return_value = [[0.1] * 384]
        mock_embedding_client.get_dimension.return_value = 384
        
        pipeline.generate_embeddings(batch_size=1)
        
        mock_embedding_client.get_embeddings.assert_called()
        session.commit.assert_called()

    def test_generate_embeddings_all_have_embeddings(self, pipeline, mock_db):
        """Test generating embeddings when all chunks already have embeddings."""
        session = Mock()
        query = Mock()
        query.filter.return_value.count.return_value = 0
        session.query.return_value = query
        mock_db.get_session.return_value = session
        
        pipeline.generate_embeddings()
        
        # Should return early without processing

    def test_parse_and_store(self, pipeline):
        """Test parse_and_store method."""
        pipeline._clear_data = Mock()
        pipeline.parse_and_store_messages = Mock()
        pipeline.parse_and_store_chunks = Mock()
        
        test_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        test_file.write("test")
        test_file.close()
        
        try:
            pipeline.parse_and_store(test_file.name, clear_existing=True)
            
            pipeline._clear_data.assert_called_once()
            pipeline.parse_and_store_messages.assert_called_once_with(test_file.name)
            pipeline.parse_and_store_chunks.assert_called_once()
        finally:
            os.unlink(test_file.name)

    def test_run(self, pipeline):
        """Test run method."""
        pipeline._clear_data = Mock()
        pipeline.parse_and_store = Mock()
        pipeline.generate_embeddings = Mock()
        
        test_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        test_file.write("test")
        test_file.close()
        
        try:
            pipeline.run(test_file.name, clear_existing=True)
            
            pipeline._clear_data.assert_called_once()
            pipeline.parse_and_store.assert_called_once_with(test_file.name, clear_existing=False)
            pipeline.generate_embeddings.assert_called_once_with(batch_size=128)
        finally:
            os.unlink(test_file.name)

    def test_run_no_file(self, pipeline):
        """Test run method without file."""
        with pytest.raises(ValueError, match="file_path must be provided"):
            pipeline.run(None, clear_existing=False)

    def test_list_topics(self, pipeline, mock_db, capsys):
        """Test listing topics."""
        mock_db.get_all_topics_l1.return_value = []
        mock_db.get_all_topics_l2.return_value = []
        
        pipeline.list_topics()
        
        captured = capsys.readouterr()
        assert "No topics found" in captured.out

    def test_show_topic(self, pipeline, mock_db, capsys):
        """Test showing topic."""
        mock_db.get_all_topics_l1.return_value = []
        
        pipeline.show_topic(1)
        
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_get_ingest_info(self, pipeline, mock_db, mock_vector_store):
        """Test getting ingest info."""
        from src.storage.db import MessageModel, ChunkModel
        
        msg1 = MessageModel(msg_id="chat1_1", chat_id="chat1", ts="2023-01-01", from_id="user1", text="msg1")
        
        chunk1 = ChunkModel(id="chunk1", text="text1", chat_id="chat1")
        chunk1.embedding_json = json.dumps([0.1] * 384)
        
        session = Mock()
        session.query.return_value.order_by.return_value.all.return_value = [msg1]
        mock_db.get_session.return_value = session
        
        session2 = Mock()
        session2.query.return_value.count.return_value = 1
        session2.query.return_value.filter.return_value.count.return_value = 1
        mock_db.get_session.side_effect = [session, session2]
        
        mock_vector_store.get_all_embeddings.return_value = {
            "ids": ["chunk1"],
            "embeddings": [[0.1] * 384]
        }
        
        info = pipeline.get_ingest_info()
        
        assert "ingest info:" in info
        assert "stage0 messages:" in info
        assert "stage1 chunks:" in info
        assert "stage2 embeddings" in info
        assert "stage3 vector_db" in info
