
import pytest
from unittest.mock import MagicMock, patch, ANY
import os
import shutil
import uuid
from datetime import datetime
from src.ingestion.pipeline import IngestionPipeline
from src.storage.db import ChunkModel

@pytest.fixture
def mock_dependencies():
    with patch('src.ingestion.pipeline.Database') as MockDB, \
         patch('src.ingestion.pipeline.VectorStore') as MockVS, \
         patch('src.ingestion.pipeline.ChatParser') as MockParser, \
         patch('src.ingestion.pipeline.MessageChunker') as MockChunker, \
         patch('src.ingestion.pipeline.create_embedding_client') as MockEmbedder:
        yield {
            'db': MockDB,
            'vector_store': MockVS,
            'parser': MockParser,
            'chunker': MockChunker,
            'embedder': MockEmbedder
        }

@pytest.fixture
def pipeline(mock_dependencies):
    return IngestionPipeline("sqlite:///test.db", "test_vector_db")

def test_init(pipeline, mock_dependencies):
    mock_dependencies['db'].assert_called_with("sqlite:///test.db")
    # VectorStore is now called with embedding_client parameter
    assert mock_dependencies['vector_store'].called
    call_kwargs = mock_dependencies['vector_store'].call_args[1] if mock_dependencies['vector_store'].call_args else {}
    assert call_kwargs.get('persist_directory') == "test_vector_db"
    assert call_kwargs.get('collection_name') == "default"

def test_clear_data(pipeline):
    # Setup mocks
    pipeline.db.count_chunks.side_effect = [10, 0] # before, after
    pipeline.db.clear.return_value = 10
    
    pipeline.vector_store.count.side_effect = [5, 0] # before, after
    pipeline.vector_store.clear.return_value = 5
    
    pipeline._clear_data()
    
    # Verify calls
    pipeline.db.clear.assert_called_once()
    pipeline.vector_store.clear.assert_called_once()

def test_run_clear_only(pipeline):
    with patch.object(pipeline, '_clear_data') as mock_clear:
        pipeline.run(clear_existing=True)
        mock_clear.assert_called_once()

def test_run_missing_file_error(pipeline):
    with pytest.raises(ValueError, match="file_path must be provided"):
        pipeline.run(file_path=None, clear_existing=False)

def test_run_success(pipeline, mock_dependencies):
    # Setup mocks
    mock_parser = pipeline.parser
    # Messages should be objects with .id attribute
    mock_msg = MagicMock()
    mock_msg.id = "123"
    mock_parser.parse_file.return_value = [mock_msg]
    
    mock_chunker = pipeline.chunker
    mock_chunk = MagicMock()
    mock_chunk.text = "chunk text"
    # Metadata should be an object with attributes, not a dict
    mock_metadata = MagicMock()
    mock_metadata.message_count = 5
    mock_metadata.ts_from = datetime(2023, 1, 1, 10, 0)
    mock_metadata.ts_to = datetime(2023, 1, 1, 12, 0)
    mock_metadata.msg_id_start = "1"
    mock_metadata.msg_id_end = "2"
    mock_chunk.metadata = mock_metadata
    mock_chunker.chunk_messages.return_value = [mock_chunk]
    
    mock_session = MagicMock()
    pipeline.db.get_session.return_value = mock_session
    
    # Get the mocked embedding client (created in fixture)
    mock_embedder_instance = mock_dependencies['embedder'].return_value
    mock_embedder_instance.get_embeddings_batched.return_value = [[0.1, 0.2]]
    
    # Mock vector store get_all_embeddings to return empty (no existing embeddings)
    pipeline.vector_store.get_all_embeddings.return_value = {"ids": []}
    
    # Mock ChunkModel query
    mock_chunk_model = MagicMock()
    mock_chunk_model.id = str(uuid.uuid4())
    mock_chunk_model.text = "chunk text"
    mock_chunk_model.metadata_json = '{"message_count": 5, "start_date": "2023-01-01T10:00:00", "end_date": "2023-01-01T12:00:00"}'
    pipeline.db.get_session.return_value.query.return_value.all.return_value = [mock_chunk_model]
    
    # Run
    pipeline.run("test.json")
    
    # Verify
    mock_parser.parse_file.assert_called_with("test.json")
    mock_chunker.chunk_messages.assert_called()
    # Check that it was called with messages list
    call_args = mock_chunker.chunk_messages.call_args[0]
    assert len(call_args) > 0
    
    # Check DB save (from parse_and_store)
    mock_session.add_all.assert_called()
    mock_session.commit.assert_called()
    mock_session.close.assert_called()
    
    # Check Embeddings (from generate_embeddings)
    mock_embedder_instance.get_embeddings_batched.assert_called()
    pipeline.vector_store.add_documents_with_embeddings.assert_called_with(
        ids=ANY,
        documents=ANY,
        embeddings=[[0.1, 0.2]],
        metadatas=ANY,  # Metadata format may vary
        show_progress=True
    )

def test_run_db_error(pipeline, mock_dependencies):
    # Setup mocks
    mock_msg = MagicMock()
    mock_msg.id = "123"
    pipeline.parser.parse_file.return_value = [mock_msg]
    
    mock_chunk = MagicMock()
    mock_chunk.text = "chunk text"
    # Metadata should be an object with attributes
    mock_metadata = MagicMock()
    mock_metadata.message_count = 5
    mock_metadata.ts_from = datetime(2023, 1, 1, 10, 0)
    mock_metadata.ts_to = datetime(2023, 1, 1, 12, 0)
    mock_metadata.msg_id_start = "1"
    mock_metadata.msg_id_end = "2"
    mock_chunk.metadata = mock_metadata
    pipeline.chunker.chunk_messages.return_value = [mock_chunk]
    
    mock_session = MagicMock()
    pipeline.db.get_session.return_value = mock_session
    mock_session.commit.side_effect = Exception("DB Fail")
    
    with pytest.raises(Exception) as exc_info:
        pipeline.run("test.json")
    
    assert "DB Fail" in str(exc_info.value)
        
    mock_session.rollback.assert_called()
    mock_session.close.assert_called()

def test_run_no_chunks_generated(pipeline, mock_dependencies):
    mock_msg = MagicMock()
    mock_msg.id = "123"
    pipeline.parser.parse_file.return_value = [mock_msg]
    pipeline.chunker.chunk_messages.return_value = [] # No chunks
    
    pipeline.run("test.json")
    
    # Embedder is called during initialization, but embed_and_save_jsonl should NOT be called
    # when no chunks are generated
    if hasattr(pipeline.embedding_client, 'embed_and_save_jsonl'):
        pipeline.embedding_client.embed_and_save_jsonl.assert_not_called()
    pipeline.vector_store.add_documents_with_embeddings.assert_not_called()

# Integration test from original file (preserved but slightly modified)
def test_integration_ingestion_pipeline(tmp_path):
    # Setup
    db_path = tmp_path / "test.db"
    vector_db_path = tmp_path / "test_chroma_db"
    input_file = tmp_path / "chat.txt"
    input_file.write_text("User1: Hello\nUser2: Hi there\nUser1: How are you?")
    
    # Use real classes but maybe mock embedding client to avoid API calls?
    # For now let's rely on standard mocks for embedding if needed, or if the original test ran without auth, 
    # it implies existing code can run without API keys if not embedding?
    # Wait, the code calls EmbeddingClient which tries to connect. We should mock EmbeddingClient even for integration test 
    # unless we have a dummy local embedder.
    
    with patch('src.ingestion.pipeline.EmbeddingClient') as MockEmbedder, \
         patch('src.ingestion.pipeline.VectorStore') as MockVS:
            
        MockEmbedder.return_value.embed_and_save_jsonl.return_value = [[0.1, 0.2], [0.3, 0.4]] # Return check dummy embeddings

        pipeline = IngestionPipeline(
            db_url=f"sqlite:///{db_path}",
            vector_db_path=str(vector_db_path)
        )
        
        # Run pipeline
        pipeline.run(str(input_file))
        
        # Verify SQL DB
        session = pipeline.db.get_session()
        chunks = session.query(ChunkModel).all()
        assert len(chunks) > 0
        session.close()

