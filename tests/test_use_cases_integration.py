"""
Integration tests for core use cases using in-memory implementations.

These tests verify that core use cases work correctly with in-memory
implementations of interfaces, without dependencies on SQLite, ChromaDB,
Telethon, FastAPI, or any external services.
"""

import pytest
from pathlib import Path
from datetime import datetime
from typing import List

from src.core.ingest_use_cases.ingest_messages import IngestMessages
from src.core.ingest_use_cases.process_chunks import ProcessChunks
from src.core.ingest_use_cases.generate_embeddings import GenerateEmbeddings
from src.core.ingest_use_cases.sync_to_vector_store import SyncToVectorStore
from src.core.ingest_use_cases.pipeline_orchestrator import PipelineOrchestrator
from src.core.domain import Message, Chunk
from src.ingestion.parser import ChatParser, ChatMessage
from src.ingestion.chunker import MessageChunker

from tests.fixtures.in_memory_stores import (
    InMemoryMessageStore,
    InMemoryChunkStore,
    InMemoryVectorIndex,
    InMemoryEmbedder,
    InMemoryFTSIndex,
    InMemoryLLM,
    InMemoryConfigProvider,
)


class TestIngestMessagesIntegration:
    """Integration tests for IngestMessages use case."""
    
    @pytest.fixture
    def message_store(self):
        """Create in-memory message store."""
        return InMemoryMessageStore()
    
    @pytest.fixture
    def use_case(self, message_store):
        """Create use case instance."""
        return IngestMessages(message_store)
    
    def test_execute_parses_and_saves_messages(self, use_case, tmp_path):
        """Test that execute parses file and saves messages."""
        # Create a test file with chat messages
        test_file = tmp_path / "telegram_dump_12345.json"
        test_file.write_text("""[
            {"id": "1", "timestamp": "2024-01-01T00:00:00", "sender": "user1", "content": "Hello"},
            {"id": "2", "timestamp": "2024-01-01T00:01:00", "sender": "user2", "content": "Hi there"}
        ]""")
        
        # Execute use case
        saved_count = use_case.execute(str(test_file))
        
        # Verify messages were saved
        assert saved_count == 2
        assert use_case.message_store.count() == 2
        
        # Verify message content
        messages = use_case.message_store.get_by_chat("12345", limit=10, offset=0)
        assert len(messages) == 2
        assert messages[0].text == "Hello"
        assert messages[1].text == "Hi there"
    
    def test_execute_handles_duplicate_messages(self, use_case, tmp_path):
        """Test that execute skips duplicate messages."""
        test_file = tmp_path / "telegram_dump_12345.json"
        test_file.write_text("""[
            {"id": "1", "timestamp": "2024-01-01T00:00:00", "sender": "user1", "content": "Hello"}
        ]""")
        
        # Execute twice
        saved1 = use_case.execute(str(test_file))
        saved2 = use_case.execute(str(test_file))
        
        # First execution saves, second skips
        assert saved1 == 1
        assert saved2 == 0
        assert use_case.message_store.count() == 1


class TestProcessChunksIntegration:
    """Integration tests for ProcessChunks use case."""
    
    @pytest.fixture
    def message_store(self):
        """Create in-memory message store with test messages."""
        store = InMemoryMessageStore()
        # Add some test messages
        messages = [
            Message(
                id="chat1_1",
                chat_id="chat1",
                from_id="user1",
                text="First message",
                timestamp=datetime(2024, 1, 1, 0, 0, 0),
                meta={}
            ),
            Message(
                id="chat1_2",
                chat_id="chat1",
                from_id="user2",
                text="Second message",
                timestamp=datetime(2024, 1, 1, 0, 1, 0),
                meta={}
            ),
        ]
        store.save_batch(messages)
        return store
    
    @pytest.fixture
    def chunk_store(self):
        """Create in-memory chunk store."""
        return InMemoryChunkStore()
    
    @pytest.fixture
    def chunker(self):
        """Create chunker with test config."""
        return MessageChunker(
            chunk_token_min=10,
            chunk_token_max=100,
            chunk_overlap_ratio=0.1
        )
    
    @pytest.fixture
    def use_case(self, message_store, chunk_store, chunker):
        """Create use case instance."""
        return ProcessChunks(message_store, chunk_store, chunker)
    
    def test_execute_creates_and_saves_chunks(self, use_case):
        """Test that execute creates chunks from messages and saves them."""
        # Execute use case
        saved_count = use_case.execute(chat_id="chat1")
        
        # Verify chunks were created and saved
        assert saved_count > 0
        
        # Verify chunks can be retrieved
        chunks = use_case.chunk_store.get_by_ids([chunk.id for chunk in use_case.chunk_store._chunks.values()])
        assert len(chunks) == saved_count
        
        # Verify chunk content
        for chunk in chunks:
            assert chunk.text is not None
            assert len(chunk.text) > 0
            assert chunk.msg_ids is not None


class TestGenerateEmbeddingsIntegration:
    """Integration tests for GenerateEmbeddings use case."""
    
    @pytest.fixture
    def chunk_store(self):
        """Create in-memory chunk store with test chunks."""
        store = InMemoryChunkStore()
        chunks = [
            Chunk(
                id="chunk1",
                text="Test chunk 1",
                msg_ids=("chat1_1", "chat1_2"),
                valid_period=(datetime(2024, 1, 1), datetime(2024, 1, 2)),
                embedding=None,
                metadata={}
            ),
            Chunk(
                id="chunk2",
                text="Test chunk 2",
                msg_ids=("chat1_3", "chat1_4"),
                valid_period=(datetime(2024, 1, 2), datetime(2024, 1, 3)),
                embedding=None,
                metadata={}
            ),
        ]
        store.save_batch(chunks)
        return store
    
    @pytest.fixture
    def embedder(self):
        """Create in-memory embedder."""
        return InMemoryEmbedder(dimension=384)
    
    @pytest.fixture
    def use_case(self, chunk_store, embedder):
        """Create use case instance."""
        return GenerateEmbeddings(chunk_store, embedder)
    
    def test_execute_generates_and_saves_embeddings(self, use_case):
        """Test that execute generates embeddings and saves them."""
        # Execute use case
        saved_count = use_case.execute(chunk_ids=["chunk1", "chunk2"])
        
        # Verify embeddings were generated
        assert saved_count == 2
        
        # Verify chunks have embeddings
        chunks = use_case.chunk_store.get_by_ids(["chunk1", "chunk2"])
        for chunk in chunks:
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 384
    
    def test_execute_skips_chunks_with_embeddings(self, use_case):
        """Test that execute skips chunks that already have embeddings."""
        # Set embedding for one chunk
        chunks = use_case.chunk_store.get_by_ids(["chunk1"])
        chunks[0].embedding = [0.1] * 384
        use_case.chunk_store.save_batch(chunks)
        
        # Execute use case
        saved_count = use_case.execute(chunk_ids=["chunk1", "chunk2"])
        
        # Only chunk2 should be processed
        assert saved_count == 1


class TestSyncToVectorStoreIntegration:
    """Integration tests for SyncToVectorStore use case."""
    
    @pytest.fixture
    def chunk_store(self):
        """Create in-memory chunk store with chunks that have embeddings."""
        store = InMemoryChunkStore()
        chunks = [
            Chunk(
                id="chunk1",
                text="Test chunk 1",
                msg_ids=("chat1_1", "chat1_2"),
                valid_period=(datetime(2024, 1, 1), datetime(2024, 1, 2)),
                embedding=[0.1] * 384,
                metadata={"chat_id": "chat1"}
            ),
            Chunk(
                id="chunk2",
                text="Test chunk 2",
                msg_ids=("chat1_3", "chat1_4"),
                valid_period=(datetime(2024, 1, 2), datetime(2024, 1, 3)),
                embedding=[0.2] * 384,
                metadata={"chat_id": "chat1"}
            ),
        ]
        store.save_batch(chunks)
        return store
    
    @pytest.fixture
    def vector_index(self):
        """Create in-memory vector index."""
        return InMemoryVectorIndex(dimension=384)
    
    @pytest.fixture
    def use_case(self, chunk_store, vector_index):
        """Create use case instance."""
        return SyncToVectorStore(chunk_store, vector_index)
    
    def test_execute_syncs_chunks_to_vector_store(self, use_case):
        """Test that execute syncs chunks to vector store."""
        # Execute use case
        synced_count = use_case.execute(chunk_ids=["chunk1", "chunk2"])
        
        # Verify chunks were synced
        assert synced_count == 2
        
        # Verify vector index has documents
        assert use_case.vector_index.count() == 2
        
        # Verify documents can be queried
        query_vector = [0.1] * 384
        results = use_case.vector_index.query(query_vector, top_k=2)
        assert len(results) == 2
    
    def test_execute_skips_chunks_without_embeddings(self, use_case):
        """Test that execute skips chunks without embeddings."""
        # Add chunk without embedding
        chunk_no_embedding = Chunk(
            id="chunk3",
            text="Test chunk 3",
            msg_ids=("chat1_5", "chat1_6"),
            valid_period=(datetime(2024, 1, 3), datetime(2024, 1, 4)),
            embedding=None,
            metadata={}
        )
        use_case.chunk_store.save_batch([chunk_no_embedding])
        
        # Execute use case
        synced_count = use_case.execute(chunk_ids=["chunk1", "chunk2", "chunk3"])
        
        # Only chunks with embeddings should be synced
        assert synced_count == 2
        assert use_case.vector_index.count() == 2


class TestPipelineOrchestratorIntegration:
    """Integration tests for PipelineOrchestrator use case."""
    
    @pytest.fixture
    def message_store(self):
        """Create in-memory message store."""
        return InMemoryMessageStore()
    
    @pytest.fixture
    def chunk_store(self):
        """Create in-memory chunk store."""
        return InMemoryChunkStore()
    
    @pytest.fixture
    def vector_index(self):
        """Create in-memory vector index."""
        return InMemoryVectorIndex(dimension=384)
    
    @pytest.fixture
    def embedder(self):
        """Create in-memory embedder."""
        return InMemoryEmbedder(dimension=384)
    
    @pytest.fixture
    def chunker(self):
        """Create chunker."""
        return MessageChunker(
            chunk_token_min=10,
            chunk_token_max=100,
            chunk_overlap_ratio=0.1
        )
    
    @pytest.fixture
    def ingest_messages(self, message_store):
        """Create IngestMessages use case."""
        return IngestMessages(message_store)
    
    @pytest.fixture
    def process_chunks(self, message_store, chunk_store, chunker):
        """Create ProcessChunks use case."""
        return ProcessChunks(message_store, chunk_store, chunker)
    
    @pytest.fixture
    def generate_embeddings(self, chunk_store, embedder):
        """Create GenerateEmbeddings use case."""
        return GenerateEmbeddings(chunk_store, embedder)
    
    @pytest.fixture
    def sync_to_vector_store(self, chunk_store, vector_index):
        """Create SyncToVectorStore use case."""
        return SyncToVectorStore(chunk_store, vector_index)
    
    @pytest.fixture
    def orchestrator(
        self,
        ingest_messages,
        process_chunks,
        generate_embeddings,
        sync_to_vector_store
    ):
        """Create PipelineOrchestrator."""
        return PipelineOrchestrator(
            ingest_messages=ingest_messages,
            process_chunks=process_chunks,
            generate_embeddings=generate_embeddings,
            sync_to_vector_store=sync_to_vector_store
        )
    
    def test_run_all_executes_all_stages(self, orchestrator, tmp_path):
        """Test that run_all executes all pipeline stages."""
        # Create test file
        test_file = tmp_path / "telegram_dump_12345.json"
        test_file.write_text("""[
            {"id": "1", "timestamp": "2024-01-01T00:00:00", "sender": "user1", "content": "Hello world"},
            {"id": "2", "timestamp": "2024-01-01T00:01:00", "sender": "user2", "content": "Hi there"}
        ]""")
        
        # Execute orchestrator
        stats = orchestrator.run_all(str(test_file))
        
        # Verify stage 0 completed
        assert "stage0_messages_saved" in stats
        assert stats["stage0_messages_saved"] == 2
        
        # Verify stage 1 completed
        assert "stage1_chunks_saved" in stats
        assert stats["stage1_chunks_saved"] > 0

