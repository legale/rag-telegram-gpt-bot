"""
Contract tests for adapters.

These tests verify that adapters correctly implement their interfaces.
Contract tests ensure that any implementation of an interface will work
correctly with use cases that depend on that interface.
"""

import pytest
from datetime import datetime, timedelta
from typing import List
from pathlib import Path
import tempfile
import shutil

from src.core.domain import Message, Chunk, TopicUpdate
from src.core.interfaces import MessageStore, ChunkStore, VectorIndex, Embedder, VectorDoc, ScoredDoc
from src.adapters.persistence import SqliteMessageStore, SqliteChunkStore
from src.adapters.vector import ChromaVectorIndex
from src.adapters.embedding import EmbedderAdapter
from src.storage.db import Database
from src.storage.vector_store import VectorStore
from src.core.embedding import LocalEmbeddingClient


class TestMessageStoreContract:
    """Contract tests for MessageStore interface implementations."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary database for testing."""
        db_path = tmp_path / "test.db"
        db_url = f"sqlite:///{db_path}"
        db = Database(db_url)
        yield db
        # Cleanup handled by tmp_path fixture

    @pytest.fixture
    def message_store(self, temp_db):
        """Create a MessageStore implementation for testing."""
        return SqliteMessageStore(temp_db)

    @pytest.fixture
    def sample_messages(self):
        """Create sample messages for testing."""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        return [
            Message(
                id=f"msg_{i}",
                chat_id="chat_1",
                from_id="user_1",
                text=f"Message {i}",
                timestamp=base_time + timedelta(minutes=i),
                meta={"test": True}
            )
            for i in range(5)
        ]

    def test_save_batch(self, message_store: MessageStore, sample_messages: List[Message]):
        """Test that save_batch saves messages and returns count."""
        count = message_store.save_batch(sample_messages)
        assert count == len(sample_messages)
        
        # Verify messages were saved
        total_count = message_store.count()
        assert total_count == len(sample_messages)

    def test_save_batch_duplicates(self, message_store: MessageStore, sample_messages: List[Message]):
        """Test that save_batch handles duplicates correctly."""
        # Save first time
        count1 = message_store.save_batch(sample_messages)
        assert count1 == len(sample_messages)
        
        # Save again (duplicates)
        count2 = message_store.save_batch(sample_messages)
        assert count2 == 0  # No new messages saved
        
        # Total count should remain the same
        assert message_store.count() == len(sample_messages)

    def test_get_by_chat(self, message_store: MessageStore, sample_messages: List[Message]):
        """Test that get_by_chat retrieves messages for a specific chat."""
        message_store.save_batch(sample_messages)
        
        # Get all messages
        messages = message_store.get_by_chat("chat_1", limit=10, offset=0)
        assert len(messages) == len(sample_messages)
        
        # Check ordering (should be by timestamp)
        timestamps = [msg.timestamp for msg in messages]
        assert timestamps == sorted(timestamps)

    def test_get_by_chat_pagination(self, message_store: MessageStore, sample_messages: List[Message]):
        """Test that get_by_chat supports pagination."""
        message_store.save_batch(sample_messages)
        
        # Get first 2 messages
        messages1 = message_store.get_by_chat("chat_1", limit=2, offset=0)
        assert len(messages1) == 2
        
        # Get next 2 messages
        messages2 = message_store.get_by_chat("chat_1", limit=2, offset=2)
        assert len(messages2) == 2
        
        # Messages should be different
        assert messages1[0].id != messages2[0].id

    def test_get_context(self, message_store: MessageStore, sample_messages: List[Message]):
        """Test that get_context retrieves messages within time window."""
        message_store.save_batch(sample_messages)
        
        # Get context around middle message (index 2)
        time_point = sample_messages[2].timestamp
        window_sec = 120  # 2 minutes
        
        context = message_store.get_context("chat_1", time_point, window_sec)
        
        # Should include messages within window
        assert len(context) > 0
        for msg in context:
            time_diff = abs((msg.timestamp - time_point).total_seconds())
            assert time_diff <= window_sec

    def test_count(self, message_store: MessageStore, sample_messages: List[Message]):
        """Test that count returns total number of messages."""
        assert message_store.count() == 0
        
        message_store.save_batch(sample_messages)
        assert message_store.count() == len(sample_messages)


class TestChunkStoreContract:
    """Contract tests for ChunkStore interface implementations."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary database for testing."""
        db_path = tmp_path / "test.db"
        db_url = f"sqlite:///{db_path}"
        db = Database(db_url)
        yield db

    @pytest.fixture
    def chunk_store(self, temp_db):
        """Create a ChunkStore implementation for testing."""
        return SqliteChunkStore(temp_db)

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        return [
            Chunk(
                id=f"chunk_{i}",
                text=f"Chunk text {i}",
                msg_ids=(f"msg_{i}_start", f"msg_{i}_end"),
                valid_period=(base_time + timedelta(minutes=i), base_time + timedelta(minutes=i+1)),
                embedding=[0.1 * i] * 10,  # Simple embedding
                metadata={"chat_id": "chat_1", "test": True}
            )
            for i in range(5)
        ]

    def test_save_batch(self, chunk_store: ChunkStore, sample_chunks: List[Chunk]):
        """Test that save_batch saves chunks and returns count."""
        count = chunk_store.save_batch(sample_chunks)
        assert count == len(sample_chunks)

    def test_save_batch_duplicates(self, chunk_store: ChunkStore, sample_chunks: List[Chunk]):
        """Test that save_batch handles duplicates correctly."""
        # Save first time
        count1 = chunk_store.save_batch(sample_chunks)
        assert count1 == len(sample_chunks)
        
        # Save again (duplicates)
        count2 = chunk_store.save_batch(sample_chunks)
        assert count2 == 0  # No new chunks saved

    def test_get_by_ids(self, chunk_store: ChunkStore, sample_chunks: List[Chunk]):
        """Test that get_by_ids retrieves chunks by their IDs."""
        chunk_store.save_batch(sample_chunks)
        
        # Get some chunks by ID
        ids = [sample_chunks[0].id, sample_chunks[2].id]
        retrieved = chunk_store.get_by_ids(ids)
        
        assert len(retrieved) == 2
        assert {chunk.id for chunk in retrieved} == set(ids)

    def test_get_by_ids_nonexistent(self, chunk_store: ChunkStore, sample_chunks: List[Chunk]):
        """Test that get_by_ids handles nonexistent IDs gracefully."""
        chunk_store.save_batch(sample_chunks)
        
        # Get nonexistent ID
        retrieved = chunk_store.get_by_ids(["nonexistent_id"])
        assert len(retrieved) == 0

    def test_update_topics(self, chunk_store: ChunkStore, sample_chunks: List[Chunk]):
        """Test that update_topics updates chunk topic information."""
        chunk_store.save_batch(sample_chunks)
        
        # Update topics for some chunks
        updates = {
            sample_chunks[0].id: TopicUpdate(topic_ids=["topic_1", "topic_2"]),
            sample_chunks[1].id: TopicUpdate(topic_ids=["topic_3"]),
        }
        
        chunk_store.update_topics(updates)
        
        # Verify updates (by retrieving chunks)
        retrieved = chunk_store.get_by_ids([sample_chunks[0].id, sample_chunks[1].id])
        assert len(retrieved) == 2
        # Note: Topic updates are stored in metadata, so we check metadata
        assert "topic_ids" in retrieved[0].metadata or "topic_ids" in retrieved[1].metadata

    def test_clear(self, chunk_store: ChunkStore, sample_chunks: List[Chunk]):
        """Test that clear removes all chunks."""
        chunk_store.save_batch(sample_chunks)
        
        chunk_store.clear()
        
        # Verify all chunks are gone
        retrieved = chunk_store.get_by_ids([chunk.id for chunk in sample_chunks])
        assert len(retrieved) == 0


class TestVectorIndexContract:
    """Contract tests for VectorIndex interface implementations."""

    @pytest.fixture
    def temp_vector_db(self, tmp_path):
        """Create a temporary vector database for testing."""
        vector_db_path = tmp_path / "vector_db"
        vector_db_path.mkdir()
        
        # Create embedding client
        embedding_client = LocalEmbeddingClient(model="paraphrase-multilingual-mpnet-base-v2")
        
        vector_store = VectorStore(
            persist_directory=str(vector_db_path),
            collection_name="test_collection",
            embedding_client=embedding_client
        )
        
        yield vector_store
        
        # Cleanup
        shutil.rmtree(vector_db_path, ignore_errors=True)

    @pytest.fixture
    def vector_index(self, temp_vector_db):
        """Create a VectorIndex implementation for testing."""
        return ChromaVectorIndex(temp_vector_db)

    @pytest.fixture
    def sample_vector_docs(self, temp_vector_db):
        """Create sample vector documents for testing."""
        # Get dimension from embedding client
        dimension = temp_vector_db.embedding_client.get_dimension()
        return [
            VectorDoc(
                id=f"doc_{i}",
                vector=[0.1 * (i + 1)] * dimension,  # Use actual dimension
                meta={"text": f"Document {i}", "chat_id": "chat_1"}
            )
            for i in range(5)
        ]

    def test_upsert(self, vector_index: VectorIndex, sample_vector_docs: List[VectorDoc], temp_vector_db):
        """Test that upsert stores vector documents."""
        vector_index.upsert(sample_vector_docs)
        
        # Verify by querying (use dimension from embedding client)
        dimension = temp_vector_db.embedding_client.get_dimension()
        query_vector = [0.1] * dimension
        results = vector_index.query(query_vector, top_k=10)
        assert len(results) > 0

    def test_query(self, vector_index: VectorIndex, sample_vector_docs: List[VectorDoc], temp_vector_db):
        """Test that query retrieves similar documents."""
        vector_index.upsert(sample_vector_docs)
        
        # Query with similar vector (use dimension from embedding client)
        dimension = temp_vector_db.embedding_client.get_dimension()
        query_vector = [0.1] * dimension
        results = vector_index.query(query_vector, top_k=3)
        
        assert len(results) > 0
        assert all(isinstance(result, ScoredDoc) for result in results)
        assert all(hasattr(result, 'id') and hasattr(result, 'score') for result in results)

    def test_query_with_filter(self, vector_index: VectorIndex, sample_vector_docs: List[VectorDoc], temp_vector_db):
        """Test that query supports filtering by metadata."""
        vector_index.upsert(sample_vector_docs)
        
        # Query with filter (use dimension from embedding client)
        dimension = temp_vector_db.embedding_client.get_dimension()
        query_vector = [0.1] * dimension
        results = vector_index.query(
            query_vector,
            top_k=10,
            filter={"chat_id": "chat_1"}
        )
        
        # All results should match filter
        assert len(results) > 0
        assert all(result.meta.get("chat_id") == "chat_1" for result in results)

    def test_delete(self, vector_index: VectorIndex, sample_vector_docs: List[VectorDoc], temp_vector_db):
        """Test that delete removes documents by ID."""
        vector_index.upsert(sample_vector_docs)
        
        # Delete some documents
        ids_to_delete = [sample_vector_docs[0].id, sample_vector_docs[1].id]
        vector_index.delete(ids_to_delete)
        
        # Query should not return deleted documents (use dimension from embedding client)
        dimension = temp_vector_db.embedding_client.get_dimension()
        query_vector = [0.1] * dimension
        results = vector_index.query(query_vector, top_k=10)
        result_ids = {result.id for result in results}
        assert not any(id in result_ids for id in ids_to_delete)


class TestEmbedderContract:
    """Contract tests for Embedder interface implementations."""

    @pytest.fixture
    def embedder(self):
        """Create an Embedder implementation for testing."""
        embedding_client = LocalEmbeddingClient(model="paraphrase-multilingual-mpnet-base-v2")
        return EmbedderAdapter(embedding_client)

    def test_embed_documents(self, embedder: Embedder):
        """Test that embed_documents generates embeddings for multiple texts."""
        texts = ["First document", "Second document", "Third document"]
        embeddings = embedder.embed_documents(texts)
        
        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
        # All embeddings should have same dimension
        assert len(set(len(emb) for emb in embeddings)) == 1

    def test_embed_query(self, embedder: Embedder):
        """Test that embed_query generates embedding for a single query text."""
        query = "What is the meaning of life?"
        embedding = embedder.embed_query(query)
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_query_and_documents_same_dimension(self, embedder: Embedder):
        """Test that query and document embeddings have same dimension."""
        query_emb = embedder.embed_query("test query")
        doc_embs = embedder.embed_documents(["test document"])
        
        assert len(query_emb) == len(doc_embs[0])

