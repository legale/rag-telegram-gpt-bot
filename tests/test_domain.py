"""Tests for src/core/domain.py"""

import pytest
from datetime import datetime
from src.core.domain import (
    Message,
    Chunk,
    SearchResult,
    IngestionJob,
    ProfileConfig,
    TopicUpdate
)


class TestMessage:
    """Tests for Message dataclass"""
    
    def test_message_creation(self):
        """Test creating a Message"""
        msg = Message(
            id="msg1",
            chat_id="chat1",
            from_id="user1",
            text="Hello",
            timestamp=datetime.now()
        )
        
        assert msg.id == "msg1"
        assert msg.chat_id == "chat1"
        assert msg.from_id == "user1"
        assert msg.text == "Hello"
        assert isinstance(msg.timestamp, datetime)
        assert msg.meta == {}
    
    def test_message_with_meta(self):
        """Test creating a Message with metadata"""
        msg = Message(
            id="msg1",
            chat_id="chat1",
            from_id="user1",
            text="Hello",
            timestamp=datetime.now(),
            meta={"key": "value"}
        )
        
        assert msg.meta["key"] == "value"


class TestChunk:
    """Tests for Chunk dataclass"""
    
    def test_chunk_creation(self):
        """Test creating a Chunk"""
        chunk = Chunk(
            id="chunk1",
            text="Test text"
        )
        
        assert chunk.id == "chunk1"
        assert chunk.text == "Test text"
        assert chunk.msg_ids is None
        assert chunk.valid_period is None
        assert chunk.embedding is None
        assert chunk.metadata == {}
    
    def test_chunk_with_all_fields(self):
        """Test creating a Chunk with all fields"""
        msg_ids = ("msg1", "msg2")
        valid_period = (datetime.now(), datetime.now())
        embedding = [0.1, 0.2, 0.3]
        metadata = {"key": "value"}
        
        chunk = Chunk(
            id="chunk1",
            text="Test",
            msg_ids=msg_ids,
            valid_period=valid_period,
            embedding=embedding,
            metadata=metadata
        )
        
        assert chunk.msg_ids == msg_ids
        assert chunk.valid_period == valid_period
        assert chunk.embedding == embedding
        assert chunk.metadata == metadata


class TestSearchResult:
    """Tests for SearchResult dataclass"""
    
    def test_search_result_creation(self):
        """Test creating a SearchResult"""
        chunk = Chunk(id="chunk1", text="Test")
        result = SearchResult(chunk=chunk, score=0.8)
        
        assert result.chunk == chunk
        assert result.score == 0.8
        assert result.original_messages == []
        assert result.topics == []
    
    def test_search_result_with_messages(self):
        """Test creating a SearchResult with messages"""
        chunk = Chunk(id="chunk1", text="Test")
        messages = [
            Message(id="msg1", chat_id="chat1", from_id="user1", text="Hello", timestamp=datetime.now())
        ]
        result = SearchResult(chunk=chunk, score=0.8, original_messages=messages)
        
        assert len(result.original_messages) == 1
    
    def test_search_result_with_topics(self):
        """Test creating a SearchResult with topics"""
        chunk = Chunk(id="chunk1", text="Test")
        result = SearchResult(chunk=chunk, score=0.8, topics=["topic1", "topic2"])
        
        assert result.topics == ["topic1", "topic2"]


class TestIngestionJob:
    """Tests for IngestionJob dataclass"""
    
    def test_ingestion_job_creation(self):
        """Test creating an IngestionJob"""
        job = IngestionJob(status="running")
        
        assert job.status == "running"
        assert job.stage is None
        assert job.stats == {}
    
    def test_ingestion_job_with_stage(self):
        """Test creating an IngestionJob with stage"""
        job = IngestionJob(status="running", stage="processing")
        
        assert job.stage == "processing"
    
    def test_ingestion_job_with_stats(self):
        """Test creating an IngestionJob with stats"""
        stats = {"processed": 100, "errors": 0}
        job = IngestionJob(status="completed", stats=stats)
        
        assert job.stats == stats


class TestProfileConfig:
    """Tests for ProfileConfig dataclass"""
    
    def test_profile_config_creation(self):
        """Test creating a ProfileConfig"""
        config = ProfileConfig(
            model_name="gpt-4",
            embedding_provider="local"
        )
        
        assert config.model_name == "gpt-4"
        assert config.embedding_provider == "local"
        assert config.paths == {}
    
    def test_profile_config_with_paths(self):
        """Test creating a ProfileConfig with paths"""
        paths = {"db": "/path/to/db", "vector": "/path/to/vector"}
        config = ProfileConfig(
            model_name="gpt-4",
            embedding_provider="local",
            paths=paths
        )
        
        assert config.paths == paths


class TestTopicUpdate:
    """Tests for TopicUpdate dataclass"""
    
    def test_topic_update_creation(self):
        """Test creating a TopicUpdate"""
        update = TopicUpdate()
        
        assert update.topic_ids == []
        assert update.metadata == {}
    
    def test_topic_update_with_ids(self):
        """Test creating a TopicUpdate with topic IDs"""
        update = TopicUpdate(topic_ids=["topic1", "topic2"])
        
        assert update.topic_ids == ["topic1", "topic2"]
    
    def test_topic_update_with_metadata(self):
        """Test creating a TopicUpdate with metadata"""
        metadata = {"key": "value"}
        update = TopicUpdate(metadata=metadata)
        
        assert update.metadata == metadata

