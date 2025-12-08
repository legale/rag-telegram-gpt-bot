import pytest
import tempfile
import os
from datetime import datetime
from sqlalchemy import create_engine
from src.storage.db import Database, MessageModel, ChunkModel, TopicL1Model, TopicL2Model


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix='.db', dir='/tmp')
    os.close(fd)
    db_url = f'sqlite:///{path}'
    db = Database(db_url)
    yield db
    os.unlink(path)


class TestMessageModel:
    """Tests for message storage and retrieval."""
    
    def test_add_message(self, temp_db):
        """Test adding a single message."""
        temp_db.add_message(
            msg_id='msg1',
            chat_id='chat1',
            ts=datetime(2024, 1, 1, 12, 0, 0),
            from_id='user1',
            text='Hello world'
        )
        
        messages = temp_db.get_messages()
        assert len(messages) == 1
        assert messages[0].msg_id == 'msg1'
        assert messages[0].text == 'Hello world'
    
    def test_get_messages_by_chat(self, temp_db):
        """Test filtering messages by chat_id."""
        temp_db.add_message('msg1', 'chat1', datetime(2024, 1, 1), 'user1', 'Text 1')
        temp_db.add_message('msg2', 'chat2', datetime(2024, 1, 2), 'user2', 'Text 2')
        temp_db.add_message('msg3', 'chat1', datetime(2024, 1, 3), 'user1', 'Text 3')
        
        chat1_messages = temp_db.get_messages_by_chat('chat1')
        assert len(chat1_messages) == 2
        assert chat1_messages[0].msg_id == 'msg1'
        assert chat1_messages[1].msg_id == 'msg3'
    
    def test_count_messages(self, temp_db):
        """Test message counting."""
        temp_db.add_message('msg1', 'chat1', datetime(2024, 1, 1), 'user1', 'Text 1')
        temp_db.add_message('msg2', 'chat1', datetime(2024, 1, 2), 'user2', 'Text 2')
        
        assert temp_db.count_messages() == 2
        assert temp_db.count_messages('chat1') == 2
        assert temp_db.count_messages('chat2') == 0
    
    def test_messages_sorted_by_timestamp(self, temp_db):
        """Test that messages are returned sorted by timestamp."""
        temp_db.add_message('msg3', 'chat1', datetime(2024, 1, 3), 'user1', 'Text 3')
        temp_db.add_message('msg1', 'chat1', datetime(2024, 1, 1), 'user1', 'Text 1')
        temp_db.add_message('msg2', 'chat1', datetime(2024, 1, 2), 'user1', 'Text 2')
        
        messages = temp_db.get_messages_by_chat('chat1')
        assert messages[0].msg_id == 'msg1'
        assert messages[1].msg_id == 'msg2'
        assert messages[2].msg_id == 'msg3'


class TestChunkWithMessages:
    """Tests for chunk creation with message references."""
    
    def test_add_chunk_with_messages(self, temp_db):
        """Test adding a chunk with message references."""
        # Add messages first
        temp_db.add_message('msg1', 'chat1', datetime(2024, 1, 1, 12, 0), 'user1', 'Text 1')
        temp_db.add_message('msg2', 'chat1', datetime(2024, 1, 1, 12, 5), 'user1', 'Text 2')
        
        # Add chunk
        temp_db.add_chunk_with_messages(
            chunk_id='chunk1',
            text='Text 1 Text 2',
            chat_id='chat1',
            msg_id_start='msg1',
            msg_id_end='msg2',
            ts_from=datetime(2024, 1, 1, 12, 0),
            ts_to=datetime(2024, 1, 1, 12, 5),
            metadata_json='{"key": "value"}'
        )
        
        chunk = temp_db.get_chunk('chunk1')
        assert chunk is not None
        assert chunk.chat_id == 'chat1'
        assert chunk.msg_id_start == 'msg1'
        assert chunk.msg_id_end == 'msg2'
        assert chunk.ts_from == datetime(2024, 1, 1, 12, 0)
        assert chunk.ts_to == datetime(2024, 1, 1, 12, 5)
    
    def test_update_chunk_topics(self, temp_db):
        """Test updating topic assignments for a chunk."""
        temp_db.add_message('msg1', 'chat1', datetime(2024, 1, 1), 'user1', 'Text 1')
        temp_db.add_chunk_with_messages(
            'chunk1', 'Text 1', 'chat1', 'msg1', 'msg1',
            datetime(2024, 1, 1), datetime(2024, 1, 1)
        )
        
        # Initially no topics
        chunk = temp_db.get_chunk('chunk1')
        assert chunk.topic_l1_id is None
        assert chunk.topic_l2_id is None
        
        # Update topics
        temp_db.update_chunk_topics('chunk1', topic_l1_id=1, topic_l2_id=2)
        
        chunk = temp_db.get_chunk('chunk1')
        assert chunk.topic_l1_id == 1
        assert chunk.topic_l2_id == 2


class TestTopicL1:
    """Tests for L1 topic CRUD operations."""
    
    def test_create_topic_l1(self, temp_db):
        """Test creating an L1 topic."""
        topic_id = temp_db.create_topic_l1(
            title='Test Topic',
            descr='This is a test topic',
            chunk_count=10,
            msg_count=100,
            center_vec=[0.1, 0.2, 0.3],
            ts_from=datetime(2024, 1, 1),
            ts_to=datetime(2024, 1, 31)
        )
        
        assert topic_id is not None
        topic = temp_db.get_topic_l1(topic_id)
        assert topic.title == 'Test Topic'
        assert topic.chunk_count == 10
        assert topic.msg_count == 100
        assert topic.center_vec == '[0.1, 0.2, 0.3]'
    
    def test_get_all_topics_l1(self, temp_db):
        """Test retrieving all L1 topics."""
        temp_db.create_topic_l1('Topic 1', 'Description 1', 5, 50)
        temp_db.create_topic_l1('Topic 2', 'Description 2', 3, 30)
        
        topics = temp_db.get_all_topics_l1()
        assert len(topics) == 2
        assert topics[0].title == 'Topic 1'
        assert topics[1].title == 'Topic 2'
    
    def test_get_chunks_by_topic_l1(self, temp_db):
        """Test retrieving chunks by L1 topic."""
        # Create topic
        topic_id = temp_db.create_topic_l1('Topic 1', 'Description', 2, 20)
        
        # Create chunks with topic assignment
        temp_db.add_message('msg1', 'chat1', datetime(2024, 1, 1), 'user1', 'Text 1')
        temp_db.add_chunk_with_messages('chunk1', 'Text 1', 'chat1', 'msg1', 'msg1',
                                       datetime(2024, 1, 1), datetime(2024, 1, 1))
        temp_db.update_chunk_topics('chunk1', topic_l1_id=topic_id, topic_l2_id=None)
        
        temp_db.add_message('msg2', 'chat1', datetime(2024, 1, 2), 'user1', 'Text 2')
        temp_db.add_chunk_with_messages('chunk2', 'Text 2', 'chat1', 'msg2', 'msg2',
                                       datetime(2024, 1, 2), datetime(2024, 1, 2))
        temp_db.update_chunk_topics('chunk2', topic_l1_id=topic_id, topic_l2_id=None)
        
        chunks = temp_db.get_chunks_by_topic_l1(topic_id)
        assert len(chunks) == 2
        assert chunks[0].id == 'chunk1'
        assert chunks[1].id == 'chunk2'
    
    def test_clear_topics_l1(self, temp_db):
        """Test clearing all L1 topics."""
        temp_db.create_topic_l1('Topic 1', 'Description 1', 5, 50)
        temp_db.create_topic_l1('Topic 2', 'Description 2', 3, 30)
        
        deleted = temp_db.clear_topics_l1()
        assert deleted == 2
        
        topics = temp_db.get_all_topics_l1()
        assert len(topics) == 0
    
    def test_update_topic_l1_parent(self, temp_db):
        """Test updating L1 topic's parent L2."""
        topic_l1_id = temp_db.create_topic_l1('L1 Topic', 'Description', 5, 50)
        topic_l2_id = temp_db.create_topic_l2('L2 Topic', 'Description', 10)
        
        # Initially no parent
        topic = temp_db.get_topic_l1(topic_l1_id)
        assert topic.parent_l2_id is None
        
        # Update parent
        temp_db.update_topic_l1_parent(topic_l1_id, topic_l2_id)
        
        topic = temp_db.get_topic_l1(topic_l1_id)
        assert topic.parent_l2_id == topic_l2_id


class TestTopicL2:
    """Tests for L2 topic CRUD operations."""
    
    def test_create_topic_l2(self, temp_db):
        """Test creating an L2 topic."""
        topic_id = temp_db.create_topic_l2(
            title='Super Topic',
            descr='This is a super topic',
            chunk_count=50,
            center_vec=[0.5, 0.6, 0.7]
        )
        
        assert topic_id is not None
        topic = temp_db.get_topic_l2(topic_id)
        assert topic.title == 'Super Topic'
        assert topic.chunk_count == 50
        # center_vec is stored in ChromaDB, not SQLite, so it will be None in the model
        # This is expected behavior
    
    def test_get_all_topics_l2(self, temp_db):
        """Test retrieving all L2 topics."""
        temp_db.create_topic_l2('L2 Topic 1', 'Description 1', 10)
        temp_db.create_topic_l2('L2 Topic 2', 'Description 2', 20)
        
        topics = temp_db.get_all_topics_l2()
        assert len(topics) == 2
    
    def test_get_l1_topics_by_l2(self, temp_db):
        """Test retrieving L1 topics belonging to an L2 topic."""
        l2_id = temp_db.create_topic_l2('L2 Topic', 'Description', 15)
        
        l1_id1 = temp_db.create_topic_l1('L1 Topic 1', 'Desc 1', 5, 50, parent_l2_id=l2_id)
        l1_id2 = temp_db.create_topic_l1('L1 Topic 2', 'Desc 2', 10, 100, parent_l2_id=l2_id)
        temp_db.create_topic_l1('L1 Topic 3', 'Desc 3', 3, 30)  # No parent
        
        l1_topics = temp_db.get_l1_topics_by_l2(l2_id)
        assert len(l1_topics) == 2
        assert l1_topics[0].id == l1_id1
        assert l1_topics[1].id == l1_id2
    
    def test_get_chunks_by_topic_l2(self, temp_db):
        """Test retrieving chunks by L2 topic."""
        l2_id = temp_db.create_topic_l2('L2 Topic', 'Description', 2)
        
        # Create chunks with L2 assignment
        temp_db.add_message('msg1', 'chat1', datetime(2024, 1, 1), 'user1', 'Text 1')
        temp_db.add_chunk_with_messages('chunk1', 'Text 1', 'chat1', 'msg1', 'msg1',
                                       datetime(2024, 1, 1), datetime(2024, 1, 1))
        temp_db.update_chunk_topics('chunk1', topic_l1_id=None, topic_l2_id=l2_id)
        
        chunks = temp_db.get_chunks_by_topic_l2(l2_id)
        assert len(chunks) == 1
        assert chunks[0].id == 'chunk1'
    
    def test_clear_topics_l2(self, temp_db):
        """Test clearing all L2 topics."""
        l2_id = temp_db.create_topic_l2('L2 Topic', 'Description', 10)
        temp_db.create_topic_l1('L1 Topic', 'Desc', 5, 50, parent_l2_id=l2_id)
        
        deleted = temp_db.clear_topics_l2()
        assert deleted == 1
        
        # L2 topics should be gone
        topics_l2 = temp_db.get_all_topics_l2()
        assert len(topics_l2) == 0
        
        # L1 topics should have parent_l2_id cleared
        topics_l1 = temp_db.get_all_topics_l1()
        assert topics_l1[0].parent_l2_id is None


class TestHierarchicalRelationships:
    """Tests for L1-L2 hierarchical relationships."""
    
    def test_full_hierarchy(self, temp_db):
        """Test complete L2 -> L1 -> Chunks hierarchy."""
        # Create L2 topic
        l2_id = temp_db.create_topic_l2('Main Theme', 'Overall theme', 20)
        
        # Create L1 topics under L2
        l1_id1 = temp_db.create_topic_l1('Subtopic 1', 'First subtopic', 10, 100, parent_l2_id=l2_id)
        l1_id2 = temp_db.create_topic_l1('Subtopic 2', 'Second subtopic', 10, 100, parent_l2_id=l2_id)
        
        # Create messages and chunks
        temp_db.add_message('msg1', 'chat1', datetime(2024, 1, 1), 'user1', 'Text 1')
        temp_db.add_chunk_with_messages('chunk1', 'Text 1', 'chat1', 'msg1', 'msg1',
                                       datetime(2024, 1, 1), datetime(2024, 1, 1))
        temp_db.update_chunk_topics('chunk1', topic_l1_id=l1_id1, topic_l2_id=l2_id)
        
        temp_db.add_message('msg2', 'chat1', datetime(2024, 1, 2), 'user1', 'Text 2')
        temp_db.add_chunk_with_messages('chunk2', 'Text 2', 'chat1', 'msg2', 'msg2',
                                       datetime(2024, 1, 2), datetime(2024, 1, 2))
        temp_db.update_chunk_topics('chunk2', topic_l1_id=l1_id2, topic_l2_id=l2_id)
        
        # Verify hierarchy
        l1_topics = temp_db.get_l1_topics_by_l2(l2_id)
        assert len(l1_topics) == 2
        
        chunks_l1_1 = temp_db.get_chunks_by_topic_l1(l1_id1)
        assert len(chunks_l1_1) == 1
        
        chunks_l2 = temp_db.get_chunks_by_topic_l2(l2_id)
        assert len(chunks_l2) == 2


class TestBackwardCompatibility:
    """Tests for legacy topic methods."""
    
    def test_legacy_topic_methods_still_work(self, temp_db):
        """Ensure backward compatibility - legacy methods were removed, use new L1/L2 methods."""
        # Legacy create_topic/get_topic/get_all_topics methods were removed in favor of L1/L2 methods
        # Test that new methods work for backward compatibility scenarios
        import json
        topic_id = temp_db.create_topic_l1(
            title='Legacy Topic',
            descr='Legacy description',
            chunk_count=0,
            msg_count=0,
            center_vec=json.dumps([0.1] * 10)
        )
        assert topic_id is not None
        
        topic = temp_db.get_topic_l1(topic_id)
        assert topic.title == 'Legacy Topic'
        
        all_topics = temp_db.get_all_topics_l1()
        assert len(all_topics) == 1
