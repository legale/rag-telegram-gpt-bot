
import pytest
from sqlalchemy.exc import DatabaseError
from unittest.mock import MagicMock, patch
from datetime import datetime
from src.storage.db import Database, ChunkModel, MessageModel

def test_init_validation():
    with pytest.raises(ValueError):
        Database(db_url="")
    with pytest.raises(ValueError):
        Database(db_url=None)

def test_database_creation(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_url=f"sqlite:///{db_path}")
    
    session = db.get_session()
    chunk = ChunkModel(id="1", text="Test chunk")
    session.add(chunk)
    session.commit()
    
    saved_chunk = session.query(ChunkModel).filter_by(id="1").first()
    assert saved_chunk is not None
    assert saved_chunk.text == "Test chunk"
    session.close()

def test_count_chunks(tmp_path):
    db_path = tmp_path / "test_count.db"
    db = Database(db_url=f"sqlite:///{db_path}")
    
    session = db.get_session()
    session.add(ChunkModel(id="1", text="Chunk 1"))
    session.add(ChunkModel(id="2", text="Chunk 2"))
    session.commit()
    session.close()
    
    assert db.count_chunks() == 2

def test_clear(tmp_path):
    db_path = tmp_path / "test_clear.db"
    db = Database(db_url=f"sqlite:///{db_path}")
    
    session = db.get_session()
    session.add(ChunkModel(id="1", text="Chunk 1"))
    session.add(ChunkModel(id="2", text="Chunk 2"))
    session.commit()
    session.close()
    
    assert db.count_chunks() == 2
    deleted = db.clear()
    assert deleted == 2
    assert db.count_chunks() == 0

def test_clear_error(tmp_path):
    # Mock session to raise exception during commit
    db_path = tmp_path / "test_error.db"
    db = Database(db_url=f"sqlite:///{db_path}")
    
    with patch.object(db, 'get_session') as mock_get_session:
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        
        # Simulate query().delete() works but commit() fails
        mock_session.query.return_value.delete.return_value = 5
        mock_session.commit.side_effect = Exception("DB Error")
        
        with pytest.raises(Exception):
            db.clear()
            
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()


def test_get_message_by_id(tmp_path):
    """Test getting message by ID."""
    db_path = tmp_path / "test_get_message.db"
    db = Database(db_url=f"sqlite:///{db_path}")
    
    session = db.get_session()
    msg = MessageModel(
        msg_id="123456_100",
        chat_id="123456",
        ts=datetime(2025, 12, 9, 0, 49, 22),
        from_id="User1",
        text="Test message"
    )
    session.add(msg)
    session.commit()
    session.close()
    
    result = db.get_message_by_id("123456_100")
    assert result is not None
    assert result.text == "Test message"
    assert result.from_id == "User1"


def test_get_message_by_id_not_found(tmp_path):
    """Test getting non-existent message."""
    db_path = tmp_path / "test_get_message_nf.db"
    db = Database(db_url=f"sqlite:///{db_path}")
    
    result = db.get_message_by_id("nonexistent")
    assert result is None


def test_get_messages_by_chunk_single_message(tmp_path):
    """Test getting messages from chunk with single message."""
    db_path = tmp_path / "test_get_messages_chunk.db"
    db = Database(db_url=f"sqlite:///{db_path}")
    
    session = db.get_session()
    
    # Create message
    msg = MessageModel(
        msg_id="123456_100",
        chat_id="123456",
        ts=datetime(2025, 12, 9, 0, 49, 22),
        from_id="User1",
        text="Test message"
    )
    session.add(msg)
    
    # Create chunk
    chunk = ChunkModel(
        id="chunk1",
        text="Chunk text",
        chat_id="123456",
        msg_id_start="123456_100",
        msg_id_end=None
    )
    session.add(chunk)
    session.commit()
    session.close()
    
    result = db.get_messages_by_chunk("chunk1")
    assert len(result) == 1
    assert result[0].msg_id == "123456_100"
    assert result[0].text == "Test message"


def test_get_messages_by_chunk_range(tmp_path):
    """Test getting messages from chunk with range (msg_id_start to msg_id_end)."""
    db_path = tmp_path / "test_get_messages_range.db"
    db = Database(db_url=f"sqlite:///{db_path}")
    
    session = db.get_session()
    
    # Create messages
    msg1 = MessageModel(
        msg_id="123456_100",
        chat_id="123456",
        ts=datetime(2025, 12, 9, 0, 49, 22),
        from_id="User1",
        text="Message 1"
    )
    msg2 = MessageModel(
        msg_id="123456_101",
        chat_id="123456",
        ts=datetime(2025, 12, 9, 0, 49, 30),
        from_id="User2",
        text="Message 2"
    )
    msg3 = MessageModel(
        msg_id="123456_102",
        chat_id="123456",
        ts=datetime(2025, 12, 9, 0, 49, 40),
        from_id="User3",
        text="Message 3"
    )
    session.add_all([msg1, msg2, msg3])
    
    # Create chunk with range
    chunk = ChunkModel(
        id="chunk1",
        text="Chunk text",
        chat_id="123456",
        msg_id_start="123456_100",
        msg_id_end="123456_102"
    )
    session.add(chunk)
    session.commit()
    session.close()
    
    result = db.get_messages_by_chunk("chunk1")
    # Should return all messages in range (ordered by timestamp)
    assert len(result) >= 1  # At least start message
    assert result[0].msg_id == "123456_100"
    # Messages should be ordered by timestamp
    for i in range(len(result) - 1):
        assert result[i].ts <= result[i + 1].ts


def test_get_messages_by_chunk_not_found(tmp_path):
    """Test getting messages from non-existent chunk."""
    db_path = tmp_path / "test_get_messages_nf.db"
    db = Database(db_url=f"sqlite:///{db_path}")
    
    result = db.get_messages_by_chunk("nonexistent")
    assert result == []


def test_get_messages_by_chunk_no_msg_id_start(tmp_path):
    """Test getting messages from chunk without msg_id_start."""
    db_path = tmp_path / "test_get_messages_no_start.db"
    db = Database(db_url=f"sqlite:///{db_path}")
    
    session = db.get_session()
    chunk = ChunkModel(
        id="chunk1",
        text="Chunk text",
        chat_id="123456",
        msg_id_start=None,
        msg_id_end=None
    )
    session.add(chunk)
    session.commit()
    session.close()
    
    result = db.get_messages_by_chunk("chunk1")
    assert result == []


def test_get_chunk_link_info_success(tmp_path):
    """Test successful extraction of all chunk link info fields."""
    import json
    db_path = tmp_path / "test_chunk_link_info.db"
    db = Database(db_url=f"sqlite:///{db_path}")
    
    session = db.get_session()
    chunk = ChunkModel(
        id="chunk1",
        text="Test chunk",
        chat_id="123456",
        msg_id_start="123456_789",
        metadata_json=json.dumps({"chat_username": "test_chat"})
    )
    session.add(chunk)
    session.commit()
    session.close()
    
    chat_id, msg_id, chat_username = db.get_chunk_link_info("chunk1")
    assert chat_id == 123456
    assert msg_id == 789
    assert chat_username == "test_chat"


def test_get_chunk_link_info_not_found(tmp_path):
    """Test get_chunk_link_info for non-existent chunk."""
    db_path = tmp_path / "test_chunk_link_info_nf.db"
    db = Database(db_url=f"sqlite:///{db_path}")
    
    chat_id, msg_id, chat_username = db.get_chunk_link_info("nonexistent")
    assert chat_id is None
    assert msg_id is None
    assert chat_username is None


def test_get_chunk_link_info_no_fields(tmp_path):
    """Test get_chunk_link_info with chunk having no link fields."""
    db_path = tmp_path / "test_chunk_link_info_empty.db"
    db = Database(db_url=f"sqlite:///{db_path}")
    
    session = db.get_session()
    chunk = ChunkModel(
        id="chunk1",
        text="Test chunk",
        chat_id=None,
        msg_id_start=None,
        metadata_json=None
    )
    session.add(chunk)
    session.commit()
    session.close()
    
    chat_id, msg_id, chat_username = db.get_chunk_link_info("chunk1")
    assert chat_id is None
    assert msg_id is None
    assert chat_username is None


def test_get_chunk_link_info_msg_id_format_with_underscore(tmp_path):
    """Test parsing msg_id_start in format '{chat_id}_{msg_id}'."""
    db_path = tmp_path / "test_chunk_link_info_format.db"
    db = Database(db_url=f"sqlite:///{db_path}")
    
    session = db.get_session()
    chunk = ChunkModel(
        id="chunk1",
        text="Test chunk",
        chat_id="123456",
        msg_id_start="123456_999",
        metadata_json=None
    )
    session.add(chunk)
    session.commit()
    session.close()
    
    chat_id, msg_id, chat_username = db.get_chunk_link_info("chunk1")
    assert chat_id == 123456
    assert msg_id == 999
    assert chat_username is None


def test_get_chunk_link_info_msg_id_format_no_underscore(tmp_path):
    """Test parsing msg_id_start without underscore (single number)."""
    db_path = tmp_path / "test_chunk_link_info_no_underscore.db"
    db = Database(db_url=f"sqlite:///{db_path}")
    
    session = db.get_session()
    chunk = ChunkModel(
        id="chunk1",
        text="Test chunk",
        chat_id="123456",
        msg_id_start="999",
        metadata_json=None
    )
    session.add(chunk)
    session.commit()
    session.close()
    
    chat_id, msg_id, chat_username = db.get_chunk_link_info("chunk1")
    assert chat_id == 123456
    assert msg_id == 999
    assert chat_username is None


def test_get_chunk_link_info_invalid_chat_id(tmp_path):
    """Test handling of invalid chat_id (non-numeric string)."""
    db_path = tmp_path / "test_chunk_link_info_invalid_chat.db"
    db = Database(db_url=f"sqlite:///{db_path}")
    
    session = db.get_session()
    chunk = ChunkModel(
        id="chunk1",
        text="Test chunk",
        chat_id="invalid_chat_id",
        msg_id_start="123456_789",
        metadata_json=None
    )
    session.add(chunk)
    session.commit()
    session.close()
    
    chat_id, msg_id, chat_username = db.get_chunk_link_info("chunk1")
    assert chat_id is None  # Should fail to convert
    assert msg_id == 789
    assert chat_username is None


def test_get_chunk_link_info_invalid_msg_id(tmp_path):
    """Test handling of invalid msg_id_start format."""
    db_path = tmp_path / "test_chunk_link_info_invalid_msg.db"
    db = Database(db_url=f"sqlite:///{db_path}")
    
    session = db.get_session()
    chunk = ChunkModel(
        id="chunk1",
        text="Test chunk",
        chat_id="123456",
        msg_id_start="invalid_msg_id",
        metadata_json=None
    )
    session.add(chunk)
    session.commit()
    session.close()
    
    chat_id, msg_id, chat_username = db.get_chunk_link_info("chunk1")
    assert chat_id == 123456
    assert msg_id is None  # Should fail to parse
    assert chat_username is None


def test_get_chunk_link_info_metadata_json_valid(tmp_path):
    """Test extraction of chat_username from valid metadata_json."""
    import json
    db_path = tmp_path / "test_chunk_link_info_meta.db"
    db = Database(db_url=f"sqlite:///{db_path}")
    
    session = db.get_session()
    chunk = ChunkModel(
        id="chunk1",
        text="Test chunk",
        chat_id="123456",
        msg_id_start="123456_789",
        metadata_json=json.dumps({"chat_username": "my_chat", "other_field": "value"})
    )
    session.add(chunk)
    session.commit()
    session.close()
    
    chat_id, msg_id, chat_username = db.get_chunk_link_info("chunk1")
    assert chat_id == 123456
    assert msg_id == 789
    assert chat_username == "my_chat"


def test_get_chunk_link_info_metadata_json_invalid(tmp_path):
    """Test handling of invalid metadata_json."""
    db_path = tmp_path / "test_chunk_link_info_invalid_meta.db"
    db = Database(db_url=f"sqlite:///{db_path}")
    
    session = db.get_session()
    chunk = ChunkModel(
        id="chunk1",
        text="Test chunk",
        chat_id="123456",
        msg_id_start="123456_789",
        metadata_json="invalid json {"
    )
    session.add(chunk)
    session.commit()
    session.close()
    
    chat_id, msg_id, chat_username = db.get_chunk_link_info("chunk1")
    assert chat_id == 123456
    assert msg_id == 789
    assert chat_username is None  # Should fail to parse JSON


def test_get_chunk_link_info_metadata_json_not_dict(tmp_path):
    """Test handling of metadata_json that is not a dict."""
    import json
    db_path = tmp_path / "test_chunk_link_info_meta_not_dict.db"
    db = Database(db_url=f"sqlite:///{db_path}")
    
    session = db.get_session()
    chunk = ChunkModel(
        id="chunk1",
        text="Test chunk",
        chat_id="123456",
        msg_id_start="123456_789",
        metadata_json=json.dumps(["array", "not", "dict"])
    )
    session.add(chunk)
    session.commit()
    session.close()
    
    chat_id, msg_id, chat_username = db.get_chunk_link_info("chunk1")
    assert chat_id == 123456
    assert msg_id == 789
    assert chat_username is None  # Should fail because not a dict


def test_get_chunk_link_info_metadata_json_no_username(tmp_path):
    """Test metadata_json without chat_username field."""
    import json
    db_path = tmp_path / "test_chunk_link_info_meta_no_user.db"
    db = Database(db_url=f"sqlite:///{db_path}")
    
    session = db.get_session()
    chunk = ChunkModel(
        id="chunk1",
        text="Test chunk",
        chat_id="123456",
        msg_id_start="123456_789",
        metadata_json=json.dumps({"other_field": "value"})
    )
    session.add(chunk)
    session.commit()
    session.close()
    
    chat_id, msg_id, chat_username = db.get_chunk_link_info("chunk1")
    assert chat_id == 123456
    assert msg_id == 789
    assert chat_username is None
