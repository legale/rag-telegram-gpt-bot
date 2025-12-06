
import pytest
from sqlalchemy.exc import SQLAlchemyError
from unittest.mock import MagicMock, patch
from src.storage.db import Database, ChunkModel

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
