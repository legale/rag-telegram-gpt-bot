import pytest
from src.storage.db import Database, ChunkModel

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
