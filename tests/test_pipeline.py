import pytest
from src.ingestion.pipeline import IngestionPipeline
import os
import shutil

def test_ingestion_pipeline(tmp_path):
    # Setup
    db_path = tmp_path / "test.db"
    vector_db_path = tmp_path / "test_chroma_db"
    input_file = tmp_path / "chat.txt"
    input_file.write_text("User1: Hello\nUser2: Hi there\nUser1: How are you?")
    
    pipeline = IngestionPipeline(
        db_url=f"sqlite:///{db_path}",
        vector_db_path=str(vector_db_path)
    )
    
    # Run pipeline
    pipeline.run(str(input_file))
    
    # Verify SQL DB
    session = pipeline.db.get_session()
    from src.storage.db import ChunkModel
    chunks = session.query(ChunkModel).all()
    assert len(chunks) > 0
    session.close()
    
    # Verify Vector DB
    assert pipeline.vector_store.count() > 0
    
    # Cleanup (ChromaDB creates a directory)
    if os.path.exists(str(vector_db_path)):
        shutil.rmtree(str(vector_db_path))
