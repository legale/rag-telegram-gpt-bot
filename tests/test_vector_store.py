import pytest
from src.storage.vector_store import VectorStore
import shutil
import os

@pytest.fixture
def test_vector_store():
    # Use a temporary directory for the test database
    test_dir = "test_chroma_db"
    store = VectorStore(collection_name="test_collection", persist_directory=test_dir)
    yield store
    # Cleanup
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

def test_add_and_query(test_vector_store):
    ids = ["1", "2"]
    documents = ["This is a legal document about unions.", "This is a recipe for cake."]
    metadatas = [{"type": "legal"}, {"type": "food"}]
    
    test_vector_store.add_documents(ids=ids, documents=documents, metadatas=metadatas)
    
    assert test_vector_store.count() == 2
    
    results = test_vector_store.query("union lawyer", n_results=1)
    
    assert len(results['ids'][0]) == 1
    assert results['ids'][0][0] == "1"
    assert "legal" in results['documents'][0][0]
