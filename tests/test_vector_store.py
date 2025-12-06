import pytest
from src.storage.vector_store import VectorStore
import shutil
import os

@pytest.fixture
def test_vector_store():
    # Use a temporary directory for the test database
    test_dir = "test_chroma_db"
    # Clean up before test
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    store = VectorStore(collection_name="test_collection", persist_directory=test_dir)
    yield store
    
    # Cleanup after test
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

def test_vector_store_initialization(test_vector_store):
    """Test that vector store initializes correctly."""
    assert test_vector_store.count() == 0
    assert test_vector_store.collection_name == "test_collection"
    assert test_vector_store.persist_directory == "test_chroma_db"
