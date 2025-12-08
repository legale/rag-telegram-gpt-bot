
import pytest
from src.storage.vector_store import VectorStore
import shutil
import os
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_embedding_client():
    client = MagicMock()
    client.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
    return client

@pytest.fixture
def test_vector_store(tmp_path, mock_embedding_client):
    test_dir = tmp_path / "test_chroma_db"
    store = VectorStore(
        collection_name="test_collection", 
        persist_directory=str(test_dir),
        embedding_client=mock_embedding_client
    )
    yield store

def test_vector_store_initialization(test_vector_store):
    assert test_vector_store.count() == 0
    assert test_vector_store.collection_name == "test_collection"

def test_add_documents_and_query(test_vector_store):
    # Add documents
    ids = ["1", "2"]
    docs = ["Hello world", "Python programming"]
    metadatas = [{"source": "a"}, {"source": "b"}]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    
    test_vector_store.add_documents_with_embeddings(
        ids=ids,
        documents=docs,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    assert test_vector_store.count() == 2
    
    # Mock collection.query to verify it's called
    with patch.object(test_vector_store.collection, 'query') as mock_query:
        mock_query.return_value = {"ids": [["1"]], "documents": ["doc"], "distances": [0.0]}
        
        results = test_vector_store.query(
            query_texts=["Hello"],
            n_results=1
        )
        
        test_vector_store.embedding_client.get_embeddings.assert_called_with(["Hello"])
        mock_query.assert_called_with(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=1
        )
        assert results["ids"][0][0] == "1"

def test_clear(test_vector_store):
    # Use valid metadata or None
    test_vector_store.add_documents_with_embeddings(
        ids=["1"],
        documents=["doc"],
        embeddings=[[0.1]],
        metadatas=[{"k": "v"}]
    )
    
    # Verify add worked (mock collection.add might be needed if we don't want real chroma, but real chroma is better for integration if light)
    # The previous error "ValueError: Expected metadata to be a non-empty dict" implies we are using real Chroma validation.
    # So we must use valid data.
    
    # Note: real chroma needs consistent dimensions.
    # If we initialized with dim=3 in previous test, reuse might be issue if same collection?
    # tmp_path ensures unique dir.
    # But embeddings=[[0.1]] is dim=1.
    
    # Just mock collection methods to avoid chroma internals in unit test
    with patch.object(test_vector_store.collection, 'count') as mock_count, \
         patch.object(test_vector_store.collection, 'delete') as mock_delete:
         
        mock_count.side_effect = [1, 0] # before, after
        
        deleted = test_vector_store.clear()
        
        assert deleted == 1
        # clear() now calls delete with ids, not where
        # Check that delete was called (with ids parameter)
        assert mock_delete.called

def test_add_empty(test_vector_store):
    test_vector_store.add_documents_with_embeddings([], [], [], [])
    assert test_vector_store.count() == 0

def test_add_mismatch(test_vector_store):
    with pytest.raises(ValueError):
        test_vector_store.add_documents_with_embeddings(["1"], ["doc"], [])

def test_add_meta_mismatch(test_vector_store):
    with pytest.raises(ValueError):
        test_vector_store.add_documents_with_embeddings(
            ["1"], ["doc"], [[1.0]], metadatas=[{}, {}]
        )
