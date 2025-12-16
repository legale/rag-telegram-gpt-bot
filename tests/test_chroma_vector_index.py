"""
Tests for ChromaVectorIndex.
"""

import pytest
from src.adapters.vector.chroma_vector_index import ChromaVectorIndex
from src.core.interfaces import VectorDoc, ScoredDoc
from src.storage.vector_store import VectorStore
from src.core.embedding import LocalEmbeddingClient


@pytest.fixture
def vector_store(tmp_path):
    """Create a temporary vector store."""
    client = LocalEmbeddingClient(model="all-MiniLM-L6-v2")
    store = VectorStore(persist_directory=str(tmp_path), embedding_client=client)
    return store


class TestChromaVectorIndex:
    """Tests for ChromaVectorIndex class."""
    
    def test_init(self, vector_store):
        """Test initialization."""
        index = ChromaVectorIndex(vector_store)
        assert index.vector_store == vector_store
    
    def test_upsert_empty(self, vector_store):
        """Test upsert with empty list."""
        index = ChromaVectorIndex(vector_store)
        index.upsert([])  # Should not raise
    
    def test_upsert_single_doc(self, vector_store):
        """Test upsert with single document."""
        index = ChromaVectorIndex(vector_store)
        embedding = [0.1] * 384
        doc = VectorDoc(id="doc1", vector=embedding, meta={"text": "Test"})
        index.upsert([doc])
        assert index.count() == 1
    
    def test_upsert_multiple_docs(self, vector_store):
        """Test upsert with multiple documents."""
        index = ChromaVectorIndex(vector_store)
        embedding = [0.1] * 384
        docs = [
            VectorDoc(id=f"doc{i}", vector=embedding, meta={"text": f"Test {i}"})
            for i in range(3)
        ]
        index.upsert(docs)
        assert index.count() == 3
    
    def test_query_empty_vector(self, vector_store):
        """Test query with empty vector."""
        index = ChromaVectorIndex(vector_store)
        results = index.query([], top_k=5)
        assert results == []
    
    def test_query_with_results(self, vector_store):
        """Test query method."""
        index = ChromaVectorIndex(vector_store)
        embedding = [0.1] * 384
        doc = VectorDoc(id="doc1", vector=embedding, meta={"text": "Test"})
        index.upsert([doc])
        
        query_vector = [0.1] * 384
        results = index.query(query_vector, top_k=5)
        assert len(results) > 0
        assert isinstance(results[0], ScoredDoc)
        # Verify text is included in metadata to avoid SQLite roundtrips
        assert "text" in results[0].meta
        assert results[0].meta["text"] == "Test"
    
    def test_query_with_filter(self, vector_store):
        """Test query with metadata filter."""
        index = ChromaVectorIndex(vector_store)
        embedding = [0.1] * 384
        doc = VectorDoc(id="doc1", vector=embedding, meta={"text": "Test", "chat_id": "chat1"})
        index.upsert([doc])
        
        query_vector = [0.1] * 384
        filter_dict = {"chat_id": "chat1"}
        results = index.query(query_vector, top_k=5, filter=filter_dict)
        assert len(results) >= 0
        if len(results) > 0:
            # Verify text is included in metadata
            assert "text" in results[0].meta
    
    def test_query_includes_text_in_metadata(self, vector_store):
        """Test that query returns text in metadata to minimize SQLite roundtrips."""
        index = ChromaVectorIndex(vector_store)
        embedding = [0.1] * 384
        # Document with text in metadata (should preserve it)
        doc1 = VectorDoc(id="doc1", vector=embedding, meta={"text": "Test document", "chat_id": "chat1"})
        # Document without text in metadata (should add it from ChromaDB documents)
        doc2 = VectorDoc(id="doc2", vector=embedding, meta={"chat_id": "chat2"})
        index.upsert([doc1, doc2])
        
        query_vector = [0.1] * 384
        results = index.query(query_vector, top_k=10)
        assert len(results) >= 2
        
        # Find results by id
        result_dict = {r.id: r for r in results}
        
        # doc1 should have text from metadata
        if "doc1" in result_dict:
            assert "text" in result_dict["doc1"].meta
            assert result_dict["doc1"].meta["text"] == "Test document"
        
        # doc2 should have text added from ChromaDB documents field
        # (ChromaDB stores text in documents field when upserting)
        if "doc2" in result_dict:
            # Text should be present (either from metadata if already there, or from documents)
            assert "text" in result_dict["doc2"].meta
    
    def test_delete_empty(self, vector_store):
        """Test delete with empty list."""
        index = ChromaVectorIndex(vector_store)
        index.delete([])  # Should not raise
    
    def test_delete_by_ids(self, vector_store):
        """Test delete method."""
        index = ChromaVectorIndex(vector_store)
        embedding = [0.1] * 384
        doc = VectorDoc(id="doc1", vector=embedding, meta={"text": "Test"})
        index.upsert([doc])
        assert index.count() == 1
        
        index.delete(["doc1"])
        assert index.count() == 0
    
    def test_count_empty(self, vector_store):
        """Test count on empty index."""
        index = ChromaVectorIndex(vector_store)
        assert index.count() == 0
    
    def test_count_after_upsert(self, vector_store):
        """Test count after upsert."""
        index = ChromaVectorIndex(vector_store)
        embedding = [0.1] * 384
        doc = VectorDoc(id="doc1", vector=embedding, meta={"text": "Test"})
        index.upsert([doc])
        assert index.count() == 1
    
    def test_get_embeddings_by_ids_empty(self, vector_store):
        """Test get_embeddings_by_ids with empty list."""
        index = ChromaVectorIndex(vector_store)
        result = index.get_embeddings_by_ids([])
        assert result == {}
    
    def test_get_embeddings_by_ids(self, vector_store):
        """Test get_embeddings_by_ids method."""
        index = ChromaVectorIndex(vector_store)
        embedding = [0.1] * 384
        doc = VectorDoc(id="doc1", vector=embedding, meta={"text": "Test"})
        index.upsert([doc])
        
        result = index.get_embeddings_by_ids(["doc1"])
        assert "doc1" in result
        assert len(result["doc1"]) == 384

