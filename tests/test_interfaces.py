"""Tests for src/core/interfaces.py"""

import pytest
from datetime import datetime
from src.core.interfaces import (
    VectorDoc,
    ScoredDoc,
    SearchFilters,
    MessageStore,
    ChunkStore,
    VectorIndex,
    Embedder,
    FTSIndex,
    LLM,
    ConfigProvider,
    TransactionManager
)
from src.core.domain import Message, Chunk, TopicUpdate, ProfileConfig


class TestVectorDoc:
    """Tests for VectorDoc dataclass"""
    
    def test_vector_doc_creation(self):
        """Test creating a VectorDoc"""
        doc = VectorDoc(
            id="doc1",
            vector=[0.1, 0.2, 0.3]
        )
        
        assert doc.id == "doc1"
        assert doc.vector == [0.1, 0.2, 0.3]
        assert doc.meta == {}
    
    def test_vector_doc_with_meta(self):
        """Test creating a VectorDoc with metadata"""
        doc = VectorDoc(
            id="doc1",
            vector=[0.1, 0.2],
            meta={"key": "value"}
        )
        
        assert doc.meta["key"] == "value"


class TestScoredDoc:
    """Tests for ScoredDoc dataclass"""
    
    def test_scored_doc_creation(self):
        """Test creating a ScoredDoc"""
        doc = ScoredDoc(id="doc1", score=0.8)
        
        assert doc.id == "doc1"
        assert doc.score == 0.8
        assert doc.meta == {}
    
    def test_scored_doc_with_meta(self):
        """Test creating a ScoredDoc with metadata"""
        doc = ScoredDoc(
            id="doc1",
            score=0.8,
            meta={"key": "value"}
        )
        
        assert doc.meta["key"] == "value"


class TestSearchFilters:
    """Tests for SearchFilters dataclass"""
    
    def test_search_filters_creation(self):
        """Test creating SearchFilters"""
        filters = SearchFilters()
        
        assert filters.author is None
        assert filters.time_from is None
        assert filters.time_to is None
        assert filters.chat_id is None
    
    def test_search_filters_with_all_fields(self):
        """Test creating SearchFilters with all fields"""
        time_from = datetime.now()
        time_to = datetime.now()
        filters = SearchFilters(
            author="user1",
            time_from=time_from,
            time_to=time_to,
            chat_id="chat1"
        )
        
        assert filters.author == "user1"
        assert filters.time_from == time_from
        assert filters.time_to == time_to
        assert filters.chat_id == "chat1"


class TestProtocols:
    """Tests for Protocol interfaces"""
    
    def test_message_store_protocol(self):
        """Test MessageStore protocol can be implemented"""
        class MockMessageStore:
            def save_batch(self, messages):
                return len(messages)
            
            def get_by_chat(self, chat_id, limit, offset):
                return []
            
            def get_context(self, chat_id, time_point, window_sec):
                return []
            
            def count(self):
                return 0
        
        store = MockMessageStore()
        # Protocol check - just verify methods exist
        assert hasattr(store, 'save_batch')
        assert hasattr(store, 'get_by_chat')
        assert hasattr(store, 'get_context')
        assert hasattr(store, 'count')
    
    def test_chunk_store_protocol(self):
        """Test ChunkStore protocol can be implemented"""
        class MockChunkStore:
            def save_batch(self, chunks):
                return len(chunks)
            
            def get_by_ids(self, ids):
                return []
            
            def update_topics(self, updates):
                pass
            
            def clear(self):
                pass
        
        store = MockChunkStore()
        # Protocol check - just verify methods exist
        assert hasattr(store, 'save_batch')
        assert hasattr(store, 'get_by_ids')
        assert hasattr(store, 'update_topics')
        assert hasattr(store, 'clear')
    
    def test_vector_index_protocol(self):
        """Test VectorIndex protocol can be implemented"""
        class MockVectorIndex:
            def upsert(self, items):
                pass
            
            def query(self, vector, top_k, filter=None):
                return []
            
            def delete(self, ids):
                pass
            
            def count(self):
                return 0
            
            def get_embeddings_by_ids(self, ids):
                return {}
        
        index = MockVectorIndex()
        # Protocol check - just verify methods exist
        assert hasattr(index, 'upsert')
        assert hasattr(index, 'query')
        assert hasattr(index, 'delete')
        assert hasattr(index, 'count')
        assert hasattr(index, 'get_embeddings_by_ids')
    
    def test_embedder_protocol(self):
        """Test Embedder protocol can be implemented"""
        class MockEmbedder:
            def embed_documents(self, texts):
                return [[0.1] * 10 for _ in texts]
            
            def embed_query(self, text):
                return [0.1] * 10
        
        embedder = MockEmbedder()
        # Protocol check - just verify methods exist
        assert hasattr(embedder, 'embed_documents')
        assert hasattr(embedder, 'embed_query')
    
    def test_fts_index_protocol(self):
        """Test FTSIndex protocol can be implemented"""
        class MockFTSIndex:
            def search(self, query, top_k, filters=None):
                return []
            
            def normalize_text(self, text):
                return text.lower()
        
        index = MockFTSIndex()
        # Protocol check - just verify methods exist
        assert hasattr(index, 'search')
        assert hasattr(index, 'normalize_text')
    
    def test_llm_protocol(self):
        """Test LLM protocol can be implemented"""
        class MockLLM:
            def complete(self, prompt, system=None, **kwargs):
                return "Response"
        
        llm = MockLLM()
        # Protocol check - just verify methods exist
        assert hasattr(llm, 'complete')
    
    def test_config_provider_protocol(self):
        """Test ConfigProvider protocol can be implemented"""
        class MockConfigProvider:
            def get_profile_config(self, profile_name):
                return ProfileConfig(model_name="test", embedding_provider="local")
        
        provider = MockConfigProvider()
        # Protocol check - just verify methods exist
        assert hasattr(provider, 'get_profile_config')
    
    def test_transaction_manager_protocol(self):
        """Test TransactionManager protocol can be implemented"""
        from contextlib import contextmanager
        
        class MockTransactionManager:
            @contextmanager
            def atomic(self):
                yield None
        
        manager = MockTransactionManager()
        # Protocol check - just verify methods exist
        assert hasattr(manager, 'atomic')

