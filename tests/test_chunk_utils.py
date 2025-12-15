"""
Tests for chunk_utils module.
"""

import pytest
from src.core.chunk_utils import build_chunk_dict_from_model, build_chunk_dict_from_domain_chunk
from src.storage.db import ChunkModel
from src.core.domain import Chunk
from datetime import datetime
import json


class TestBuildChunkDictFromModel:
    """Tests for build_chunk_dict_from_model function."""
    
    def test_build_basic(self):
        """Test building basic chunk dict."""
        chunk = ChunkModel(
            id="chunk1",
            text="Test text",
            metadata_json=None
        )
        result = build_chunk_dict_from_model(chunk, similarity=0.8)
        
        assert result["id"] == "chunk1"
        assert result["text"] == "Test text"
        assert result["score"] == 0.8
        assert result["source"] == "vector"
        assert "metadata" in result
    
    def test_build_with_metadata(self):
        """Test building with metadata."""
        metadata = {"chat_id": "chat1", "msg_id": "msg1"}
        chunk = ChunkModel(
            id="chunk1",
            text="Test text",
            metadata_json=json.dumps(metadata)
        )
        result = build_chunk_dict_from_model(chunk, similarity=0.8)
        
        assert result["metadata"] == metadata
    
    def test_build_with_distance(self):
        """Test building with distance."""
        chunk = ChunkModel(id="chunk1", text="Test", metadata_json=None)
        result = build_chunk_dict_from_model(chunk, similarity=0.8, distance=0.2)
        
        assert result["distance"] == 0.2
    
    def test_build_with_source(self):
        """Test building with custom source."""
        chunk = ChunkModel(id="chunk1", text="Test", metadata_json=None)
        result = build_chunk_dict_from_model(chunk, similarity=0.8, source="fts")
        
        assert result["source"] == "fts"
    
    def test_build_with_invalid_json(self):
        """Test building with invalid JSON metadata."""
        chunk = ChunkModel(
            id="chunk1",
            text="Test text",
            metadata_json="invalid json"
        )
        result = build_chunk_dict_from_model(chunk, similarity=0.8)
        
        assert result["metadata"] == {}


class TestBuildChunkDictFromDomainChunk:
    """Tests for build_chunk_dict_from_domain_chunk function."""
    
    def test_build_basic(self):
        """Test building basic chunk dict from domain chunk."""
        chunk = Chunk(
            id="chunk1",
            text="Test text",
            metadata={}
        )
        result = build_chunk_dict_from_domain_chunk(chunk, similarity=0.8)
        
        assert result["id"] == "chunk1"
        assert result["text"] == "Test text"
        assert result["score"] == 0.8
        assert result["source"] == "vector"
    
    def test_build_with_metadata(self):
        """Test building with metadata."""
        chunk = Chunk(
            id="chunk1",
            text="Test text",
            metadata={"chat_id": "chat1"}
        )
        result = build_chunk_dict_from_domain_chunk(chunk, similarity=0.8)
        
        assert result["metadata"] == {"chat_id": "chat1"}
    
    def test_build_with_distance(self):
        """Test building with distance."""
        chunk = Chunk(id="chunk1", text="Test", metadata={})
        result = build_chunk_dict_from_domain_chunk(chunk, similarity=0.8, distance=0.2)
        
        assert result["distance"] == 0.2
    
    def test_build_with_source(self):
        """Test building with custom source."""
        chunk = Chunk(id="chunk1", text="Test", metadata={})
        result = build_chunk_dict_from_domain_chunk(chunk, similarity=0.8, source="hybrid")
        
        assert result["source"] == "hybrid"
    
    def test_build_with_none_metadata(self):
        """Test building with None metadata."""
        chunk = Chunk(id="chunk1", text="Test", metadata=None)
        result = build_chunk_dict_from_domain_chunk(chunk, similarity=0.8)
        
        assert result["metadata"] == {}

