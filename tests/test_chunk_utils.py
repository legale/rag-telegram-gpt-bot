"""Tests for src/core/chunk_utils.py"""

import pytest
import json
from unittest.mock import Mock
from src.core.chunk_utils import (
    build_chunk_dict_from_model,
    build_chunk_dict_from_domain_chunk
)
from src.storage.db import ChunkModel
from src.core.domain import Chunk


class TestBuildChunkDictFromModel:
    """Tests for build_chunk_dict_from_model"""
    
    def test_build_with_metadata(self):
        """Test building chunk dict with metadata"""
        chunk = Mock(spec=ChunkModel)
        chunk.id = "chunk1"
        chunk.text = "Test text"
        chunk.metadata_json = json.dumps({"key": "value"})
        
        result = build_chunk_dict_from_model(chunk, similarity=0.8)
        
        assert result["id"] == "chunk1"
        assert result["text"] == "Test text"
        assert result["metadata"]["key"] == "value"
        assert result["score"] == 0.8
        assert result["source"] == "vector"
    
    def test_build_with_distance(self):
        """Test building chunk dict with distance"""
        chunk = Mock(spec=ChunkModel)
        chunk.id = "chunk1"
        chunk.text = "Test"
        chunk.metadata_json = None
        
        result = build_chunk_dict_from_model(chunk, similarity=0.7, distance=0.3)
        
        assert result["distance"] == 0.3
    
    def test_build_with_invalid_json(self):
        """Test building chunk dict with invalid JSON metadata"""
        chunk = Mock(spec=ChunkModel)
        chunk.id = "chunk1"
        chunk.text = "Test"
        chunk.metadata_json = "invalid json"
        
        result = build_chunk_dict_from_model(chunk, similarity=0.8)
        
        assert result["metadata"] == {}
    
    def test_build_with_custom_source(self):
        """Test building chunk dict with custom source"""
        chunk = Mock(spec=ChunkModel)
        chunk.id = "chunk1"
        chunk.text = "Test"
        chunk.metadata_json = None
        
        result = build_chunk_dict_from_model(chunk, similarity=0.8, source="fts")
        
        assert result["source"] == "fts"


class TestBuildChunkDictFromDomainChunk:
    """Tests for build_chunk_dict_from_domain_chunk"""
    
    def test_build_with_metadata(self):
        """Test building chunk dict from domain chunk with metadata"""
        chunk = Chunk(
            id="chunk1",
            text="Test text",
            metadata={"key": "value"}
        )
        
        result = build_chunk_dict_from_domain_chunk(chunk, similarity=0.8)
        
        assert result["id"] == "chunk1"
        assert result["text"] == "Test text"
        assert result["metadata"]["key"] == "value"
        assert result["score"] == 0.8
        assert result["source"] == "vector"
    
    def test_build_without_metadata(self):
        """Test building chunk dict from domain chunk without metadata"""
        chunk = Chunk(id="chunk1", text="Test")
        
        result = build_chunk_dict_from_domain_chunk(chunk, similarity=0.7)
        
        assert result["metadata"] == {}
    
    def test_build_with_distance(self):
        """Test building chunk dict with distance"""
        chunk = Chunk(id="chunk1", text="Test")
        
        result = build_chunk_dict_from_domain_chunk(chunk, similarity=0.7, distance=0.3)
        
        assert result["distance"] == 0.3
    
    def test_build_with_custom_source(self):
        """Test building chunk dict with custom source"""
        chunk = Chunk(id="chunk1", text="Test")
        
        result = build_chunk_dict_from_domain_chunk(chunk, similarity=0.8, source="hybrid")
        
        assert result["source"] == "hybrid"
