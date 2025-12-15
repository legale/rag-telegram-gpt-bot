"""
Chunk dictionary building utilities.

Provides functions for building chunk dictionaries from database models and domain objects.
"""

import json
from typing import Dict, Optional
from src.storage.db import ChunkModel
from src.core.domain import Chunk


def build_chunk_dict_from_model(
    chunk: ChunkModel, 
    similarity: float, 
    distance: Optional[float] = None, 
    source: str = "vector"
) -> Dict:
    """
    Build chunk dictionary with metadata and topics.
    
    Args:
        chunk: ChunkModel instance
        similarity: Similarity score (0.0-1.0)
        distance: Original distance value (optional)
        source: Source of the chunk ("vector", "two_stage", "hybrid", "fts")
        
    Returns:
        Dictionary with chunk data
    """
    meta = {}
    if chunk.metadata_json:
        try:
            meta = json.loads(chunk.metadata_json)
        except json.JSONDecodeError:
            pass
    
    # topic_l1 and topic_l2 removed - clustering is deprecated
    
    result = {
        "id": chunk.id,
        "text": chunk.text,
        "metadata": meta,
        "score": similarity,
        "source": source
    }
    
    if distance is not None:
        result["distance"] = distance
    
    return result


def _extract_chunk_metadata(chunk: Chunk) -> Dict:
    """
    Extract metadata from chunk domain object.
    
    Args:
        chunk: Chunk domain object
        
    Returns:
        Dictionary with metadata (or empty dict if no metadata)
    """
    return chunk.metadata.copy() if chunk.metadata else {}


def _create_chunk_dict(
    chunk: Chunk,
    similarity: float,
    metadata: Dict,
    distance: Optional[float] = None,
    source: str = "vector"
) -> Dict:
    """
    Create chunk dictionary from components.
    
    Args:
        chunk: Chunk domain object
        similarity: Similarity score (0.0-1.0)
        metadata: Extracted metadata dictionary
        distance: Original distance value (optional)
        source: Source of the chunk ("vector", "two_stage", "hybrid", "fts")
        
    Returns:
        Dictionary with chunk data
    """
    result = {
        "id": chunk.id,
        "text": chunk.text,
        "metadata": metadata,
        "score": similarity,
        "source": source
    }
    
    if distance is not None:
        result["distance"] = distance
    
    return result


def build_chunk_dict_from_domain_chunk(
    chunk: Chunk,
    similarity: float,
    distance: Optional[float] = None,
    source: str = "vector"
) -> Dict:
    """
    Build chunk dictionary from domain Chunk object.
    
    Args:
        chunk: Chunk domain object
        similarity: Similarity score (0.0-1.0)
        distance: Original distance value (optional)
        source: Source of the chunk ("vector", "two_stage", "hybrid", "fts")
        
    Returns:
        Dictionary with chunk data
    """
    meta = _extract_chunk_metadata(chunk)
    return _create_chunk_dict(chunk, similarity, meta, distance, source)


