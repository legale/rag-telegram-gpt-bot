"""
Chunk dictionary building utilities.

Provides functions for building chunk dictionaries from database models.
"""

import json
from typing import Dict, Optional
from src.storage.db import ChunkModel


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
        source: Source of the chunk ("vector", "two_stage", "topic_l1", "topic_l2")
        
    Returns:
        Dictionary with chunk data
    """
    meta = {}
    if chunk.metadata_json:
        try:
            meta = json.loads(chunk.metadata_json)
        except json.JSONDecodeError:
            pass
    
    if chunk.topic_l1:
        meta["topic_l1_id"] = chunk.topic_l1.id
        meta["topic_l1_title"] = chunk.topic_l1.title
    if chunk.topic_l2:
        meta["topic_l2_id"] = chunk.topic_l2.id
        meta["topic_l2_title"] = chunk.topic_l2.title
    
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


