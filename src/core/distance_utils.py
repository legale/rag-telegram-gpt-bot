"""
Distance and similarity conversion utilities.

Provides functions for converting between distance and similarity metrics
used in vector search operations.
"""

from typing import List


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0.0-1.0), or 0.0 if vectors have different lengths or zero norms
    """
    if len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def distance_to_similarity(distance: float) -> float:
    """
    Convert distance to similarity score.
    
    Args:
        distance: Distance value from vector search
        
    Returns:
        Similarity score (0.0-1.0)
    """
    if distance <= 1.0:
        return 1.0 - distance
    elif distance <= 2.0:
        return 1.0 - (distance / 2.0)
    else:
        return max(0.0, 1.0 - distance)


def similarity_to_distance(similarity: float) -> float:
    """
    Convert similarity score to distance.
    
    Args:
        similarity: Similarity score (0.0-1.0)
        
    Returns:
        Distance value
    """
    return 1.0 - similarity



