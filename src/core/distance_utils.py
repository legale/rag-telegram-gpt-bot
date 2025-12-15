"""
Distance and similarity conversion utilities.

Provides functions for converting between distance and similarity metrics
used in vector search operations.
"""


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



