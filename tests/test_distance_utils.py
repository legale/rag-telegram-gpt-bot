"""
Tests for distance_utils module.
"""

import pytest
from src.core.distance_utils import distance_to_similarity, similarity_to_distance


class TestDistanceToSimilarity:
    """Tests for distance_to_similarity function."""
    
    def test_distance_zero(self):
        """Test distance 0.0."""
        result = distance_to_similarity(0.0)
        assert result == 1.0
    
    def test_distance_one(self):
        """Test distance 1.0."""
        result = distance_to_similarity(1.0)
        assert result == 0.0
    
    def test_distance_half(self):
        """Test distance 0.5."""
        result = distance_to_similarity(0.5)
        assert result == 0.5
    
    def test_distance_between_one_and_two(self):
        """Test distance between 1.0 and 2.0."""
        result = distance_to_similarity(1.5)
        assert 0.0 <= result <= 1.0
    
    def test_distance_over_two(self):
        """Test distance over 2.0."""
        result = distance_to_similarity(3.0)
        assert result >= 0.0
        assert result <= 1.0


class TestSimilarityToDistance:
    """Tests for similarity_to_distance function."""
    
    def test_similarity_one(self):
        """Test similarity 1.0."""
        result = similarity_to_distance(1.0)
        assert result == 0.0
    
    def test_similarity_zero(self):
        """Test similarity 0.0."""
        result = similarity_to_distance(0.0)
        assert result == 1.0
    
    def test_similarity_half(self):
        """Test similarity 0.5."""
        result = similarity_to_distance(0.5)
        assert result == 0.5
    
    def test_similarity_roundtrip(self):
        """Test roundtrip conversion."""
        original_distance = 0.7
        similarity = distance_to_similarity(original_distance)
        converted_distance = similarity_to_distance(similarity)
        
        # Should be approximately equal (within floating point precision)
        assert abs(original_distance - converted_distance) < 0.01

