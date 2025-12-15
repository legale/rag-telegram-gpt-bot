"""Tests for src/core/distance_utils.py"""

import pytest
from src.core.distance_utils import (
    distance_to_similarity,
    similarity_to_distance
)


class TestDistanceToSimilarity:
    """Tests for distance_to_similarity"""
    
    def test_distance_leq_one(self):
        """Test distance <= 1.0"""
        similarity = distance_to_similarity(0.5)
        assert similarity == 0.5
    
    def test_distance_one(self):
        """Test distance == 1.0"""
        similarity = distance_to_similarity(1.0)
        assert similarity == 0.0
    
    def test_distance_zero(self):
        """Test distance == 0.0"""
        similarity = distance_to_similarity(0.0)
        assert similarity == 1.0
    
    def test_distance_between_one_and_two(self):
        """Test distance between 1.0 and 2.0"""
        similarity = distance_to_similarity(1.5)
        assert similarity == 0.25  # 1.0 - (1.5 / 2.0)
    
    def test_distance_two(self):
        """Test distance == 2.0"""
        similarity = distance_to_similarity(2.0)
        assert similarity == 0.0  # 1.0 - (2.0 / 2.0)
    
    def test_distance_gt_two(self):
        """Test distance > 2.0"""
        similarity = distance_to_similarity(3.0)
        assert similarity == 0.0  # max(0.0, 1.0 - 3.0)
    
    def test_distance_very_large(self):
        """Test very large distance"""
        similarity = distance_to_similarity(100.0)
        assert similarity == 0.0


class TestSimilarityToDistance:
    """Tests for similarity_to_distance"""
    
    def test_similarity_one(self):
        """Test similarity == 1.0"""
        distance = similarity_to_distance(1.0)
        assert distance == 0.0
    
    def test_similarity_zero(self):
        """Test similarity == 0.0"""
        distance = similarity_to_distance(0.0)
        assert distance == 1.0
    
    def test_similarity_half(self):
        """Test similarity == 0.5"""
        distance = similarity_to_distance(0.5)
        assert distance == 0.5
    
    def test_similarity_arbitrary(self):
        """Test arbitrary similarity value"""
        distance = similarity_to_distance(0.8)
        assert abs(distance - 0.2) < 0.0001
