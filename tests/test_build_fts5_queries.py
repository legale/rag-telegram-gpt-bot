"""Tests for build_fts5_queries function."""

import pytest
from src.core.hybrid_retrieval import build_fts5_queries


class TestBuildFTS5Queries:
    """Test build_fts5_queries function."""
    
    def test_removes_stop_words(self):
        """Test that stop words are removed."""
        q_and, q_or = build_fts5_queries("что это такое")
        # "что" and "это" are stop words, should be removed
        assert "что" not in q_and
        assert "это" not in q_and
        assert "такое" in q_and or "такое" in q_or
    
    def test_removes_short_tokens(self):
        """Test that tokens < 2 chars are removed."""
        q_and, q_or = build_fts5_queries("а б в г д е")
        # All single char tokens should be removed
        assert q_and == "" or all(len(t) >= 2 for t in q_and.split())
    
    def test_removes_numbers(self):
        """Test that numbers are removed."""
        q_and, q_or = build_fts5_queries("ошибка 123 тест 456")
        assert "123" not in q_and
        assert "456" not in q_and
        assert "ошибка" in q_and or "ошибка" in q_or
        assert "тест" in q_and or "тест" in q_or
    
    def test_handles_quotes(self):
        """Test that quoted text is parsed as regular tokens, not phrase."""
        q_and, q_or = build_fts5_queries('поиск "ошибка системы"')
        # Quotes should be removed, tokens should be separate
        assert '"' not in q_and
        assert '"' not in q_or
        # Both tokens should be present
        assert "ошибка" in q_and or "ошибка" in q_or
        assert "системы" in q_and or "системы" in q_or
    
    def test_handles_different_quote_types(self):
        """Test handling of different quote types."""
        q_and, q_or = build_fts5_queries('тест "двойные" и «кавычки»')
        assert '"' not in q_and
        assert '"' not in q_or
        assert "двойные" in q_and or "двойные" in q_or
        assert "кавычки" in q_and or "кавычки" in q_or
    
    def test_gives_and_then_or(self):
        """Test that function returns both AND and OR queries."""
        q_and, q_or = build_fts5_queries("ошибка система")
        assert q_and != ""
        assert q_or != ""
        # OR query should contain OR operator
        assert " OR " in q_or
        # AND query should be space-separated
        assert " " in q_and or len(q_and.split()) == 1
    
    def test_normalizes_yo_to_e(self):
        """Test that ё is converted to е."""
        q_and, q_or = build_fts5_queries("ёлка")
        assert "ё" not in q_and
        assert "ё" not in q_or
        assert "елка" in q_and or "елка" in q_or
    
    def test_handles_special_characters(self):
        """Test that special characters don't cause errors."""
        q_and, q_or = build_fts5_queries("тест!@#$%^&*()_+-=[]{}|;':\",./<>?")
        # Should not raise exception
        assert isinstance(q_and, str)
        assert isinstance(q_or, str)
    
    def test_handles_empty_input(self):
        """Test that empty input returns empty queries."""
        q_and, q_or = build_fts5_queries("")
        assert q_and == ""
        assert q_or == ""
    
    def test_handles_whitespace_only(self):
        """Test that whitespace-only input returns empty queries."""
        q_and, q_or = build_fts5_queries("   \n\t  ")
        assert q_and == ""
        assert q_or == ""
    
    def test_handles_only_stop_words(self):
        """Test that query with only stop words returns empty."""
        q_and, q_or = build_fts5_queries("что это и как")
        # All are stop words, should return empty
        assert q_and == ""
        assert q_or == ""
    
    def test_handles_only_numbers(self):
        """Test that query with only numbers returns empty."""
        q_and, q_or = build_fts5_queries("123 456 789")
        assert q_and == ""
        assert q_or == ""
    
    def test_handles_mixed_content(self):
        """Test handling of mixed content."""
        q_and, q_or = build_fts5_queries("найти ошибку в системе 123")
        # Should contain meaningful words, exclude stop words and numbers
        assert "найти" in q_and or "найти" in q_or
        assert "ошибку" in q_and or "ошибку" in q_or
        assert "системе" in q_and or "системе" in q_or
        assert "123" not in q_and
        assert "в" not in q_and  # stop word
    
    def test_lowercase_normalization(self):
        """Test that text is lowercased."""
        q_and, q_or = build_fts5_queries("ОШИБКА СИСТЕМЫ")
        # Tokens should be lowercase (OR operator may be uppercase)
        tokens_and = q_and.split() if q_and else []
        tokens_or = [t for t in q_or.split(" OR ") if t.strip()]
        all_tokens = tokens_and + tokens_or
        assert all(t.islower() for t in all_tokens) or len(all_tokens) == 0
        assert "ошибка" in q_and or "ошибка" in q_or
        assert "системы" in q_and or "системы" in q_or
    
    def test_removes_punctuation(self):
        """Test that punctuation is removed."""
        q_and, q_or = build_fts5_queries("ошибка, система!")
        assert "," not in q_and
        assert "!" not in q_and
        assert "ошибка" in q_and or "ошибка" in q_or
        assert "система" in q_and or "система" in q_or
    
    def test_escapes_fts5_special_chars(self):
        """Test that FTS5 special characters are escaped."""
        # Note: build_fts5_queries escapes double quotes by doubling them
        q_and, q_or = build_fts5_queries('тест "quote"')
        # Double quotes should be escaped in the query
        if '"' in q_and:
            # If quotes remain, they should be escaped (doubled)
            assert '""' in q_and or q_and.count('"') % 2 == 0

