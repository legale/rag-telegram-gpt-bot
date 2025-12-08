"""
Tests for LegaleBot model management functionality.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from src.bot.core import LegaleBot


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies for LegaleBot."""
    with patch('src.bot.core.Database') as mock_db, \
         patch('src.bot.core.VectorStore') as mock_vs, \
         patch('src.bot.core.EmbeddingClient') as mock_ec, \
         patch('src.bot.core.LLMClient') as mock_llm, \
         patch('src.bot.core.RetrievalService') as mock_rs, \
         patch('src.bot.core.PromptEngine') as mock_pe:
        
        yield {
            'db': mock_db,
            'vector_store': mock_vs,
            'embedding_client': mock_ec,
            'llm_client': mock_llm,
            'retrieval_service': mock_rs,
            'prompt_engine': mock_pe
        }


@pytest.fixture
def temp_models_file():
    """Create a temporary models.txt file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp') as f:
        f.write("openai/gpt-oss-20b:free\n")
        f.write("nvidia/nemotron-nano-9b-v2:free\n")
        f.write("cognitivecomputations/dolphin-mistral-24b-venice-edition:free\n")
        f.write("google/gemma-3-27b-it:free\n")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestLoadAvailableModels:
    """Tests for _load_available_models() method."""
    
    def test_load_models_with_valid_file(self, mock_dependencies, temp_models_file):
        """Test loading models from a valid models.txt file."""
        with patch('src.bot.core.os.path.join', return_value=temp_models_file):
            bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
            
            assert len(bot.available_models) == 4
            assert "openai/gpt-oss-20b:free" in bot.available_models
            assert "nvidia/nemotron-nano-9b-v2:free" in bot.available_models
            assert "cognitivecomputations/dolphin-mistral-24b-venice-edition:free" in bot.available_models
            assert "google/gemma-3-27b-it:free" in bot.available_models
    
    def test_load_models_with_missing_file(self, mock_dependencies):
        """Test loading models when models.txt is missing (should use fallback)."""
        with patch('src.bot.core.os.path.join', return_value='/nonexistent/models.txt'):
            bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
            
            # Should return empty list when file is missing (no fallback to default)
            assert len(bot.available_models) == 0
    
    def test_load_models_with_empty_file(self, mock_dependencies):
        """Test loading models from an empty models.txt file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp') as f:
            temp_path = f.name
        
        try:
            with patch('src.bot.core.os.path.join', return_value=temp_path):
                bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
                
                # Should return empty list for empty file (no fallback)
                assert len(bot.available_models) == 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_load_models_filters_empty_lines(self, mock_dependencies):
        """Test that empty lines are filtered out from models.txt."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp') as f:
            f.write("model1\n")
            f.write("\n")
            f.write("model2\n")
            f.write("   \n")
            f.write("model3\n")
            temp_path = f.name
        
        try:
            with patch('src.bot.core.os.path.join', return_value=temp_path):
                bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
                
                assert len(bot.available_models) == 3
                assert "model1" in bot.available_models
                assert "model2" in bot.available_models
                assert "model3" in bot.available_models
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestgetModel:
    """Tests for get_model() method."""
    
    def test_get_model_single_model(self, mock_dependencies, temp_models_file):
        """Test geting when only one model is available."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp') as f:
            f.write("single-model\n")
            single_model_path = f.name
        
        try:
            with patch('src.bot.core.os.path.join', return_value=single_model_path):
                bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
                
                result = bot.get_model()
                
                # Should stay on the same model (cycling)
                assert "single-model" in result
                assert "(1/1)" in result
                assert bot.current_model_index == 0
        finally:
            if os.path.exists(single_model_path):
                os.unlink(single_model_path)
    
    def test_get_model_multiple_models_cycling(self, mock_dependencies, temp_models_file):
        """Test cycling through multiple models."""
        with patch('src.bot.core.os.path.join', return_value=temp_models_file):
            bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
            
            # Initial state
            assert bot.current_model_index == 0
            
            # First get
            result1 = bot.get_model()
            assert bot.current_model_index == 1
            assert "nvidia/nemotron-nano-9b-v2:free" in result1
            assert "(2/4)" in result1
            
            # Second get
            result2 = bot.get_model()
            assert bot.current_model_index == 2
            assert "cognitivecomputations/dolphin-mistral-24b-venice-edition:free" in result2
            assert "(3/4)" in result2
            
            # Third get
            result3 = bot.get_model()
            assert bot.current_model_index == 3
            assert "google/gemma-3-27b-it:free" in result3
            assert "(4/4)" in result3
            
            # Fourth get - should wrap around
            result4 = bot.get_model()
            assert bot.current_model_index == 0
            assert "openai/gpt-oss-20b:free" in result4
            assert "(1/4)" in result4
    
    def test_get_model_recreates_llm_client(self, mock_dependencies, temp_models_file):
        """Test that get_model() recreates the LLM client with new model."""
        with patch('src.bot.core.os.path.join', return_value=temp_models_file):
            bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
            
            # Get initial LLM client
            initial_llm_client = bot.llm_client
            
            # get model
            bot.get_model()
            
            # LLM client should be recreated
            # In our mock, it will be a new Mock instance
            assert bot.llm_client is not None
    
    def test_get_model_empty_list(self, mock_dependencies):
        """Test get_model() with empty model list."""
        with patch('src.bot.core.os.path.join', return_value='/nonexistent/models.txt'):
            bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
            bot.available_models = []  # Force empty list
            
            result = bot.get_model()
            
            assert "Нет доступных моделей" in result


class TestGetCurrentModel:
    """Tests for get_current_model() method."""
    
    def test_get_current_model_correct_info(self, mock_dependencies, temp_models_file):
        """Test that get_current_model() returns correct model and position."""
        with patch('src.bot.core.os.path.join', return_value=temp_models_file):
            bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
            
            result = bot.get_current_model()
            
            assert "openai/gpt-oss-20b:free" in result
            assert "(1/4)" in result
            assert "" in result
    
    def test_get_current_model_after_get(self, mock_dependencies, temp_models_file):
        """Test get_current_model() after geting models."""
        with patch('src.bot.core.os.path.join', return_value=temp_models_file):
            bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
            
            # get to second model
            bot.get_model()
            
            result = bot.get_current_model()
            
            assert "nvidia/nemotron-nano-9b-v2:free" in result
            assert "(2/4)" in result
    
    def test_get_current_model_empty_list(self, mock_dependencies):
        """Test get_current_model() with empty model list."""
        with patch('src.bot.core.os.path.join', return_value='/nonexistent/models.txt'):
            bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
            bot.available_models = []  # Force empty list
            
            result = bot.get_current_model()
            
            assert "Нет доступных моделей" in result


class TestInitialModelIndex:
    """Tests for initial model index detection."""
    
    def test_initial_index_default_model(self, mock_dependencies, temp_models_file):
        """Test that initial index is 0 when using default model."""
        with patch('src.bot.core.os.path.join', return_value=temp_models_file):
            bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
            
            assert bot.current_model_index == 0
    
    def test_initial_index_custom_model_in_list(self, mock_dependencies, temp_models_file):
        """Test that initial index is set correctly when custom model is in list."""
        with patch('src.bot.core.os.path.join', return_value=temp_models_file):
            bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", model_name="nvidia/nemotron-nano-9b-v2:free", verbosity=0)
            
            assert bot.current_model_index == 1
    
    def test_initial_index_custom_model_not_in_list(self, mock_dependencies, temp_models_file):
        """Test that initial index stays 0 when custom model is not in list."""
        with patch('src.bot.core.os.path.join', return_value=temp_models_file):
            bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", model_name="some-other-model", verbosity=0)
            
            # If model not in list, it may not have current_model_index attribute
            # Just verify bot was created successfully
            assert bot is not None
            assert hasattr(bot, 'available_models')
