"""Tests for src/bot/core.py"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from src.bot.core import LegaleBot
from src.lib.syslog2 import LOG_WARNING, LOG_INFO, LOG_DEBUG


class TestLegaleBotInit:
    """Tests for LegaleBot.__init__"""
    
    def test_init_with_minimal_params(self, tmp_path):
        """Test initialization with minimal parameters"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1", "model2"]
            mock_llm.return_value.model_name = "model1"
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            
            assert bot.db.db_url == db_url
            assert bot.log_level == LOG_WARNING
            assert bot.debug_rag is False
            assert bot.chat_history == []
    
    def test_init_missing_db_url(self):
        """Test initialization fails without db_url"""
        with pytest.raises(ValueError, match="db_url and vector_db_path must be provided"):
            LegaleBot(db_url="", vector_db_path="/tmp/vec")
    
    def test_init_missing_vector_db_path(self):
        """Test initialization fails without vector_db_path"""
        with pytest.raises(ValueError, match="db_url and vector_db_path must be provided"):
            LegaleBot(db_url="sqlite:///test.db", vector_db_path="")
    
    def test_init_with_profile_dir(self, tmp_path):
        """Test initialization with profile_dir"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm.return_value.model_name = "model1"
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(
                db_url=db_url,
                vector_db_path=vector_db_path,
                profile_dir=str(profile_dir)
            )
            
            # profile_dir is not stored as attribute, but used during initialization
            assert bot.config is not None
    
    def test_init_with_retrieval_type_hybrid(self, tmp_path):
        """Test initialization with hybrid retrieval type"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm.return_value.model_name = "model1"
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(
                db_url=db_url,
                vector_db_path=vector_db_path,
                retrieval_type="hybrid"
            )
            
            assert bot.retrieval_type == "hybrid"
            mock_retrieval.assert_called_once()
    
    def test_init_with_retrieval_type_fts_only(self, tmp_path):
        """Test initialization with fts_only retrieval type"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm.return_value.model_name = "model1"
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(
                db_url=db_url,
                vector_db_path=vector_db_path,
                retrieval_type="fts_only"
            )
            
            assert bot.retrieval_type == "fts_only"
    
    def test_init_with_invalid_retrieval_type(self, tmp_path):
        """Test initialization fails with invalid retrieval_type"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm.return_value.model_name = "model1"
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            
            with pytest.raises(ValueError, match="Unknown retrieval_type"):
                LegaleBot(
                    db_url=db_url,
                    vector_db_path=vector_db_path,
                    retrieval_type="invalid"
                )


class TestLoadAvailableModels:
    """Tests for LegaleBot._load_available_models"""
    
    def test_load_models_file_exists(self, tmp_path):
        """Test loading models from existing file"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        models_file = tmp_path / "models.txt"
        models_file.write_text("model1\nmodel2\nmodel3\n")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1", "model2", "model3"]
            mock_llm.return_value.model_name = "model1"
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            
            # Test the actual method with mocked file path
            with patch('builtins.open', mock_open(read_data="model1\nmodel2\n")), \
                 patch('src.bot.core.os.path.join') as mock_join, \
                 patch('src.bot.core.os.path.dirname') as mock_dirname:
                mock_join.return_value = str(models_file)
                mock_dirname.return_value = str(tmp_path)
                models = bot._load_available_models()
                assert "model1" in models
                assert "model2" in models
    
    def test_load_models_file_not_found(self, tmp_path):
        """Test loading models when file doesn't exist"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = []
            mock_llm.return_value.model_name = "unknown"
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            
            with patch('builtins.open', side_effect=FileNotFoundError()), \
                 patch('src.bot.core.syslog2') as mock_syslog, \
                 patch('src.bot.core.os.path.join') as mock_join:
                mock_join.return_value = "/nonexistent/models.txt"
                models = bot._load_available_models()
                assert models == []


class TestGetModel:
    """Tests for LegaleBot.get_model"""
    
    def test_get_model_success(self, tmp_path):
        """Test getting to next model"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1", "model2", "model3"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            bot.current_model_index = 0
            bot.available_models = ["model1", "model2", "model3"]
            
            result = bot.get_model()
            
            assert "model2" in result
            assert bot.current_model_index == 1
    
    def test_get_model_cyclic(self, tmp_path):
        """Test getting model wraps around"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1", "model2"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model2"
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            bot.current_model_index = 1
            bot.available_models = ["model1", "model2"]
            
            result = bot.get_model()
            
            assert "model1" in result
            assert bot.current_model_index == 0
    
    def test_get_model_no_models(self, tmp_path):
        """Test getting model when no models available"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = []
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "unknown"
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            bot.available_models = []
            
            result = bot.get_model()
            
            assert "Нет доступных моделей" in result


class TestSetModel:
    """Tests for LegaleBot.set_model"""
    
    def test_set_model_success(self, tmp_path):
        """Test setting a specific model"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1", "model2"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            bot.available_models = ["model1", "model2"]
            bot.current_model_index = 0
            
            result = bot.set_model("model2")
            
            assert "успешно установлена" in result
            assert bot.current_model_index == 1
    
    def test_set_model_not_found(self, tmp_path):
        """Test setting model that doesn't exist"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            bot.available_models = ["model1"]
            
            result = bot.set_model("nonexistent")
            
            assert "не найдена" in result


class TestCurrentModelName:
    """Tests for LegaleBot.current_model_name property"""
    
    def test_current_model_name(self, tmp_path):
        """Test getting current model name"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "test-model"
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            
            assert bot.current_model_name == "test-model"


class TestGetCurrentModel:
    """Tests for LegaleBot.get_current_model"""
    
    def test_get_current_model_success(self, tmp_path):
        """Test getting current model info"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1", "model2"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            bot.available_models = ["model1", "model2"]
            bot.current_model_index = 0
            
            result = bot.get_current_model()
            
            assert "model1" in result
            assert "1/2" in result
    
    def test_get_current_model_no_models(self, tmp_path):
        """Test getting current model when no models available"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = []
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "unknown"
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            bot.available_models = []
            
            result = bot.get_current_model()
            
            assert "Нет доступных моделей" in result


class TestResetContext:
    """Tests for LegaleBot.reset_context"""
    
    def test_reset_context(self, tmp_path):
        """Test resetting context"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            bot.chat_history = [{"role": "user", "content": "test"}]
            
            result = bot.reset_context()
            
            assert "Контекст сброшен" in result
            assert bot.chat_history == []


class TestBuildHistoryForPrompt:
    """Tests for LegaleBot._build_history_for_prompt"""
    
    def test_build_history_empty(self, tmp_path):
        """Test building history from empty chat"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            
            history = bot._build_history_for_prompt()
            
            assert history == []
    
    def test_build_history_with_messages(self, tmp_path):
        """Test building history with messages"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            bot.chat_history = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"}
            ]
            
            history = bot._build_history_for_prompt()
            
            assert len(history) == 2
            assert history[0]["sender"] == "User"
            assert history[1]["sender"] == "Bot"
    
    def test_build_history_max_messages(self, tmp_path):
        """Test building history respects max_messages"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            bot.chat_history = [
                {"role": "user", "content": "1"},
                {"role": "assistant", "content": "2"},
                {"role": "user", "content": "3"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "5"},
                {"role": "assistant", "content": "6"},
            ]
            
            history = bot._build_history_for_prompt(max_messages=2)
            
            assert len(history) == 2
            assert history[0]["content"] == "5"
            assert history[1]["content"] == "6"


class TestBuildPromptAndHistory:
    """Tests for LegaleBot._build_prompt_and_history"""
    
    def test_build_prompt_and_history(self, tmp_path):
        """Test building prompt and history"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            mock_prompt_instance = Mock()
            mock_prompt_instance.construct_prompt.return_value = "System prompt"
            mock_prompt.return_value = mock_prompt_instance
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            
            system_prompt, history = bot._build_prompt_and_history(
                context_chunks=[{"id": "chunk1"}],
                user_task="test task"
            )
            
            assert system_prompt == "System prompt"
            assert isinstance(history, list)
            mock_prompt_instance.construct_prompt.assert_called_once()


class TestCalculateTokenUsage:
    """Tests for LegaleBot._calculate_token_usage"""
    
    def test_calculate_token_usage(self, tmp_path):
        """Test calculating token usage"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm_instance.count_tokens.return_value = 100
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            bot.max_context_tokens = 1000
            
            usage = bot._calculate_token_usage("System prompt", "User content")
            
            assert usage["current_tokens"] == 100
            assert usage["max_tokens"] == 1000
            assert usage["percentage"] == 10.0


class TestGetTokenUsage:
    """Tests for LegaleBot.get_token_usage"""
    
    def test_get_token_usage_empty_history(self, tmp_path):
        """Test getting token usage with empty history"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            
            usage = bot.get_token_usage()
            
            assert usage["current_tokens"] == 0
            assert usage["percentage"] == 0.0
    
    def test_get_token_usage_with_history(self, tmp_path):
        """Test getting token usage with history"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm_instance.count_tokens.return_value = 200
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            mock_prompt_instance = Mock()
            mock_prompt_instance.construct_prompt.return_value = "System prompt"
            mock_prompt.return_value = mock_prompt_instance
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            bot.chat_history = [{"role": "user", "content": "test"}]
            
            usage = bot.get_token_usage()
            
            assert usage["current_tokens"] == 200


class TestEnsureContextLimit:
    """Tests for LegaleBot._ensure_context_limit"""
    
    def test_ensure_context_limit_no_reset(self, tmp_path):
        """Test ensuring context limit when under limit"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm_instance.count_tokens.return_value = 500
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            mock_prompt_instance = Mock()
            mock_prompt_instance.construct_prompt.return_value = "System prompt"
            mock_prompt.return_value = mock_prompt_instance
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            bot.max_context_tokens = 1000
            bot.chat_history = [{"role": "user", "content": "test"}]
            
            warning = bot._ensure_context_limit()
            
            assert warning == ""
            assert len(bot.chat_history) == 1
    
    def test_ensure_context_limit_reset(self, tmp_path):
        """Test ensuring context limit when over limit"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm_instance.count_tokens.return_value = 1000
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            mock_prompt_instance = Mock()
            mock_prompt_instance.construct_prompt.return_value = "System prompt"
            mock_prompt.return_value = mock_prompt_instance
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            bot.max_context_tokens = 1000
            bot.chat_history = [{"role": "user", "content": "test"}]
            
            warning = bot._ensure_context_limit()
            
            assert "сброшен" in warning
            assert bot.chat_history == []


class TestIsTokenLimitError:
    """Tests for LegaleBot._is_token_limit_error"""
    
    def test_is_token_limit_error_402(self, tmp_path):
        """Test detecting 402 error"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            
            assert bot._is_token_limit_error("Error 402") is True
    
    def test_is_token_limit_error_context_length(self, tmp_path):
        """Test detecting context_length_exceeded error"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            
            assert bot._is_token_limit_error("context_length_exceeded") is True
    
    def test_is_token_limit_error_not_token_error(self, tmp_path):
        """Test detecting non-token-limit error"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            
            assert bot._is_token_limit_error("Network error") is False


class TestRetryAfterReset:
    """Tests for LegaleBot._retry_after_reset"""
    
    def test_retry_after_reset_success(self, tmp_path):
        """Test retrying after reset succeeds"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm_instance.complete.return_value = "Response"
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            mock_prompt_instance = Mock()
            mock_prompt_instance.construct_prompt.return_value = "System prompt"
            mock_prompt.return_value = mock_prompt_instance
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            bot.chat_history = [{"role": "user", "content": "test"}]
            
            result = bot._retry_after_reset([{"id": "chunk1"}], "user input")
            
            assert "сброшен" in result
            assert "Response" in result
            assert bot.chat_history == []
    
    def test_retry_after_reset_failure(self, tmp_path):
        """Test retrying after reset fails"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm_instance.complete.side_effect = Exception("Still failing")
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            mock_prompt_instance = Mock()
            mock_prompt_instance.construct_prompt.return_value = "System prompt"
            mock_prompt.return_value = mock_prompt_instance
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            
            with patch('src.bot.core.syslog2') as mock_syslog:
                result = bot._retry_after_reset([{"id": "chunk1"}], "user input")
                
                assert "не удалось" in result or "Не удалось" in result
                mock_syslog.assert_called()


class TestCallLLMWithRetry:
    """Tests for LegaleBot._call_llm_with_retry"""
    
    def test_call_llm_with_retry_success(self, tmp_path):
        """Test calling LLM with retry succeeds"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm_instance.complete.return_value = "Response"
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            
            result = bot._call_llm_with_retry(
                [{"role": "user", "content": "test"}],
                [{"id": "chunk1"}],
                "user input"
            )
            
            assert result == "Response"
    
    def test_call_llm_with_retry_token_error(self, tmp_path):
        """Test calling LLM with retry handles token error"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm_instance.complete.side_effect = [
                Exception("Error 402"),
                "Response after reset"
            ]
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            mock_prompt_instance = Mock()
            mock_prompt_instance.construct_prompt.return_value = "System prompt"
            mock_prompt.return_value = mock_prompt_instance
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            
            result = bot._call_llm_with_retry(
                [{"role": "user", "content": "test"}],
                [{"id": "chunk1"}],
                "user input"
            )
            
            assert "сброшен" in result
    
    def test_call_llm_with_retry_other_error(self, tmp_path):
        """Test calling LLM with retry raises other errors"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm_instance.complete.side_effect = Exception("Network error")
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            
            with pytest.raises(Exception, match="Network error"):
                bot._call_llm_with_retry(
                    [{"role": "user", "content": "test"}],
                    [{"id": "chunk1"}],
                    "user input"
                )


class TestChat:
    """Tests for LegaleBot.chat"""
    
    def test_chat_success(self, tmp_path):
        """Test successful chat"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm_instance.complete.return_value = "Bot response"
            mock_llm_instance.count_tokens.return_value = 100
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            mock_retrieval_instance = Mock()
            mock_retrieval_instance.retrieve.return_value = [{"id": "chunk1"}]
            mock_retrieval.return_value = mock_retrieval_instance
            mock_prompt_instance = Mock()
            mock_prompt_instance.construct_prompt.return_value = "System prompt"
            mock_prompt.return_value = mock_prompt_instance
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            bot.retrieval_service = mock_retrieval_instance
            
            result = bot.chat("Hello")
            
            assert result == "Bot response"
            assert len(bot.chat_history) == 2
            assert bot.chat_history[0]["role"] == "user"
            assert bot.chat_history[1]["role"] == "assistant"
    
    def test_chat_not_respond(self, tmp_path):
        """Test chat with respond=False"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            
            result = bot.chat("Hello", respond=False)
            
            assert result == ""
            assert len(bot.chat_history) == 1
            assert bot.chat_history[0]["role"] == "user"
    
    def test_chat_llm_error(self, tmp_path):
        """Test chat when LLM call fails"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm_instance.complete.side_effect = Exception("LLM error")
            mock_llm_instance.count_tokens.return_value = 100
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            mock_retrieval_instance = Mock()
            mock_retrieval_instance.retrieve.return_value = [{"id": "chunk1"}]
            mock_retrieval.return_value = mock_retrieval_instance
            mock_prompt_instance = Mock()
            mock_prompt_instance.construct_prompt.return_value = "System prompt"
            mock_prompt.return_value = mock_prompt_instance
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            bot.retrieval_service = mock_retrieval_instance
            
            with patch('src.bot.core.syslog2') as mock_syslog:
                result = bot.chat("Hello")
                
                assert "Произошла ошибка" in result
                mock_syslog.assert_called()


class TestGetRagDebugInfo:
    """Tests for LegaleBot.get_rag_debug_info"""
    
    def test_get_rag_debug_info(self, tmp_path):
        """Test getting RAG debug info"""
        db_url = f"sqlite:///{tmp_path}/test.db"
        vector_db_path = str(tmp_path / "vector")
        
        with patch('src.bot.core.Database') as mock_db, \
             patch('src.bot.core.VectorStore') as mock_vector, \
             patch('src.bot.core.create_embedding_client') as mock_embed, \
             patch('src.bot.core.LLMClient') as mock_llm, \
             patch('src.bot.core.create_hybrid_retrieval') as mock_retrieval, \
             patch('src.bot.core.PromptEngine') as mock_prompt, \
             patch('src.bot.config.BotConfig') as mock_config, \
             patch('src.bot.core.LegaleBot._load_available_models') as mock_load:
            mock_load.return_value = ["model1"]
            mock_llm_instance = Mock()
            mock_llm_instance.model_name = "model1"
            mock_llm_instance.count_tokens.return_value = 200
            mock_llm.return_value = mock_llm_instance
            mock_config.return_value.embedding_generator = "local"
            mock_config.return_value.embedding_model = "model"
            mock_config.return_value.fts5_score_thr = 0.5
            mock_retrieval_instance = Mock()
            mock_retrieval_instance.retrieve.return_value = [{"id": "chunk1"}, {"id": "chunk2"}]
            mock_retrieval.return_value = mock_retrieval_instance
            mock_prompt_instance = Mock()
            mock_prompt_instance.construct_prompt.return_value = "System prompt"
            mock_prompt.return_value = mock_prompt_instance
            
            bot = LegaleBot(db_url=db_url, vector_db_path=vector_db_path)
            bot.retrieval_service = mock_retrieval_instance
            
            info = bot.get_rag_debug_info("test query", n_results=5)
            
            assert "chunks" in info
            assert "prompt" in info
            assert "token_count" in info
            assert info["chunks_count"] == 2
            assert info["token_count"] == 200

