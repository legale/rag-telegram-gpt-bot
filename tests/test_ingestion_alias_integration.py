
import os
import sys
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.ingestion.pipeline import IngestionPipeline, IngestionBotAdapter
from src.storage.db import Database

# Mock config
class MockBotConfig:
    def __init__(self, profile_dir):
        self.chunk_token_min = 100
        self.chunk_token_max = 500
        self.chunk_overlap_ratio = 0.1
        self.data = {
            "embedding_model": "test_model",
            "current_model": "test_llm"
        }
        self.config_file = "config.json"

@pytest.fixture
def mock_db_url(tmp_path):
    db_path = tmp_path / "test.db"
    return f"sqlite:///{db_path}"

@pytest.fixture
def mock_pipeline(mock_db_url, tmp_path):
    with patch("src.ingestion.pipeline.BotConfig", MockBotConfig), \
         patch("src.ingestion.pipeline.create_embedding_client") as mock_create_emb, \
         patch("src.ingestion.pipeline.MessageChunker") as mock_chunker:
        
        # Mock embedding client
        mock_create_emb.return_value = Mock()
        
        pipeline = IngestionPipeline(
            db_url=mock_db_url,
            vector_db_path=str(tmp_path / "chroma"),
            profile_dir=tmp_path
        )
        return pipeline

def test_ingestion_alias_discovery_flow(mock_pipeline):
    """Test that ingestion pipeline calls alias discovery correctly."""
    
    # 1. Setup mocks
    # Mock parser to return some dummy messages
    mock_message = Mock()
    mock_message.id = "1"
    mock_message.timestamp = datetime.now()
    mock_message.sender = "Alice"
    mock_message.content = "Hello world"
    
    mock_pipeline.parser.parse_file = Mock(return_value=[mock_message])
    
    # Mock DB methods used in _run_alias_discovery
    # We need real DB interaction for populate_users usually, but here we can mock the higher level calls
    # or rely on the real DB since we provided a file-based sqlite url.
    # Let's rely on the real DB methods of the pipeline instance, but mock the LLM part.
    
    # Mock LLM Client creation
    mock_llm_client = Mock()
    mock_pipeline._get_llm_client = Mock(return_value=mock_llm_client)
    
    # Mock AliasDiscoveryService
    with patch("src.ingestion.pipeline.AliasDiscoveryService") as MockService:
        mock_service_instance = AsyncMock()
        MockService.return_value = mock_service_instance
        
        # Setup what discover_aliases returns
        mock_service_instance.discover_aliases.return_value = ["Ally", "A-Dog"]
        
        # 2. Run parse_and_store_messages (Stage 0)
        # We need to mock os.path.exists for the file check in parse_and_store_messages -> parse_file
        
        mock_pipeline.parse_and_store_messages("dummy_dump.json")
        
        # Verify user is NOT yet in DB or at least populated?
        # populate_users_from_messages is called in run_alias_discovery. 
        # But wait, senders are in messages table. Users table is populated in run_alias_discovery.
        user_before = mock_pipeline.db.get_user("Alice")
        assert user_before is None, "User should not be in users table before alias discovery"

        # 3. Run run_alias_discovery (Stage 4)
        mock_pipeline.run_alias_discovery()
        
        # 4. Verify
        
        # Check if user "Alice" is in `users` table
        user = mock_pipeline.db.get_user("Alice")
        assert user is not None, "User Alice should have been populated in DB"
        assert user.username == "Alice"
        
        # Verify AliasDiscoveryService was initialized
        MockService.assert_called_once()
        
        # Verify discover_aliases was called for Alice
        mock_service_instance.discover_aliases.assert_called_with("Alice")
        
        # Verify save_aliases was called
        mock_service_instance.save_aliases.assert_called_with("Alice", ["Ally", "A-Dog"])
        
