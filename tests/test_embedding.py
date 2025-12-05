import pytest
from unittest.mock import MagicMock, patch
from src.core.embedding import EmbeddingClient

@pytest.fixture
def mock_openai_client():
    with patch("src.core.embedding.OpenAI") as mock_openai:
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance
        yield mock_instance

def test_embedding_client_init(mock_openai_client):
    client = EmbeddingClient(api_key="test-key")
    assert client.model == "text-embedding-3-small"
    mock_openai_client.assert_not_called() # OpenAI init is called inside __init__, wait.
    # Actually OpenAI() is called.
    
def test_get_embeddings(mock_openai_client):
    client = EmbeddingClient(api_key="test-key")
    
    # Mock response
    mock_response = MagicMock()
    mock_data_item = MagicMock()
    mock_data_item.embedding = [0.1, 0.2, 0.3]
    mock_response.data = [mock_data_item]
    
    client.client.embeddings.create.return_value = mock_response
    
    embeddings = client.get_embeddings(["test text"])
    
    assert len(embeddings) == 1
    assert embeddings[0] == [0.1, 0.2, 0.3]
    client.client.embeddings.create.assert_called_once()
    
def test_get_embedding_single(mock_openai_client):
    client = EmbeddingClient(api_key="test-key")
    
    # Mock response
    mock_response = MagicMock()
    mock_data_item = MagicMock()
    mock_data_item.embedding = [0.1, 0.2, 0.3]
    mock_response.data = [mock_data_item]
    
    client.client.embeddings.create.return_value = mock_response
    
    embedding = client.get_embedding("test text")
    
    assert embedding == [0.1, 0.2, 0.3]
