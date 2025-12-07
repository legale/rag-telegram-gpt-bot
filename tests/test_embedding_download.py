
import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
from src.core.embedding import LocalEmbeddingClient, LOG_WARNING, LOG_INFO, LOG_ERR

class TestLocalEmbeddingClient(unittest.TestCase):
    def setUp(self):
        # Clean up any existing environment variables
        if "HF_HUB_OFFLINE" in os.environ:
            del os.environ["HF_HUB_OFFLINE"]
            
        self.model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        self.client = LocalEmbeddingClient(model=self.model_name)

    @patch("src.core.embedding.SentenceTransformer")
    @patch("src.core.embedding.syslog2")
    @patch("subprocess.check_call")
    def test_offline_load_success(self, mock_subprocess, mock_syslog, mock_transformer):
        # Simulate successful offline load
        os.environ["HF_HUB_OFFLINE"] = "1"
        
        # Access model property to trigger load
        model = self.client.model
        
        # Verify attempt to load offline
        mock_transformer.assert_called_with(self.model_name)
        # Should not trigger download
        mock_subprocess.assert_not_called()
        self.assertIsNotNone(model)

    @patch("src.core.embedding.SentenceTransformer")
    @patch("src.core.embedding.syslog2")
    @patch("subprocess.check_call")
    def test_offline_load_fail_then_download(self, mock_subprocess, mock_syslog, mock_transformer):
        # Simulate offline load failure first, then success after download
        os.environ["HF_HUB_OFFLINE"] = "1"
        
        # Side effect: first call raises Exception, second call returns mock
        mock_transformer.side_effect = [Exception("Offline load failed"), MagicMock()]
        
        # Access model property
        model = self.client.model
        
        # Verify download was triggered
        mock_subprocess.assert_called_once()
        
        # Verify it retried loading the model (offline)
        self.assertEqual(mock_transformer.call_count, 2)

    def test_get_embedding(self):
        # Mock the model behavior
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        self.client._model = mock_model
        
        emb = self.client.get_embedding("hello world")
        
        self.assertEqual(emb, [0.1, 0.2, 0.3])
        mock_model.encode.assert_called_with(["hello world"], show_progress_bar=False)

if __name__ == "__main__":
    unittest.main()
