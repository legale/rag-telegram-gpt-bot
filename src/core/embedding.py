from openai import OpenAI
from typing import List, Optional
import os

class EmbeddingClient:
    """Client for generating text embeddings using an OpenAI-compatible API (e.g., OpenRouter)."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = "text-embedding-3-small"):
        """
        Initialize the EmbeddingClient.
        
        Args:
            api_key: API key. If None, tries to read from OPENAI_API_KEY or OPENROUTER_API_KEY env var.
            base_url: Base URL for the API. Defaults to OpenRouter if not provided, or OpenAI if specified.
                      Actually, let's default to OpenRouter's URL if we want to be "autonomous" from OpenAI defaults,
                      but standard OpenAI client defaults to openai.com. 
                      User said "API models will be taken from openrouter".
            model: The model to use for embeddings.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        self.model = model
            
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of strings to embed.
            
        Returns:
            List of embedding vectors (lists of floats).
        """
        cleaned_texts = [text.replace("\n", " ") for text in texts]
        
        # OpenRouter (and others) might have different response structures or limitations.
        # Assuming standard OpenAI format compatibility.
        response = self.client.embeddings.create(
            input=cleaned_texts,
            model=self.model
        )
        
        return [data.embedding for data in response.data]

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: String to embed.
            
        Returns:
            Embedding vector.
        """
        return self.get_embeddings([text])[0]
