"""Adapter wrapping EmbeddingClient to implement Embedder protocol."""

from __future__ import annotations

from typing import List

from src.core.interfaces import Embedder
from src.core.embedding import EmbeddingClient, LocalEmbeddingClient


class EmbedderAdapter:
    """Adapter that wraps EmbeddingClient to implement Embedder protocol."""

    def __init__(self, embedding_client: EmbeddingClient | LocalEmbeddingClient):
        """
        Initialize adapter with an EmbeddingClient instance.

        Args:
            embedding_client: EmbeddingClient or LocalEmbeddingClient instance
        """
        self.client = embedding_client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If texts is empty or contains invalid values
        """
        if not texts:
            raise ValueError("texts list cannot be empty")
        
        # Validate that all texts are non-empty strings
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"texts[{i}] must be a string, got {type(text).__name__}")
            if not text.strip():
                raise ValueError(f"texts[{i}] cannot be empty or whitespace-only")
        
        return self.client.get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: Query text string

        Returns:
            Embedding vector
            
        Raises:
            ValueError: If text is empty or invalid
        """
        if not isinstance(text, str):
            raise ValueError(f"text must be a string, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("text cannot be empty or whitespace-only")
        
        return self.client.get_embedding(text)

