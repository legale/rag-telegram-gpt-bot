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

    def _validate_documents(self, texts: List[str]) -> None:
        """
        Validate input documents for embedding.
        
        Args:
            texts: List of text strings to validate
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(texts, list):
            raise ValueError("texts must be a list")
        if not texts:
            raise ValueError("texts list cannot be empty")
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"texts[{i}] must be a string, got {type(text).__name__}")
            if not text.strip():
                raise ValueError(f"texts[{i}] cannot be empty or whitespace-only")

    def _validate_query(self, text: str) -> None:
        """
        Validate input query for embedding.
        
        Args:
            text: Query text string to validate
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(text, str):
            raise ValueError(f"text must be a string, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("text cannot be empty or whitespace-only")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If input validation fails
        """
        self._validate_documents(texts)
        return self.client.get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: Query text string

        Returns:
            Embedding vector
            
        Raises:
            ValueError: If input validation fails
        """
        self._validate_query(text)
        return self.client.get_embedding(text)
