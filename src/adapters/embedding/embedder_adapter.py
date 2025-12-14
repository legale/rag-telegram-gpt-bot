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
        """
        return self.client.get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: Query text string

        Returns:
            Embedding vector
        """
        return self.client.get_embedding(text)

