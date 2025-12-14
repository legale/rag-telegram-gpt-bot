"""Bootstrap module for dependency injection and application composition."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.storage.db import Database
from src.storage.vector_store import VectorStore
from src.core.embedding import EmbeddingClient, LocalEmbeddingClient, create_embedding_client

from src.adapters.persistence import SqliteMessageStore, SqliteChunkStore
from src.adapters.vector import ChromaVectorIndex
from src.adapters.embedding import EmbedderAdapter
from src.core.use_cases.search import HybridSearch


def create_hybrid_search(
    db_url: str,
    vector_db_path: str,
    embedding_client: Optional[EmbeddingClient | LocalEmbeddingClient] = None,
    profile_dir: Optional[str | Path] = None,
) -> HybridSearch:
    """
    Create and configure HybridSearch use case with all dependencies.

    This function serves as the composition root, creating all adapters and
    wiring them together to create a fully configured HybridSearch instance.

    Args:
        db_url: SQLite database URL (e.g., "sqlite:///path/to/db")
        vector_db_path: Path to ChromaDB persistent directory
        embedding_client: Optional pre-configured embedding client.
                         If None, creates default EmbeddingClient.
        profile_dir: Optional profile directory for loading embedding config

    Returns:
        Configured HybridSearch use case instance
    """
    # Create infrastructure instances
    database = Database(db_url)
    vector_store = VectorStore(
        persist_directory=vector_db_path,
        embedding_client=embedding_client  # Pass through if provided
    )

    # Create or use provided embedding client
    if embedding_client is None:
        # Try to load from profile config if available
        if profile_dir:
            profile_path = Path(profile_dir)
            if profile_path.exists():
                try:
                    from src.bot.config import BotConfig
                    config = BotConfig(profile_path)
                    embedding_client = create_embedding_client(
                        generator=config.embedding_generator,
                        model=config.embedding_model
                    )
                except Exception:
                    # Fall back to default if config load fails
                    embedding_client = EmbeddingClient()
        else:
            embedding_client = EmbeddingClient()

    # Create adapters
    message_store = SqliteMessageStore(database)
    chunk_store = SqliteChunkStore(database)
    vector_index = ChromaVectorIndex(vector_store)
    embedder = EmbedderAdapter(embedding_client)

    # Create and return use case
    hybrid_search = HybridSearch(
        embedder=embedder,
        vector_index=vector_index,
        chunk_store=chunk_store,
        message_store=message_store,
    )

    return hybrid_search

