"""Bootstrap module for dependency injection and application composition."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.storage.db import Database
from src.storage.vector_store import VectorStore
from src.core.embedding import EmbeddingClient, LocalEmbeddingClient, create_embedding_client

from src.adapters.persistence import SqliteMessageStore, SqliteChunkStore, SqliteFTSIndex
from src.adapters.vector import ChromaVectorIndex, ChromaTopicProvider
from src.adapters.embedding import EmbedderAdapter
from src.adapters.llm.llm_adapter import LLMAdapter
from src.core.use_cases.search import HybridSearch
from src.core.use_cases.hybrid_retrieval import HybridRetrievalService
from src.core.retrieval import RetrievalService
from src.core.llm import LLMClient
from src.lib.syslog2 import *


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


def create_retrieval_service(
    db_url: str,
    vector_db_path: str,
    embedding_client: Optional[EmbeddingClient | LocalEmbeddingClient] = None,
    llm_client: Optional[LLMClient] = None,
    profile_dir: Optional[str | Path] = None,
    log_level: int = 4,  # LOG_WARNING
    use_topic_retrieval: bool = True,
    topic_retrieval_weight: float = 0.3,
    search_mode: str = "two_stage",
    l2_top_k: int = 5,
    chunk_top_k: int = 50,
    debug_rag: bool = False,
    rag_ntop: int = 0
) -> RetrievalService:
    """
    Create and configure RetrievalService with all dependencies.

    This function serves as the composition root, creating all adapters and
    wiring them together to create a fully configured RetrievalService instance.

    Args:
        db_url: SQLite database URL (e.g., "sqlite:///path/to/db")
        vector_db_path: Path to ChromaDB persistent directory
        embedding_client: Optional pre-configured embedding client.
                         If None, creates default EmbeddingClient.
        llm_client: Optional LLM client for query rephrasing
        profile_dir: Optional profile directory for loading embedding config
        log_level: Logging level (LOG_WARNING=4 by default)
        use_topic_retrieval: Enable hierarchical topic-based retrieval
        topic_retrieval_weight: Weight for topic-based results (0.0-1.0)
        search_mode: "two_stage" (L2→L1) or "direct" (direct chunk search)
        l2_top_k: Number of L2 topics to select in two-stage search
        chunk_top_k: Number of chunks to return in two-stage search
        debug_rag: Enable detailed rag debug logging
        rag_ntop: Number of top results to limit (if > 0, otherwise no limit)

    Returns:
        Configured RetrievalService instance
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
    topic_provider = ChromaTopicProvider(vector_store)
    
    # Create LLM adapter if llm_client is provided
    llm = None
    if llm_client:
        llm = LLMAdapter(llm_client)

    # Create and return RetrievalService
    retrieval_service = RetrievalService(
        vector_index=vector_index,
        chunk_store=chunk_store,
        message_store=message_store,
        embedder=embedder,
        topic_provider=topic_provider,
        llm=llm,
        log_level=log_level,
        use_topic_retrieval=use_topic_retrieval,
        topic_retrieval_weight=topic_retrieval_weight,
        search_mode=search_mode,
        l2_top_k=l2_top_k,
        chunk_top_k=chunk_top_k,
        debug_rag=debug_rag,
        rag_ntop=rag_ntop
    )

    return retrieval_service


def create_hybrid_retrieval(
    db_url: str,
    vector_db_path: str,
    embedding_client: Optional[EmbeddingClient | LocalEmbeddingClient] = None,
    profile_dir: Optional[str | Path] = None,
    log_level: int = LOG_WARNING,
    fts_only: bool = False,
) -> HybridRetrievalService:
    """
    Create and configure HybridRetrievalService with all dependencies.

    This function creates FTS5 + vector hybrid retrieval service.

    Args:
        db_url: SQLite database URL (e.g., "sqlite:///path/to/db")
        vector_db_path: Path to ChromaDB persistent directory
        embedding_client: Optional pre-configured embedding client.
                         If None, creates default EmbeddingClient.
        profile_dir: Optional profile directory for loading embedding config
        log_level: Logging level (LOG_WARNING=4 by default)
        fts_only: If True, skip vector reranking and use FTS-only mode

    Returns:
        Configured HybridRetrievalService instance
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
    fts_index = SqliteFTSIndex(database)

    # Create and return HybridRetrievalService
    hybrid_retrieval = HybridRetrievalService(
        fts_index=fts_index,
        vector_index=vector_index,
        embedder=embedder,
        chunk_store=chunk_store,
        message_store=message_store,
        log_level=log_level,
        fts_only=fts_only,
    )

    return hybrid_retrieval

