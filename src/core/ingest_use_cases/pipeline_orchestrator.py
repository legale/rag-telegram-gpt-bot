"""Orchestrator use case that coordinates ingestion pipeline stages."""

from __future__ import annotations

from typing import Optional, Dict, Any

from .ingest_messages import IngestMessages
from .process_chunks import ProcessChunks
from .generate_embeddings import GenerateEmbeddings
from .sync_to_vector_store import SyncToVectorStore
from src.lib.syslog2 import *


class PipelineOrchestrator:
    """
    Orchestrator that coordinates multiple ingestion use cases.
    
    This replaces the monolithic IngestionPipeline.run_all() method
    by composing simpler use cases.
    """

    def __init__(
        self,
        ingest_messages: IngestMessages,
        process_chunks: ProcessChunks,
        generate_embeddings: GenerateEmbeddings,
        sync_to_vector_store: SyncToVectorStore,
    ):
        """
        Initialize orchestrator with use cases.

        Args:
            ingest_messages: IngestMessages use case
            process_chunks: ProcessChunks use case
            generate_embeddings: GenerateEmbeddings use case
            sync_to_vector_store: SyncToVectorStore use case
        """
        self.ingest_messages = ingest_messages
        self.process_chunks = process_chunks
        self.generate_embeddings = generate_embeddings
        self.sync_to_vector_store = sync_to_vector_store

    def run_all(
        self,
        file_path: str,
        model: Optional[str] = None,
        batch_size: int = 128,
        **clustering_params
    ) -> Dict[str, Any]:
        """
        Run all ingestion stages in sequence.

        Args:
            file_path: Path to chat dump file
            model: Optional embedding model override
            batch_size: Batch size for embedding generation
            **clustering_params: Clustering parameters (for future stages)

        Returns:
            Dictionary with statistics about the ingestion process
        """
        stats = {}

        # Stage 0: Parse and store messages
        syslog2(LOG_NOTICE, "running stage0: parse and store messages")
        messages_saved = self.ingest_messages.execute(file_path)
        stats["stage0_messages_saved"] = messages_saved

        # Stage 1: Create and store chunks
        syslog2(LOG_NOTICE, "running stage1: create and store chunks")
        chunks_saved = self.process_chunks.execute()
        stats["stage1_chunks_saved"] = chunks_saved

        # Stage 2: Generate embeddings
        # Note: This requires getting chunk IDs first, which needs ChunkStore extension
        # For now, this is a placeholder showing the structure
        syslog2(LOG_NOTICE, "running stage2: generate embeddings for chunks (save to SQLite)")
        # stats["stage2_embeddings_generated"] = self.generate_embeddings.execute(...)

        # Stage 3: Sync chunks to vector database
        # Note: This also requires getting chunk IDs first
        syslog2(LOG_NOTICE, "running stage3: sync chunks to vector database")
        # stats["stage3_chunks_synced"] = self.sync_to_vector_store.execute(...)

        # Stages 4-9 (clustering and topic naming) are not yet migrated to use cases
        # They remain in IngestionPipeline for now

        syslog2(LOG_NOTICE, "pipeline orchestrator complete", **stats)
        return stats

