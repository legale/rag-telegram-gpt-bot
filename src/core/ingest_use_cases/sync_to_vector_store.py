"""Use case for syncing chunks with embeddings to vector store."""

from __future__ import annotations

from typing import List

from typing import Optional, List

from src.core.interfaces import ChunkStore, VectorIndex, VectorDoc
from src.lib.syslog2 import *


class SyncToVectorStore:
    """
    Use case for syncing chunks with embeddings to vector index.
    
    This use case depends only on ChunkStore and VectorIndex interfaces.
    """

    def __init__(
        self,
        chunk_store: ChunkStore,
        vector_index: VectorIndex,
    ):
        """
        Initialize use case.

        Args:
            chunk_store: ChunkStore interface implementation
            vector_index: VectorIndex interface implementation
        """
        self.chunk_store = chunk_store
        self.vector_index = vector_index

    def execute(self, chunk_ids: Optional[List[str]] = None) -> int:
        """
        Sync chunks with embeddings to vector index.

        Args:
            chunk_ids: Optional list of chunk IDs to sync (if None, syncs all chunks with embeddings)

        Returns:
            Number of chunks synced
        """
        syslog2(LOG_NOTICE, "starting sync to vector store")

        # Prepare vector data
        vector_docs = self._prepare_vector_data(chunk_ids)
        
        if not vector_docs:
            syslog2(LOG_NOTICE, "no chunks with embeddings found")
            return 0

        # Sync to vector store
        synced_count = self._sync_to_vector_store(vector_docs)
        
        syslog2(LOG_NOTICE, "sync to vector store complete", synced=synced_count)
        return synced_count
    
    def _prepare_vector_data(self, chunk_ids: Optional[List[str]]) -> List[VectorDoc]:
        """
        Prepare vector data from chunks.
        
        Args:
            chunk_ids: Optional list of chunk IDs to sync
            
        Returns:
            List of VectorDoc objects ready for sync
            
        Raises:
            NotImplementedError: If chunk_ids is None (getting all chunks not supported)
        """
        # Note: This is a simplified implementation.
        # Full implementation would need ChunkStore to support:
        # - get_chunks_with_embeddings() method
        # - Or iterate through all chunks and filter
        
        if chunk_ids is None:
            raise NotImplementedError(
                "Getting all chunks with embeddings requires extending ChunkStore interface. "
                "For now, provide explicit chunk_ids."
            )

        # Get chunks by IDs
        chunks = self.chunk_store.get_by_ids(chunk_ids)
        
        # Filter chunks with embeddings
        chunks_to_sync = [chunk for chunk in chunks if chunk.embedding is not None]
        
        if not chunks_to_sync:
            return []

        syslog2(LOG_NOTICE, "chunks to sync", total=len(chunks_to_sync))

        # Convert chunks to VectorDoc objects
        vector_docs = []
        for chunk in chunks_to_sync:
            # Prepare metadata from chunk metadata
            meta = chunk.metadata.copy() if chunk.metadata else {}
            
            vector_doc = VectorDoc(
                id=chunk.id,
                vector=chunk.embedding,
                meta=meta
            )
            vector_docs.append(vector_doc)
        
        return vector_docs
    
    def _sync_to_vector_store(self, vector_docs: List[VectorDoc]) -> int:
        """
        Sync vector documents to vector index.
        
        Args:
            vector_docs: List of VectorDoc objects to sync
            
        Returns:
            Number of documents synced
        """
        # Upsert to vector index
        self.vector_index.upsert(vector_docs)
        return len(vector_docs)

