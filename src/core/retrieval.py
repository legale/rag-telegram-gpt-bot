# src/core/retrieval.py
from typing import List, Dict, Optional
from src.storage.vector_store import VectorStore
from src.storage.db import Database, ChunkModel
from src.core.embedding import EmbeddingClient
from src.core.syslog2 import *


class RetrievalService:
    """Service to retrieve relevant context for a query."""
    
    def __init__(self, vector_store: VectorStore, db: Database, embedding_client: EmbeddingClient, verbosity: int = 0):
        self.vector_store = vector_store
        self.db = db
        self.embedding_client = embedding_client
        self.verbosity = verbosity
        
    def retrieve(self, query: str, n_results: int = 5, score_threshold: float = 0.5) -> List[Dict]:
        """
        Retrieve relevant chunks for a given query.
        
        Args:
            query: User query string.
            n_results: Number of results to return.
            score_threshold: Minimum similarity score.
                             
        Returns:
            List of dictionaries containing chunk text and metadata.
        """
        if self.verbosity >= 1:
            syslog2(LOG_DEBUG, "retrieval query", query=query)

        if self.verbosity >= 2:
            syslog2(LOG_DEBUG, "computing query embedding")
        
        # 1. embed query with the same embedding client (same dim as stored vectors)
        query_embs = self.embedding_client.get_embeddings([query])

        if self.verbosity >= 2:
            syslog2(LOG_DEBUG, "searching vector store", collection=self.vector_store.collection.name)

        # 2. query vector store using precomputed query embedding
        results = self.vector_store.collection.query(
            query_embeddings=query_embs,
            n_results=n_results,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )
        
        if self.verbosity >= 2:
            count = len(results['ids'][0]) if results['ids'] else 0
            syslog2(LOG_DEBUG, "retrieval candidates found", count=count)

        retrieved_chunks: List[Dict] = []
        
        if not results["ids"] or not results["ids"][0]:
            return []
            
        ids = results["ids"][0]
        distances = results["distances"][0] if "distances" in results else []
        metadatas = results["metadatas"][0] if "metadatas" in results else []
        
        if self.verbosity >= 2:
            syslog2(LOG_DEBUG, "retrieval candidates detail", ids=ids, distances=distances)

        if self.verbosity >= 2:
            syslog2(LOG_DEBUG, "fetching full text from sqlite")

        session = self.db.get_session()
        
        try:
            for i, chunk_id in enumerate(ids):
                distance = distances[i] if i < len(distances) else 0

                db_chunk = session.query(ChunkModel).filter_by(id=chunk_id).first()
                if db_chunk:
                    retrieved_chunks.append(
                        {
                            "id": chunk_id,
                            "text": db_chunk.text,
                            "metadata": db_chunk.metadata_json,
                            "score": distance,
                        }
                    )
        finally:
            session.close()

        return retrieved_chunks
