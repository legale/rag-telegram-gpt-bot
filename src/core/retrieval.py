from typing import List, Dict, Optional
from src.storage.vector_store import VectorStore
from src.storage.db import Database, ChunkModel
from src.core.embedding import EmbeddingClient

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
            print(f"\n[Retrieval] Query: '{query}'")

        # 1. Query Vector Store using raw text
        # ChromaDB will use its default embedding function (all-MiniLM-L6-v2) which matches
        # what was used during ingestion (since we didn't provide embeddings there either).
        results = self.vector_store.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances', 'embeddings'] if self.verbosity >= 2 else ['documents', 'metadatas', 'distances']
        )
        
        if self.verbosity >= 2:
             if 'embeddings' in results and results['embeddings']:
                 # Just show the first few dimensions of the query embedding if available (it's not returned for query_texts usually)
                 # But results['embeddings'] are the embeddings of the retrieved docs.
                 pass
             print(f"[Retrieval] Found {len(results['ids'][0]) if results['ids'] else 0} potential matches.")

        # 2. Process results and fetch full text from SQL DB
        retrieved_chunks = []
        
        if not results['ids'] or not results['ids'][0]:
            return []
            
        ids = results['ids'][0]
        distances = results['distances'][0] if 'distances' in results else []
        metadatas = results['metadatas'][0] if 'metadatas' in results else []
        
        if self.verbosity >= 2:
            print("[Retrieval] Candidates:")
            for i, (chunk_id, dist) in enumerate(zip(ids, distances)):
                print(f"  - ID: {chunk_id}, Distance: {dist:.4f}")

        session = self.db.get_session()
        
        for i, chunk_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else 0
            
            # Simple threshold check (assuming L2 distance, lower is better)
            # Adjust this logic based on actual metric used (Cosine vs L2)
            # For now, we'll just return everything found by KNN
            
            db_chunk = session.query(ChunkModel).filter_by(id=chunk_id).first()
            if db_chunk:
                retrieved_chunks.append({
                    "id": chunk_id,
                    "text": db_chunk.text,
                    "metadata": db_chunk.metadata_json,
                    "score": distance
                })
                
        session.close()
        return retrieved_chunks
