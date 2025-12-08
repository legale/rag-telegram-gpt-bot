# src/core/retrieval.py
from typing import List, Dict, Optional, Tuple
import json
import numpy as np
from sqlalchemy.orm import joinedload
from src.storage.vector_store import VectorStore
from src.storage.db import Database, ChunkModel, TopicL1Model, TopicL2Model
from src.core.embedding import EmbeddingClient
from src.core.syslog2 import *


class RetrievalService:
    """Service to retrieve relevant context for a query."""
    
    def __init__(
        self, 
        vector_store: VectorStore, 
        db: Database, 
        embedding_client: EmbeddingClient, 
        verbosity: int = 0,
        use_topic_retrieval: bool = True,
        topic_retrieval_weight: float = 0.3
    ):
        """
        Initialize RetrievalService.
        
        Args:
            vector_store: Vector store for semantic search
            db: Database for chunk and topic storage
            embedding_client: Client for computing embeddings
            verbosity: Logging verbosity level
            use_topic_retrieval: Enable hierarchical topic-based retrieval
            topic_retrieval_weight: Weight for topic-based results (0.0-1.0)
        """
        self.vector_store = vector_store
        self.db = db
        self.embedding_client = embedding_client
        self.verbosity = verbosity
        self.use_topic_retrieval = use_topic_retrieval
        self.topic_retrieval_weight = topic_retrieval_weight

        
    def _find_similar_topics(
        self, 
        query_embedding: List[float], 
        topic_type: str = "l1",
        n_topics: int = 3,
        similarity_threshold: float = 0.5
    ) -> List[Tuple[int, float]]:
        """
        Find topics similar to the query embedding by comparing with topic centroids.
        
        Args:
            query_embedding: Query embedding vector
            topic_type: "l1" or "l2"
            n_topics: Number of top topics to return
            similarity_threshold: Minimum cosine similarity threshold
            
        Returns:
            List of (topic_id, similarity_score) tuples, sorted by similarity descending
        """
        session = self.db.get_session()
        try:
            if topic_type == "l1":
                topics = session.query(TopicL1Model).filter(
                    TopicL1Model.center_vec.isnot(None)
                ).all()
            else:
                topics = session.query(TopicL2Model).filter(
                    TopicL2Model.center_vec.isnot(None)
                ).all()
            
            if not topics:
                return []
            
            query_vec = np.array(query_embedding)
            similarities = []
            
            for topic in topics:
                try:
                    center_vec = json.loads(topic.center_vec)
                    center_vec = np.array(center_vec)
                    
                    # Compute cosine similarity
                    dot_product = np.dot(query_vec, center_vec)
                    norm_query = np.linalg.norm(query_vec)
                    norm_center = np.linalg.norm(center_vec)
                    
                    if norm_query > 0 and norm_center > 0:
                        similarity = dot_product / (norm_query * norm_center)
                        if similarity >= similarity_threshold:
                            similarities.append((topic.id, float(similarity)))
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    if self.verbosity >= 2:
                        syslog2(LOG_DEBUG, "topic centroid parse error", topic_id=topic.id, error=str(e))
                    continue
            
            # Sort by similarity descending and return top n
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:n_topics]
        finally:
            session.close()

    def _retrieve_chunks_from_topics(
        self,
        topic_ids_l1: List[int],
        topic_ids_l2: List[int],
        max_chunks_per_topic: int = 5
    ) -> List[Dict]:
        """
        Retrieve chunks from specified topics.
        
        Args:
            topic_ids_l1: List of L1 topic IDs
            topic_ids_l2: List of L2 topic IDs
            max_chunks_per_topic: Maximum chunks to retrieve per topic
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        session = self.db.get_session()
        retrieved_chunks = []
        seen_chunk_ids = set()
        
        try:
            # Retrieve chunks from L1 topics
            for topic_id in topic_ids_l1:
                chunks = session.query(ChunkModel)\
                    .options(joinedload(ChunkModel.topic_l1), joinedload(ChunkModel.topic_l2))\
                    .filter(ChunkModel.topic_l1_id == topic_id)\
                    .limit(max_chunks_per_topic)\
                    .all()
                
                for chunk in chunks:
                    if chunk.id in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(chunk.id)
                    
                    meta = {}
                    if chunk.metadata_json:
                        try:
                            meta = json.loads(chunk.metadata_json)
                        except json.JSONDecodeError:
                            pass
                    
                    if chunk.topic_l1:
                        meta["topic_l1_id"] = chunk.topic_l1.id
                        meta["topic_l1_title"] = chunk.topic_l1.title
                    if chunk.topic_l2:
                        meta["topic_l2_id"] = chunk.topic_l2.id
                        meta["topic_l2_title"] = chunk.topic_l2.title
                    
                    retrieved_chunks.append({
                        "id": chunk.id,
                        "text": chunk.text,
                        "metadata": meta,
                        "score": 0.8,  # Topic-based chunks get high score
                        "source": "topic_l1"
                    })
            
            # Retrieve chunks from L2 topics
            for topic_id in topic_ids_l2:
                chunks = session.query(ChunkModel)\
                    .options(joinedload(ChunkModel.topic_l1), joinedload(ChunkModel.topic_l2))\
                    .filter(ChunkModel.topic_l2_id == topic_id)\
                    .limit(max_chunks_per_topic)\
                    .all()
                
                for chunk in chunks:
                    if chunk.id in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(chunk.id)
                    
                    meta = {}
                    if chunk.metadata_json:
                        try:
                            meta = json.loads(chunk.metadata_json)
                        except json.JSONDecodeError:
                            pass
                    
                    if chunk.topic_l1:
                        meta["topic_l1_id"] = chunk.topic_l1.id
                        meta["topic_l1_title"] = chunk.topic_l1.title
                    if chunk.topic_l2:
                        meta["topic_l2_id"] = chunk.topic_l2.id
                        meta["topic_l2_title"] = chunk.topic_l2.title
                    
                    retrieved_chunks.append({
                        "id": chunk.id,
                        "text": chunk.text,
                        "metadata": meta,
                        "score": 0.75,  # L2 topics slightly lower than L1
                        "source": "topic_l2"
                    })
        finally:
            session.close()
        
        return retrieved_chunks

    def retrieve(
        self, 
        query: str, 
        n_results: int = 5, 
        score_threshold: float = 0.5,
        use_topics: Optional[bool] = None
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a given query using hybrid vector + topic search.
        
        Args:
            query: User query string.
            n_results: Number of results to return.
            score_threshold: Minimum similarity score for vector search.
            use_topics: Override use_topic_retrieval setting (None = use instance setting)
                             
        Returns:
            List of dictionaries containing chunk text and metadata.
        """
        use_topics = use_topics if use_topics is not None else self.use_topic_retrieval
        
        if self.verbosity >= 1:
            syslog2(LOG_DEBUG, "retrieval query", query=query, use_topics=use_topics)
        
        if self.verbosity >= 2:
            syslog2(LOG_DEBUG, "computing query embedding")
        
        # 1. Embed query
        query_embs = self.embedding_client.get_embeddings([query])
        query_emb = query_embs[0] if query_embs else []
        
        if not query_emb:
            return []
        
        # 2. Vector-based retrieval (always performed)
        if self.verbosity >= 2:
            collection_count = self.vector_store.collection.count()
            syslog2(LOG_DEBUG, "searching vector store", 
                   collection=self.vector_store.collection.name,
                   total_documents=collection_count)
        
        if self.vector_store.collection.count() == 0:
            if self.verbosity >= 1:
                syslog2(LOG_WARNING, "vector store is empty", action="skipping vector retrieval")
            vector_results = {"ids": [[]], "distances": [[]]}
        else:
            vector_results = self.vector_store.collection.query(
                query_embeddings=[query_emb],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        
        if self.verbosity >= 2:
            syslog2(LOG_DEBUG, "vector store query result", 
                   has_ids=bool(vector_results.get("ids")),
                   ids_count=len(vector_results.get("ids", [[]])[0]) if vector_results.get("ids") and len(vector_results.get("ids", [])) > 0 else 0,
                   result_keys=list(vector_results.keys()))
        
        vector_chunks: List[Dict] = []
        # Check if we have results: vector_results["ids"] should be a list of lists
        has_results = (vector_results.get("ids") and 
                      len(vector_results["ids"]) > 0 and 
                      len(vector_results["ids"][0]) > 0)
        
        if has_results:
            ids = vector_results["ids"][0]
            distances = vector_results["distances"][0] if "distances" in vector_results else []
            
            if self.verbosity >= 2:
                syslog2(LOG_DEBUG, "vector store returned", ids_count=len(ids), distances_count=len(distances))
            
            if self.verbosity >= 2:
                syslog2(LOG_DEBUG, "fetching full text from sqlite", count=len(ids))
            
            session = self.db.get_session()
            try:
                for i, chunk_id in enumerate(ids):
                    distance = distances[i] if i < len(distances) else 0
                    
                    # Convert distance to similarity score
                    # ChromaDB with cosine metric returns distances in range [0, 2]
                    # For cosine: similarity = 1 - (distance / 2) or just 1 - distance if normalized
                    # Actually, cosine distance = 1 - cosine_similarity, so similarity = 1 - distance
                    # But need to handle cases where distance might be > 1
                    if distance <= 1.0:
                        similarity = 1.0 - distance
                    elif distance <= 2.0:
                        # If distance is in [1, 2], it's still cosine distance, normalize
                        similarity = 1.0 - (distance / 2.0)
                    else:
                        similarity = max(0.0, 1.0 - distance)
                    
                    if self.verbosity >= 3:
                        syslog2(LOG_DEBUG, "chunk similarity", chunk_id=chunk_id, distance=distance, similarity=similarity, threshold=score_threshold)
                    
                    # Lower threshold or remove it - let all results through, we'll sort by score
                    # if similarity < score_threshold:
                    #     if self.verbosity >= 2:
                    #         syslog2(LOG_DEBUG, "chunk filtered by threshold", chunk_id=chunk_id, similarity=similarity)
                    #     continue
                    
                    db_chunk = session.query(ChunkModel)\
                        .options(joinedload(ChunkModel.topic_l1), joinedload(ChunkModel.topic_l2))\
                        .filter_by(id=chunk_id).first()
                    
                    if db_chunk:
                        meta = {}
                        if db_chunk.metadata_json:
                            try:
                                meta = json.loads(db_chunk.metadata_json)
                            except json.JSONDecodeError:
                                pass
                        
                        if db_chunk.topic_l1:
                            meta["topic_l1_id"] = db_chunk.topic_l1.id
                            meta["topic_l1_title"] = db_chunk.topic_l1.title
                        if db_chunk.topic_l2:
                            meta["topic_l2_id"] = db_chunk.topic_l2.id
                            meta["topic_l2_title"] = db_chunk.topic_l2.title
                        
                        vector_chunks.append({
                            "id": chunk_id,
                            "text": db_chunk.text,
                            "metadata": meta,
                            "score": similarity,
                            "source": "vector"
                        })
            finally:
                session.close()
        
        # 3. Topic-based retrieval (if enabled)
        topic_chunks: List[Dict] = []
        if use_topics:
            if self.verbosity >= 2:
                syslog2(LOG_DEBUG, "searching topics")
            
            # Find similar L1 and L2 topics
            similar_l1 = self._find_similar_topics(
                query_emb, 
                topic_type="l1", 
                n_topics=2,
                similarity_threshold=0.5
            )
            similar_l2 = self._find_similar_topics(
                query_emb,
                topic_type="l2",
                n_topics=1,
                similarity_threshold=0.5
            )
            
            if similar_l1 or similar_l2:
                topic_ids_l1 = [tid for tid, _ in similar_l1]
                topic_ids_l2 = [tid for tid, _ in similar_l2]
                
                if self.verbosity >= 2:
                    syslog2(LOG_DEBUG, "found similar topics", l1_count=len(topic_ids_l1), l2_count=len(topic_ids_l2))
                
                topic_chunks = self._retrieve_chunks_from_topics(
                    topic_ids_l1=topic_ids_l1,
                    topic_ids_l2=topic_ids_l2,
                    max_chunks_per_topic=3
                )
        
        # 4. Merge and deduplicate results
        all_chunks: Dict[str, Dict] = {}
        
        # Add vector chunks (weighted by vector weight)
        vector_weight = 1.0 - self.topic_retrieval_weight
        for chunk in vector_chunks:
            chunk_id = chunk["id"]
            if chunk_id not in all_chunks:
                chunk["score"] *= vector_weight
                all_chunks[chunk_id] = chunk
            else:
                # If chunk appears in both, take the higher score
                existing_score = all_chunks[chunk_id]["score"]
                new_score = chunk["score"] * vector_weight
                if new_score > existing_score:
                    all_chunks[chunk_id] = chunk
                    all_chunks[chunk_id]["score"] = new_score
        
        # Add topic chunks (weighted by topic weight)
        topic_weight = self.topic_retrieval_weight
        for chunk in topic_chunks:
            chunk_id = chunk["id"]
            if chunk_id not in all_chunks:
                chunk["score"] *= topic_weight
                all_chunks[chunk_id] = chunk
            else:
                # Boost score if chunk appears in both sources
                existing_score = all_chunks[chunk_id]["score"]
                topic_score = chunk["score"] * topic_weight
                all_chunks[chunk_id]["score"] = existing_score + topic_score * 0.5  # Partial boost
        
        # 5. Sort by score and return top n_results
        final_chunks = sorted(all_chunks.values(), key=lambda x: x["score"], reverse=True)
        
        if self.verbosity >= 2:
            syslog2(LOG_DEBUG, "retrieval complete", 
                   vector_count=len(vector_chunks),
                   topic_count=len(topic_chunks),
                   final_count=len(final_chunks))
        
        return final_chunks[:n_results]
