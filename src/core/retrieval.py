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
        topic_retrieval_weight: float = 0.3,
        search_mode: str = "two_stage",
        l2_top_k: int = 5,
        chunk_top_k: int = 50
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
            search_mode: "two_stage" (L2→L1) or "direct" (direct chunk search)
            l2_top_k: Number of L2 topics to select in two-stage search
            chunk_top_k: Number of chunks to return in two-stage search
        """
        self.vector_store = vector_store
        self.db = db
        self.embedding_client = embedding_client
        self.verbosity = verbosity
        self.use_topic_retrieval = use_topic_retrieval
        self.topic_retrieval_weight = topic_retrieval_weight
        self.search_mode = search_mode
        self.l2_top_k = l2_top_k
        self.chunk_top_k = chunk_top_k
        
        # Get topics_l2 collection
        self.topics_l2_collection = vector_store.get_topics_l2_collection()

        
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

    def _two_stage_search(
        self,
        query_embedding: List[float],
        n_results: int = 50
    ) -> List[Dict]:
        """
        Two-stage search: L2 topics → chunks within selected L2 topics.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of chunks to return
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Step 1: Search for relevant L2 topics
        if self.verbosity >= 2:
            syslog2(LOG_DEBUG, "two_stage_search: searching l2 topics", l2_top_k=self.l2_top_k)
        
        try:
            l2_results = self.topics_l2_collection.query(
                query_embeddings=[query_embedding],
                n_results=self.l2_top_k,
                include=["metadatas", "distances"]
            )
        except Exception as e:
            if self.verbosity >= 1:
                syslog2(LOG_WARNING, "failed to query l2 topics, falling back to direct search", error=str(e))
            return []
        
        if not l2_results or not l2_results.get("ids") or not l2_results["ids"][0]:
            if self.verbosity >= 2:
                syslog2(LOG_DEBUG, "no l2 topics found, falling back to direct search")
            return []
        
        # Extract L2 topic IDs
        l2_ids = []
        l2_metadatas = l2_results.get("metadatas", [[]])[0] if l2_results.get("metadatas") else []
        for meta in l2_metadatas:
            if meta and "topic_l2_id" in meta:
                l2_ids.append(meta["topic_l2_id"])
        
        if not l2_ids:
            if self.verbosity >= 2:
                syslog2(LOG_DEBUG, "no valid l2 topic ids found, falling back to direct search")
            return []
        
        if self.verbosity >= 2:
            syslog2(LOG_DEBUG, "two_stage_search: found l2 topics", l2_ids=l2_ids, count=len(l2_ids))
        
        # Step 2: Search chunks within selected L2 topics
        # Try Option A: Filter by topic_l2_id in chroma metadata
        try:
            # Check if chunks have topic_l2_id in metadata
            # Try to query with filter
            chunk_results = self.vector_store.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where={"topic_l2_id": {"$in": l2_ids}},
                include=["documents", "metadatas", "distances"]
            )
            
            if chunk_results and chunk_results.get("ids") and chunk_results["ids"][0]:
                # Successfully used filter
                if self.verbosity >= 2:
                    syslog2(LOG_DEBUG, "two_stage_search: found chunks via chroma filter", count=len(chunk_results["ids"][0]))
                
                return self._process_chunk_results(chunk_results, query_embedding)
        except Exception as e:
            if self.verbosity >= 2:
                syslog2(LOG_DEBUG, "chroma filter not available, using sqlite fallback", error=str(e))
        
        # Option B: Get chunks via SQLite, then compute similarity
        if self.verbosity >= 2:
            syslog2(LOG_DEBUG, "two_stage_search: using sqlite fallback")
        
        all_chunk_ids = []
        for l2_id in l2_ids:
            chunks = self.db.get_chunks_by_topic_l2(l2_id)
            all_chunk_ids.extend([chunk.id for chunk in chunks])
        
        if not all_chunk_ids:
            if self.verbosity >= 2:
                syslog2(LOG_DEBUG, "no chunks found for l2 topics")
            return []
        
        # Remove duplicates
        all_chunk_ids = list(set(all_chunk_ids))
        
        if self.verbosity >= 2:
            syslog2(LOG_DEBUG, "two_stage_search: found chunks via sqlite", count=len(all_chunk_ids))
        
        # Get embeddings for these chunks
        try:
            embeddings_data = self.vector_store.get_embeddings_by_ids(all_chunk_ids)
            chunk_embeddings = embeddings_data.get("embeddings", [])
            chunk_ids_with_emb = embeddings_data.get("ids", [])
            
            if not chunk_embeddings:
                return []
            
            # Compute cosine similarity
            query_vec = np.array(query_embedding)
            similarities = []
            
            for i, chunk_id in enumerate(chunk_ids_with_emb):
                if i < len(chunk_embeddings):
                    chunk_vec = np.array(chunk_embeddings[i])
                    # Cosine similarity
                    dot_product = np.dot(query_vec, chunk_vec)
                    norm_query = np.linalg.norm(query_vec)
                    norm_chunk = np.linalg.norm(chunk_vec)
                    
                    if norm_query > 0 and norm_chunk > 0:
                        similarity = dot_product / (norm_query * norm_chunk)
                        similarities.append((chunk_id, float(similarity)))
            
            # Sort by similarity and take top n_results
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_chunk_ids = [cid for cid, _ in similarities[:n_results]]
            
            # Get full chunk data from database
            return self._get_chunks_by_ids(top_chunk_ids, similarities[:n_results])
            
        except Exception as e:
            if self.verbosity >= 1:
                syslog2(LOG_WARNING, "failed to get chunk embeddings for two_stage_search", error=str(e))
            return []

    def _process_chunk_results(self, chunk_results: Dict, query_embedding: List[float]) -> List[Dict]:
        """Process chunk results from chroma query."""
        if not chunk_results or not chunk_results.get("ids") or not chunk_results["ids"][0]:
            return []
        
        ids = chunk_results["ids"][0]
        distances = chunk_results.get("distances", [[]])[0] if chunk_results.get("distances") else []
        metadatas = chunk_results.get("metadatas", [[]])[0] if chunk_results.get("metadatas") else []
        
        # Convert distances to similarities
        similarities = []
        for i, chunk_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else 0
            if distance <= 1.0:
                similarity = 1.0 - distance
            elif distance <= 2.0:
                similarity = 1.0 - (distance / 2.0)
            else:
                similarity = max(0.0, 1.0 - distance)
            similarities.append((chunk_id, similarity))
        
        return self._get_chunks_by_ids(ids, similarities)

    def _get_chunks_by_ids(self, chunk_ids: List[str], similarities: List[Tuple[str, float]]) -> List[Dict]:
        """Get full chunk data from database by IDs."""
        if not chunk_ids:
            return []
        
        # Create similarity map
        sim_map = {cid: sim for cid, sim in similarities}
        
        session = self.db.get_session()
        try:
            chunks = session.query(ChunkModel)\
                .options(joinedload(ChunkModel.topic_l1), joinedload(ChunkModel.topic_l2))\
                .filter(ChunkModel.id.in_(chunk_ids))\
                .all()
            
            result = []
            for chunk in chunks:
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
                
                similarity = sim_map.get(chunk.id, 0.0)
                
                result.append({
                    "id": chunk.id,
                    "text": chunk.text,
                    "metadata": meta,
                    "score": similarity,
                    "source": "two_stage"
                })
            
            # Sort by similarity (already sorted, but ensure)
            result.sort(key=lambda x: x["score"], reverse=True)
            return result
        finally:
            session.close()

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
        
        # 2. Choose search strategy
        if self.search_mode == "two_stage":
            if self.verbosity >= 1:
                syslog2(LOG_DEBUG, "using two_stage search mode")
            
            two_stage_results = self._two_stage_search(query_emb, n_results=self.chunk_top_k)
            
            if two_stage_results:
                if self.verbosity >= 2:
                    syslog2(LOG_DEBUG, "two_stage search returned results", count=len(two_stage_results))
                return two_stage_results[:n_results]
            else:
                # Fallback to direct search if two_stage found nothing
                if self.verbosity >= 1:
                    syslog2(LOG_DEBUG, "two_stage search found nothing, falling back to direct search")
        
        # 3. Direct vector-based retrieval (fallback or default mode)
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
        
        # 4. Topic-based retrieval (if enabled)
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
        
        # 5. Merge and deduplicate results
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
        
        # 6. Sort by score and return top n_results
        final_chunks = sorted(all_chunks.values(), key=lambda x: x["score"], reverse=True)
        
        if self.verbosity >= 2:
            syslog2(LOG_DEBUG, "retrieval complete", 
                   vector_count=len(vector_chunks),
                   topic_count=len(topic_chunks),
                   final_count=len(final_chunks))
        
        return final_chunks[:n_results]
