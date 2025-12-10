# src/core/retrieval.py
from typing import List, Dict, Optional, Tuple
import json
import numpy as np
from sqlalchemy.orm import joinedload
from src.storage.vector_store import VectorStore
from src.storage.db import Database, ChunkModel, TopicL1Model, TopicL2Model, MessageModel
from src.core.embedding import EmbeddingClient
from src.core.syslog2 import *


class RetrievalService:
    """Service to retrieve relevant context for a query."""
    
    def __init__(
        self, 
        vector_store: VectorStore, 
        db: Database, 
        embedding_client: EmbeddingClient, 
        log_level: int = LOG_WARNING,
        use_topic_retrieval: bool = True,
        topic_retrieval_weight: float = 0.3,
        search_mode: str = "two_stage",
        l2_top_k: int = 5,
        chunk_top_k: int = 50,
        debug_rag: bool = False,
        rag_strategy_use_ntop: bool = False,
        rag_ntop: int = 5
    ):
        """
        Initialize RetrievalService.
        
        Args:
            vector_store: Vector store for semantic search
            db: Database for chunk and topic storage
            embedding_client: Client for computing embeddings
            log_level: Logging level (LOG_ALERT=1, LOG_CRIT=2, LOG_ERR=3, LOG_WARNING=4, LOG_NOTICE=5, LOG_INFO=6, LOG_DEBUG=7)
            use_topic_retrieval: Enable hierarchical topic-based retrieval
            topic_retrieval_weight: Weight for topic-based results (0.0-1.0)
            search_mode: "two_stage" (L2→L1) or "direct" (direct chunk search)
            l2_top_k: Number of L2 topics to select in two-stage search
            chunk_top_k: Number of chunks to return in two-stage search
            debug_rag: enable detailed rag debug logging
            rag_strategy_use_ntop: If True, take top rag_ntop results regardless of similarity threshold
            rag_ntop: Number of top results to take when rag_strategy_use_ntop is enabled
        """
        self.vector_store = vector_store
        self.db = db
        self.embedding_client = embedding_client
        self.log_level = log_level
        self.use_topic_retrieval = use_topic_retrieval
        self.topic_retrieval_weight = topic_retrieval_weight
        self.search_mode = search_mode
        self.l2_top_k = l2_top_k
        self.chunk_top_k = chunk_top_k
        self.debug_rag = debug_rag
        self.rag_strategy_use_ntop = rag_strategy_use_ntop
        self.rag_ntop = rag_ntop
        
        # Get topics collections
        self.topics_l2_collection = vector_store.get_topics_l2_collection()
        self.topics_l1_collection = vector_store.get_topics_l1_collection()

    def _convert_distance_to_similarity(self, distance: float) -> float:
        """
        Convert distance to similarity score.
        
        Args:
            distance: Distance value from vector search
            
        Returns:
            Similarity score (0.0-1.0)
        """
        if distance <= 1.0:
            return 1.0 - distance
        elif distance <= 2.0:
            return 1.0 - (distance / 2.0)
        else:
            # For large distances, use inverse relationship to keep similarity informative
            # Using 1/(1+distance) ensures similarity decreases smoothly but stays > 0
            return 1.0 / (1.0 + distance)
    
    def _build_chunk_dict(
        self, 
        chunk: ChunkModel, 
        similarity: float, 
        distance: Optional[float] = None, 
        source: str = "vector"
    ) -> Dict:
        """
        Build chunk dictionary with metadata and topics.
        
        Args:
            chunk: ChunkModel instance
            similarity: Similarity score (0.0-1.0)
            distance: Original distance value (optional)
            source: Source of the chunk ("vector", "two_stage", "topic_l1", "topic_l2")
            
        Returns:
            Dictionary with chunk data
        """
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
        
        result = {
            "id": chunk.id,
            "text": chunk.text,
            "metadata": meta,
            "score": similarity,
            "source": source
        }
        
        if distance is not None:
            result["distance"] = distance
        
        return result

    def _debug_log_chunk_messages(self, session, chunk: ChunkModel):
        """log messages belonging to chunk for rag debug"""
        if not self.debug_rag:
            return
        try:
            msgs_query = session.query(MessageModel)
            if chunk.chat_id and chunk.ts_from and chunk.ts_to:
                msgs_query = msgs_query.filter(
                    MessageModel.chat_id == chunk.chat_id,
                    MessageModel.ts >= chunk.ts_from,
                    MessageModel.ts <= chunk.ts_to,
                )
            elif chunk.chat_id and chunk.msg_id_start and chunk.msg_id_end:
                # fallback by msg id range if timestamps are missing
                msgs_query = msgs_query.filter(
                    MessageModel.chat_id == chunk.chat_id,
                    MessageModel.msg_id >= chunk.msg_id_start,
                    MessageModel.msg_id <= chunk.msg_id_end,
                )
            elif chunk.msg_id_start:
                msgs_query = msgs_query.filter(
                    MessageModel.chat_id == chunk.chat_id,
                    MessageModel.msg_id == chunk.msg_id_start,
                )
            else:
                return

            msgs = msgs_query.order_by(MessageModel.ts).all()
            for msg in msgs:
                txt = msg.text or ""
                txt_snip = txt[:64]
                syslog2(
                    LOG_DEBUG,
                    "rag chunk msg",
                    chunk_id=chunk.id,
                    msg_id=msg.msg_id,
                    user_id=str(msg.from_id) if msg.from_id is not None else "",
                    text_snippet=txt_snip,
                )
        except Exception as e:
            syslog2(LOG_DEBUG, "rag chunk msg log failed", chunk_id=chunk.id, error=str(e))

    def _find_similar_topics(
        self, 
        query_embedding: List[float], 
        topic_type: str = "l1",
        n_topics: int = 3,
        similarity_threshold: float = 0.5
    ) -> List[Tuple[int, float]]:
        """
        Find topics similar to the query embedding by comparing with topic centroids.
        Reads center vectors from ChromaDB collections.
        
        Args:
            query_embedding: Query embedding vector
            topic_type: "l1" or "l2"
            n_topics: Number of top topics to return
            similarity_threshold: Minimum cosine similarity threshold
            
        Returns:
            List of (topic_id, similarity_score) tuples, sorted by similarity descending
        """
        try:
            # Get topics from ChromaDB
            if topic_type == "l1":
                collection = self.topics_l1_collection
                topic_prefix = "l1-"
            else:
                collection = self.topics_l2_collection
                topic_prefix = "l2-"
            
            # Get all topics from collection
            all_topics = collection.get(include=["embeddings", "metadatas"])
            
            if not all_topics or not all_topics.get("ids") or len(all_topics["ids"]) == 0:
                return []
            
            ids = all_topics["ids"]
            embeddings = all_topics.get("embeddings", [])
            metadatas = all_topics.get("metadatas", [])
            
            if len(embeddings) == 0 or len(embeddings) != len(ids):
                return []
            
            # Ensure 1D float vectors
            query_vec = np.asarray(query_embedding, dtype=float).ravel()
            similarities = []
            
            for idx, topic_id_str in enumerate(ids):
                try:
                    # Extract numeric topic ID from "l1-123" or "l2-123" format
                    topic_id = int(topic_id_str.replace(topic_prefix, ""))
                    center_vec = np.asarray(embeddings[idx], dtype=float).ravel()
                    
                    # Compute cosine similarity
                    dot_product = float(np.dot(query_vec, center_vec))
                    norm_query = float(np.linalg.norm(query_vec))
                    norm_center = float(np.linalg.norm(center_vec))
                    
                    if norm_query > 0 and norm_center > 0:
                        similarity = float(dot_product / (norm_query * norm_center))
                        if similarity >= similarity_threshold:
                            similarities.append((topic_id, float(similarity)))
                except (ValueError, TypeError, IndexError) as e:
                    if self.debug_rag:
                        syslog2(LOG_DEBUG, "topic centroid parse error", topic_id_str=topic_id_str, error=str(e))
                    continue
            
            # Sort by similarity descending and return top n
            similarities.sort(key=lambda x: x[1], reverse=True)

            if self.debug_rag and similarities:
                for idx, (tid, sim) in enumerate(similarities[:n_topics]):
                    syslog2(
                        LOG_DEBUG,
                        "rag topic match",
                        topic_type=topic_type,
                        idx=idx,
                        topic_id=tid,
                        similarity=sim,
                    )

            return similarities[:n_topics]
        except Exception as e:
            if self.log_level <= LOG_WARNING:
                syslog2(LOG_WARNING, "failed to find similar topics", topic_type=topic_type, error=str(e))
            return []

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
                    
                    # Use _build_chunk_dict to avoid duplicating metadata parsing logic
                    chunk_dict = self._build_chunk_dict(
                        chunk,
                        similarity=0.8,  # Topic-based chunks get high score
                        source="topic_l1"
                    )
                    retrieved_chunks.append(chunk_dict)
            
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
                    
                    # Use _build_chunk_dict to avoid duplicating metadata parsing logic
                    chunk_dict = self._build_chunk_dict(
                        chunk,
                        similarity=0.75,  # L2 topics slightly lower than L1
                        source="topic_l2"
                    )
                    retrieved_chunks.append(chunk_dict)
        finally:
            session.close()
        
        return retrieved_chunks

    def _query_l2_topics(self, query_embedding: List[float]) -> List[int]:
        """
        Query L2 topics and extract topic IDs.
        
        Args:
            query_embedding: Query embedding vector
            
        Returns:
            List of L2 topic IDs, empty list if no topics found
        """
        if self.debug_rag:
            syslog2(LOG_DEBUG, "two_stage_search: searching l2 topics", l2_top_k=self.l2_top_k)
        
        try:
            l2_results = self.topics_l2_collection.query(
                query_embeddings=[query_embedding],
                n_results=self.l2_top_k,
                include=["metadatas", "distances"]
            )
        except Exception as e:
            if self.log_level <= LOG_INFO or self.debug_rag:
                syslog2(LOG_WARNING, "failed to query l2 topics, falling back to direct search", error=str(e))
            return []
        
        if not l2_results or not l2_results.get("ids") or not l2_results["ids"][0]:
            if self.debug_rag:
                syslog2(LOG_DEBUG, "no l2 topics found, falling back to direct search")
            return []
        
        # Extract L2 topic IDs and log distances
        l2_ids = []
        l2_metadatas = l2_results.get("metadatas", [[]])[0] if l2_results.get("metadatas") else []
        l2_distances = l2_results.get("distances", [[]])[0] if l2_results.get("distances") else []
        
        for idx, meta in enumerate(l2_metadatas):
            if meta and "topic_l2_id" in meta:
                l2_id = meta["topic_l2_id"]
                l2_ids.append(l2_id)
                distance = l2_distances[idx] if idx < len(l2_distances) else 0.0
                # Always log L2 topic distance (not just in debug mode)
                syslog2(
                    LOG_ALERT,
                    "rag l2 topic match distance",
                    idx=idx,
                    topic_l2_id=l2_id,
                    distance=distance,
                )
                if self.debug_rag:
                    syslog2(
                        LOG_DEBUG,
                        "rag l2 topic match details",
                        idx=idx,
                        topic_l2_id=l2_id,
                        metadata=meta,
                    )
        
        if not l2_ids:
            if self.debug_rag:
                syslog2(LOG_DEBUG, "no valid l2 topic ids found, falling back to direct search")
            return []
        
        if self.debug_rag:
            syslog2(LOG_DEBUG, "two_stage_search: found l2 topics", l2_ids=l2_ids, count=len(l2_ids))
        
        return l2_ids
    
    def _filter_chunks_by_topics(
        self,
        query_embedding: List[float],
        l2_ids: List[int],
        n_results: int
    ) -> Optional[List[Dict]]:
        """
        Filter chunks by L2 topics using ChromaDB metadata filter.
        
        Args:
            query_embedding: Query embedding vector
            l2_ids: List of L2 topic IDs to filter by
            n_results: Number of chunks to return
            
        Returns:
            List of chunk dictionaries if successful, None if filter not available
        """
        try:
            chunk_results = self.vector_store.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where={"topic_l2_id": {"$in": l2_ids}},
                include=["documents", "metadatas", "distances"]
            )
            
            if chunk_results and chunk_results.get("ids") and chunk_results["ids"][0]:
                chunk_ids = chunk_results["ids"][0]
                chunk_distances = chunk_results.get("distances", [[]])[0] if chunk_results.get("distances") else []
                
                syslog2(LOG_ALERT, "two_stage_search: found chunks via chroma filter", count=len(chunk_ids))
                # Always log distances for chunks from ChromaDB
                for idx, chunk_id in enumerate(chunk_ids):
                    distance = chunk_distances[idx] if idx < len(chunk_distances) else 0.0
                    syslog2(
                        LOG_ALERT,
                        "rag two_stage chunk chroma distance",
                        idx=idx,
                        chunk_id=chunk_id,
                        distance=distance,
                    )
                
                res = self._process_chunk_results(chunk_results, query_embedding)

                if self.debug_rag and res:
                    for idx, item in enumerate(res):
                        syslog2(
                            LOG_DEBUG,
                            "rag two_stage chunk processed",
                            idx=idx,
                            chunk_id=item["id"],
                            score=item.get("score", 0.0),
                            distance=1.0 - item.get("score", 0.0),  # Convert score to distance
                            source=item.get("source", ""),
                        )

                return res
        except Exception as e:
            if self.debug_rag:
                syslog2(LOG_DEBUG, "chroma filter not available, using sqlite fallback", error=str(e))
        
        return None
    
    def _fallback_sqlite_search(
        self,
        query_embedding: List[float],
        l2_ids: List[int],
        n_results: int
    ) -> List[Dict]:
        """
        Fallback search using SQLite: get chunks by topic, then compute similarity.
        
        Args:
            query_embedding: Query embedding vector
            l2_ids: List of L2 topic IDs
            n_results: Number of chunks to return
            
        Returns:
            List of chunk dictionaries sorted by similarity
        """
        if self.debug_rag:
            syslog2(LOG_DEBUG, "two_stage_search: using sqlite fallback")
        
        all_chunk_ids = []
        for l2_id in l2_ids:
            chunks = self.db.get_chunks_by_topic_l2(l2_id)
            all_chunk_ids.extend([chunk.id for chunk in chunks])
        
        if not all_chunk_ids:
            if self.debug_rag:
                syslog2(LOG_DEBUG, "no chunks found for l2 topics")
            return []
        
        all_chunk_ids = list(set(all_chunk_ids))
        
        if self.debug_rag:
            syslog2(LOG_DEBUG, "two_stage_search: found chunks via sqlite", count=len(all_chunk_ids))
        
        try:
            embeddings_data = self.vector_store.get_embeddings_by_ids(all_chunk_ids)
            chunk_embeddings = embeddings_data.get("embeddings", [])
            chunk_ids_with_emb = embeddings_data.get("ids", [])
            
            if not chunk_embeddings:
                return []
            
            query_vec = np.array(query_embedding)
            similarities = []
            original_distances = []
            
            for i, chunk_id in enumerate(chunk_ids_with_emb):
                if i < len(chunk_embeddings):
                    chunk_vec = np.array(chunk_embeddings[i])
                    dot_product = np.dot(query_vec, chunk_vec)
                    norm_query = np.linalg.norm(query_vec)
                    norm_chunk = np.linalg.norm(chunk_vec)
                    
                    if norm_query > 0 and norm_chunk > 0:
                        similarity = dot_product / (norm_query * norm_chunk)
                        distance = 1.0 - similarity  # Convert similarity to distance
                        similarities.append((chunk_id, float(similarity)))
                        original_distances.append((chunk_id, float(distance)))
                        
                        # Always log distance for SQLite fallback chunks
                        syslog2(
                            LOG_ALERT,
                            "rag two_stage chunk sqlite distance",
                            chunk_id=chunk_id,
                            similarity=similarity,
                            distance=distance,
                        )
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_chunk_ids = [cid for cid, _ in similarities[:n_results]]
            # Create distance map for quick lookup, then preserve order of top_chunk_ids
            distance_map = {cid: dist for cid, dist in original_distances}
            top_distances = [(cid, distance_map.get(cid, 0.0)) for cid in top_chunk_ids]
            
            res = self._get_chunks_by_ids(top_chunk_ids, similarities[:n_results], top_distances)

            if self.debug_rag and res:
                for idx, item in enumerate(res):
                    syslog2(
                        LOG_DEBUG,
                        "rag two_stage chunk sqlite processed",
                        idx=idx,
                        chunk_id=item["id"],
                        score=item.get("score", 0.0),
                        distance=1.0 - item.get("score", 0.0),  # Convert score to distance
                        source=item.get("source", ""),
                    )

            return res
            
        except Exception as e:
            if self.log_level <= LOG_INFO or self.debug_rag:
                syslog2(LOG_WARNING, "failed to get chunk embeddings for two_stage_search", error=str(e))
            return []
    
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
        # Step 1: Query L2 topics
        l2_ids = self._query_l2_topics(query_embedding)
        if not l2_ids:
            return []
        
        # Step 2: Try filtering chunks by topics in ChromaDB
        chroma_results = self._filter_chunks_by_topics(query_embedding, l2_ids, n_results)
        if chroma_results is not None:
            return chroma_results
        
        # Step 3: Fallback to SQLite search
        return self._fallback_sqlite_search(query_embedding, l2_ids, n_results)

    def _process_chunk_results(self, chunk_results: Dict, query_embedding: List[float]) -> List[Dict]:
        """Process chunk results from chroma query."""
        if not chunk_results or not chunk_results.get("ids") or not chunk_results["ids"][0]:
            return []
        
        ids = chunk_results["ids"][0]
        distances = chunk_results.get("distances", [[]])[0] if chunk_results.get("distances") else []
        metadatas = chunk_results.get("metadatas", [[]])[0] if chunk_results.get("metadatas") else []
        
        similarities = []
        original_distances = []
        for i, chunk_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else 0
            original_distances.append((chunk_id, float(distance)))
            similarity = self._convert_distance_to_similarity(distance)
            similarities.append((chunk_id, similarity))
            
            # Always log distance for processed chunk results
            syslog2(
                LOG_ALERT,
                "rag process_chunk_result distance",
                idx=i,
                chunk_id=chunk_id,
                distance=distance,
                similarity=similarity,
            )
            
            if self.debug_rag:
                metadata = metadatas[i] if i < len(metadatas) else {}
                syslog2(
                    LOG_DEBUG,
                    "rag process_chunk_result details",
                    idx=i,
                    chunk_id=chunk_id,
                    metadata=metadata,
                )
        
        return self._get_chunks_by_ids(ids, similarities, original_distances)

    def _get_chunks_by_ids(self, chunk_ids: List[str], similarities: List[Tuple[str, float]], original_distances: Optional[List[Tuple[str, float]]] = None) -> List[Dict]:
        """Get full chunk data from database by IDs."""
        if not chunk_ids:
            return []
        
        sim_map = {cid: sim for cid, sim in similarities}
        distance_map = {cid: dist for cid, dist in (original_distances or [])}
        
        session = self.db.get_session()
        try:
            chunks = session.query(ChunkModel)\
                .options(joinedload(ChunkModel.topic_l1), joinedload(ChunkModel.topic_l2))\
                .filter(ChunkModel.id.in_(chunk_ids))\
                .all()
            
            result = []
            for chunk in chunks:
                similarity = sim_map.get(chunk.id, 0.0)
                original_distance = distance_map.get(chunk.id)

                if self.debug_rag:
                    # Build chunk dict first to get metadata with topics
                    temp_dict = self._build_chunk_dict(
                        chunk,
                        similarity,
                        distance=original_distance,
                        source="two_stage"
                    )
                    meta = temp_dict.get("metadata", {})
                    syslog2(
                        LOG_DEBUG,
                        "rag chunk result",
                        chunk_id=chunk.id,
                        score=similarity,
                        original_distance=original_distance,
                        topic_l1_id=meta.get("topic_l1_id"),
                        topic_l2_id=meta.get("topic_l2_id"),
                        source="two_stage",
                    )
                    self._debug_log_chunk_messages(session, chunk)
                
                result_item = self._build_chunk_dict(
                    chunk, 
                    similarity, 
                    distance=original_distance, 
                    source="two_stage"
                )
                result.append(result_item)
            
            result.sort(key=lambda x: x["score"], reverse=True)
            return result
        finally:
            session.close()

    def _direct_chunk_query(self, query_emb: List[float], n_results: int) -> Dict:
        """
        Direct chunk query without post-processing.
        
        Args:
            query_emb: Query embedding vector
            n_results: Number of results to return
            
        Returns:
            Dictionary with keys: ids, distances, metadatas (ChromaDB format)
        """
        if self.vector_store.collection.count() == 0:
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}
        
        return self.vector_store.collection.query(
            query_embeddings=[query_emb],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

    def search_chunks_basic(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Simple chunk search without LLM processing.
        
        Args:
            query: Search query string
            n_results: Number of results to return
            
        Returns:
            List of dictionaries with keys: id, distance, metadata
            Sorted by distance (ascending)
        """
        if self.debug_rag:
            syslog2(LOG_DEBUG, "basic search start", query=query, n_results=n_results)

        query_embs = self.embedding_client.get_embeddings([query])
        if not query_embs:
            syslog2(LOG_DEBUG, "basic search", query=query, results=0, error="no embeddings")
            return []
        
        query_emb = query_embs[0]
        
        result = self._direct_chunk_query(query_emb, n_results)
        
        ids = result.get("ids", [[]])
        distances = result.get("distances", [[]])
        metadatas = result.get("metadatas", [[]])
        
        if not ids or not ids[0]:
            syslog2(LOG_DEBUG, "basic search", query=query, results=0)
            return []
        
        chunk_list = []
        for i, chunk_id in enumerate(ids[0]):
            if distances and distances[0]:
                if i >= len(distances[0]):
                    continue
                distance = distances[0][i]
            else:
                distance = 0.0
            metadata = metadatas[0][i] if metadatas and metadatas[0] and i < len(metadatas[0]) else {}
            
            chunk_list.append({
                "id": chunk_id,
                "distance": float(distance),
                "metadata": metadata if metadata else {}
            })
        
        chunk_list.sort(key=lambda x: x["distance"])
        chunk_list = chunk_list[:n_results]
        
        # Always log distance for basic search results
        for idx, item in enumerate(chunk_list):
            syslog2(
                LOG_ALERT,
                "basic search result distance",
                idx=idx,
                chunk_id=item["id"],
                distance=item["distance"],
            )
        
        if self.debug_rag:
            for idx, item in enumerate(chunk_list):
                syslog2(
                    LOG_DEBUG,
                    "basic search result details",
                    idx=idx,
                    chunk_id=item["id"],
                    metadata=item.get("metadata", {}),
                )

        syslog2(LOG_DEBUG, "basic search", query=query, results=len(chunk_list))
        return chunk_list

    def _compute_query_embedding(self, query: str) -> Optional[List[float]]:
        """
        Compute query embedding.
        
        Args:
            query: Query string
            
        Returns:
            Query embedding vector or None if failed
        """
        if self.debug_rag:
            syslog2(LOG_DEBUG, "computing query embedding")
        
        query_embs = self.embedding_client.get_embeddings([query])
        query_emb = query_embs[0] if query_embs else []
        
        if not query_emb:
            return None
        
        return query_emb
    
    def _select_search_mode(self, query_emb: List[float], n_results: int) -> List[Dict]:
        """
        Select search mode (two_stage or direct) and execute search.
        
        Args:
            query_emb: Query embedding vector
            n_results: Number of results to return
            
        Returns:
            List of chunk dictionaries
        """
        if self.search_mode == "two_stage":
            if self.log_level <= LOG_INFO or self.debug_rag:
                syslog2(LOG_DEBUG, "using two_stage search mode")
            
            two_stage_results = self._two_stage_search(query_emb, n_results=self.chunk_top_k)
            
            if two_stage_results:
                if self.debug_rag:
                    syslog2(LOG_DEBUG, "two_stage search returned results", count=len(two_stage_results))
                    for idx, item in enumerate(two_stage_results[:n_results]):
                        syslog2(
                            LOG_DEBUG,
                            "rag result two_stage top",
                            idx=idx,
                            chunk_id=item["id"],
                            score=item.get("score", 0.0),
                            source=item.get("source", ""),
                        )
                return two_stage_results[:n_results]
            else:
                if self.log_level <= LOG_INFO or self.debug_rag:
                    syslog2(LOG_DEBUG, "two_stage search found nothing, falling back to direct search")
        
        # Direct search fallback or default mode
        if self.debug_rag:
            collection_count = self.vector_store.collection.count()
            syslog2(LOG_DEBUG, "searching vector store", 
                   collection=self.vector_store.collection.name,
                   total_documents=collection_count)
        
        vector_results = self._direct_chunk_query(query_emb, n_results)
        
        if self.debug_rag:
            syslog2(LOG_DEBUG, "vector store query result", 
                   has_ids=bool(vector_results.get("ids")),
                   ids_count=len(vector_results.get("ids", [[]])[0]) if vector_results.get("ids") and len(vector_results.get("ids", [])) > 0 else 0,
                   result_keys=list(vector_results.keys()))
        
        vector_chunks: List[Dict] = []
        has_results = (vector_results.get("ids") and 
                      len(vector_results["ids"]) > 0 and 
                      len(vector_results["ids"][0]) > 0)
        
        if has_results:
            ids = vector_results["ids"][0]
            distances = vector_results["distances"][0] if "distances" in vector_results else []
            
            if self.debug_rag:
                syslog2(LOG_DEBUG, "vector store returned", ids_count=len(ids), distances_count=len(distances))
                for idx, cid in enumerate(ids):
                    d = distances[idx] if idx < len(distances) else 0.0
                    syslog2(LOG_DEBUG, "rag raw distance", idx=idx, chunk_id=cid, distance=d)
            
            if self.debug_rag:
                syslog2(LOG_DEBUG, "fetching full text from sqlite", count=len(ids))
            
            session = self.db.get_session()
            try:
                for i, chunk_id in enumerate(ids):
                    distance = distances[i] if i < len(distances) else 0
                    similarity = self._convert_distance_to_similarity(distance)
                    
                    if self.log_level <= LOG_DEBUG or self.debug_rag:
                        syslog2(LOG_DEBUG, "chunk similarity", chunk_id=chunk_id, distance=distance, similarity=similarity)
                    
                    db_chunk = session.query(ChunkModel)\
                        .options(joinedload(ChunkModel.topic_l1), joinedload(ChunkModel.topic_l2))\
                        .filter_by(id=chunk_id).first()
                    
                    if db_chunk:
                        if self.debug_rag:
                            meta = {}
                            if db_chunk.metadata_json:
                                try:
                                    meta = json.loads(db_chunk.metadata_json)
                                except json.JSONDecodeError:
                                    pass
                            syslog2(
                                LOG_DEBUG,
                                "rag chunk result",
                                chunk_id=db_chunk.id,
                                score=similarity,
                                topic_l1_id=meta.get("topic_l1_id") if db_chunk.topic_l1 else None,
                                topic_l2_id=meta.get("topic_l2_id") if db_chunk.topic_l2 else None,
                                source="vector",
                            )
                            self._debug_log_chunk_messages(session, db_chunk)
                        
                        vector_chunks.append(
                            self._build_chunk_dict(db_chunk, similarity, distance=distance, source="vector")
                        )
            finally:
                session.close()
        
        return vector_chunks
    
    def _merge_retrieval_results(
        self, 
        vector_chunks: List[Dict], 
        topic_chunks: List[Dict], 
        n_results: int
    ) -> List[Dict]:
        """
        Merge vector and topic-based retrieval results with weights.
        
        Args:
            vector_chunks: Chunks from vector search
            topic_chunks: Chunks from topic-based search
            n_results: Number of final results to return
            
        Returns:
            Merged and sorted list of chunk dictionaries
        """
        all_chunks: Dict[str, Dict] = {}
        
        vector_weight = 1.0 - self.topic_retrieval_weight
        for chunk in vector_chunks:
            chunk_id = chunk["id"]
            if chunk_id not in all_chunks:
                chunk["score"] *= vector_weight
                all_chunks[chunk_id] = chunk
            else:
                existing_score = all_chunks[chunk_id]["score"]
                new_score = chunk["score"] * vector_weight
                if new_score > existing_score:
                    all_chunks[chunk_id] = chunk
                    all_chunks[chunk_id]["score"] = new_score
        
        topic_weight = self.topic_retrieval_weight
        for chunk in topic_chunks:
            chunk_id = chunk["id"]
            if chunk_id not in all_chunks:
                chunk["score"] *= topic_weight
                all_chunks[chunk_id] = chunk
            else:
                existing_score = all_chunks[chunk_id]["score"]
                topic_score = chunk["score"] * topic_weight
                all_chunks[chunk_id]["score"] = existing_score + topic_score * 0.5
        
        final_chunks = sorted(all_chunks.values(), key=lambda x: x["score"], reverse=True)
        
        # Apply ntop strategy if enabled
        if self.rag_strategy_use_ntop:
            result_count = min(self.rag_ntop, len(final_chunks))
        else:
            result_count = min(n_results, len(final_chunks))
        
        if self.debug_rag:
            syslog2(LOG_DEBUG, "retrieval complete", 
                   vector_count=len(vector_chunks),
                   topic_count=len(topic_chunks),
                   final_count=len(final_chunks),
                   rag_strategy_use_ntop=self.rag_strategy_use_ntop,
                   result_count=result_count)
            for idx, item in enumerate(final_chunks[:result_count]):
                meta = item.get("metadata", {}) or {}
                syslog2(
                    LOG_DEBUG,
                    "rag final result",
                    idx=idx,
                    chunk_id=item["id"],
                    score=item.get("score", 0.0),
                    source=item.get("source", ""),
                    topic_l1_id=meta.get("topic_l1_id"),
                    topic_l2_id=meta.get("topic_l2_id"),
                )
        
        return final_chunks[:result_count]
    
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
        
        if self.log_level <= LOG_INFO or self.debug_rag:
            syslog2(LOG_DEBUG, "retrieval query", query=query, use_topics=use_topics, mode=self.search_mode, n_results=n_results)
        
        # Step 1: Compute query embedding
        query_emb = self._compute_query_embedding(query)
        if not query_emb:
            return []
        
        # Step 2: Select search mode and get vector chunks
        vector_chunks = self._select_search_mode(query_emb, n_results)
        
        # Step 3: Get topic-based chunks if enabled
        topic_chunks: List[Dict] = []
        if use_topics:
            if self.debug_rag:
                syslog2(LOG_DEBUG, "searching topics")
            
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
                
                if self.debug_rag:
                    syslog2(LOG_DEBUG, "found similar topics", l1_count=len(topic_ids_l1), l2_count=len(topic_ids_l2))
                
                topic_chunks = self._retrieve_chunks_from_topics(
                    topic_ids_l1=topic_ids_l1,
                    topic_ids_l2=topic_ids_l2,
                    max_chunks_per_topic=3
                )
        
        # Step 4: Merge results
        return self._merge_retrieval_results(vector_chunks, topic_chunks, n_results)
