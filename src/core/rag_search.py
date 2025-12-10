# src/core/rag_search.py

from typing import List, Dict, Optional, Tuple, Any
from src.storage.db import Database, ChunkModel, MessageModel, TopicL1Model, TopicL2Model
from src.storage.vector_store import VectorStore
from src.core.embedding import EmbeddingClient
from src.core.syslog2 import *


class RAGSearch:
    """
    High-level RAG search service that combines vector search with database lookups.
    Provides a unified interface for searching similar chunks with full context.
    """
    
    def __init__(
        self,
        db: Database,
        vector_store: VectorStore,
        embedding_client: EmbeddingClient
    ):
        """
        Initialize RAGSearch service.
        
        Args:
            db: Database instance for chunk and topic storage
            vector_store: Vector store for semantic search
            embedding_client: Client for computing query embeddings
        """
        self.db = db
        self.vector_store = vector_store
        self.embedding_client = embedding_client
    
    def search_similar_chunks(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks and return full context including messages, topics, and link info.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with structure:
            {
                "chunk": ChunkModel,
                "messages": List[MessageModel],
                "topic_l1": Optional[TopicL1Model],
                "topic_l2": Optional[TopicL2Model],
                "link_info": Tuple[chat_id, msg_id, chat_username],
                "similarity": float  # 1 - distance (higher is better)
            }
        """
        if not query:
            return []
        
        # Step 1: Vector search
        vector_results = self.vector_store.query(query_texts=[query], n_results=top_k)
        
        if not vector_results or not vector_results.get("ids") or not vector_results["ids"][0]:
            return []
        
        chunk_ids = vector_results["ids"][0]
        distances = vector_results.get("distances", [[1.0] * len(chunk_ids)])[0]
        
        # Step 2: Retrieve chunks from database with topics
        session = self.db.get_session()
        results = []
        
        try:
            for chunk_id, distance in zip(chunk_ids, distances):
                # Get chunk with topic relationships
                chunk = session.query(ChunkModel).filter(ChunkModel.id == chunk_id).first()
                if not chunk:
                    continue
                
                # Get messages for this chunk
                messages = self._get_chunk_messages(session, chunk)
                
                # Get topics
                topic_l1 = None
                topic_l2 = None
                if chunk.topic_l1_id:
                    topic_l1 = session.query(TopicL1Model).filter(
                        TopicL1Model.id == chunk.topic_l1_id
                    ).first()
                    if topic_l1 and topic_l1.parent_l2_id:
                        topic_l2 = session.query(TopicL2Model).filter(
                            TopicL2Model.id == topic_l1.parent_l2_id
                        ).first()
                elif chunk.topic_l2_id:
                    topic_l2 = session.query(TopicL2Model).filter(
                        TopicL2Model.id == chunk.topic_l2_id
                    ).first()
                
                # Get link info
                link_info = self.db.get_chunk_link_info(chunk_id)
                
                # Convert distance to similarity (1 - distance, higher is better)
                similarity = 1.0 - distance if distance <= 1.0 else 0.0
                
                results.append({
                    "chunk": chunk,
                    "messages": messages,
                    "topic_l1": topic_l1,
                    "topic_l2": topic_l2,
                    "link_info": link_info,
                    "similarity": similarity
                })
        finally:
            session.close()
        
        return results
    
    def _get_chunk_messages(self, session, chunk: ChunkModel) -> List[MessageModel]:
        """
        Get all messages belonging to a chunk.
        
        Args:
            session: Database session
            chunk: ChunkModel instance
            
        Returns:
            List of MessageModel instances
        """
        if not chunk.chat_id:
            return []
        
        # Try to get messages by timestamp range (most reliable)
        if chunk.ts_from and chunk.ts_to:
            messages = session.query(MessageModel).filter(
                MessageModel.chat_id == chunk.chat_id,
                MessageModel.ts >= chunk.ts_from,
                MessageModel.ts <= chunk.ts_to
            ).order_by(MessageModel.ts).all()
            if messages:
                return messages
        
        # Fallback: get messages by msg_id range
        if chunk.msg_id_start and chunk.msg_id_end:
            messages = session.query(MessageModel).filter(
                MessageModel.chat_id == chunk.chat_id,
                MessageModel.msg_id >= chunk.msg_id_start,
                MessageModel.msg_id <= chunk.msg_id_end
            ).order_by(MessageModel.ts).all()
            if messages:
                return messages
        
        # Last resort: get single message by msg_id_start
        if chunk.msg_id_start:
            message = session.query(MessageModel).filter(
                MessageModel.msg_id == chunk.msg_id_start
            ).first()
            if message:
                return [message]
        
        return []

