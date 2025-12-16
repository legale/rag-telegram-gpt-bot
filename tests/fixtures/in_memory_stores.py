"""
In-memory implementations of core interfaces for testing.

These implementations allow testing core use cases without dependencies on
SQLite, ChromaDB, Telethon, FastAPI, or any external services.
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any
from datetime import datetime
from collections import defaultdict

from src.core.domain import Message, Chunk, ProfileConfig
from src.core.interfaces import (
    MessageStore,
    ChunkStore,
    VectorIndex,
    Embedder,
    FTSIndex,
    LLM,
    ConfigProvider,
    VectorDoc,
    ScoredDoc,
    SearchFilters,
)


class InMemoryMessageStore:
    """In-memory implementation of MessageStore interface."""
    
    def __init__(self):
        self._messages: Dict[str, Message] = {}
        self._by_chat: Dict[str, List[str]] = defaultdict(list)
    
    def save_batch(self, messages: List[Message]) -> int:
        """Save messages to in-memory store."""
        saved = 0
        for msg in messages:
            if msg.id not in self._messages:
                self._messages[msg.id] = msg
                self._by_chat[msg.chat_id].append(msg.id)
                saved += 1
        return saved
    
    def get_by_chat(self, chat_id: str, limit: int, offset: int) -> List[Message]:
        """Get messages by chat ID."""
        if chat_id == "":
            # Return all messages
            all_ids = list(self._messages.keys())
            ids = all_ids[offset:offset + limit] if limit > 0 else all_ids[offset:]
            return [self._messages[id] for id in ids]
        
        if chat_id not in self._by_chat:
            return []
        
        ids = self._by_chat[chat_id][offset:offset + limit] if limit > 0 else self._by_chat[chat_id][offset:]
        return [self._messages[id] for id in ids]
    
    def get_context(self, chat_id: str, time_point: datetime, window_sec: int) -> List[Message]:
        """Get messages in time window."""
        if chat_id not in self._by_chat:
            return []
        
        time_from = time_point.timestamp() - window_sec
        time_to = time_point.timestamp() + window_sec
        
        messages = []
        for msg_id in self._by_chat[chat_id]:
            msg = self._messages[msg_id]
            msg_time = msg.timestamp.timestamp() if isinstance(msg.timestamp, datetime) else msg.timestamp
            if time_from <= msg_time <= time_to:
                messages.append(msg)
        
        return sorted(messages, key=lambda m: m.timestamp if isinstance(m.timestamp, datetime) else m.timestamp)
    
    def count(self) -> int:
        """Get total message count."""
        return len(self._messages)


class InMemoryChunkStore:
    """In-memory implementation of ChunkStore interface."""
    
    def __init__(self):
        self._chunks: Dict[str, Chunk] = {}
    
    def save_batch(self, chunks: List[Chunk]) -> int:
        """Save chunks to in-memory store."""
        saved = 0
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
            saved += 1
        return saved
    
    def get_by_ids(self, ids: List[str]) -> List[Chunk]:
        """Get chunks by IDs."""
        return [self._chunks[id] for id in ids if id in self._chunks]
    
    def update_topics(self, updates: Dict[str, Any]) -> None:
        """Update topics for chunks."""
        for chunk_id, update in updates.items():
            if chunk_id in self._chunks:
                chunk = self._chunks[chunk_id]
                if hasattr(update, 'topic_l1_id'):
                    if chunk.metadata is None:
                        chunk.metadata = {}
                    chunk.metadata['topic_l1_id'] = update.topic_l1_id
                if hasattr(update, 'topic_l2_id'):
                    if chunk.metadata is None:
                        chunk.metadata = {}
                    chunk.metadata['topic_l2_id'] = update.topic_l2_id
    
    def clear(self) -> None:
        """Clear all chunks."""
        self._chunks.clear()


class InMemoryVectorIndex:
    """In-memory implementation of VectorIndex interface."""
    
    def __init__(self, dimension: int = 384):
        self._docs: Dict[str, VectorDoc] = {}
        self._dimension = dimension
    
    def upsert(self, items: List[VectorDoc]) -> None:
        """Upsert vector documents."""
        for doc in items:
            if len(doc.vector) != self._dimension:
                raise ValueError(f"Vector dimension mismatch: expected {self._dimension}, got {len(doc.vector)}")
            self._docs[doc.id] = doc
    
    def query(self, vector: List[float], top_k: int, filter: Optional[Dict] = None) -> List[ScoredDoc]:
        """Query similar vectors using cosine similarity."""
        if len(vector) != self._dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self._dimension}, got {len(vector)}")
        
        # Simple cosine similarity
        results = []
        for doc_id, doc in self._docs.items():
            if filter:
                # Simple filter matching
                match = True
                for key, value in filter.items():
                    if key not in doc.meta or doc.meta[key] != value:
                        match = False
                        break
                if not match:
                    continue
            
            # Cosine similarity
            dot_product = sum(a * b for a, b in zip(vector, doc.vector))
            norm_a = sum(a * a for a in vector) ** 0.5
            norm_b = sum(b * b for b in doc.vector) ** 0.5
            similarity = dot_product / (norm_a * norm_b) if (norm_a * norm_b) > 0 else 0.0
            
            results.append(ScoredDoc(id=doc_id, score=similarity, meta=doc.meta))
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents by IDs."""
        for doc_id in ids:
            self._docs.pop(doc_id, None)
    
    def count(self) -> int:
        """Get total document count."""
        return len(self._docs)
    
    def get_embeddings_by_ids(self, ids: List[str]) -> Dict[str, List[float]]:
        """Get embeddings by document IDs."""
        result = {}
        for doc_id in ids:
            if doc_id in self._docs:
                result[doc_id] = self._docs[doc_id].vector
        return result


class InMemoryEmbedder:
    """In-memory implementation of Embedder interface."""
    
    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._call_count = 0
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents."""
        self._call_count += 1
        # Simple deterministic embedding based on text hash
        embeddings = []
        for text in texts:
            # Create a deterministic embedding based on text
            embedding = [0.0] * self._dimension
            text_hash = hash(text)
            for i in range(self._dimension):
                embedding[i] = (text_hash % (i + 1)) / (i + 1) / 100.0
            # Normalize
            norm = sum(x * x for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for query."""
        return self.embed_documents([text])[0]


class InMemoryFTSIndex:
    """In-memory implementation of FTSIndex interface."""
    
    def __init__(self):
        self._index: Dict[str, List[str]] = defaultdict(list)  # word -> [doc_ids]
        self._doc_texts: Dict[str, str] = {}  # doc_id -> text
    
    def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[SearchFilters] = None
    ) -> List[ScoredDoc]:
        """Search using simple keyword matching."""
        query_words = self.normalize_text(query).split()
        if not query_words:
            return []
        
        # Score documents by word matches
        doc_scores: Dict[str, float] = defaultdict(float)
        for word in query_words:
            if word in self._index:
                for doc_id in self._index[word]:
                    doc_scores[doc_id] += 1.0
        
        # Convert to ScoredDoc list
        results = [
            ScoredDoc(id=doc_id, score=score, meta={})
            for doc_id, score in doc_scores.items()
        ]
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for indexing."""
        # Lowercase
        text = text.lower()
        # Replace ё with е
        text = text.replace('ё', 'е')
        # Remove punctuation (simple)
        import string
        text = ''.join(c for c in text if c not in string.punctuation)
        return text
    
    def index_document(self, doc_id: str, text: str) -> None:
        """Index a document (helper method for tests)."""
        self._doc_texts[doc_id] = text
        words = self.normalize_text(text).split()
        for word in words:
            if word:
                self._index[word].append(doc_id)


class InMemoryLLM:
    """In-memory implementation of LLM interface."""
    
    def __init__(self):
        self._call_count = 0
        self._responses: Dict[str, str] = {}
    
    def complete(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """Generate completion."""
        self._call_count += 1
        # Return a simple response
        if prompt in self._responses:
            return self._responses[prompt]
        return f"Response to: {prompt[:50]}..."
    
    def set_response(self, prompt: str, response: str) -> None:
        """Set a custom response for a prompt (helper for tests)."""
        self._responses[prompt] = response


class InMemoryConfigProvider:
    """In-memory implementation of ConfigProvider interface."""
    
    def __init__(self):
        self._configs: Dict[str, ProfileConfig] = {}
    
    def get_profile_config(self, profile_name: str) -> ProfileConfig:
        """Get profile configuration."""
        if profile_name not in self._configs:
            # Return default config
            return ProfileConfig(
                profile_name=profile_name,
                embedding_model="test-model",
                embedding_generator="local",
                current_model="openai/gpt-4",
                chunk_token_min=50,
                chunk_token_max=200,
                chunk_overlap_ratio=0.1,
            )
        return self._configs[profile_name]
    
    def set_config(self, profile_name: str, config: ProfileConfig) -> None:
        """Set configuration (helper for tests)."""
        self._configs[profile_name] = config

