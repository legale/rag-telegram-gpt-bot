"""ChromaDB implementation of TopicIndexProvider interface."""

from __future__ import annotations

from typing import Optional

from src.core.interfaces import TopicIndexProvider, TopicIndex
from src.adapters.vector.chroma_topic_index import ChromaTopicIndex
from src.storage.vector_store import VectorStore


class ChromaTopicProvider:
    """ChromaDB adapter implementing TopicIndexProvider protocol."""

    def __init__(self, vector_store: VectorStore):
        """
        Initialize the provider with a VectorStore instance.

        Args:
            vector_store: VectorStore instance that provides topic collections
        """
        self.vector_store = vector_store
        self._l1_index: Optional[TopicIndex] = None
        self._l2_index: Optional[TopicIndex] = None

    def get_l1_index(self) -> TopicIndex:
        """
        Get L1 topic index.

        Returns:
            TopicIndex instance for L1 topics
        """
        if self._l1_index is None:
            l1_collection = self.vector_store.get_topics_l1_collection()
            self._l1_index = ChromaTopicIndex(l1_collection)
        return self._l1_index

    def get_l2_index(self) -> TopicIndex:
        """
        Get L2 topic index.

        Returns:
            TopicIndex instance for L2 topics
        """
        if self._l2_index is None:
            l2_collection = self.vector_store.get_topics_l2_collection()
            self._l2_index = ChromaTopicIndex(l2_collection)
        return self._l2_index

