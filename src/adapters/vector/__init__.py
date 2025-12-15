"""Vector adapters - ChromaDB implementations of vector index interfaces."""

from .chroma_vector_index import ChromaVectorIndex

# ChromaTopicIndex and ChromaTopicProvider removed - clustering is deprecated

__all__ = ["ChromaVectorIndex"]
