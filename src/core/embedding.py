# src/core/embedding.py

from openai import OpenAI
from typing import List, Optional, Tuple
import os
import json
try:
    from chromadb import EmbeddingFunction, Documents, Embeddings
except ImportError:
    EmbeddingFunction = object
    Documents = List[str]
    Embeddings = List[List[float]]

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class EmbeddingClient:
    """client for generating text embeddings using an openai-compatible api (e.g., openrouter)"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "text-embedding-3-small",
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        self.model = model
            
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """single api call for given batch"""
        cleaned_texts = [text.replace("\n", " ") for text in texts]
        resp = self.client.embeddings.create(
            input=cleaned_texts,
            model=self.model,
        )
        return [d.embedding for d in resp.data]

    def get_embeddings_batched(
        self,
        texts: List[str],
        batch_size: int = 128,
        show_progress: bool = True,
    ) -> List[List[float]]:
        """batched embeddings with simple progress"""
        total = len(texts)
        if total == 0:
            return []

        all_embs: List[List[float]] = []
        done = 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = texts[start:end]

            batch_embs = self.get_embeddings(batch)
            all_embs.extend(batch_embs)

            done = end
            if show_progress:
                print(f"\rembeddings: {done}/{total} ({done * 100 // total}%)", end="", flush=True)

        return all_embs

    def get_embedding(self, text: str) -> List[float]:
        return self.get_embeddings([text])[0]

    def embed_and_save_jsonl(
        self,
        ids: List[str],
        texts: List[str],
        out_path: str,
        batch_size: int = 128,
        show_progress: bool = True,
    ) -> List[List[float]]:
        """
        offline phase: compute embeddings with progress and save to jsonl
        each line: {"id": ..., "embedding": [...]}
        returns full embeddings list в том же порядке, что ids/texts
        """
        if len(ids) != len(texts):
            raise ValueError("ids and texts must have same length")

        total = len(texts)
        if total == 0:
            with open(out_path, "w", encoding="utf-8"):
                pass
            return []

        print(f"embedding {total} texts -> {out_path}")
        all_embs: List[List[float]] = []
        done = 0

        with open(out_path, "w", encoding="utf-8") as f:
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                batch_ids = ids[start:end]
                batch_texts = texts[start:end]

                batch_embs = self.get_embeddings(batch_texts)
                all_embs.extend(batch_embs)

                for cid, emb in zip(batch_ids, batch_embs):
                    rec = {"id": cid, "embedding": emb}
                    f.write(json.dumps(rec, ensure_ascii=False))
                    f.write("\n")

                done = end
                if show_progress:
                    print(f"embeddings: {done}/{total} ({done * 100 // total}%)")

        return all_embs

    @staticmethod
    def load_embeddings_jsonl(path: str) -> Tuple[List[str], List[List[float]]]:
        """load ids and embeddings back from jsonl produced by embed_and_save_jsonl"""
        ids: List[str] = []
        embs: List[List[float]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                ids.append(obj["id"])
                embs.append(obj["embedding"])
        return ids, embs


class LocalEmbeddingClient:
    """Client for generating text embeddings locally using sentence-transformers."""
    
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        """
        Initialize local embedding client.
        
        Args:
            model: Model name from sentence-transformers (e.g., "all-MiniLM-L6-v2")
        """
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            )
        
        self.model_name = model
        self._model = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        cleaned_texts = [text.replace("\n", " ") for text in texts]
        embeddings = self.model.encode(cleaned_texts, show_progress_bar=False)
        return embeddings.tolist()
    
    def get_embeddings_batched(
        self,
        texts: List[str],
        batch_size: int = 128,
        show_progress: bool = True,
    ) -> List[List[float]]:
        """Batched embeddings with progress."""
        total = len(texts)
        if total == 0:
            return []

        all_embs: List[List[float]] = []
        done = 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = texts[start:end]

            batch_embs = self.get_embeddings(batch)
            all_embs.extend(batch_embs)

            done = end
            if show_progress:
                print(f"\rembeddings: {done}/{total} ({done * 100 // total}%)", end="", flush=True)

        return all_embs
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.get_embeddings([text])[0]
    
    def embed_and_save_jsonl(
        self,
        ids: List[str],
        texts: List[str],
        out_path: str,
        batch_size: int = 128,
        show_progress: bool = True,
    ) -> List[List[float]]:
        """
        Offline phase: compute embeddings with progress and save to jsonl.
        Each line: {"id": ..., "embedding": [...]}
        Returns full embeddings list in the same order as ids/texts.
        """
        if len(ids) != len(texts):
            raise ValueError("ids and texts must have same length")

        total = len(texts)
        if total == 0:
            with open(out_path, "w", encoding="utf-8"):
                pass
            return []

        print(f"embedding {total} texts -> {out_path}")
        all_embs: List[List[float]] = []
        done = 0

        with open(out_path, "w", encoding="utf-8") as f:
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                batch_ids = ids[start:end]
                batch_texts = texts[start:end]

                batch_embs = self.get_embeddings(batch_texts)
                all_embs.extend(batch_embs)

                for cid, emb in zip(batch_ids, batch_embs):
                    rec = {"id": cid, "embedding": emb}
                    f.write(json.dumps(rec, ensure_ascii=False))
                    f.write("\n")

                done = end
                if show_progress:
                    print(f"embeddings: {done}/{total} ({done * 100 // total}%)")

        return all_embs


class OpenRouterEmbeddingFunction(EmbeddingFunction):
    """если где-то ещё нужен embedding_function для chroma"""
    def __init__(self, embedding_client: EmbeddingClient):
        self.embedding_client = embedding_client

    def __call__(self, input: Documents) -> Embeddings:
        return self.embedding_client.get_embeddings(input)


class LocalEmbeddingFunction(EmbeddingFunction):
    """Embedding function for ChromaDB using local sentence-transformers."""
    def __init__(self, embedding_client: LocalEmbeddingClient):
        self.embedding_client = embedding_client

    def __call__(self, input: Documents) -> Embeddings:
        return self.embedding_client.get_embeddings(input)


def get_embedding_function(
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> Optional[EmbeddingFunction]:
    """
    Get embedding function for ChromaDB.
    
    Args:
        provider: Provider name ("openrouter", "openai", "local")
        model: Model name (for API: "text-embedding-3-small", for local: "all-MiniLM-L6-v2")
    
    Returns:
        EmbeddingFunction instance or None
    """
    if provider is None:
        provider = os.getenv("EMBEDDING_PROVIDER", "local")
    
    provider_lower = provider.lower()
    
    if provider_lower in ["openrouter", "openai", "current"]:
        api_model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        client = EmbeddingClient(model=api_model)
        return OpenRouterEmbeddingFunction(client)
    elif provider_lower == "local":
        local_model = model or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        client = LocalEmbeddingClient(model=local_model)
        return LocalEmbeddingFunction(client)
    
    return None


def create_embedding_client(
    generator: Optional[str] = None,
    model: Optional[str] = None
):
    """
    Create embedding client based on generator type.
    
    Args:
        generator: Generator type ("openrouter", "openai", "local")
        model: Model name
    
    Returns:
        EmbeddingClient or LocalEmbeddingClient instance
    """
    if generator is None:
        generator = os.getenv("EMBEDDING_PROVIDER", "openrouter")
    
    generator_lower = generator.lower()
    
    if generator_lower in ["openrouter", "openai", "current"]:
        api_model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        return EmbeddingClient(model=api_model)
    elif generator_lower == "local":
        local_model = model or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        try:
            return LocalEmbeddingClient(model=local_model)
        except ImportError:
            import sys
            print("Error: sentence-transformers is not installed.", file=sys.stderr)
            print("Install it with: poetry install", file=sys.stderr)
            print("Or: pip install sentence-transformers", file=sys.stderr)
            sys.exit(1)
    else:
        # Default to openrouter for backward compatibility
        api_model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        return EmbeddingClient(model=api_model)

