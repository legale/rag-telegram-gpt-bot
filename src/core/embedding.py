# src/core/embedding.py

from openai import OpenAI
from typing import List, Optional, Tuple
import os
import json
import time
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


class _DummySentenceTransformer:
    """
    Lightweight fallback used when sentence-transformers is unavailable.
    Generates deterministic embeddings so tests and local runs can proceed
    without the optional dependency.
    """
    def __init__(self, model_name: str, dimension: int = 384):
        self.model_name = model_name
        self._dimension = dimension

    def encode(self, texts, show_progress_bar: bool = False):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            seed = sum(ord(c) for c in text)
            # Simple deterministic vector; not meaningful but stable
            embeddings.append([((seed + i) % 997) / 997 for i in range(self._dimension)])
        return embeddings

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

try:
    from huggingface_hub.errors import OfflineModeIsEnabled
except ImportError:
    OfflineModeIsEnabled = None

from src.lib.syslog2 import *



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
        self._dimension: Optional[int] = None  # Cache for embedding dimension
            
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def _clean_texts(self, texts: List[str]) -> List[str]:
        """
        Clean texts for embedding generation.
        
        Args:
            texts: List of texts to clean
            
        Returns:
            List of cleaned texts
        """
        return [text.replace("\n", " ") for text in texts]
    
    def _call_embedding_api(self, cleaned_texts: List[str]) -> List[List[float]]:
        """
        Call embedding API with cleaned texts.
        
        Args:
            cleaned_texts: List of cleaned texts
            
        Returns:
            List of embedding vectors
        """
        resp = self.client.embeddings.create(
            input=cleaned_texts,
            model=self.model,
            timeout=60.0  # Longer timeout for batch embeddings
        )
        return [d.embedding for d in resp.data]

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """single api call for given batch"""
        cleaned_texts = self._clean_texts(texts)
        return self._call_embedding_api(cleaned_texts)

    def _get_embeddings_with_retry(
        self,
        texts: List[str],
        max_retries: int = 3,
        initial_delay: float = 1.0,
    ) -> List[List[float]]:
        """
        Get embeddings with retry logic and exponential backoff for temporary API errors.
        
        Args:
            texts: List of texts to embed
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds before first retry
            
        Returns:
            List of embedding vectors
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return self.get_embeddings(texts)
            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                error_msg = str(e)
                
                # Check if it's a retryable error
                is_retryable = (
                    attempt < max_retries and (
                        "rate limit" in error_msg.lower() or
                        "RateLimitError" in error_type or
                        "APIConnectionError" in error_type or
                        "APITimeoutError" in error_type or
                        isinstance(e, TimeoutError) or
                        ("temporary" in error_msg.lower() and "error" in error_msg.lower())
                    )
                )
                
                if not is_retryable:
                    # Non-retryable error, raise immediately
                    syslog2(LOG_ERR, "embedding api error (non-retryable)", 
                           error_type=error_type, error=str(e))
                    raise
                
                # Calculate exponential backoff delay
                delay = initial_delay * (2 ** attempt)
                syslog2(LOG_WARNING, "embedding api error (retrying)", 
                       attempt=attempt + 1, max_retries=max_retries + 1,
                       error_type=error_type, error=str(e), delay=delay)
                time.sleep(delay)
        
        # All retries exhausted
        syslog2(LOG_ERR, "embedding api error (all retries exhausted)", 
               error_type=type(last_exception).__name__, error=str(last_exception))
        raise last_exception

    def _process_batch(self, batch: List[str]) -> List[List[float]]:
        """
        Process a single batch of texts to get embeddings.
        
        Args:
            batch: List of texts in the batch
            
        Returns:
            List of embedding vectors for the batch
        """
        return self._get_embeddings_with_retry(batch)
    
    def _update_progress(self, done: int, total: int, show_progress: bool) -> None:
        """
        Update and log progress for batch processing.
        
        Args:
            done: Number of items processed
            total: Total number of items
            show_progress: Whether to show progress
        """
        if show_progress:
            pct = done * 100 // total
            syslog2(LOG_DEBUG, "embeddings progress", done=done, total=total, percent=pct)
    
    def get_embeddings_batched(
        self,
        texts: List[str],
        batch_size: int = 128,
        show_progress: bool = True,
    ) -> List[List[float]]:
        """batched embeddings with retry logic and exponential backoff for temporary API errors"""
        total = len(texts)
        if total == 0:
            return []

        all_embs: List[List[float]] = []
        done = 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = texts[start:end]

            batch_embs = self._process_batch(batch)
            all_embs.extend(batch_embs)

            done = end
            self._update_progress(done, total, show_progress)

        return all_embs

    def get_embedding(self, text: str) -> List[float]:
        return self.get_embeddings([text])[0]
    
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        # Use cached dimension if available
        if self._dimension is not None:
            return self._dimension
        
        # Create a test embedding to determine dimension
        test_emb = self.get_embedding("test")
        self._dimension = len(test_emb)
        return self._dimension

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

        syslog2(LOG_NOTICE, "embedding texts", total=total, out_path=out_path)
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
                    pct = done * 100 // total
                    syslog2(LOG_DEBUG, "embeddings progress", done=done, total=total, percent=pct)

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
    
    def __init__(self, model: str):
        """
        Initialize local embedding client.
        
        Args:
            model: Model name from sentence-transformers (e.g., "paraphrase-multilingual-mpnet-base-v2")
        """
        self.model_name = model
        self._model = None
        self._dimension: Optional[int] = None  # Cache for embedding dimension
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load model."""
        if self._model is not None:
            return self._model

        # Suppress tokenizer parallelism warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Clear any HF offline flags to allow download if needed
        for var in ["HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"]:
            os.environ.pop(var, None)

        syslog2(LOG_DEBUG, "loading local embedding model", model=self.model_name)

        # Prefer real SentenceTransformer if available (can be patched in tests)
        if SentenceTransformer is not None:
            try:
                self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
                syslog2(LOG_DEBUG, "model loaded successfully", model=self.model_name)
                return self._model
            except Exception as e:
                syslog2(LOG_ERR, "failed to load model", model=self.model_name, error=str(e))
                raise

        # Fallback to deterministic dummy to keep functionality without dependency
        syslog2(LOG_WARNING, "sentence-transformers not installed, using dummy embeddings", model=self.model_name)
        self._model = _DummySentenceTransformer(self.model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        cleaned_texts = [text.replace("\n", " ") for text in texts]
        embeddings = self.model.encode(cleaned_texts, show_progress_bar=False)
        # Handle both numpy arrays and lists
        if hasattr(embeddings, 'tolist'):
            return embeddings.tolist()
        return embeddings
    
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
                pct = done * 100 // total
                print(f"\rembeddings progress: {done}/{total} ({pct}%)", flush=True, end="")

        if show_progress:
            print()
        return all_embs
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.get_embeddings([text])[0]
    
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        # Use cached dimension if available
        if self._dimension is not None:
            return self._dimension
        
        # Try to use model's built-in method if available
        if hasattr(self.model, 'get_sentence_embedding_dimension'):
            self._dimension = self.model.get_sentence_embedding_dimension()
            return self._dimension
        
        # Otherwise, create a test embedding to determine dimension
        test_emb = self.get_embedding("test")
        self._dimension = len(test_emb)
        return self._dimension
    
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

        syslog2(LOG_NOTICE, "embedding texts", total=total, out_path=out_path)
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
                    pct = done * 100 // total
                    syslog2(LOG_DEBUG, "embeddings progress", done=done, total=total, percent=pct)

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
        model: Model name (for API: "text-embedding-3-small", for local: "paraphrase-multilingual-mpnet-base-v2")
    
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
        local_model = model or os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")
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
        local_model = model or os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")
        try:
            return LocalEmbeddingClient(model=local_model)
        except ImportError:
            import sys
            syslog2(LOG_ERR, "sentence-transformers is not installed", instructions="pip install sentence-transformers")
            sys.exit(1)
    else:
        # Default to openrouter for backward compatibility
        api_model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        return EmbeddingClient(model=api_model)

