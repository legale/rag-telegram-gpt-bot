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


class OpenRouterEmbeddingFunction(EmbeddingFunction):
    """если где-то ещё нужен embedding_function для chroma"""
    def __init__(self, embedding_client: EmbeddingClient):
        self.embedding_client = embedding_client

    def __call__(self, input: Documents) -> Embeddings:
        return self.embedding_client.get_embeddings(input)


def get_embedding_function(provider: Optional[str] = None) -> Optional[EmbeddingFunction]:
    if provider is None:
        provider = os.getenv("EMBEDDING_PROVIDER", "local")
    if provider.lower() in ["openrouter", "openai", "current"]:
        client = EmbeddingClient()
        return OpenRouterEmbeddingFunction(client)
    return None

