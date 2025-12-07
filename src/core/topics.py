import numpy as np
import json
import logging
from typing import List, Tuple, Dict, Optional, Any
import os

from src.storage.db import Database
from src.storage.vector_store import VectorStore
from src.core.llm import LLMClient
from src.core.syslog2 import *

# Configurable constants
DEFAULT_CLUSTERS = 15  # Default number of topics if not dynamic
MAX_REPRESENTATIVE_CHUNKS = 5
MAX_CHUNK_CHARS = 300  # Truncate for prompt

class SimpleKMeans:
    """
    A simple implementation of K-Means clustering using NumPy.
    Used when scikit-learn is not available.
    """
    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 100,
        tol: float = 1e-4,
        seed: int = 42,
        show_progress: bool = False,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.centroids = None
        self.labels_ = None
        self.show_progress = show_progress

    def fit(self, X: np.ndarray):
        np.random.seed(self.seed)
        n_samples, n_features = X.shape
        
        # If fewer samples than clusters, adjust k
        n_clusters = min(self.n_clusters, n_samples)
        
        # Initialize centroids randomly
        indices = np.random.choice(n_samples, n_clusters, replace=False)
        self.centroids = X[indices]
        
        for it in range(self.max_iter):
            # Optimize distance calculation: ||x-c||^2 = ||x||^2 + ||c||^2 - 2<x,c>
            # X: (N, D)
            # Centroids: (K, D)
            
            X_sq = (X**2).sum(axis=1)[:, np.newaxis] # (N, 1)
            C_sq = (self.centroids**2).sum(axis=1)   # (K,)
            
            # (N, K) distance matrix squared
            # Note: we don't strictly need sqrt for argmin
            dists_sq = X_sq + C_sq - 2 * np.dot(X, self.centroids.T)
            
            labels = np.argmin(dists_sq, axis=1)
            
            new_centroids = np.zeros_like(self.centroids)
            for k in range(n_clusters):
                mask = labels == k
                if np.any(mask):
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    # Re-initialize empty cluster
                    new_centroids[k] = X[np.random.choice(n_samples)]
            
            if self.show_progress:
                print(f"\rkmeans: iter {it+1}/{self.max_iter}", end="", flush=True)
            
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                self.centroids = new_centroids
                self.labels_ = labels
                break
                
            self.centroids = new_centroids
            self.labels_ = labels

        if self.show_progress:
            print()
            
        return self

class TopicSummarizer:
    """Helper to generate title and description for a topic using LLM."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def generate_topic_title_and_description(self, representative_texts: List[str]) -> Tuple[str, str]:
        """
        Generates a short title and one-sentence description.
        
        Args:
            representative_texts: List of text chunks from the cluster.
            
        Returns:
            start (title, description) tuple.
        """
        # Prepare context
        # Truncate texts to avoid huge prompts
        texts = [t[:MAX_CHUNK_CHARS] + "..." if len(t) > MAX_CHUNK_CHARS else t for t in representative_texts]
        
        examples_str = "\n---\n".join(texts)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that summarizes topics from chat logs. "
                    "Output must be valid JSON with keys: 'title', 'description'."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Analyze the following {len(texts)} text snippets which belong to the same topic:\n\n"
                    f"{examples_str}\n\n"
                    "Provide:\n"
                    "1. A short topic title (3-7 words), suitable for a menu.\n"
                    "2. A single concise sentence describing the covered subject.\n\n"
                    "Respond ONLY with a JSON object: {\"title\": \"...\", \"description\": \"...\"}"
                )
            }
        ]
        
        try:
            response = self.llm.complete(messages, temperature=0.3, max_tokens=150)
            # Parse JSON
            # Clean up potential markdown formatting like ```json ... ```
            clean_resp = response.strip()
            if clean_resp.startswith("```"):
                clean_resp = clean_resp.split("```", 2)[1]
                if clean_resp.startswith("json"):
                    clean_resp = clean_resp[4:]
            
            data = json.loads(clean_resp.strip())
            title = data.get("title", "Untitled Topic").strip()
            description = data.get("description", "No description provided.").strip()
            
            return title, description
            
        except Exception as e:
            syslog2(LOG_ERR, "topic summary failed", error=str(e))
            return "Unknown Topic", "Could not generate description."

class TopicBuilder:
    """Pipeline to build topics from existing embeddings."""
    
    def __init__(self, db_url: str, vector_store_path: str):
        self.db = Database(db_url)
        self.vector_store = VectorStore(vector_store_path)
        self.llm_client = LLMClient()
        
    def build_topics(self, clear_existing: bool = True, show_progress: bool = True) -> int:
        """
        Main pipeline: load vectors -> cluster -> summarize -> save.
        Returns number of topics created.
        """
        if show_progress:
            print("Loading embeddings from ChromaDB...")
        collection_data = self.vector_store.get_all_embeddings()
        
        embeddings = collection_data.get("embeddings") # List[List[float]]
        ids = collection_data.get("ids")             # List[str]
        
        if embeddings is None or len(embeddings) == 0 or ids is None or len(ids) == 0:
            if show_progress:
                print("No embeddings found in Vector Store.")
            else:
                # silent mode still returns 0
                pass
            return 0
            
        X = np.array(embeddings)
        valid_ids = np.array(ids)
        
        n_samples = len(X)
        if show_progress:
            print(f"Found {n_samples} vectors. Starting clustering...")
        
        # Determine K (simple heuristic: sqrt(N/2), capped at 50)
        if n_samples < 5:
            k = 1
        else:
            k = int(np.sqrt(n_samples / 2))
            k = max(2, min(k, 50)) # Clamp between 2 and 50
            
        if show_progress:
            print(f"Clustering into approx {k} topics using SimpleKMeans...")
        
        kmeans = SimpleKMeans(n_clusters=k, show_progress=show_progress)
        kmeans.fit(X)
        labels = kmeans.labels_
        
        # Prepare DB
        if clear_existing:
            if show_progress:
                print("Clearing existing topics...")
            self.db.clear_topics()
            
        summarizer = TopicSummarizer(self.llm_client)
        
        topics_created = 0
        
        for idx, cluster_id in enumerate(range(k)):
            # Get indices for this cluster
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) < 3:
                # Skip tiny clusters (noise)
                if show_progress:
                    print(
                        f"\rtopics: processed {idx+1}/{k} clusters, "
                        f"topics={topics_created} (cluster {cluster_id} skipped: {len(cluster_indices)} chunks)",
                        end="",
                        flush=True,
                    )
                continue
                
            cluster_ids = valid_ids[cluster_indices]
            
            # Find representative chunks (closest to centroid, or just first N)
            centroid = kmeans.centroids[cluster_id]
            cluster_vectors = X[cluster_indices]
            dists = np.linalg.norm(cluster_vectors - centroid, axis=1)
            sorted_local_indices = np.argsort(dists)
            top_n_indices = sorted_local_indices[:MAX_REPRESENTATIVE_CHUNKS]
            top_ids = cluster_ids[top_n_indices]
            
            # Retrieve text from DB
            texts = []
            for cid in top_ids:
                txt = self.db.get_chunk_text(cid)
                if txt:
                    texts.append(txt)
            
            if not texts:
                if show_progress:
                    print(
                        f"\rtopics: processed {idx+1}/{k} clusters, "
                        f"topics={topics_created} (cluster {cluster_id} has no texts)",
                        end="",
                        flush=True,
                    )
                continue
                
            if show_progress:
                print(
                    f"\rSummarizing topic {cluster_id+1}/{k} "
                    f"({len(cluster_indices)} chunks, reps={len(texts)})...",
                    end="",
                    flush=True,
                )
            
            title, description = summarizer.generate_topic_title_and_description(texts)
            
            # Save to DB
            topic_id = self.db.create_topic(title, description)
            
            # Link chunks
            self.db.add_topic_chunks(topic_id, cluster_ids.tolist())
            
            topics_created += 1
            
            if show_progress:
                print(
                    f"\rtopics: processed {idx+1}/{k} clusters, "
                    f"topics={topics_created} (last='{title}')",
                    end="",
                    flush=True,
                )
            
        if show_progress:
            print()
            print(f"Done. Created {topics_created} topics.")
        return topics_created
