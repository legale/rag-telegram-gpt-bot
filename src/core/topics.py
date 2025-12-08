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
                syslog2(LOG_DEBUG, "kmeans iteration", iteration=it+1, max_iter=self.max_iter)
            
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                self.centroids = new_centroids
                self.labels_ = labels
                break
                
            self.centroids = new_centroids
            self.labels_ = labels

        if self.show_progress:
            syslog2(LOG_DEBUG, "kmeans complete")
            
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
