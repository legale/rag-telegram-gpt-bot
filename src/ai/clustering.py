
import numpy as np
import hdbscan
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
import random
import warnings

from src.storage.db import Database
from src.storage.vector_store import VectorStore
from src.core.syslog2 import syslog2, LOG_INFO, LOG_WARNING, LOG_ERR, LOG_DEBUG, LOG_NOTICE
from src.core.llm import LLMClient
from src.core.prompt import TOPIC_L1_NAMING_PROMPT, TOPIC_L2_NAMING_PROMPT

# Suppress sklearn deprecation warnings from hdbscan
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

class TopicClusterer:
    """
    Handles hierarchical topic clustering logic.
    Phase 14.3: L1 Clustering (Fine-grained)
    Phase 14.4: L2 Clustering (Super-topics)
    Phase 14.5: Topic Naming (LLM)
    """
    def __init__(self, db: Database, vector_store: VectorStore, llm_client: Optional[LLMClient] = None):
        self.db = db
        self.vector_store = vector_store
        self.llm_client = llm_client

    def perform_l1_clustering(
        self, 
        min_cluster_size: int = 2, 
        min_samples: int = 1,
        metric: str = 'cosine',
        cluster_selection_method: str = 'eom',
        cluster_selection_epsilon: float = 0.0
    ):
        """
        Fetches all chunks, clusters them using HDBSCAN, and saves L1 topics.
        
        Args:
            min_cluster_size: Minimum number of chunks in a cluster (default: 2)
                Lower values = more topics, but may create noise clusters
            min_samples: Minimum samples in neighborhood (default: 1)
                Lower values = more topics, but may be less stable
            metric: Distance metric ('cosine', 'euclidean', 'manhattan', etc.)
                'cosine' works well for embeddings, 'euclidean' may find more clusters
            cluster_selection_method: 'eom' (Excess of Mass) or 'leaf' (Leaf)
                'leaf' tends to create more, smaller clusters
                'eom' is more conservative and creates fewer, larger clusters
            cluster_selection_epsilon: A distance threshold (0.0 = automatic)
                Higher values merge more clusters, creating fewer topics
        """
        syslog2(LOG_INFO, "starting l1 clustering", 
                min_cluster_size=min_cluster_size, 
                min_samples=min_samples,
                metric=metric,
                method=cluster_selection_method)

        # 1. Fetch data
        data = self.vector_store.get_all_embeddings()
        ids = data['ids']
        embeddings = data['embeddings']
        metadatas = data['metadatas']

        if not ids or embeddings is None or (isinstance(embeddings, list) and not embeddings) or (hasattr(embeddings, 'size') and embeddings.size == 0):
            syslog2(LOG_WARNING, "no data for clustering")
            return

        X = np.array(embeddings)
        
        # Check dimensionality
        if len(X.shape) != 2:
             syslog2(LOG_ERR, "invalid embeddings shape", shape=str(X.shape))
             return

        n_samples = len(ids)
        syslog2(LOG_INFO, "clustering chunks", count=n_samples)

        # Validate min_cluster_size (HDBSCAN requires >= 2)
        if min_cluster_size < 2:
            syslog2(LOG_ERR, "min_cluster_size must be at least 2 (HDBSCAN requirement)", 
                    provided=min_cluster_size)
            raise ValueError(f"min_cluster_size must be at least 2, got {min_cluster_size}")

        # Auto-adjust min_cluster_size if too large for dataset
        if min_cluster_size > n_samples:
            syslog2(LOG_WARNING, "min_cluster_size too large, adjusting", 
                    original=min_cluster_size, adjusted=n_samples)
            min_cluster_size = max(2, n_samples // 2)

        # Normalize embeddings if using cosine metric
        # HDBSCAN doesn't support cosine directly, but euclidean on normalized vectors is equivalent
        actual_metric = metric
        if metric == 'cosine':
            # Normalize vectors to unit length
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.where(norms == 0, 1, norms)
            X = X / norms
            actual_metric = 'euclidean'
            syslog2(LOG_DEBUG, "normalized embeddings for cosine similarity, using euclidean metric")

        # 2. Run HDBSCAN
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=actual_metric,
                cluster_selection_method=cluster_selection_method,
                cluster_selection_epsilon=cluster_selection_epsilon
            )
            labels = clusterer.fit_predict(X)
        except (ValueError, KeyError) as e:
            error_msg = str(e)
            if "Min cluster size must be greater than one" in error_msg:
                syslog2(LOG_ERR, "HDBSCAN requires min_cluster_size >= 2", provided=min_cluster_size)
                raise ValueError(f"min_cluster_size must be at least 2, got {min_cluster_size}") from None
            elif "metric" in error_msg.lower() or "unrecognized" in error_msg.lower():
                # Metric not supported - this shouldn't happen if we normalized for cosine
                raise ValueError(f"Metric '{actual_metric}' not supported by HDBSCAN: {error_msg}") from None
            else:
                raise
        
        # 3. Process clusters
        # Group chunk IDs by label
        clusters: Dict[int, List[int]] = {}  # label -> list of indices in 'ids' array
        
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
        
        # Statistics
        noise_count = len(clusters.get(-1, []))
        valid_clusters = len(clusters) - (1 if -1 in clusters else 0)
        noise_percentage = (noise_count / n_samples * 100) if n_samples > 0 else 0
        
        syslog2(LOG_INFO, "clustering complete", 
                clusters_found=valid_clusters,
                noise_count=noise_count,
                noise_percentage=f"{noise_percentage:.1f}%")
        
        # Warn if too much noise
        if noise_percentage > 50:
            syslog2(LOG_WARNING, "high noise percentage, consider lowering min_cluster_size or min_samples",
                    noise_percentage=f"{noise_percentage:.1f}%")

        # 4. Save topics and update chunks
        # First, clear existing L1 topics? Maybe we should be additive or replace?
        # For now, let's assume a full rebuild is requested (safer for consistency).
        self.db.clear_topics_l1()
        
        for label, indices in clusters.items():
            if label == -1:
                # Noise - basically "unclustered"
                # We update chunks to have no topic
                for idx in indices:
                    chunk_id = ids[idx]
                    self.db.update_chunk_topics(chunk_id, topic_l1_id=None, topic_l2_id=None)
                continue
                
            # Process real cluster
            cluster_ids = [ids[i] for i in indices]
            cluster_embeddings = X[indices]
            cluster_metas = [metadatas[i] for i in indices]
            
            # Calculate stats
            centroid = np.mean(cluster_embeddings, axis=0).tolist()
            chunk_count = len(cluster_ids)
            
            # Aggregate message counts and dates
            total_msg_count = 0
            ts_start_list = []
            ts_end_list = []
            
            for meta in cluster_metas:
                if meta: # meta might be None
                    total_msg_count += int(meta.get('message_count', 0))
                    s_date = meta.get('start_date')
                    e_date = meta.get('end_date')
                    if s_date: ts_start_list.append(s_date)
                    if e_date: ts_end_list.append(e_date)
            
            ts_from = None
            ts_to = None
            
            if ts_start_list:
                try:
                    ts_from = datetime.fromisoformat(min(ts_start_list))
                except ValueError: pass
                
            if ts_end_list:
                try:
                    ts_to = datetime.fromisoformat(max(ts_end_list))
                except ValueError: pass
                
            # Create Topic L1
            # For now, title is placeholder "Topic {label}" -> will be named by LLM in Phase 14.5
            topic_id = self.db.create_topic_l1(
                title=f"Topic L1-{label}",
                descr="Pending description...",
                chunk_count=chunk_count,
                msg_count=total_msg_count,
                center_vec=centroid,
                ts_from=ts_from,
                ts_to=ts_to
            )
            
            # Assign chunks to this topic
            for chunk_id in cluster_ids:
                self.db.update_chunk_topics(chunk_id, topic_l1_id=topic_id, topic_l2_id=None)
                
        syslog2(LOG_INFO, "l1 topics saved")

    def perform_l2_clustering(
        self, 
        min_cluster_size: int = 3, 
        min_samples: int = 1,
        metric: str = 'cosine',
        cluster_selection_method: str = 'eom',
        cluster_selection_epsilon: float = 0.0
    ):
        """
        Fetches L1 topics, clusters their centroids to form L2 topics.
        
        Args:
            min_cluster_size: Minimum L1 topics per super-topic (default: 3)
            min_samples: Minimum samples in neighborhood (default: 1)
            metric: Distance metric ('cosine', 'euclidean', etc.)
            cluster_selection_method: 'eom' or 'leaf'
            cluster_selection_epsilon: Distance threshold (0.0 = automatic)
        """
        syslog2(LOG_INFO, "starting l2 clustering")
        
        # 1. Fetch L1 topics
        l1_topics = self.db.get_all_topics_l1()
        if not l1_topics:
            syslog2(LOG_WARNING, "no l1 topics found for l2 clustering")
            return

        # Extract centroids and IDs
        ids = []
        embeddings = []
        for t in l1_topics:
            if t.center_vec:
                try:
                    vec = json.loads(t.center_vec)
                    embeddings.append(vec)
                    ids.append(t.id)
                except:
                    pass
        
        if not embeddings:
            syslog2(LOG_WARNING, "no l1 topic centroids found")
            return
            
        X = np.array(embeddings)
        n_l1_topics = len(X)
        
        # Validate min_cluster_size (HDBSCAN requires >= 2)
        if min_cluster_size < 2:
            syslog2(LOG_ERR, "min_cluster_size must be at least 2 (HDBSCAN requirement)", 
                    provided=min_cluster_size)
            raise ValueError(f"min_cluster_size must be at least 2, got {min_cluster_size}")
        
        # If too few points for clustering, maybe skip or just put all in one if requested?
        if n_l1_topics < min_cluster_size:
             syslog2(LOG_NOTICE, "not enough l1 topics for clustering", 
                     count=n_l1_topics, min_required=min_cluster_size)
             return

        syslog2(LOG_INFO, "clustering l1 topics", count=n_l1_topics)

        # Normalize embeddings if using cosine metric
        actual_metric = metric
        if metric == 'cosine':
            # Normalize vectors to unit length
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.where(norms == 0, 1, norms)
            X = X / norms
            actual_metric = 'euclidean'
            syslog2(LOG_DEBUG, "normalized embeddings for cosine similarity, using euclidean metric")

        # 2. Run HDBSCAN
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=actual_metric,
                cluster_selection_method=cluster_selection_method,
                cluster_selection_epsilon=cluster_selection_epsilon
            )
            labels = clusterer.fit_predict(X)
        except (ValueError, KeyError) as e:
            error_msg = str(e)
            if "Min cluster size must be greater than one" in error_msg:
                syslog2(LOG_ERR, "HDBSCAN requires min_cluster_size >= 2", provided=min_cluster_size)
                raise ValueError(f"min_cluster_size must be at least 2, got {min_cluster_size}") from None
            elif "metric" in error_msg.lower() or "unrecognized" in error_msg.lower():
                # Metric not supported - this shouldn't happen if we normalized for cosine
                raise ValueError(f"Metric '{actual_metric}' not supported by HDBSCAN: {error_msg}") from None
            else:
                raise

        # 3. Process Clusters
        self.db.clear_topics_l2()
        
        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
        
        noise_count = len(clusters.get(-1, []))
        valid_clusters = len(clusters) - (1 if -1 in clusters else 0)
        noise_percentage = (noise_count / n_l1_topics * 100) if n_l1_topics > 0 else 0
            
        syslog2(LOG_INFO, "l2 clustering complete", 
                clusters_found=valid_clusters,
                noise_count=noise_count,
                noise_percentage=f"{noise_percentage:.1f}%")

        # 4. Save L2 topics and update L1 parents
        for label, indices in clusters.items():
            if label == -1:
                # Noise - L1 topics remain orphans
                for idx in indices:
                    l1_id = ids[idx]
                    self.db.update_topic_l1_parent(l1_id, parent_l2_id=None)
                    # Use bulk update for chunks
                    self.db.update_chunks_parent_l2(l1_id, topic_l2_id=None)
                continue
                
            # Valid Super-Topic
            cluster_indices = indices # indices in X/ids lists
            cluster_l1_ids = [ids[i] for i in cluster_indices]
            cluster_embeddings = X[cluster_indices]
            
            # Stats
            centroid = np.mean(cluster_embeddings, axis=0).tolist()
            
            # Sum chunk count from member L1 topics
            total_chunk_count = 0
            for i in cluster_indices:
                l1_topic = next(t for t in l1_topics if t.id == ids[i])
                total_chunk_count += l1_topic.chunk_count
                
            l2_id = self.db.create_topic_l2(
                title=f"Topic L2-{label}",
                descr="Pending description...",
                chunk_count=total_chunk_count,
                center_vec=centroid
            )
            
            # Update L1 parents and Chunks
            for l1_id in cluster_l1_ids:
                self.db.update_topic_l1_parent(l1_id, parent_l2_id=l2_id)
                self.db.update_chunks_parent_l2(l1_id, topic_l2_id=l2_id)
                    
        syslog2(LOG_INFO, "l2 topics saved")

    def name_topics(self):
        """
        Generates names for L1 and L2 topics using LLM.
        """
        if not self.llm_client:
            syslog2(LOG_WARNING, "llm client not available for topic naming")
            return

        syslog2(LOG_INFO, "starting topic naming")
        
        # 1. Name L1 Topics
        l1_topics = self.db.get_all_topics_l1()
        syslog2(LOG_INFO, "naming l1 topics", count=len(l1_topics))
        
        for topic in l1_topics:
            self._name_l1_topic(topic)
            
        # 2. Name L2 Topics
        l2_topics = self.db.get_all_topics_l2()
        syslog2(LOG_INFO, "naming l2 topics", count=len(l2_topics))
        
        for topic in l2_topics:
            self._name_l2_topic(topic)
            
        syslog2(LOG_INFO, "topic naming complete")

    def _name_l1_topic(self, topic) -> None:
        """Helper to name a single L1 topic."""
        # Fetch chunks for this topic
        chunks = self.db.get_chunks_by_topic_l1(topic.id)
        if not chunks:
            return

        # Simple sampling: take up to 5 chunks, maybe random?
        sample_size = min(len(chunks), 5)
        # Using sorted or random? Random is better for diversity if chunks are time-ordered
        sample_chunks = random.sample(chunks, sample_size)
        
        messages_text = "\n\n".join([f"Chunk {i+1}:\n{c.text[:200]}..." for i, c in enumerate(sample_chunks)])
        
        prompt = TOPIC_L1_NAMING_PROMPT.format(messages=messages_text)
        
        try:
            response_json = self._call_llm_json(prompt)
            if response_json:
                title = response_json.get("title", f"Topic {topic.id}")
                description = response_json.get("description", "No description generated.")
                
                # Update DB directly (we need a method in DB or use session directly, 
                # but DB class is preferred abstraction)
                # Adding a dedicated update method would be clean, but for now we can do raw update via session if needed
                # Actually let's add a specialized update method to DB class later or just direct access here?
                # DB class has update_topic_l1_parent... let's add update_topic_l1_info
                
                # IMPORTANT: Since I cannot modify DB class inside this tool call, 
                # I will assume I can add this method or use a generic one. 
                # For now let's hack it via direct session usage or assume method exists and add it next step.
                # Actually, I'll add the method to DB in the next step.
                # For now, I will use a placeholder call.
                self.db.update_topic_l1_info(topic.id, title, description)
                
        except Exception as e:
            syslog2(LOG_ERR, "failed to name l1 topic", id=topic.id, error=str(e))

    def _name_l2_topic(self, topic) -> None:
        """Helper to name a single L2 topic."""
        # Fetch L1 subtopics
        subtopics = self.db.get_l1_topics_by_l2(topic.id)
        if not subtopics:
            return
            
        subtopics_text = "\n".join([f"- {t.title}: {t.descr}" for t in subtopics])
        
        prompt = TOPIC_L2_NAMING_PROMPT.format(subtopics=subtopics_text)
        
        try:
            response_json = self._call_llm_json(prompt)
            if response_json:
                title = response_json.get("title", f"Super-Topic {topic.id}")
                description = response_json.get("description", "No description generated.")
                
                self.db.update_topic_l2_info(topic.id, title, description)
                
        except Exception as e:
            syslog2(LOG_ERR, "failed to name l2 topic", id=topic.id, error=str(e))

    def _call_llm_json(self, prompt: str) -> Optional[Dict]:
        """Helper to call LLM and parse JSON response."""
        try:
            # We use a dummy message structure for the LLMClient
            # Assuming LLMClient.complete takes list of messages
            messages = [{"role": "user", "content": prompt}]
            
            # Since LLMClient might not support JSON mode explicitly yet, we just ask for it in prompt.
            response = self.llm_client.complete(messages)
            
            # Try to find JSON block in response
            # Sometimes LLMs wrap in ```json ... ```
            clean_response = response.strip()
            if "```json" in clean_response:
                clean_response = clean_response.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_response:
                clean_response = clean_response.split("```")[1].split("```")[0].strip()
            
            # Remove any leading/trailing non-json chars?
            # Ideally try to parse
            return json.loads(clean_response)
        except json.JSONDecodeError:
            syslog2(LOG_WARNING, "failed to parse json from llm", response=response[:100])
            return None
        except Exception as e:
            syslog2(LOG_ERR, "llm call error", error=str(e))
            return None
