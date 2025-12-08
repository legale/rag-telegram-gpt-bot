"""
Migration utility to transfer existing L2 topic center_vec from SQLite to chroma_db.

This should be run once after updating the code to move L2 topic embeddings
from SQLite (where they were stored as JSON) to chroma_db (where they are now stored).
"""

import json
from src.storage.db import Database
from src.storage.vector_store import VectorStore
from src.core.syslog2 import *


def migrate_l2_topics_to_chroma(db: Database, vector_store: VectorStore) -> int:
    """
    Migrate all existing L2 topics from SQLite to chroma_db.
    
    Args:
        db: Database instance
        vector_store: VectorStore instance
        
    Returns:
        Number of topics migrated
    """
    l2_topics = db.get_all_topics_l2()
    
    if not l2_topics:
        syslog2(LOG_INFO, "no l2 topics to migrate")
        return 0
    
    migrated_count = 0
    skipped_count = 0
    
    for topic in l2_topics:
        # Check if already in chroma_db
        try:
            existing = vector_store.topics_l2_collection.get(ids=[f"l2-{topic.id}"])
            if existing and existing.get("ids"):
                syslog2(LOG_DEBUG, "l2 topic already in chroma_db, skipping", topic_id=topic.id)
                skipped_count += 1
                continue
        except Exception:
            pass  # Not found, proceed with migration
        
        # Check if center_vec exists in SQLite (old format)
        if topic.center_vec:
            try:
                center_vec = json.loads(topic.center_vec)
                
                # Save to chroma_db
                vector_store.topics_l2_collection.add(
                    ids=[f"l2-{topic.id}"],
                    embeddings=[center_vec],
                    metadatas=[{
                        "topic_l2_id": topic.id,
                        "title": topic.title,
                        "chunk_count": topic.chunk_count
                    }]
                )
                
                migrated_count += 1
                syslog2(LOG_DEBUG, "migrated l2 topic to chroma_db", topic_id=topic.id, title=topic.title)
            except (json.JSONDecodeError, ValueError) as e:
                syslog2(LOG_WARNING, "failed to parse center_vec for l2 topic", topic_id=topic.id, error=str(e))
                skipped_count += 1
        else:
            syslog2(LOG_DEBUG, "l2 topic has no center_vec, skipping", topic_id=topic.id)
            skipped_count += 1
    
    syslog2(LOG_INFO, "l2 migration complete", migrated=migrated_count, skipped=skipped_count, total=len(l2_topics))
    return migrated_count


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate L2 topics from SQLite to chroma_db")
    parser.add_argument("--db-url", required=True, help="Database URL (e.g., sqlite:///path/to/db.sqlite)")
    parser.add_argument("--vector-db-path", required=True, help="Path to chroma_db directory")
    parser.add_argument("--collection", default="default", help="ChromaDB collection name for chunks")
    
    args = parser.parse_args()
    
    db = Database(args.db_url)
    vector_store = VectorStore(
        persist_directory=args.vector_db_path,
        collection_name=args.collection
    )
    
    count = migrate_l2_topics_to_chroma(db, vector_store)
    print(f"Migration completed. Migrated {count} topics.")

