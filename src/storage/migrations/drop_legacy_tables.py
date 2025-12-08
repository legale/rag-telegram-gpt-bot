"""
Migration to drop legacy topic tables (topics and topic_chunks).

These tables were used by the old simple clustering system and are no longer needed
since we now use hierarchical topics (topics_l1 and topics_l2).
"""

from sqlalchemy import create_engine, text
from src.core.syslog2 import *


def drop_legacy_tables(db_url: str) -> None:
    """
    Drop legacy topic tables if they exist.
    
    Args:
        db_url: Database URL (e.g., 'sqlite:///path/to/db.sqlite')
    """
    engine = create_engine(db_url)
    
    with engine.connect() as conn:
        try:
            # Drop topic_chunks first (has foreign key to topics)
            conn.execute(text("DROP TABLE IF EXISTS topic_chunks"))
            conn.commit()
            syslog2(LOG_NOTICE, "dropped legacy table", table="topic_chunks")
        except Exception as e:
            syslog2(LOG_WARNING, "failed to drop topic_chunks", error=str(e))
            conn.rollback()
        
        try:
            # Drop topics table
            conn.execute(text("DROP TABLE IF EXISTS topics"))
            conn.commit()
            syslog2(LOG_NOTICE, "dropped legacy table", table="topics")
        except Exception as e:
            syslog2(LOG_WARNING, "failed to drop topics", error=str(e))
            conn.rollback()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python drop_legacy_tables.py <db_url>")
        sys.exit(1)
    
    db_url = sys.argv[1]
    drop_legacy_tables(db_url)
    print("Migration completed.")

