"""
Database statistics utilities for Legale Bot.

Provides centralized access to database statistics including:
- Chunk counts
- Database sizes
- Date ranges
- Vector store information
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from src.core.syslog2 import *


class DatabaseStatsService:
    """Service for retrieving database statistics."""
    
    @staticmethod
    def get_chunk_count(db_path: Path) -> int:
        """
        Get the number of chunks in the database.
        
        Args:
            db_path: Path to the SQLite database
            
        Returns:
            Number of chunks, or 0 if error
        """
        if not db_path.exists():
            return 0
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunks")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            syslog2(LOG_ERR, "get chunk count failed", error=str(e))
            return 0
    
    @staticmethod
    def get_database_size(db_path: Path) -> float:
        """
        Get the database file size in MB.
        
        Args:
            db_path: Path to the SQLite database
            
        Returns:
            Size in MB, or 0.0 if file doesn't exist
        """
        if not db_path.exists():
            return 0.0
        
        try:
            size_bytes = db_path.stat().st_size
            return size_bytes / (1024 * 1024)
        except Exception as e:
            syslog2(LOG_ERR, "get database size failed", error=str(e))
            return 0.0
    
    @staticmethod
    def get_date_range(db_path: Path) -> Optional[Tuple[str, str]]:
        """
        Get the date range of chunks in the database.
        
        Args:
            db_path: Path to the SQLite database
            
        Returns:
            Tuple of (min_date, max_date) as strings, or None if error
        """
        if not db_path.exists():
            return None
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    MIN(json_extract(metadata_json, '$.date')) as min_date,
                    MAX(json_extract(metadata_json, '$.date')) as max_date
                FROM chunks
                WHERE json_extract(metadata_json, '$.date') IS NOT NULL
            """)
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0] and result[1]:
                return (result[0][:10], result[1][:10])
            return None
        except Exception as e:
            syslog2(LOG_ERR, "get date range failed", error=str(e))
            return None
    
    @staticmethod
    def get_vector_store_size(vector_path: Path) -> float:
        """
        Get the total size of vector store directory in MB.
        
        Args:
            vector_path: Path to the vector store directory
            
        Returns:
            Size in MB, or 0.0 if directory doesn't exist
        """
        if not vector_path.exists():
            return 0.0
        
        try:
            total_size = sum(
                f.stat().st_size 
                for f in vector_path.rglob('*') 
                if f.is_file()
            )
            return total_size / (1024 * 1024)
        except Exception as e:
            syslog2(LOG_ERR, "get vector store size failed", error=str(e))
            return 0.0
    
    @staticmethod
    def check_database_health(db_path: Path) -> bool:
        """
        Check if database is accessible and healthy.
        
        Args:
            db_path: Path to the SQLite database
            
        Returns:
            True if database is healthy, False otherwise
        """
        if not db_path.exists():
            return False
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            # PRAGMA quick_check returns 'ok' if healthy
            cursor.execute("PRAGMA quick_check")
            result = cursor.fetchone()
            conn.close()
            return result and result[0] == 'ok'
        except Exception as e:
            syslog2(LOG_ERR, "database health check failed", error=str(e))
            return False
    
    @staticmethod
    def get_database_stats(db_path: Path) -> Dict[str, Any]:
        """
        Get comprehensive database statistics.
        
        Args:
            db_path: Path to the SQLite database
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            "exists": db_path.exists(),
            "size_mb": 0.0,
            "chunk_count": 0,
            "date_range": None,
            "healthy": False,
        }
        
        if not db_path.exists():
            return stats
        
        stats["size_mb"] = DatabaseStatsService.get_database_size(db_path)
        stats["chunk_count"] = DatabaseStatsService.get_chunk_count(db_path)
        stats["date_range"] = DatabaseStatsService.get_date_range(db_path)
        stats["healthy"] = DatabaseStatsService.check_database_health(db_path)
        
        return stats
    
    @staticmethod
    def get_vector_store_stats(vector_path: Path) -> Dict[str, Any]:
        """
        Get vector store statistics.
        
        Args:
            vector_path: Path to the vector store directory
            
        Returns:
            Dictionary with statistics
        """
        return {
            "exists": vector_path.exists(),
            "size_mb": DatabaseStatsService.get_vector_store_size(vector_path),
        }
