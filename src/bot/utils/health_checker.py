"""
Health check utilities for Legale Bot.

Provides system health checking including:
- Database connectivity
- Vector store status
- API key validation
- System resources
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


class HealthChecker:
    """Utility for performing system health checks."""
    
    @staticmethod
    def check_database(db_path: Path) -> Tuple[str, str]:
        """
        Check database health.
        
        Args:
            db_path: Path to SQLite database
            
        Returns:
            Tuple of (check_name, status_message)
        """
        if not db_path.exists():
            return ("💾 База данных", "⚠️ Не создана")
        
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            return ("💾 База данных", "✅ OK")
        except Exception as e:
            return ("💾 База данных", f"❌ Ошибка: {e}")
    
    @staticmethod
    def check_vector_store(vector_path: Path) -> Tuple[str, str]:
        """
        Check vector store health.
        
        Args:
            vector_path: Path to vector store directory
            
        Returns:
            Tuple of (check_name, status_message)
        """
        if not vector_path.exists():
            return ("🔍 Векторное хранилище", "⚠️ Не создано")
        
        return ("🔍 Векторное хранилище", "✅ OK")
    
    @staticmethod
    def check_llm_api_key() -> Tuple[str, str]:
        """
        Check LLM API key availability.
        
        Returns:
            Tuple of (check_name, status_message)
        """
        try:
            api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            if api_key:
                return ("🤖 LLM API ключ", "✅ Установлен")
            else:
                return ("🤖 LLM API ключ", "❌ Не установлен")
        except Exception as e:
            return ("🤖 LLM API ключ", f"❌ Ошибка: {e}")
    
    @staticmethod
    def check_embedding_api_key() -> Tuple[str, str]:
        """
        Check Embedding API key availability.
        
        Returns:
            Tuple of (check_name, status_message)
        """
        try:
            # Uses same key as LLM currently
            emb_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            if emb_key:
                return ("🔤 Embedding API ключ", "✅ Установлен")
            else:
                return ("🔤 Embedding API ключ", "❌ Не установлен")
        except Exception as e:
            return ("🔤 Embedding API ключ", f"❌ Ошибка: {e}")
    
    @staticmethod
    def check_memory() -> Tuple[str, str]:
        """
        Check system memory usage.
        
        Returns:
            Tuple of (check_name, status_message)
        """
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent < 90:
                return ("💾 Память", f"✅ {memory.percent:.1f}% использовано")
            else:
                return ("💾 Память", f"⚠️ {memory.percent:.1f}% использовано")
        except Exception as e:
            return ("💾 Память", f"❌ Ошибка: {e}")
    
    @staticmethod
    def check_disk(path: Path) -> Tuple[str, str]:
        """
        Check disk space.
        
        Args:
            path: Path to check disk space for
            
        Returns:
            Tuple of (check_name, status_message)
        """
        try:
            import psutil
            disk = psutil.disk_usage(str(path))
            if disk.percent < 90:
                return ("💿 Диск", f"✅ {disk.percent:.1f}% использовано")
            else:
                return ("💿 Диск", f"⚠️ {disk.percent:.1f}% использовано")
        except Exception as e:
            return ("💿 Диск", f"❌ Ошибка: {e}")
    
    @staticmethod
    def run_all_checks(db_path: Path, vector_path: Path, profile_dir: Path) -> List[Tuple[str, str]]:
        """
        Run all health checks.
        
        Args:
            db_path: Path to database
            vector_path: Path to vector store
            profile_dir: Profile directory path
            
        Returns:
            List of (check_name, status_message) tuples
        """
        checks = [
            HealthChecker.check_database(db_path),
            HealthChecker.check_vector_store(vector_path),
            HealthChecker.check_llm_api_key(),
            HealthChecker.check_embedding_api_key(),
            HealthChecker.check_memory(),
            HealthChecker.check_disk(profile_dir),
        ]
        
        return checks
    
    @staticmethod
    def format_health_report(checks: List[Tuple[str, str]]) -> str:
        """
        Format health check results into a report.
        
        Args:
            checks: List of (check_name, status_message) tuples
            
        Returns:
            Formatted health report string
        """
        response = "🏥 **Проверка здоровья системы**\n\n"
        
        for name, status in checks:
            response += f"{name}: {status}\n"
        
        # Overall status
        failed = sum(1 for _, status in checks if "❌" in status)
        warnings = sum(1 for _, status in checks if "⚠️" in status)
        
        response += "\n"
        if failed == 0 and warnings == 0:
            response += "✅ **Все системы работают нормально**"
        elif failed == 0:
            response += f"⚠️ **Обнаружено {warnings} предупреждений**"
        else:
            response += f"❌ **Обнаружено {failed} ошибок, {warnings} предупреждений**"
        
        return response
