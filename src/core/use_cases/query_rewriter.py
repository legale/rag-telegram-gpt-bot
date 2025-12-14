"""Query rewriting and expansion for better search."""

from __future__ import annotations

from typing import List, Optional
from ..interfaces import LLM
from src.lib.syslog2 import *


class QueryRewriter:
    """
    Query rewriter for expansion and rephrasing.
    
    Provides:
    - Query expansion (generate variants)
    - Rephrasing for better embedding search
    """

    def __init__(self, llm: Optional[LLM] = None, log_level: int = LOG_WARNING):
        """
        Initialize QueryRewriter.

        Args:
            llm: Optional LLM for query expansion/rephrasing
            log_level: Logging level
        """
        self.llm = llm
        self.log_level = log_level

    def expand(self, query: str) -> List[str]:
        """
        Generate query variants using LLM.

        Args:
            query: Original query

        Returns:
            List of query variants (including original)
        """
        if not self.llm:
            # No LLM available, return original only
            return [query]

        try:
            prompt = f"""Пользователь ищет: "{query}"

Сгенерируй 2-3 варианта этого запроса с разными формулировками, синонимами, или более конкретными/общими формулировками.

Верни только варианты запросов, каждый с новой строки, без нумерации и дополнительных комментариев."""

            response = self.llm.complete(prompt, system="Ты помощник для расширения поисковых запросов.")
            
            # Parse response - split by newlines
            variants = [line.strip() for line in response.split('\n') if line.strip()]
            
            # Add original query
            variants.insert(0, query)
            
            # Remove duplicates
            seen = set()
            unique_variants = []
            for v in variants:
                if v.lower() not in seen:
                    seen.add(v.lower())
                    unique_variants.append(v)
            
            return unique_variants[:4]  # Max 4 variants
            
        except Exception as e:
            if self.log_level <= LOG_WARNING:
                syslog2(LOG_WARNING, "query_rewriter: expansion failed", error=str(e))
            return [query]

    def rephrase_for_embedding(self, query: str) -> str:
        """
        Rephrase query for better embedding search.

        Args:
            query: Original query

        Returns:
            Rephrased query optimized for semantic search
        """
        if not self.llm:
            return query

        try:
            prompt = f"""Пользователь просит: {query}

Твоя задача выполнить rephrasing для повышения точности эмбединга запроса для поиска в локальной истории чата.

Верни только перефразированный запрос, без дополнительных комментариев."""

            response = self.llm.complete(prompt, system="Ты помощник для перефразирования поисковых запросов.")
            
            # Clean response
            rephrased = response.strip()
            if not rephrased:
                return query
            
            return rephrased
            
        except Exception as e:
            if self.log_level <= LOG_WARNING:
                syslog2(LOG_WARNING, "query_rewriter: rephrasing failed", error=str(e))
            return query

