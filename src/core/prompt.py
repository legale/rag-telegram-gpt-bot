from typing import List, Dict, Optional
from src.lib.syslog2 import *

TOPIC_L1_NAMING_PROMPT = """
You are a summarization assistant.
Analyze the following chat messages that have been grouped into a single topic.
Generate a concise, human-readable Title (max 5-7 words) and a one-sentence Description.
Return JSON format: {{"title": "...", "description": "..."}}

Messages:
{messages}
"""

TOPIC_L2_NAMING_PROMPT = """
You are a summarization assistant.
Analyze the following sub-topics that belong to a super-topic.
Generate a high-level Super-Topic Title (max 3-5 words) and a one-sentence Description of the broader category.
Return JSON format: {{"title": "...", "description": "..."}}

Sub-topics:
{subtopics}
"""

class PromptEngine:
    """Constructs system prompts for the bot."""

    
    SYSTEM_PROMPT_TEMPLATE = """
You are a librarian assistant. Be short, precise, and consistent.

context = retrieved RAG chunks selected for relevance
history = last chat messages showing the current situation

Your rules:
- Always start with a brief summary of context, then react to history.
- Use only plain text, ASCII only.
- No tables, no markup, no formatting blocks.
- Prefer factual compression: extract meaning, avoid long quotes.
- If context contradicts history, history has priority.
- If context is too large or repetitive, summarize it in 1 to 3 short lines.
- When answering, rely on RAG context whenever it improves accuracy.
- If context is missing or irrelevant, answer based on history alone.
- Keep answers minimal unless explicitly asked for details.

Task:
{task}

Context:
{context}

History:
{history}
"""

    def construct_prompt(self, context_chunks: List[Dict], chat_history: List[Dict], user_task: str, max_context_chars: int = 8000, custom_template: str = None, log_level: int = LOG_WARNING) -> str:
        """
        Constructs the full system prompt.
        
        Args:
            context_chunks: List of retrieved chunks with 'text' and 'metadata'.
            chat_history: List of recent chat messages (dictionaries with 'sender', 'content').
            user_task: The specific instruction for the bot.
            max_context_chars: Maximum characters for context (to prevent token overflow).
            custom_template: Optional custom template string overriding the default.
            
        Returns:
            Formatted prompt string.
        """
        # Format context with size limit
        if log_level <= LOG_DEBUG:
            syslog2(LOG_DEBUG, "prompt: construct_prompt called", 
                   context_chunks_count=len(context_chunks),
                   chat_history_count=len(chat_history),
                   user_task=user_task[:50],
                   max_context_chars=max_context_chars)
        
        context_str = ""
        total_chars = 0
        import json
        chunks_without_text = 0
        
        for i, chunk in enumerate(context_chunks):
            # Check if chunk has text
            chunk_text_value = chunk.get('text', '')
            if not chunk_text_value:
                if log_level <= LOG_DEBUG:
                    syslog2(LOG_DEBUG, "prompt: chunk has no text, skipping", 
                           chunk_id=chunk.get('id'),
                           chunk_index=i)
                chunks_without_text += 1
                continue
            
            # Extract metadata
            meta = chunk.get('metadata')
            if isinstance(meta, str) and meta:
                try:
                    meta = json.loads(meta)
                except:
                    meta = {}
            if not isinstance(meta, dict):
                meta = {}
                
            # Format topic header
            topic_header = ""
            # topic_l1_title and topic_l2_title removed - clustering is deprecated
            
            if l2_title:
                topic_header += f"Category: {l2_title} > "
            if l1_title:
                topic_header += f"Topic: {l1_title}"
            
            if topic_header:
                chunk_header = f"--- Chunk {i+1} ({topic_header}) ---\n"
            else:
                chunk_header = f"--- Chunk {i+1} ---\n"
                
            chunk_text = f"{chunk_header}{chunk_text_value}\n\n"
            
            if total_chars + len(chunk_text) > max_context_chars:
                # Truncate if we exceed the limit
                remaining = max_context_chars - total_chars
                if remaining > 100:  # Only add if we have meaningful space left
                     context_str += chunk_text[:remaining] + "...\n"
                if log_level <= LOG_DEBUG:
                    syslog2(LOG_DEBUG, "prompt: context truncated", 
                           total_chars=total_chars,
                           chunk_text_len=len(chunk_text),
                           remaining=remaining)
                break
            context_str += chunk_text
            total_chars += len(chunk_text)
            
            if log_level <= LOG_DEBUG:
                syslog2(LOG_DEBUG, "prompt: chunk added to context", 
                       chunk_index=i,
                       chunk_id=chunk.get('id'),
                       text_length=len(chunk_text_value),
                       total_chars=total_chars)
        
        if not context_str:
            context_str = "Нет релевантного контекста."
            if log_level <= LOG_DEBUG:
                syslog2(LOG_DEBUG, "prompt: no context string generated", 
                       chunks_without_text=chunks_without_text,
                       total_chunks=len(context_chunks))
        else:
            if log_level <= LOG_DEBUG:
                syslog2(LOG_DEBUG, "prompt: context string generated", 
                       context_length=len(context_str),
                       total_chars=total_chars,
                       chunks_processed=len(context_chunks) - chunks_without_text)
            
        # Format history
        history_str = ""
        for msg in chat_history:
            sender = msg.get('sender', 'Unknown')
            content = msg.get('content', '')
            history_str += f"{sender}: {content}\n"
            
        if not history_str:
            history_str = "Нет недавних сообщений."
            
        template = custom_template if custom_template else self.SYSTEM_PROMPT_TEMPLATE
        
        # Ensure template has necessary keys if using custom one? 
        # For now assume user provides correct format or we handle error if format fails.
        # But to be safe let's wrap formatted.
        
        try:
            return template.format(
                context=context_str.strip(),
                history=history_str.strip(),
                task=user_task
            )
        except KeyError as e:
            # Fallback if custom template is broken
             return f"Error in system prompt template: {e}\nUsing default.\n" + self.SYSTEM_PROMPT_TEMPLATE.format(
                context=context_str.strip(),
                history=history_str.strip(),
                task=user_task
            )
