from typing import List, Dict

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
You are librarian assistant. Be short and precise. Always start with found history context and last messages in chat. Do not use pseudo tables just text.

Контекст из истории чата (наиболее релевантные сообщения):
{context}

Последние сообщения в чате (текущая ситуация):
{history}

Твоя задача:
{task}
"""

    def construct_prompt(self, context_chunks: List[Dict], chat_history: List[Dict], user_task: str, max_context_chars: int = 8000, custom_template: str = None) -> str:
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
        context_str = ""
        total_chars = 0
        import json
        
        for i, chunk in enumerate(context_chunks):
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
            l2_title = meta.get('topic_l2_title')
            l1_title = meta.get('topic_l1_title')
            
            if l2_title:
                topic_header += f"Category: {l2_title} > "
            if l1_title:
                topic_header += f"Topic: {l1_title}"
            
            if topic_header:
                chunk_header = f"--- Chunk {i+1} ({topic_header}) ---\n"
            else:
                chunk_header = f"--- Chunk {i+1} ---\n"
                
            chunk_text = f"{chunk_header}{chunk['text']}\n\n"
            
            if total_chars + len(chunk_text) > max_context_chars:
                # Truncate if we exceed the limit
                remaining = max_context_chars - total_chars
                if remaining > 100:  # Only add if we have meaningful space left
                     context_str += chunk_text[:remaining] + "...\n"
                break
            context_str += chunk_text
            total_chars += len(chunk_text)
            
        if not context_str:
            context_str = "Нет релевантного контекста."
            
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
