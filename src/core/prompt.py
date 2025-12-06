from typing import List, Dict

class PromptEngine:
    """Constructs system prompts for the Union Lawyer bot."""
    
    SYSTEM_PROMPT_TEMPLATE = """
ты мотивирующий помощник с юмором висельника, который читает все сообщения в чате и подбадривает или подкалывает участников чата, цитаты делай через ``` ```


Контекст из истории чата (наиболее релевантные сообщения):
{context}

Последние сообщения в чате (текущая ситуация):
{history}

Твоя задача:
{task}
"""

    def construct_prompt(self, context_chunks: List[Dict], chat_history: List[Dict], user_task: str, max_context_chars: int = 8000) -> str:
        """
        Constructs the full system prompt.
        
        Args:
            context_chunks: List of retrieved chunks with 'text' and 'metadata'.
            chat_history: List of recent chat messages (dictionaries with 'sender', 'content').
            user_task: The specific instruction for the bot.
            max_context_chars: Maximum characters for context (to prevent token overflow).
            
        Returns:
            Formatted prompt string.
        """
        # Format context with size limit
        context_str = ""
        total_chars = 0
        for i, chunk in enumerate(context_chunks):
            chunk_text = f"--- Чанк {i+1} ---\n{chunk['text']}\n\n"
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
            
        return self.SYSTEM_PROMPT_TEMPLATE.format(
            context=context_str.strip(),
            history=history_str.strip(),
            task=user_task
        )
