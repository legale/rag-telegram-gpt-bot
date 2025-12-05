from typing import List, Dict

class PromptEngine:
    """Constructs system prompts for the Union Lawyer bot."""
    
    SYSTEM_PROMPT_TEMPLATE = """
Ты помощник в чате, находишь по контексту исторические сообщения, анализируешь и выносишь суждения или гипотезы. Опирайся на факты из переписки. Пиши обычным текстом без таблиц, эмодзи. цитаты делай через ``` ```


Контекст из истории чата (наиболее релевантные сообщения):
{context}

Последние сообщения в чате (текущая ситуация):
{history}

Твоя задача:
{task}
"""

    def construct_prompt(self, context_chunks: List[Dict], chat_history: List[Dict], user_task: str) -> str:
        """
        Constructs the full system prompt.
        
        Args:
            context_chunks: List of retrieved chunks with 'text' and 'metadata'.
            chat_history: List of recent chat messages (dictionaries with 'sender', 'content').
            user_task: The specific instruction for the bot.
            
        Returns:
            Formatted prompt string.
        """
        # Format context
        context_str = ""
        for i, chunk in enumerate(context_chunks):
            context_str += f"--- Чанк {i+1} ---\n{chunk['text']}\n\n"
            
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
