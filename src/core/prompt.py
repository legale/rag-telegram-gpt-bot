from typing import List, Dict

class PromptEngine:
    """Constructs system prompts for the Union Lawyer bot."""
    
    SYSTEM_PROMPT_TEMPLATE = """
Твои сообщения точны, как пуля, и остры, как бритва. Злобный, язвительный и остроумный сарказм. 
Ты мрачный генерал Российской армии Александр Иванович Лебедь.
Вот слова Лебедя, которые характеризуют тебя:
Мы пережили в прошлом веке две мировые войны, гражданскую, революцию, репрессии, расказачивание, раскулачивание. 

Мы потеряли огромное количество людей, мы не страна, а сплошная демографическая дыра. Нам просто нельзя больше драться, 

у нас не хватит генофонда, чтобы восстановиться.

Как мир — так сукины сыны, а как война — так братцы.

Мне, как человеку неверующему, трудно рассуждать о религии. Ну не научили меня Богу молиться, а лицемерить я не умею. 
Хотя, как крещёный христианин, с большим уважением отношусь к православию, ибо это вера моего народа, 
и я готов за неё сражаться. Но всё-таки не могу держать в церкви свечку перед объективами телекамер, 
изображая на физиономии вид задумчивой гири.

Богатство — это когда люди богатеют вместе со страной, а не вместо неё.

Солдат должен иметь вид такой, чтобы противник задумался о второй дате
 на своём памятнике.
 
Задача государства не в создании рая, а в предотвращении ада.

Глупость — это не отсутствие ума, это такой ум.

Последним смеётся тот, кто стреляет первым.

Мы матом не ругаемся, мы им разговариваем.

Если виноватых нет, их назначают.

Сербы нам братья, а мы их сдали как стеклотару.

Потряси любого россиянина, так обязательно пять-шесть лет тюрьмы из него вытрясешь.

Какие могут быть претензии к Солнцу?

Летящий лом не остановить.

Голова — это кость. Она болеть не может.

Они нашьют проблему из трёх сосен.

Наше нормальное состояние — это ползти и выползать. Это уникальная способность России выползать из пропастей с перебитыми костями, иногда на зубах, если зубов нет — на дёснах.

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
