"""User-facing command handlers."""

from __future__ import annotations

from typing import Optional

from src.core.dispatcher import CommandContext, CommandHandler, AsyncCommandHandler, CommandResult


class StartCommandHandler(CommandHandler):
    """Handler for /start command."""

    def handle(self, context: CommandContext) -> CommandResult:
        return CommandResult(success=True, message="Привет!\n\nИспользуйте /help для справки.")


class HelpCommandHandler(CommandHandler):
    """Handler for /help command."""

    def handle(self, context: CommandContext) -> CommandResult:
        message = (
            "Я анализирую историю чата и отвечаю на вопросы.\n\n"
            "Доступные команды:\n"
            "• /start — приветствие\n"
            "• /help — эта справка\n"
            "• /reset — сбросить контекст разговора\n"
            "• /tokens — показать использование токенов\n"
            "• /model — переключить модель LLM\n"
            "• /find <rag_method> <запрос> — поиск сообщений\n"
            "  /find <rag_method> list — показать список методов\n"
            "  Методы: hybrid, vector_only, fts_only\n\n"
            "Просто напишите свой вопрос!"
        )
        return CommandResult(success=True, message=message)


class ResetCommandHandler(CommandHandler):
    """Handler for /reset command."""

    def __init__(self, bot):
        self.bot = bot

    def handle(self, context: CommandContext) -> CommandResult:
        try:
            message = self.bot.reset_context()
            return CommandResult(success=True, message=message)
        except Exception as e:
            from src.lib.syslog2 import LOG_ERR, syslog2

            syslog2(LOG_ERR, "reset context failed", error=str(e))
            return CommandResult(success=False, message="Ошибка при сбросе контекста.", error=str(e))


class TokensCommandHandler(CommandHandler):
    """Handler for /tokens command."""

    def __init__(self, bot):
        self.bot = bot

    def handle(self, context: CommandContext) -> CommandResult:
        try:
            usage = self.bot.get_token_usage()
            response = (
                "Использование токенов:\n\n"
                f"Текущее: {usage['current_tokens']:,}\n"
                f"Максимум: {usage['max_tokens']:,}\n"
                f"Использовано: {usage['percentage']}%\n\n"
            )
            if usage["percentage"] > 80:
                response += "Приближаетесь к лимиту! Используйте /reset для сброса."
            elif usage["percentage"] > 50:
                response += "Контекст заполнен наполовину."
            else:
                response += "Достаточно места для разговора."
            return CommandResult(success=True, message=response, data=usage)
        except Exception as e:
            from src.lib.syslog2 import LOG_ERR, syslog2

            syslog2(LOG_ERR, "get token usage failed", error=str(e))
            return CommandResult(success=False, message="Ошибка при получении информации о токенах.", error=str(e))


class ModelCommandHandler(CommandHandler):
    """Handler for /model command."""

    def __init__(self, bot, admin_manager=None):
        self.bot = bot
        self.admin_manager = admin_manager

    def handle(self, context: CommandContext) -> CommandResult:
        try:
            if not context.args:
                return self._show_help()
            
            subcommand = context.args[0].lower()
            
            if subcommand == "list":
                return self._show_list()
            elif subcommand == "get":
                return self._get_current()
            elif subcommand == "set":
                if len(context.args) < 2:
                    return CommandResult(success=False, message="Укажите имя модели: /model set <имя_модели>")
                return self._set_model(context.args[1])
            elif subcommand == "help":
                return self._show_help()
            else:
                 return self._show_help()

        except Exception as e:
            from src.lib.syslog2 import LOG_ERR, syslog2

            syslog2(LOG_ERR, "model command failed", error=str(e))
            return CommandResult(success=False, message="Ошибка при выполнении команды модели.", error=str(e))
            
    def _show_help(self) -> CommandResult:
        message = (
            "Команды управления моделью:\n\n"
            "• /model list — показать список моделей и их параметры\n"
            "• /model get — показать текущую модель\n"
            "• /model set <имя> — установить модель\n"
            "• /model help — эта справка"
        )
        return CommandResult(success=True, message=message)

    def _show_list(self) -> CommandResult:
        models = self.bot.available_models  # Dict[str, int]
        if not models:
             return CommandResult(success=True, message="Нет доступных моделей.")
             
        lines = ["Доступные модели:"]
        current = self.bot.current_model_name
        
        for model_name, max_tokens in models.items():
            is_current = " (текущая)" if model_name == current else ""
            lines.append(f"• {model_name} [context: {max_tokens}]{is_current}")
            
        return CommandResult(success=True, message="\n".join(lines))

    def _get_current(self) -> CommandResult:
        return CommandResult(success=True, message=self.bot.get_current_model())
        
    def _set_model(self, model_name: str) -> CommandResult:
        message = self.bot.set_model(model_name)
        self._save_model_to_config()
        return CommandResult(success=True, message=message, data={"model": self.bot.current_model_name})

    def _save_model_to_config(self) -> None:
        if self.admin_manager:
            try:
                self.admin_manager.config.current_model = self.bot.current_model_name
            except Exception as e:
                from src.lib.syslog2 import LOG_WARNING, syslog2

                syslog2(LOG_WARNING, "failed to save model to config", error=str(e))


class FindCommandHandler(CommandHandler):
    """Handler for /find command."""

    def __init__(self, bot, admin_manager=None, debug_rag: bool = False):
        self.bot = bot
        self.admin_manager = admin_manager
        self.debug_rag = debug_rag

    def handle(self, context: CommandContext) -> CommandResult:
        from src.lib.syslog2 import LOG_ERR, syslog2

        try:
            rag_method, action, query_or_error = self._parse_find_args(context)

            if rag_method is None:
                return CommandResult(
                    success=False,
                    message=query_or_error,
                    error="Invalid arguments",
                )

            if action == "list":
                message = (
                    "Доступные методы RAG:\n\n"
                    "• hybrid - гибридный поиск (FTS5 + векторный rerank)\n"
                    "  Использует FTS5 для первичного поиска, затем векторный поиск для ранжирования\n\n"
                    "• vector_only - только векторный поиск\n"
                    "  Использует только векторные embeddings для поиска\n\n"
                    "• fts_only - только FTS5 поиск\n"
                    "  Использует только полнотекстовый поиск SQLite FTS5\n\n"
                    f"Текущий метод: {rag_method}"
                )
                return CommandResult(
                    success=True,
                    message=message,
                    data={"rag_method": rag_method, "action": "list"},
                )

            query = query_or_error
            retrieval_service = self._create_retrieval_service(rag_method)
            return self._execute_search(retrieval_service, query, rag_method)

        except Exception as e:
            syslog2(LOG_ERR, "find command failed", error=str(e))
            return CommandResult(
                success=False,
                message=f"Ошибка при выполнении поиска: {e}",
                error=str(e),
            )

    def _parse_find_args(self, context: CommandContext) -> tuple[Optional[str], str, str]:
        from src.bot.command_parser import parse_find_command_args

        args_text = " ".join(context.args) if context.args else ""
        rag_method, action, query_or_error = parse_find_command_args(args_text, self.admin_manager)
        return rag_method, action, query_or_error

    def _create_retrieval_service(self, rag_method: str):
        from src.app.bootstrap import create_hybrid_retrieval

        db_url = self.bot.db.db_url
        vector_db_path = self.bot.vector_store.persist_directory
        profile_dir = str(self.bot.profile_dir) if hasattr(self.bot, "profile_dir") and self.bot.profile_dir else None

        return create_hybrid_retrieval(
            db_url=db_url,
            vector_db_path=vector_db_path,
            embedding_client=self.bot.embedding_client,
            profile_dir=profile_dir,
            log_level=self.bot.log_level,
            retrieval_mode=rag_method,
            llm_client=self.bot.llm_client if hasattr(self.bot, "llm_client") else None,
        )

    def _execute_search(self, retrieval_service, query: str, rag_method: str) -> CommandResult:
        from src.core.message_search import search_message_contents

        threshold = 1.5
        if self.admin_manager:
            threshold = self.admin_manager.config.cosine_distance_thr

        message_parts_list = search_message_contents(
            retrieval=retrieval_service,
            db=self.bot.db,
            query=query,
            top_k=100,
            threshold=threshold,
            debug_rag=self.debug_rag,
        )

        if not message_parts_list:
            return CommandResult(
                success=True,
                message=f'по запросу "{query}" ничего не найдено (метод: {rag_method})',
                data={"results_count": 0, "rag_method": rag_method},
            )

        return CommandResult(
            success=True,
            message="",
            data={
                "message_parts_list": message_parts_list,
                "query": query,
                "rag_method": rag_method,
                "results_count": len(message_parts_list),
                "needs_formatting": True,
            },
        )


class ProfileCommandHandler(AsyncCommandHandler):
    """Handler for /userprofile command."""

    SYSTEM_PROMPT = """
Ты — опытный аналитик и хедхантер, специализирующийся на создании психологических и профессиональных профилей на основе текстовых коммуникаций. Твоя задача — проанализировать предоставленную историю сообщений пользователя и составить максимально объективный, детализированный и полезный профиль.

КРИТИЧЕСКИ ВАЖНЫЕ ПРИНЦИПЫ АНАЛИЗА:
1.  Используй ТОЛЬКО предоставленные данные: Все выводы должны быть напрямую подтверждены цитатами или четкими паттернами из истории переписки. НЕ выдумывай информацию. Если данных недостаточно для какого-то пункта, напиши "недостаточно данных" вместо выдумывания.
2.  Анализируй ВСЕ сообщения: Внимательно проанализируй все сообщения из предоставленного контекста. Учитывай контекст каждого сообщения (к кому обращено, в рамках какой темы, эмоциональный фон дискуссии).
3.  Иерархия доказательств: Прямое утверждение пользователя о себе > повторяющиеся паттерны поведения > единичные, но яркие примеры > косвенные указания.
4.  Баланс: Отмечай как сильные, так и слабые стороны.

СТРУКТУРА ПРОФАЙЛА:
Представь результат СТРОГО В ФОРМАТЕ ОБЫЧНОГО ТЕКСТА (Plain Text).

КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать любое Markdown-форматирование:
- ЗАПРЕЩЕНО использовать жирный шрифт (**текст** или __текст__)
- ЗАПРЕЩЕНО использовать курсив (*текст* или _текст_)
- ЗАПРЕЩЕНО использовать заголовки (### Текст)
- ЗАПРЕЩЕНО использовать маркеры списков с * (используй цифры 1. 2. 3. или просто отступы)

Используй ТОЛЬКО простой текст, переносы строк и нумерацию пунктов. Каждый пункт должен содержать конкретику и примеры. Если данных недостаточно, пиши "недостаточно данных".

0. Общая оценка и рекомендации:
   Потенциал для вербовки: Высокий / Средний / Низкий. Обоснование выбранного потенциала.
   Рекомендуемый подход к взаимодействию: Например, "через общие профессиональные интересы", "через предложение сотрудничества в сфере хобби", "через профессиональные возможности".
   Темы для установления контакта: Конкретные темы, которые, судя по истории, его глубоко интересуют или затрагивают эмоционально.

1. Роль в команде (на основе наблюдаемого поведения):
   Определи роль или несколько ролей: Лидер, Инициатор, Исполнитель, Критик/«Дьявольский адвокат», Медиатор/Миротворец, Эксперт/Наставник, Наблюдатель.
   Обоснование: Какие сообщения демонстрируют эту роль? (приведи 1-2 ключевых примера).

2. Навыки (выведенные из контекста и самоописаний):
   Hard Skills (профессиональные): Упоминание технологий, методик, языков, инструментов. Оценка уровня (дилетант, компетентный, эксперт) на основе глубины суждений.
   Soft Skills (коммуникативные и социальные): Убеждение, аргументация, эмпатия, работа с конфликтами, юмор, ясность изложения, адаптивность. Подтверди примерами.

3. Сильные стороны:
   Что делает его ценным? (Например: уникальные технические навыки, аналитический склад ума, высокая мотивация по теме, влиятельные связи, стрессоустойчивость, обучаемость).
   Подтверждение из текста.

4. Слабые стороны и особенности:
   Что может представлять сложность или требует внимания? (Например: склонность к риску или излишняя осторожность, потребность в признании, особенности коммуникации).
   Подтверждение из текста.

5. Особенности темперамента и эмоционального интеллекта:
   Поведение под давлением: Агрессия, уход в себя, сарказм, хладнокровие.
   Доминирующий эмоциональный фон: Нейтрально-аналитический, циничный, энтузиастичный, тревожный, нестабильный.
   Реакция на критику: Конструктивная, оборонительная, игнорирующая.
   Примеры, иллюстрирующие эти черты.

6. Прочие особенности:
   Ценности и убеждения: Политические, социальные, профессиональные взгляды (только если явно выражены в сообщениях).
   Мотиваторы: Что им движет? (Деньги, статус, идеология, азарт, познание, принадлежность к группе).
   Демографические и биографические данные: (Только если прямо указано или однозначно следует из контекста): примерный возраст, род деятельности, географические упоминания, язык общения.
   Паттерны общения: Формальный/неформальный стиль, использование жаргона, грамматические особенности, активность.

7. Характерные сообщения (прямые цитаты-ключи):
   Приведи 3-5 самых показательных, коротких цитат пользователя, которые ярко иллюстрируют его личность, мотивацию или особенности. Каждую цитату сопроводи пояснением, почему она значима.

"""

    def __init__(self, bot):
        self.bot = bot

    async def handle(self, context: CommandContext) -> CommandResult:
        from src.lib.syslog2 import LOG_WARNING, LOG_DEBUG, LOG_ERR, syslog2
        
        if not context.args:
            return CommandResult(success=False, message="Укажите username или alias: /userprofile <name>")
            
        target_name = " ".join(context.args).strip()
        
        if self.bot.log_level >= LOG_DEBUG:
            syslog2(LOG_DEBUG, "profile command started", target_name=target_name)
        
        try:
            # 1. Resolve User
            user = self.bot.db.get_user(target_name)
            found_by = "username"
            if not user:
                user = self.bot.db.get_user_by_alias(target_name)
                found_by = "alias"
            
            if not user:
                return CommandResult(success=False, message=f"Пользователь '{target_name}' не найден.")
            
            real_username = user.username
            
            if self.bot.log_level >= LOG_DEBUG:
                syslog2(LOG_DEBUG, "profile user resolved", target_name=target_name, real_username=real_username, found_by=found_by)
            
            # 2. Gather Context
            # Send initial message as this might take time
            # Note: We rely on caller to show "typing" or wait
            
            if self.bot.log_level >= LOG_DEBUG:
                syslog2(LOG_DEBUG, "profile context gathering started", username=real_username)
            
            # Determine max tokens settings
            config_limit = 0
            
            # 1. Try to get limit from config
            if hasattr(self.bot, 'config') and self.bot.config:
                 if hasattr(self.bot.config, 'llm_max_tokens'):
                     config_limit = self.bot.config.llm_max_tokens
            elif hasattr(self.bot, 'profile_dir') and self.bot.profile_dir:
                try:
                    from src.bot.config import BotConfig
                    config = BotConfig(self.bot.profile_dir)
                    config_limit = config.llm_max_tokens
                except Exception as e:
                    syslog2(LOG_ERR, "failed to load profile config", error=str(e))

            # 2. Determine effective limit
            if config_limit > 0:
                effective_limit = config_limit
            else:
                # Fallback to model's max tokens
                current_model = getattr(self.bot, 'current_model_name', 'unknown')
                available_models = getattr(self.bot, 'available_models', {})
                raw_limit = available_models.get(current_model, 140000)
                # Use raw limit, let bot.complete handle the dynamic adjustment
                effective_limit = raw_limit
                
                if self.bot.log_level >= LOG_DEBUG:
                    syslog2(LOG_DEBUG, "profile tokens using model default", model=current_model, limit=effective_limit)

            max_context_tokens = effective_limit
            llm_max_tokens = effective_limit

            full_context = await self._gather_context(real_username, max_tokens=max_context_tokens)
            
            if not full_context:
                return CommandResult(success=False, message=f"Нет сообщений для анализа пользователя '{real_username}'.")
            
            # Estimate tokens (rough: 3 chars per token)
            estimated_tokens = len(full_context) // 3
            
            if self.bot.log_level >= LOG_DEBUG:
                syslog2(LOG_DEBUG, "profile context gathered", username=real_username, context_length=len(full_context), context_chars=len(full_context), estimated_tokens=estimated_tokens)
                
            # 3. Call LLM
            # Count messages in context (approximate by counting separators)
            message_count = full_context.count("---") if full_context else 0
            
            prompt = f"""Проанализируй историю сообщений пользователя и составь детальный профиль.

Целевой пользователь: {real_username}
Количество сообщений в контексте: {message_count}

История сообщений:
{full_context}

Проанализируй все предоставленные сообщения и составь профиль согласно структуре из системного промпта."""
            
            if self.bot.log_level >= LOG_DEBUG:
                syslog2(LOG_DEBUG, "profile system prompt", prompt_length=len(self.SYSTEM_PROMPT), prompt_preview=self.SYSTEM_PROMPT[:800])
                syslog2(LOG_DEBUG, "profile final prompt", prompt_length=len(prompt), prompt_preview=prompt[:1000])
                syslog2(LOG_DEBUG, "profile llm request", temperature=0.3, system_prompt_length=len(self.SYSTEM_PROMPT), user_prompt_length=len(prompt))
            
            response = await self.bot.complete(
                prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.3, # Balanced
                max_tokens=llm_max_tokens
            )
            
            if self.bot.log_level >= LOG_DEBUG:
                syslog2(LOG_DEBUG, "profile llm response", response_length=len(response), response_preview=response[:500])
            
            return CommandResult(success=True, message=response)
            
        except Exception as e:
            syslog2(LOG_ERR, "profile command failed", error=str(e))
            return CommandResult(success=False, message=f"Ошибка при создании профиля: {e}", error=str(e))

    async def _gather_context(self, username: str, max_tokens: int = 60000) -> str:
        """Gather messages and neighbors for the user."""
        from src.lib.syslog2 import LOG_DEBUG, syslog2
        
        if self.bot.log_level >= LOG_DEBUG:
            syslog2(LOG_DEBUG, "profile gather_context started", username=username, max_tokens=max_tokens)
        
        # Get recent messages. Start with 500.
        messages = self.bot.db.get_messages_by_user(username, limit=500)
        
        if not messages:
            return ""
        
        if self.bot.log_level >= LOG_DEBUG:
            syslog2(LOG_DEBUG, "profile user messages found", username=username, message_count=len(messages))
            
        context_lines = []
        seen_ids = set()
        current_tokens = 0
        
        # Chronological order
        for i, msg in enumerate(messages):
            # Get neighbors (window=5)
            neighbors = self.bot.db.get_neighbor_messages(msg, window_count=5, max_tokens=2000)
            
            block_lines = []
            for m in neighbors:
                if m.msg_id not in seen_ids:
                    line = f"[{m.ts.strftime('%Y-%m-%d %H:%M')}] [user: {m.from_id}] {m.text}"
                    block_lines.append(line)
                    seen_ids.add(m.msg_id)
            
            if block_lines:
                block_lines.append("---")
                block_text = "\n".join(block_lines)
                tokens = len(block_text) // 3 # Rough estimate
                
                if self.bot.log_level >= LOG_DEBUG:
                    syslog2(LOG_DEBUG, "profile neighbor block", msg_id=msg.msg_id, neighbors_count=len(neighbors), block_lines=len(block_lines), block_tokens=tokens)
                
                if current_tokens + tokens > max_tokens:
                    break
                    
                context_lines.append(block_text)
                current_tokens += tokens
            
            # Log progress every 10 messages or at the end
            if self.bot.log_level >= LOG_DEBUG and ((i + 1) % 10 == 0 or i == len(messages) - 1):
                syslog2(LOG_DEBUG, "profile context progress", processed_messages=i+1, total_messages=len(messages), current_tokens=current_tokens, context_blocks=len(context_lines))
        
        result = "\n".join(context_lines)
        
        if self.bot.log_level >= LOG_DEBUG:
            syslog2(LOG_DEBUG, "profile gather_context completed", username=username, total_blocks=len(context_lines), total_tokens=current_tokens, total_chars=len(result))
        
        return result
