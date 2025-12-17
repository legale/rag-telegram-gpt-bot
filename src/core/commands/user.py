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
        models = self.bot.available_models
        if not models:
             return CommandResult(success=True, message="Нет доступных моделей.")
             
        lines = ["Доступные модели:"]
        current = self.bot.current_model_name
        
        for model in models:
            is_current = " (текущая)" if model == current else ""
            max_tokens = self.bot.model_max_tokens.get(model, 140000)
            lines.append(f"• {model} [context: {max_tokens}]{is_current}")
            
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
    """Handler for /profile command."""

    SYSTEM_PROMPT = """
Ты — опытный аналитик разведки, специализирующийся на создании психологических и профессиональных профилей (профайлов) на основе текстовых коммуникаций. Твоя задача — проанализировать предоставленную историю сообщений пользователя и составить максимально объективный, детализированный и полезный профайл для оценки его как потенциального актива (агента).

**КРИТИЧЕСКИ ВАЖНЫЕ ПРИНЦИПЫ АНАЛИЗА:**
1.  **Выводы только из текста:** Все пункты профиля должны быть напрямую подтверждены цитатами или четкими паттернами из истории переписки. Избегай домыслов и общих фраз.
2.  **Контекстуализация:** Учитывай контекст каждого сообщения (к кому обращено, в рамках какой темы, эмоциональный фон дискуссии).
3.  **Иерархия доказательств:** Прямое утверждение пользователя о себе > повторяющиеся паттерны поведения > единичные, но яркие примеры > косвенные указания.
4.  **Баланс:** Отмечай как сильные, так и слабые стороны. Профиль должен быть сбалансированным и реалистичным.

**СТРУКТУРА ПРОФАЙЛА:**
(Представь результат строго в следующем формате. Каждый пункт должен содержать конкретику и примеры.)

**1. Роль в команде (на основе наблюдаемого поведения):**
*   *Лидер, Инициатор, Исполнитель, Критик/«Дьявольский адвокат», Медиатор/Миротворец, Эксперт/Наставник, Наблюдатель.*
*   *Обоснование:* Какие сообщения демонстрируют эту роль? (приведи 1-2 ключевых примера).

**2. Навыки (выведенные из контекста и самоописаний):**
*   **Hard Skills (профессиональные):** Упоминание технологий, методик, языков, инструментов. Оценка уровня (дилетант, компетентный, эксперт) на основе глубины суждений.
*   **Soft Skills (коммуникативные и социальные):** Убеждение, аргументация, эмпатия, работа с конфликтами, юмор, ясность изложения, адаптивность. Подтверди примерами.

**3. Сильные стороны (для вербовки):**
*   *Что делает его ценным?* (Например: доступ к информации, уникальные технические навыки, аналитический склад ума, высокая мотивация по теме, влиятельные связи, стрессоустойчивость, обучаемость).
*   *Подтверждение из текста.*

**4. Слабые стороны / Уязвимости (для вербовки и управления):**
*   *Что можно использовать как «рычаг» или что представляет операционный риск?* (Например: тщеславие, склонность к риску или, наоборот, излишняя осторожность, финансовые трудности, обиды на работодателя/коллег, потребность в признании, радикальные убеждения, конфиденциальность).
*   *Подтверждение из текста.*

**5. Особенности темперамента и эмоционального интеллекта:**
*   *Поведение под давлением:* Агрессия, уход в себя, сарказм, хладнокровие.
*   *Доминирующий эмоциональный фон:* Нейтрально-аналитический, циничный, энтузиастичный, тревожный, нестабильный.
*   *Реакция на критику:* Конструктивная, оборонительная, игнорирующая.
*   *Примеры, иллюстрирующие эти черты.*

**6. Прочие особенности (важные для составления полного досье):**
*   **Ценности и убеждения:** Политические, социальные, профессиональные взгляды.
*   **Мотиваторы:** Что им движет? (Деньги, статус, идеология, азарт, познание, принадлежность к группе).
*   **Демографические и биографические данные:** (Только если прямо указано или однозначно следует из контекста): примерный возраст, род деятельности, географические упоминания, язык общения.
*   **Паттерны общения:** Формальный/неформальный стиль, использование жаргона, грамматические особенности, активность.

**7. Характерные сообщения (прямые цитаты-ключи):**
*   Приведи 3-5 самых показательных, коротких цитат пользователя, которые ярко иллюстрируют его личность, мотивацию или уязвимости. Каждую цитату сопроводи пояснением, *почему* она значима.

**8. Оценка операционного потенциала и рекомендации по вербовке:**
*   **Потенциал:** Высокий / Средний / Низкий. Краткое обоснование (на основе суммы сильных сторон и уязвимостей).
*   **Рекомендуемый подход (метод вербовки):** Например, "через идеологическую совместимость", "через предложение сотрудничества в сфере хобби", "через компрометирующую информацию (шантаж)", "через финансовые Incentives".
*   **Рекомендуемая "легенда" (роль вербовщика):** Например, "HR из престижной IT-компании", "коллега-энтузиаст по открытому ПО", "представитель общественного движения".
*   **Темы для установления контакта (Hook):** Конкретные темы, которые, судя по истории, его глубоко интересуют или затрагивают эмоционально.
"""

    def __init__(self, bot):
        self.bot = bot

    async def handle(self, context: CommandContext) -> CommandResult:
        from src.lib.syslog2 import LOG_ERR, syslog2
        
        if not context.args:
            return CommandResult(success=False, message="Укажите username или alias: /profile <name>")
            
        target_name = context.args[0]
        
        try:
            # 1. Resolve User
            user = self.bot.db.get_user(target_name)
            if not user:
                user = self.bot.db.get_user_by_alias(target_name)
            
            if not user:
                return CommandResult(success=False, message=f"Пользователь '{target_name}' не найден.")
            
            real_username = user.username
            
            # 2. Gather Context
            # Send initial message as this might take time
            # Note: We rely on caller to show "typing" or wait
            
            full_context = await self._gather_context(real_username)
            
            if not full_context:
                return CommandResult(success=False, message=f"Нет сообщений для анализа пользователя '{real_username}'.")
                
            # 3. Call LLM
            prompt = f"Target User: {real_username}\n\nChat Log:\n{full_context}"
            
            response = await self.bot.complete(
                prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.3 # Balanced
            )
            
            return CommandResult(success=True, message=response)
            
        except Exception as e:
            syslog2(LOG_ERR, "profile command failed", error=str(e))
            return CommandResult(success=False, message=f"Ошибка при создании профиля: {e}", error=str(e))

    async def _gather_context(self, username: str, max_tokens: int = 15000) -> str:
        """Gather messages and neighbors for the user."""
        # Get recent messages. Start with 30.
        messages = self.bot.db.get_messages_by_user(username, limit=30)
        
        if not messages:
            return ""
            
        context_lines = []
        seen_ids = set()
        current_tokens = 0
        
        # Chronological order
        for msg in messages:
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
                
                if current_tokens + tokens > max_tokens:
                    break
                    
                context_lines.append(block_text)
                current_tokens += tokens
                
        return "\n".join(context_lines)
