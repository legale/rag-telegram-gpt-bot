"""User-facing command handlers."""

from __future__ import annotations

from typing import Optional

from src.core.dispatcher import CommandContext, CommandHandler, CommandResult


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
