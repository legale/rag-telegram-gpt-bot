"""Command handlers for common bot commands."""

from __future__ import annotations

from typing import Optional
from src.core.dispatcher import CommandHandler, CommandContext, CommandResult


class StartCommandHandler(CommandHandler):
    """Handler for /start command."""

    def handle(self, context: CommandContext) -> CommandResult:
        return CommandResult(
            success=True,
            message="Привет!\n\nИспользуйте /help для справки."
        )


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
        """
        Initialize handler with bot instance.

        Args:
            bot: LegaleBot instance
        """
        self.bot = bot

    def handle(self, context: CommandContext) -> CommandResult:
        try:
            message = self.bot.reset_context()
            return CommandResult(success=True, message=message)
        except Exception as e:
            from src.lib.syslog2 import syslog2, LOG_ERR
            syslog2(LOG_ERR, "reset context failed", error=str(e))
            return CommandResult(
                success=False,
                message="Ошибка при сбросе контекста.",
                error=str(e)
            )


class TokensCommandHandler(CommandHandler):
    """Handler for /tokens command."""

    def __init__(self, bot):
        """
        Initialize handler with bot instance.

        Args:
            bot: LegaleBot instance
        """
        self.bot = bot

    def handle(self, context: CommandContext) -> CommandResult:
        try:
            usage = self.bot.get_token_usage()
            response = (
                f"Использование токенов:\n\n"
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
            from src.lib.syslog2 import syslog2, LOG_ERR
            syslog2(LOG_ERR, "get token usage failed", error=str(e))
            return CommandResult(
                success=False,
                message="Ошибка при получении информации о токенах.",
                error=str(e)
            )


class ModelCommandHandler(CommandHandler):
    """Handler for /model command."""

    def __init__(self, bot, admin_manager=None):
        """
        Initialize handler with bot instance.

        Args:
            bot: LegaleBot instance
            admin_manager: Optional AdminManager for saving model to config
        """
        self.bot = bot
        self.admin_manager = admin_manager

    def handle(self, context: CommandContext) -> CommandResult:
        try:
            message = self.bot.get_model()
            # Save new model to config if admin_manager is available
            if self.admin_manager:
                try:
                    self.admin_manager.config.current_model = self.bot.current_model_name
                except Exception as e:
                    from src.lib.syslog2 import syslog2, LOG_WARNING
                    syslog2(LOG_WARNING, "failed to save model to config", error=str(e))
            return CommandResult(
                success=True,
                message=message,
                data={"model": self.bot.current_model_name}
            )
        except Exception as e:
            from src.lib.syslog2 import syslog2, LOG_ERR
            syslog2(LOG_ERR, "get model failed", error=str(e))
            return CommandResult(
                success=False,
                message="Ошибка при переключении модели.",
                error=str(e)
            )


class FindCommandHandler(CommandHandler):
    """Handler for /find command."""

    def __init__(self, bot, admin_manager=None, debug_rag: bool = False):
        """
        Initialize handler with dependencies.

        Args:
            bot: LegaleBot instance
            admin_manager: Optional AdminManager for config access
            debug_rag: Whether to enable debug RAG mode
        """
        self.bot = bot
        self.admin_manager = admin_manager
        self.debug_rag = debug_rag

    def handle(self, context: CommandContext) -> CommandResult:
        """
        Handle /find command.

        Args:
            context: Command context with args containing rag_method and query or list

        Returns:
            CommandResult with search results or error
        """
        from src.bot.command_parser import parse_find_command_args
        from src.app.bootstrap import create_hybrid_retrieval
        from src.core.message_search import search_message_contents
        from src.lib.syslog2 import syslog2, LOG_ERR, LOG_ALERT

        try:
            # Parse arguments from context
            args_text = " ".join(context.args) if context.args else ""
            rag_method, action, query_or_error = parse_find_command_args(args_text, self.admin_manager)
            
            if rag_method is None:
                # query_or_error contains error message
                return CommandResult(
                    success=False,
                    message=query_or_error,
                    error="Invalid arguments"
                )

            # Handle "list" action
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
                    data={"rag_method": rag_method, "action": "list"}
                )

            # Perform search
            query = query_or_error
            
            # Get paths from bot_instance
            db_url = self.bot.db.db_url  # Fix: use db.db_url instead of db_url
            vector_db_path = self.bot.vector_store.persist_directory  # Get from vector_store
            profile_dir = str(self.bot.profile_dir) if hasattr(self.bot, 'profile_dir') and self.bot.profile_dir else None

            # Create HybridRetrievalService with specified retrieval mode
            retrieval_service = create_hybrid_retrieval(
                db_url=db_url,
                vector_db_path=vector_db_path,
                embedding_client=self.bot.embedding_client,
                profile_dir=profile_dir,
                log_level=self.bot.log_level,
                retrieval_mode=rag_method,
                llm_client=self.bot.llm_client if hasattr(self.bot, 'llm_client') else None
            )

            # Get threshold from config or use default
            threshold = 1.5
            if self.admin_manager:
                threshold = self.admin_manager.config.cosine_distance_thr

            # Perform search using search_message_contents
            message_parts_list = search_message_contents(
                retrieval=retrieval_service,
                db=self.bot.db,
                query=query,
                top_k=100,
                threshold=threshold,
                debug_rag=self.debug_rag
            )

            if not message_parts_list:
                return CommandResult(
                    success=True,
                    message=f'по запросу "{query}" ничего не найдено (метод: {rag_method})',
                    data={"results_count": 0, "rag_method": rag_method}
                )

            # For Telegram bot, we need to return a special result that will be handled
            # by the message handler to format and send results properly
            return CommandResult(
                success=True,
                message="",  # Empty message - results will be sent separately
                data={
                    "message_parts_list": message_parts_list,
                    "query": query,
                    "rag_method": rag_method,
                    "results_count": len(message_parts_list),
                    "needs_formatting": True  # Flag to indicate special handling needed
                }
            )

        except Exception as e:
            syslog2(LOG_ERR, "find command failed", error=str(e))
            return CommandResult(
                success=False,
                message=f"Ошибка при выполнении поиска: {e}",
                error=str(e)
            )

