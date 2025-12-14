"""Command handlers for common bot commands."""

from __future__ import annotations

from typing import Optional
from ..dispatcher import CommandHandler, CommandContext, CommandResult


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
            "• /find [thr] <запрос> — поиск сообщений по запросу\n\n"
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
            context: Command context with args containing query and optional threshold

        Returns:
            CommandResult with search results or error
        """
        from src.bot.command_parser import parse_find_command_args
        from src.core.message_search import convert_search_results_to_dict, _prepare_message_parts
        from src.app.bootstrap import create_hybrid_search
        from src.lib.syslog2 import syslog2, LOG_ERR, LOG_ALERT
        import os

        try:
            # Parse arguments from context
            args_text = " ".join(context.args) if context.args else ""
            threshold, query = parse_find_command_args(args_text, self.admin_manager)
            
            if threshold is None:
                return CommandResult(
                    success=False,
                    message=query,  # query is error message in this case
                    error="Invalid arguments"
                )

            # Get profile paths
            db_url = os.getenv("DATABASE_URL")
            vector_db_path = os.getenv("VECTOR_DB_PATH")
            profile_dir = os.getenv("PROFILE_DIR")
            
            if not db_url or not vector_db_path:
                return CommandResult(
                    success=False,
                    message="Ошибка: не настроены пути к базе данных.",
                    error="Missing DATABASE_URL or VECTOR_DB_PATH"
                )

            # Create HybridSearch
            hybrid_search = create_hybrid_search(
                db_url=db_url,
                vector_db_path=vector_db_path,
                embedding_client=self.bot.embedding_client,
                profile_dir=profile_dir
            )

            # Convert threshold from distance to similarity score
            similarity_threshold = 1.0 - threshold if threshold <= 1.0 else 0.0

            # Perform search
            search_results = hybrid_search.search(
                query=query,
                top_k=100,
                threshold=similarity_threshold,
                enrich_with_messages=False
            )

            if not search_results:
                return CommandResult(
                    success=True,
                    message=f'по запросу "{query}" ничего не найдено (distance <= {threshold})',
                    data={"results_count": 0}
                )

            # Convert to dict format
            filtered_results = convert_search_results_to_dict(search_results)

            # Prepare message parts
            message_parts_list = _prepare_message_parts(
                self.bot.db,
                filtered_results,
                self.debug_rag
            )

            if not message_parts_list:
                return CommandResult(
                    success=True,
                    message=f'по запросу "{query}" ничего не найдено',
                    data={"results_count": 0}
                )

            syslog2(
                LOG_ALERT,
                "find command cli",
                query=query,
                threshold=threshold,
                chunks_found=len(filtered_results),
                messages=len(message_parts_list),
            )

            # Format results for CLI output
            output_lines = [f'Найдено результатов по запросу "{query}" (threshold={threshold}):\n']
            output_lines.append("=" * 70)

            for idx, message_parts in enumerate(message_parts_list, 1):
                for part_idx, part in enumerate(message_parts, 1):
                    content = part.get("content", "")
                    # Simple HTML tag removal for console
                    import re
                    content = re.sub(r'<[^>]+>', '', content)

                    output_lines.append(f"\n--- Результат {idx}, часть {part_idx} ---")
                    output_lines.append(f"Distance: {part.get('distance', 'N/A')}")
                    output_lines.append("-" * 70)
                    output_lines.append(content)
                    output_lines.append("-" * 70)

            return CommandResult(
                success=True,
                message="\n".join(output_lines),
                data={
                    "results_count": len(filtered_results),
                    "messages_count": len(message_parts_list),
                    "query": query,
                    "threshold": threshold
                }
            )

        except Exception as e:
            syslog2(LOG_ERR, "find command cli failed", error=str(e))
            return CommandResult(
                success=False,
                message=f"Ошибка при выполнении поиска: {e}",
                error=str(e)
            )

