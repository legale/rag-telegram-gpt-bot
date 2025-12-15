from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
from src.storage.db import Database
from src.storage.vector_store import VectorStore
from src.core.embedding import EmbeddingClient, LocalEmbeddingClient, create_embedding_client
from src.core.use_cases.hybrid_retrieval import HybridRetrievalService
from src.core.prompt import PromptEngine
from src.core.llm import LLMClient
from src.app.bootstrap import create_hybrid_retrieval
import os
from src.lib.syslog2 import *

class LegaleBot:
    """Main bot class orchestrating the RAG pipeline."""
    
    def __init__(
        self,
        db_url: str,
        vector_db_path: str,
        model_name: Optional[str] = None,
        log_level: int = LOG_WARNING,
        debug_rag: bool = False,
        profile_dir: Optional[Union[str, Path]] = None,
        retrieval_type: str = "hybrid"  # "hybrid" | "fts_only"
    ):
        # Initialize components
        if not db_url or not vector_db_path:
            raise ValueError("db_url and vector_db_path must be provided")
            
        self.db = Database(db_url)
        self.log_level = log_level
        self.debug_rag = debug_rag
        # Load profile config - always create config (uses defaults from BotConfig if profile_dir not provided)
        from src.bot.config import BotConfig
        embedding_client = None
        if profile_dir:
            profile_path = Path(profile_dir)
            if profile_path.exists():
                try:
                    config = BotConfig(profile_path)
                    embedding_client = create_embedding_client(
                        generator=config.embedding_generator,
                        model=config.embedding_model
                    )
                except Exception as e:
                    if self.log_level <= LOG_INFO:
                        syslog2(LOG_WARNING, "profile config load failed", error=str(e), action="using default embedding client")
                    # Create config with defaults even if load failed
                    config = BotConfig(profile_path)
            else:
                # Profile dir doesn't exist, create config with defaults
                profile_path.mkdir(parents=True, exist_ok=True)
                config = BotConfig(profile_path)
        else:
            # No profile_dir provided, use default profile path
            default_profile_path = Path("profiles/default")
            default_profile_path.mkdir(parents=True, exist_ok=True)
            config = BotConfig(default_profile_path)
        
        # Use profile embedding client or create default
        if embedding_client is None:
            embedding_client = create_embedding_client(
                generator=config.embedding_generator,
                model=config.embedding_model
            )
        
        self.config = config
        self.embedding_client = embedding_client
        self.vector_store = VectorStore(
            persist_directory=vector_db_path,
            embedding_client=embedding_client
        )
        # self.llm_client moved to after model selection logic

        
        # Model getting support (needed before creating retrieval service)
        self.available_models = self._load_available_models()
        if not model_name and self.available_models:
            model_name = self.available_models[0]
            
        # Find initial model index
        if model_name and model_name in self.available_models:
            self.current_model_index = self.available_models.index(model_name)
            
        if not model_name:
             # This means available_models is empty and no model provided
             syslog2(LOG_WARNING, "no model configured and no models found in models.txt")
             model_name = "unknown" # LLMClient might fail or just log warning?
             
        self.llm_client = LLMClient(model=model_name, log_level=log_level)
        
        # Initialize services using bootstrap based on retrieval_type
        self.retrieval_type = retrieval_type
        
        if retrieval_type == "hybrid":
            # Use new hybrid retrieval (FTS5 + vector rerank)
            self.retrieval_service = create_hybrid_retrieval(
                db_url=db_url,
                vector_db_path=vector_db_path,
                embedding_client=self.embedding_client,
                profile_dir=profile_dir,
                log_level=log_level,
            )
            self.retrieval = self.retrieval_service
        elif retrieval_type == "fts_only":
            # FTS-only mode: skip vector reranking
            self.retrieval_service = create_hybrid_retrieval(
                db_url=db_url,
                vector_db_path=vector_db_path,
                embedding_client=self.embedding_client,
                profile_dir=profile_dir,
                log_level=log_level,
                fts_only=True,
            )
            self.retrieval = self.retrieval_service
        else:
            raise ValueError(f"Unknown retrieval_type: {retrieval_type}. Use: hybrid, fts_only")
        self.prompt_engine = PromptEngine()
        
        # Simple in-memory history for the current session
        self.chat_history: List[Dict[str, str]] = []
        
        # Token limit configuration
        self.max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", "14000"))
    
    def _load_available_models(self) -> List[str]:
        """
        Load available models from models.txt file.
        
        Returns:
            List of model names.
        """
        models_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models.txt")
        try:
            with open(models_file, 'r') as f:
                models = [line.strip() for line in f if line.strip()]
            return models if models else []
        except FileNotFoundError:
            if self.log_level <= LOG_INFO:
                syslog2(LOG_WARNING, "models file missing", path=models_file)
            return []
    
    def get_model(self) -> str:
        """
        get to the next model in the list (cyclic).
        
        Returns:
            Message with the new model name.
        """
        if not self.available_models:
            return "Нет доступных моделей для переключения."
        
        # Move to next model (cyclic)
        self.current_model_index = (self.current_model_index + 1) % len(self.available_models)
        new_model = self.available_models[self.current_model_index]
        
        # Recreate LLM client with new model
        self.llm_client = LLMClient(model=new_model, log_level=self.log_level)
        
        if self.log_level <= LOG_INFO:
            syslog2(LOG_NOTICE, "model geted", new_model=new_model)
        
        return f"Модель переключена на: {new_model}\n({self.current_model_index + 1}/{len(self.available_models)})"

    def set_model(self, model_name: str) -> str:
        """
        Set a specific model by name.
        
        Args:
            model_name: Name of the model to get to.
            
        Returns:
            Success message or error message.
        """
        if model_name not in self.available_models:
            return f"Модель `{model_name}` не найдена в списке доступных."
        
        self.current_model_index = self.available_models.index(model_name)
        
        # Recreate LLM client with new model
        self.llm_client = LLMClient(model=model_name, log_level=self.log_level)
        
        if self.log_level <= LOG_INFO:
            syslog2(LOG_NOTICE, "model set", new_model=model_name)
            
        return f"Модель успешно установлена: {model_name}"
    
    @property
    def current_model_name(self) -> str:
        """Get the name of the currently active model."""
        # If available_models is empty, self.current_model_index might not be valid
        # or llm_client might have been initialized with "unknown".
        # It's safer to get the model name directly from the llm_client.
        return self.llm_client.model_name

    def get_current_model(self) -> str:
        """
        Get current model information.
        
        Returns:
            Current model name and position in the list.
        """
        if not self.available_models:
            return "Нет доступных моделей."
        
        current_model = self.available_models[self.current_model_index]
        return f"Текущая модель: {current_model}\n({self.current_model_index + 1}/{len(self.available_models)})"
        
    def reset_context(self) -> str:
        """
        Reset the chat history/context.
        
        Returns:
            Confirmation message.
        """
        self.chat_history = []
        return "Контекст сброшен!"
    
    def _build_history_for_prompt(self, max_messages: int = 5) -> List[Dict[str, str]]:
        """
        Build history for prompt from chat history.
        
        Args:
            max_messages: Maximum number of recent messages to include
            
        Returns:
            List of history entries with sender and content
        """
        history_for_prompt = []
        for msg in self.chat_history[-max_messages:]:
            sender = "User" if msg["role"] == "user" else "Bot"
            history_for_prompt.append(
                {"sender": sender, "content": msg["content"]}
            )
        return history_for_prompt
    
    def _build_prompt_and_history(
        self, 
        context_chunks: List[Dict], 
        user_task: str, 
        max_messages: int = 5,
        custom_template: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Build system prompt and history for prompt (helper to reduce duplication).
        
        Args:
            context_chunks: Retrieved context chunks
            user_task: User task/query string
            max_messages: Maximum number of recent messages to include in history
            custom_template: Optional custom system prompt template
            
        Returns:
            Tuple of (system_prompt, history_for_prompt)
        """
        history_for_prompt = self._build_history_for_prompt(max_messages=max_messages)
        system_prompt = self.prompt_engine.construct_prompt(
            context_chunks=context_chunks,
            chat_history=history_for_prompt,
            user_task=user_task,
            custom_template=custom_template,
            log_level=self.log_level
        )
        return system_prompt, history_for_prompt
    
    def _calculate_token_usage(self, system_prompt: str, user_content: str = "") -> Dict[str, Union[int, float]]:
        """
        Calculate token usage for given messages.
        
        Args:
            system_prompt: System prompt content
            user_content: User message content (optional)
            
        Returns:
            Dictionary with current_tokens, max_tokens, and percentage
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        current_tokens = self.llm_client.count_tokens(messages)
        percentage = (current_tokens / self.max_context_tokens) * 100 if self.max_context_tokens > 0 else 0.0
        
        return {
            "current_tokens": current_tokens,
            "max_tokens": self.max_context_tokens,
            "percentage": round(percentage, 2),
        }
    
    def get_token_usage(self) -> Dict[str, int]:
        """
        Get current token usage statistics.
        """
        if not self.chat_history:
            return {
                "current_tokens": 0,
                "max_tokens": self.max_context_tokens,
                "percentage": 0.0,
            }

        # делаем системный промпт без реального task, только для оценки объема контекста
        system_prompt, _ = self._build_prompt_and_history(
            context_chunks=[],
            user_task="",
            max_messages=5
        )

        # считаем так же, как реально вызываем модель: system + пустой user
        return self._calculate_token_usage(system_prompt, user_content="")

    def _ensure_context_limit(self) -> str:
        """
        Ensure context doesn't exceed token limit by resetting if necessary.
        
        Returns:
            Warning message if context was reset, empty string otherwise
        """
        if not self.chat_history:
            return ""
        
        token_usage = self.get_token_usage()
        # можно сбрасывать не по 100%, а, например, по 0.8 * лимита
        if token_usage["current_tokens"] >= self.max_context_tokens:
            self.reset_context()
            warning = "Контекст был автоматически сброшен из-за достижения лимита токенов.\n\n"
            if self.log_level <= LOG_INFO:
                syslog2(LOG_WARNING, "auto reset context", token_usage=f"{token_usage['current_tokens']}/{self.max_context_tokens}")
            return warning
        return ""
    
    def _is_token_limit_error(self, error_msg: str) -> bool:
        """
        Check if error is related to token limit or payment issues.
        
        Args:
            error_msg: Error message string
            
        Returns:
            True if error is token limit related
        """
        return ("402" in error_msg or 
                "context_length_exceeded" in error_msg or 
                "Prompt tokens limit exceeded" in error_msg)
    
    def _retry_after_reset(self, context_chunks: List[Dict], user_input: str, 
                          system_prompt_template: Optional[str] = None) -> str:
        """
        Retry LLM call after resetting context due to token limit error.
        
        Args:
            context_chunks: Retrieved context chunks
            user_input: User input message
            system_prompt_template: Optional custom system prompt template
            
        Returns:
            Response from LLM or error message
        """
        if self.log_level <= LOG_INFO:
            syslog2(LOG_ERR, "token limit exceeded", action="resetting context and retrying")
        
        # Force reset context
        self.reset_context()
        
        # Reconstruct prompt without history
        system_prompt = self.prompt_engine.construct_prompt(
            context_chunks=context_chunks,
            chat_history=[],
            user_task=user_input,
            custom_template=system_prompt_template,
            log_level=self.log_level
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        
        try:
            # Retry
            response = self.llm_client.complete(messages)
            return "Ошибка лимита токенов/баланса. Контекст сброшен.\n\n" + response
        except Exception as retry_e:
            syslog2(LOG_ERR, "retry failed", error=str(retry_e))
            return "Ошибка: Не удалось получить ответ даже после сброса контекста (лимит токенов или баланс исчерпан)."
    
    def _call_llm_with_retry(
        self, 
        messages: List[Dict[str, str]], 
        context_chunks: List[Dict], 
        user_input: str,
        system_prompt_template: Optional[str] = None
    ) -> str:
        """
        Call LLM with automatic retry on token limit errors.
        
        Args:
            messages: Messages to send to LLM
            context_chunks: Retrieved context chunks (for retry)
            user_input: User input (for retry)
            system_prompt_template: Optional custom template (for retry)
            
        Returns:
            LLM response string
            
        Raises:
            Exception: If LLM call fails with non-token-limit error
        """
        try:
            return self.llm_client.complete(messages)
        except Exception as e:
            error_msg = str(e)
            # Check for token limit or payment issues
            if self._is_token_limit_error(error_msg):
                return self._retry_after_reset(context_chunks, user_input, system_prompt_template)
            else:
                # Re-raise other errors
                raise
    
    def chat(self, user_input: str, n_results: int = 3, respond: bool = True, system_prompt_template: str = None) -> str:
        """
        Process a user message and return the bot's response.
        """
        auto_reset_warning = self._ensure_context_limit()

        if not respond:
            self.chat_history.append({"role": "user", "content": user_input})
            return ""

        syslog2(LOG_NOTICE, "retrieving context", retrieval_type=self.retrieval_type)
        context_chunks = self.retrieval_service.retrieve(
            user_input, n_results=n_results, score_threshold=self.config.fts5_score_thr
        )

        # системный промпт: контекст + история + инструкции, но без дублирования user_input
        system_prompt, _ = self._build_prompt_and_history(
            context_chunks=context_chunks,
            user_task=user_input,
            max_messages=5,
            custom_template=system_prompt_template
        )

        if self.log_level <= LOG_DEBUG:
            syslog2(LOG_DEBUG, "system prompt constructed", length=len(system_prompt))

        syslog2(LOG_NOTICE, "querying llm")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        try:
            response = self._call_llm_with_retry(messages, context_chunks, user_input, system_prompt_template)
            # Check if retry already included warning
            if "Ошибка лимита токенов" in response:
                auto_reset_warning = ""
        except Exception as e:
            syslog2(LOG_ERR, "llm call failed", error=str(e))
            return f"Произошла ошибка при обращении к нейросети: {e}"

        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": response})

        return auto_reset_warning + response

    def get_rag_debug_info(self, user_input: str, n_results: int = 3) -> Dict:
        """
        Get debug information about RAG retrieval without actually calling the model.
        Useful for debugging what chunks are retrieved and what prompt is constructed.
        
        Args:
            user_input: User query string
            n_results: Number of chunks to retrieve
            
        Returns:
            Dictionary with 'chunks', 'prompt', and 'token_count'
        """
        # Retrieve context chunks
        context_chunks = self.retrieval_service.retrieve(
            user_input, n_results=n_results, score_threshold=self.config.fts5_score_thr
        )
        
        # Build prompt and history using helper
        system_prompt, _ = self._build_prompt_and_history(
            context_chunks=context_chunks,
            user_task=user_input,
            max_messages=5
        )
        
        # Count tokens using helper
        token_usage = self._calculate_token_usage(system_prompt, user_input)
        
        return {
            "chunks": context_chunks,
            "prompt": system_prompt,
            "token_count": token_usage["current_tokens"],
            "chunks_count": len(context_chunks)
        }
