from typing import List, Dict, Optional, Union
from pathlib import Path
from src.storage.db import Database
from src.storage.vector_store import VectorStore
from src.core.embedding import EmbeddingClient, LocalEmbeddingClient, create_embedding_client
from src.core.retrieval import RetrievalService
from src.core.prompt import PromptEngine
from src.core.llm import LLMClient
import os
from src.core.syslog2 import *

class LegaleBot:
    """Main bot class orchestrating the RAG pipeline."""
    
    def __init__(
        self,
        db_url: str,
        vector_db_path: str,
        model_name: Optional[str] = None,
        verbosity: int = 0,
        profile_dir: Optional[Union[str, Path]] = None
    ):
        # Initialize components
        if not db_url or not vector_db_path:
            raise ValueError("db_url and vector_db_path must be provided")
            
        self.db = Database(db_url)
        self.verbosity = verbosity
        
        # Load profile config if available
        embedding_client = None
        if profile_dir:
            profile_path = Path(profile_dir)
            if profile_path.exists():
                try:
                    from src.bot.config import BotConfig
                    config = BotConfig(profile_path)
                    embedding_client = create_embedding_client(
                        generator=config.embedding_generator,
                        model=config.embedding_model
                    )
                except Exception as e:
                    if verbosity >= 1:
                        syslog2(LOG_WARNING, "profile config load failed", error=str(e), action="using default embedding client")
        
        # Use profile embedding client or create default
        if embedding_client is None:
            embedding_client = EmbeddingClient()
        
        self.embedding_client = embedding_client
        self.vector_store = VectorStore(
            persist_directory=vector_db_path,
            embedding_client=embedding_client
        )
        # self.llm_client moved to after model selection logic

        
        # Initialize services
        self.retrieval_service = RetrievalService(
            vector_store=self.vector_store,
            db=self.db,
            embedding_client=self.embedding_client,
            verbosity=verbosity
        )
        self.prompt_engine = PromptEngine()
        
        # Simple in-memory history for the current session
        self.chat_history: List[Dict[str, str]] = []
        
        # Token limit configuration
        self.max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", "14000"))
        
        # Model geting support
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
             
        self.llm_client = LLMClient(model=model_name, verbosity=verbosity)
    
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
            if self.verbosity >= 1:
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
        self.llm_client = LLMClient(model=new_model, verbosity=self.verbosity)
        
        if self.verbosity >= 1:
            syslog2(LOG_INFO, "model geted", new_model=new_model)
        
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
        self.llm_client = LLMClient(model=model_name, verbosity=self.verbosity)
        
        if self.verbosity >= 1:
            syslog2(LOG_INFO, "model set", new_model=model_name)
            
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

        # берем последние N сообщений как в chat()
        history_for_prompt = []
        for msg in self.chat_history[-5:]:
            sender = "User" if msg["role"] == "user" else "Bot"
            history_for_prompt.append(
                {"sender": sender, "content": msg["content"]}
            )

        # делаем системный промпт без реального task, только для оценки объема контекста
        system_prompt = self.prompt_engine.construct_prompt(
            context_chunks=[],
            chat_history=history_for_prompt,
            user_task="",
        )

        # считаем так же, как реально вызываем модель: system + пустой user
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ""},
        ]

        current_tokens = self.llm_client.count_tokens(messages)
        percentage = (current_tokens / self.max_context_tokens) * 100 if self.max_context_tokens > 0 else 0.0

        return {
            "current_tokens": current_tokens,
            "max_tokens": self.max_context_tokens,
            "percentage": round(percentage, 2),
        }

    def chat(self, user_input: str, n_results: int = 3, respond: bool = True, system_prompt_template: str = None) -> str:
        """
        Process a user message and return the bot's response.
        """
        auto_reset_warning = ""
        if self.chat_history:
            token_usage = self.get_token_usage()
            # можно сбрасывать не по 100%, а, например, по 0.8 * лимита
            if token_usage["current_tokens"] >= self.max_context_tokens:
                self.reset_context()
                auto_reset_warning = (
                    "Контекст был автоматически сброшен из-за достижения лимита токенов.\n\n"
                )
                if self.verbosity >= 1:
                    syslog2(LOG_WARNING, "auto reset context", token_usage=f"{token_usage['current_tokens']}/{self.max_context_tokens}")

        if not respond:
            self.chat_history.append({"role": "user", "content": user_input})
            return ""

        syslog2(LOG_INFO, "retrieving context", chunks=n_results)
        context_chunks = self.retrieval_service.retrieve(
            user_input, n_results=n_results
        )

        history_for_prompt = []
        for msg in self.chat_history[-5:]:
            sender = "User" if msg["role"] == "user" else "Bot"
            history_for_prompt.append(
                {"sender": sender, "content": msg["content"]}
            )

        # системный промпт: контекст + история + инструкции, но без дублирования user_input
        system_prompt = self.prompt_engine.construct_prompt(
            context_chunks=context_chunks,
            chat_history=history_for_prompt,
            user_task=user_input,
            custom_template=system_prompt_template
        )

        if self.verbosity >= 2:
            syslog2(LOG_DEBUG, "system prompt constructed", length=len(system_prompt))

        syslog2(LOG_INFO, "querying llm")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        try:
            response = self.llm_client.complete(messages)
        except Exception as e:
            error_msg = str(e)
            # Check for token limit or payment issues (OpenRouter 402 or context length)
            if "402" in error_msg or "context_length_exceeded" in error_msg or "Prompt tokens limit exceeded" in error_msg:
                if self.verbosity >= 1:
                    syslog2(LOG_ERR, "token limit exceeded", error=str(e), action="resetting context and retrying")
                
                # Force reset context
                self.reset_context()
                
                # Reconstruct prompt without history
                system_prompt = self.prompt_engine.construct_prompt(
                    context_chunks=context_chunks,
                    chat_history=[],
                    user_task=user_input,
                    custom_template=system_prompt_template
                )
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ]
                
                # Append warning to result
                auto_reset_warning = "Ошибка лимита токенов/баланса. Контекст сброшен.\n\n"
                
                try:
                    # Retry
                    response = self.llm_client.complete(messages)
                except Exception as retry_e:
                     syslog2(LOG_ERR, "retry failed", error=str(retry_e))
                     return "Ошибка: Не удалось получить ответ даже после сброса контекста (лимит токенов или баланс исчерпан)."
            else:
                 # Other errors
                 syslog2(LOG_ERR, "llm call failed", error=str(e))
                 # In verbose mode, might want to show error, but for user safety keep it generic or specific if needed
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
            user_input, n_results=n_results
        )
        
        # Build history for prompt
        history_for_prompt = []
        for msg in self.chat_history[-5:]:
            sender = "User" if msg["role"] == "user" else "Bot"
            history_for_prompt.append(
                {"sender": sender, "content": msg["content"]}
            )
        
        # Construct prompt
        system_prompt = self.prompt_engine.construct_prompt(
            context_chunks=context_chunks,
            chat_history=history_for_prompt,
            user_task=user_input,
        )
        
        # Count tokens
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        token_count = self.llm_client.count_tokens(messages)
        
        return {
            "chunks": context_chunks,
            "prompt": system_prompt,
            "token_count": token_count,
            "chunks_count": len(context_chunks)
        }
