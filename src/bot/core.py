from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
from src.storage.db import Database
from src.storage.vector import VectorStore
from src.core.embedding import EmbeddingClient, LocalEmbeddingClient, create_embedding_client
from src.core.hybrid_retrieval import HybridRetrievalService
from src.core.prompt import PromptEngine
from src.core.llm import LLMClient
from src.app.bootstrap import create_hybrid_retrieval, create_embedding_client_from_config
from src.core.conversation_state import ConversationState
from src.core.context_provider import ContextProvider
from src.core.prompt_builder import PromptBuilder
from src.core.llm_gateway import LLMGateway
import os
from src.lib.syslog2 import *

# Import OpenAI exceptions for specific error handling
try:
    from openai import RateLimitError, APIError, APIConnectionError, APITimeoutError
except ImportError:
    # Fallback if openai is not available
    RateLimitError = type('RateLimitError', (Exception,), {})
    APIError = type('APIError', (Exception,), {})
    APIConnectionError = type('APIConnectionError', (Exception,), {})
    APITimeoutError = type('APITimeoutError', (Exception,), {})

class LegaleBot:
    """Main bot class orchestrating the RAG pipeline."""
    
    def _create_embedding_client(self, profile_dir: Optional[Union[str, Path]] = None) -> Tuple[EmbeddingClient, object]:
        """
        Create embedding client and load profile config.
        
        Args:
            profile_dir: Optional profile directory path
            
        Returns:
            Tuple of (embedding_client, config)
        """
        from src.bot.config import BotConfig
        
        # Determine profile path
        if profile_dir:
            profile_path = Path(profile_dir)
            if not profile_path.exists():
                # Profile dir doesn't exist, create config with defaults
                profile_path.mkdir(parents=True, exist_ok=True)
        else:
            # No profile_dir provided, use default profile path
            profile_path = Path("profiles/default")
            profile_path.mkdir(parents=True, exist_ok=True)
        
        # Load config (always needed for bot configuration)
        try:
            config = BotConfig(profile_path)
        except Exception as e:
            if self.log_level <= LOG_INFO:
                syslog2(LOG_WARNING, "profile config load failed", error=str(e), action="using default config")
            # Create config with defaults even if load failed
            config = BotConfig(profile_path)
        
        # Use unified function from bootstrap.py to create embedding client
        embedding_client = create_embedding_client_from_config(None, profile_path)
        
        return embedding_client, config

    def _create_retrieval_service(
        self,
        db_url: str,
        vector_db_path: str,
        profile_dir: Optional[Union[str, Path]],
        retrieval_type: str,
        llm_client: LLMClient,
        rag_ntop: int = 0
    ):
        """
        Create retrieval service based on retrieval type.
        
        Args:
            db_url: Database URL
            vector_db_path: Vector database path
            profile_dir: Optional profile directory path
            retrieval_type: Retrieval type ("hybrid", "fts_only", "vector_only")
            llm_client: LLM client instance
            rag_ntop: Configured top N results for RAG
            
        Returns:
            Retrieval service instance
        """
        if retrieval_type == "hybrid":
            # Use new hybrid retrieval (FTS5 + vector rerank)
            retrieval_service = create_hybrid_retrieval(
                db_url=db_url,
                vector_db_path=vector_db_path,
                embedding_client=self.embedding_client,
                profile_dir=profile_dir,
                log_level=self.log_level,
                fts_only=False,  # Explicitly set for hybrid
                llm_client=llm_client,  # Pass LLM for query rephrasing
                retrieval_mode="hybrid",
                rag_ntop=rag_ntop,
            )
        elif retrieval_type == "fts_only":
            # FTS-only mode: skip vector reranking
            retrieval_service = create_hybrid_retrieval(
                db_url=db_url,
                vector_db_path=vector_db_path,
                embedding_client=self.embedding_client,
                profile_dir=profile_dir,
                log_level=self.log_level,
                fts_only=True,  # Explicitly set for fts_only
                llm_client=None,  # No LLM needed for FTS-only
                retrieval_mode="fts_only",
                rag_ntop=rag_ntop,
            )
        elif retrieval_type == "vector_only":
            # Vector-only mode: skip FTS5, use only vector search
            retrieval_service = create_hybrid_retrieval(
                db_url=db_url,
                vector_db_path=vector_db_path,
                embedding_client=self.embedding_client,
                profile_dir=profile_dir,
                log_level=self.log_level,
                fts_only=False,
                llm_client=llm_client,  # Pass LLM for query rephrasing
                retrieval_mode="vector_only",
                rag_ntop=rag_ntop,
            )
        else:
            raise ValueError(f"Unknown retrieval_type: {retrieval_type}. Use: hybrid, fts_only, vector_only")
        
        return retrieval_service

    def __init__(
        self,
        db_url: str,
        vector_db_path: str,
        model_name: Optional[str] = None,
        log_level: int = LOG_WARNING,
        debug_rag: bool = False,
        profile_dir: Optional[Union[str, Path]] = None,
        retrieval_type: str = "hybrid"  # "hybrid" | "fts_only" | "vector_only"
    ):
        # Initialize components
        if not db_url or not vector_db_path:
            raise ValueError("db_url and vector_db_path must be provided")
            
        self.db = Database(db_url)
        # Expose db_url for tests/introspection even when Database is mocked
        try:
            self.db.db_url = db_url
        except Exception:
            pass
        self.log_level = log_level
        self.debug_rag = debug_rag
        
        # Create embedding client and load config
        self.embedding_client, self.config = self._create_embedding_client(profile_dir)
        
        self.vector_store = VectorStore(
            persist_directory=vector_db_path,
            embedding_client=self.embedding_client
        )
        
        # Model getting support (needed before creating retrieval service)
        self.model_max_tokens = {}
        # Initial load of models
        available = self.available_models
        if not model_name and available:
            model_name = available[0]
            
        # Find initial model index
        self.current_model_index = 0
        if model_name and available and model_name in available:
            self.current_model_index = available.index(model_name)
            
        if not model_name:
             # This means available_models is empty and no model provided
             syslog2(LOG_WARNING, "no model configured and no models found in models.txt")
             model_name = "unknown" # LLMClient might fail or just log warning?
             
        self.llm_client = LLMClient(model=model_name, log_level=log_level)
        
        # Initialize retrieval service
        self.retrieval_type = retrieval_type
        self.retrieval_service = self._create_retrieval_service(
            db_url=db_url,
            vector_db_path=vector_db_path,
            profile_dir=profile_dir,
            retrieval_type=retrieval_type,
            llm_client=self.llm_client,
            rag_ntop=self.config.rag_ntop
        )
        self.retrieval = self.retrieval_service
        
        self.prompt_engine = PromptEngine()
        
        # Initialize conversation state
        self.conversation_state = ConversationState()
        
        # Initialize context provider
        self.context_provider = ContextProvider(
            retrieval_service=self.retrieval_service,
            config=self.config,
            conversation_state=self.conversation_state,
            log_level=self.log_level
        )
        
        # Initialize prompt builder
        self.prompt_builder = PromptBuilder(
            prompt_engine=self.prompt_engine,
            conversation_state=self.conversation_state,
            log_level=self.log_level
        )
        
        # Initialize LLM gateway
        self.llm_gateway = LLMGateway(
            llm_client=self.llm_client,
            prompt_builder=self.prompt_builder,
            conversation_state=self.conversation_state,
            log_level=self.log_level
        )
        
        # Token limit configuration
        # Use value from models.txt if available, otherwise fallback to 140000
        self.max_context_tokens = self.model_max_tokens.get(model_name, 140000)
    
    # Backward compatibility properties
    @property
    def chat_history(self):
        """Backward compatibility: access conversation_state.chat_history."""
        return self.conversation_state.chat_history

    @chat_history.setter
    def chat_history(self, value):
        """Backward compatibility: allow tests/legacy code to replace chat_history."""
        self.conversation_state.chat_history = value

    @property
    def active_context_chunks(self):
        """Backward compatibility: access conversation_state.active_context_chunks."""
        return self.conversation_state.active_context_chunks

    @active_context_chunks.setter
    def active_context_chunks(self, value):
        """Backward compatibility: allow tests/legacy code to replace active_context_chunks."""
        self.conversation_state.active_context_chunks = value

    @property
    def active_context_query(self):
        """Backward compatibility: access conversation_state.active_context_query."""
        return self.conversation_state.active_context_query

    @active_context_query.setter
    def active_context_query(self, value):
        """Backward compatibility: allow tests/legacy code to replace active_context_query."""
        self.conversation_state.active_context_query = value

    @property
    def active_context_score(self):
        """Backward compatibility: access conversation_state.active_context_score."""
        return self.conversation_state.active_context_score

    @active_context_score.setter
    def active_context_score(self, value):
        """Backward compatibility: allow tests/legacy code to replace active_context_score."""
        self.conversation_state.active_context_score = value
    
    @property
    def available_models(self) -> List[str]:
        """
        Get list of available models, reloading from file on each access.
        Also updates self.model_max_tokens.
        """
        return self._load_available_models()

    def _load_available_models(self) -> List[str]:
        """
        Load available models from models.txt file.
        Also populates self.model_max_tokens.
        
        Returns:
            List of model names.
        """
        self.model_max_tokens = {}
        models_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models.txt")
        try:
            models = []
            with open(models_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    model_name = parts[0]
                    models.append(model_name)
                    
                    if len(parts) > 1:
                        try:
                            self.model_max_tokens[model_name] = int(parts[1])
                        except ValueError:
                             self.model_max_tokens[model_name] = 140000
                    else:
                        self.model_max_tokens[model_name] = 140000
            return models
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
        models = self.available_models
        if not models:
            return "Нет доступных моделей для переключения."
        
        # Recalculate index based on current model name, in case list changed
        current_name = self.current_model_name
        try:
            current_idx = models.index(current_name)
        except ValueError:
            current_idx = -1
            
        # Move to next model (cyclic)
        next_idx = (current_idx + 1) % len(models)
        new_model = models[next_idx]
        
        self.current_model_index = next_idx
        
        # Recreate LLM client with new model
        self.llm_client = LLMClient(model=new_model, log_level=self.log_level)
        
        # Update token limit
        self.max_context_tokens = self.model_max_tokens.get(new_model, 140000)
        
        if self.log_level <= LOG_INFO:
            syslog2(LOG_NOTICE, "model geted", new_model=new_model, max_tokens=self.max_context_tokens)
        
        return f"Модель переключена на: {new_model}\n({next_idx + 1}/{len(models)})"

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
        
        # Update token limit
        self.max_context_tokens = self.model_max_tokens.get(model_name, 140000)
        
        if self.log_level <= LOG_INFO:
            syslog2(LOG_NOTICE, "model set", new_model=model_name, max_tokens=self.max_context_tokens)
            
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
        
    def _clear_active_context(self, reason: str) -> None:
        """
        Clear the active RAG context cache.
        
        Args:
            reason: Reason for clearing (for logging)
        """
        self.active_context_chunks = None
        self.active_context_query = None
        self.active_context_score = None
        if self.log_level <= LOG_INFO:
            syslog2(LOG_NOTICE, "rag_context_cleared", reason=reason)
    
    def _is_good_context(self, context_chunks: List[Dict]) -> tuple[bool, float]:
        """
        Evaluate if RAG context is good enough to cache.
        
        Args:
            context_chunks: List of retrieved chunk dictionaries
            
        Returns:
            Tuple of (is_good, max_score)
        """
        if not context_chunks:
            if self.log_level <= LOG_DEBUG:
                syslog2(LOG_DEBUG, "rag_context_evaluated", good=False, max_score=0.0, chunks=0)
            return (False, 0.0)
        
        # Extract scores from chunks
        # Chunks from HybridRetrievalService have 'score' field (similarity 0.0-1.0)
        scores = []
        for chunk in context_chunks:
            score = chunk.get("score")
            if score is not None:
                scores.append(float(score))
        
        if not scores:
            if self.log_level <= LOG_DEBUG:
                syslog2(LOG_DEBUG, "rag_context_evaluated", good=False, max_score=0.0, chunks=len(context_chunks), note="no_scores")
            return (False, 0.0)
        
        max_score = max(scores)
        
        # Use threshold from config
        # cosine_distance_thr in config is distance (higher = worse), but chunks have similarity (0.0-1.0, higher = better)
        # For similarity scores, we want at least some reasonable quality
        # Default threshold: 0.3 similarity (reasonable quality)
        threshold = 0.3
        
        # Check if we have at least one chunk with good similarity
        is_good = max_score >= threshold and len(context_chunks) > 0
        
        if self.log_level <= LOG_DEBUG:
            syslog2(LOG_DEBUG, "rag_context_evaluated", good=is_good, max_score=max_score, chunks=len(context_chunks), threshold=threshold)
        
        return (is_good, max_score)
    
    def _should_refresh_context(self) -> bool:
        """
        Determine if RAG context should be refreshed.
        
        Returns:
            True if context should be refreshed (no active context), False otherwise
        """
        return self.active_context_chunks is None
    
    def _build_new_context(self, user_input: str, n_results: int) -> List[Dict]:
        """
        Build new RAG context by retrieving chunks.
        
        Args:
            user_input: User query string
            n_results: Number of chunks to retrieve
            
        Returns:
            List of context chunk dictionaries
        """
        return self.retrieval_service.retrieve(
            user_input, n_results=n_results, score_threshold=self.config.fts5_score_thr
        )
    
    def _evaluate_context_quality(self, context_chunks: List[Dict]) -> tuple[bool, float]:
        """
        Evaluate context quality and return (is_good, max_score).
        
        Args:
            context_chunks: List of context chunk dictionaries
            
        Returns:
            Tuple of (is_good, max_score)
        """
        return self._is_good_context(context_chunks)
    
    def _get_or_build_context(self, user_input: str, n_results: int) -> List[Dict]:
        """
        Get cached RAG context or build new one if needed.
        
        Args:
            user_input: User query string
            n_results: Number of chunks to retrieve
            
        Returns:
            List of context chunk dictionaries
        """
        if self._should_refresh_context():
            # Build new context
            context_chunks = self._build_new_context(user_input, n_results)
            
            # Evaluate context quality
            is_good, max_score = self._evaluate_context_quality(context_chunks)
            
            if is_good:
                # Cache the context
                self.active_context_chunks = context_chunks
                self.active_context_query = user_input
                self.active_context_score = max_score
                syslog2(LOG_NOTICE, "rag_context_new", query=user_input[:80], chunks=len(context_chunks), max_score=max_score)
            else:
                # Context not good enough, don't cache but still use it once
                self._clear_active_context(reason="context_not_good")
                if self.log_level <= LOG_DEBUG:
                    syslog2(LOG_DEBUG, "rag_context_not_cached", query=user_input[:80], chunks=len(context_chunks), max_score=max_score)
            
            return context_chunks
        else:
            # Reuse cached context
            syslog2(LOG_NOTICE, "rag_context_reused", source_query=self.active_context_query[:80] if self.active_context_query else None, chunks=len(self.active_context_chunks))
            return self.active_context_chunks
    
    def reset_context(self) -> str:
        """
        Reset the chat history/context.
        
        Returns:
            Confirmation message.
        """
        self.conversation_state.reset()
        return "Контекст сброшен!"
    
    def _build_history_for_prompt(self, max_messages: int = 5) -> List[Dict[str, str]]:
        """
        Build history for prompt from chat history.
        
        Args:
            max_messages: Maximum number of recent messages to include
            
        Returns:
            List of history entries with sender and content
        """
        return self.prompt_builder.build_history_for_prompt(max_messages)
    
    def _build_prompt_and_history(
        self, 
        context_chunks: List[Dict], 
        user_task: str, 
        custom_template: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Build system prompt and history for prompt (helper to reduce duplication).
        
        Args:
            context_chunks: Retrieved context chunks
            user_task: User task/query string
            custom_template: Optional custom system prompt template
            history: Optional pre-built history (if None, builds from chat_history)
            
        Returns:
            Tuple of (system_prompt, history_for_prompt)
        """
        return self.prompt_builder.build_prompt_and_history(
            context_chunks=context_chunks,
            user_task=user_task,
            custom_template=custom_template,
            history=history
        )
    
    def _build_messages_for_token_count(self, system_prompt: str, user_content: str = "") -> List[Dict[str, str]]:
        """
        Build messages list for token counting.
        
        Args:
            system_prompt: System prompt content
            user_content: User message content (optional)
            
        Returns:
            List of message dictionaries
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    
    def _calculate_token_usage(self, system_prompt: str, user_content: str = "") -> Dict[str, Union[int, float]]:
        """
        Calculate token usage for given messages.
        
        Args:
            system_prompt: System prompt content
            user_content: User message content (optional)
            
        Returns:
            Dictionary with current_tokens, max_tokens, and percentage
        """
        messages = self._build_messages_for_token_count(system_prompt, user_content)
        
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

        # Use real context chunks and user task if available, otherwise use empty values
        # This provides more accurate token usage estimation
        context_chunks = self.conversation_state.active_context_chunks if self.conversation_state.active_context_chunks else []
        user_task = self.conversation_state.active_context_query if self.conversation_state.active_context_query else ""
        
        # If no active context, try to get last user message from chat history
        if not user_task:
            user_task = self.conversation_state.get_last_user_message() or ""

        # Build prompt using helper method (reuses existing logic)
        system_prompt, _ = self._build_prompt_and_history(
            context_chunks=context_chunks,
            user_task=user_task
        )

        # Count tokens: system prompt + user content (same as real LLM call)
        user_content = user_task if user_task else ""
        return self._calculate_token_usage(system_prompt, user_content=user_content)

    def _check_token_limit_exceeded(self) -> bool:
        """
        Check if token limit is exceeded.
        
        Returns:
            True if token limit is exceeded, False otherwise
        """
        if not self.conversation_state.chat_history:
            return False
        
        token_usage = self.get_token_usage()
        # можно сбрасывать не по 100%, а, например, по 0.8 * лимита
        return token_usage["current_tokens"] >= self.max_context_tokens
    
    def _reset_context_if_needed(self) -> str:
        """
        Reset context if needed and return warning message.
        
        Returns:
            Warning message if context was reset, empty string otherwise
        """
        if not self._check_token_limit_exceeded():
            return ""
        
        token_usage = self.get_token_usage()
        had_active_context = self.conversation_state.active_context_chunks is not None
        self.reset_context()
        warning = "Контекст был автоматически сброшен из-за достижения лимита токенов.\n\n"
        if self.log_level <= LOG_INFO:
            syslog2(LOG_WARNING, "auto reset context", token_usage=f"{token_usage['current_tokens']}/{self.max_context_tokens}", had_active_context=had_active_context)
        return warning
    
    def _ensure_context_limit(self) -> str:
        """
        Ensure context doesn't exceed token limit by resetting if necessary.
        
        Returns:
            Warning message if context was reset, empty string otherwise
        """
        return self._reset_context_if_needed()
    
    def _is_token_limit_error(self, error_msg: str) -> bool:
        """
        Check if error is related to token limit or payment issues.
        
        Args:
            error_msg: Error message string
            
        Returns:
            True if error is token limit related
        """
        error_lower = error_msg.lower()
        token_limit_keywords = [
            "402",
            "context_length_exceeded",
            "prompt tokens limit exceeded",
            "token limit exceeded",
            "maximum context length",
            "context length exceeded",
        ]
        return any(keyword in error_lower for keyword in token_limit_keywords)
    
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
        
        # Reconstruct prompt without history using _build_prompt_and_history
        system_prompt, _ = self._build_prompt_and_history(
            context_chunks=context_chunks,
            user_task=user_input,
            custom_template=system_prompt_template,
            history=[]  # Empty history after reset
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
            from src.bot.utils.response_formatter import ResponseFormatter
            return ResponseFormatter.format_error_message("Не удалось получить ответ даже после сброса контекста (лимит токенов или баланс исчерпан).")
    
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
        except TimeoutError as e:
            # Handle timeout errors separately
            syslog2(LOG_ERR, "llm call timeout", error=str(e))
            raise
        except APITimeoutError as e:
            # Handle API timeout errors separately
            syslog2(LOG_ERR, "llm api timeout", error=str(e))
            raise
        except RateLimitError as e:
            # Handle rate limit errors separately
            syslog2(LOG_ERR, "llm rate limit error", error=str(e))
            raise
        except APIError as e:
            # Handle API errors separately
            syslog2(LOG_ERR, "llm api error", error=str(e))
            raise
        except APIConnectionError as e:
            # Handle connection errors separately
            syslog2(LOG_ERR, "llm connection error", error=str(e))
            raise
        except Exception as e:
            # Check for token limit or payment issues (existing logic)
            # This catches any other exceptions, including those that might be
            # wrapped or have token limit errors in their message
            error_msg = str(e)
            if self._is_token_limit_error(error_msg):
                return self._retry_after_reset(context_chunks, user_input, system_prompt_template)
            else:
                # Re-raise other errors
                raise
    
    def _get_context_for_query(self, user_input: str, n_results: int) -> List[Dict]:
        """
        Get context chunks for user query.
        
        Args:
            user_input: User query string
            n_results: Number of chunks to retrieve
            
        Returns:
            List of context chunk dictionaries
        """
        if self.debug_rag and self.log_level <= LOG_DEBUG:
            syslog2(LOG_DEBUG, "rag_debug_state", has_active_context=self.conversation_state.active_context_chunks is not None, active_query=self.conversation_state.active_context_query[:80] if self.conversation_state.active_context_query else None)
        
        syslog2(LOG_NOTICE, "retrieving context", retrieval_type=self.retrieval_type, cached=self.conversation_state.active_context_chunks is not None)
        context_chunks = self.context_provider.get_or_build_context(user_input, n_results)
        
        return context_chunks

    def _build_llm_messages(
        self,
        context_chunks: List[Dict],
        user_input: str,
        system_prompt_template: str = None
    ) -> List[Dict[str, str]]:
        """
        Build messages list for LLM API call.
        
        Args:
            context_chunks: List of context chunk dictionaries
            user_input: User query string
            system_prompt_template: Optional custom system prompt template
            
        Returns:
            List of message dictionaries for LLM API
        """
        return self.prompt_builder.build_messages_for_llm(
            context_chunks=context_chunks,
            user_input=user_input,
            system_prompt_template=system_prompt_template
        )

    def _call_llm_with_context(
        self,
        messages: List[Dict[str, str]],
        context_chunks: List[Dict],
        user_input: str,
        system_prompt_template: str = None
    ) -> str:
        """
        Call LLM with context and handle errors.
        
        Args:
            messages: List of message dictionaries for LLM API
            context_chunks: List of context chunk dictionaries
            user_input: User query string
            system_prompt_template: Optional custom system prompt template
            
        Returns:
            LLM response string
            
        Raises:
            Exception: If LLM call fails
        """
        return self.llm_gateway.call_with_context(messages, context_chunks, user_input, system_prompt_template)

    def chat(self, user_input: str, n_results: int = 3, respond: bool = True, system_prompt_template: str = None) -> str:
        """
        Process a user message and return the bot's response.
        """
        auto_reset_warning = self._ensure_context_limit()

        if not respond:
            self.conversation_state.add_user_message(user_input)
            return ""

        # Get context for query
        context_chunks = self._get_context_for_query(user_input, n_results)

        # Build LLM messages
        messages = self._build_llm_messages(context_chunks, user_input, system_prompt_template)

        # Call LLM with context
        try:
            response = self._call_llm_with_context(messages, context_chunks, user_input, system_prompt_template)
            # Check if retry already included warning
            if "Ошибка лимита токенов" in response:
                auto_reset_warning = ""
        except Exception as e:
            syslog2(LOG_ERR, "llm call failed", error=str(e))
            return f"Произошла ошибка при обращении к нейросети: {e}"

        self.conversation_state.add_user_message(user_input)
        self.conversation_state.add_assistant_message(response)

        return auto_reset_warning + response

    async def complete(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Async wrapper for LLM completion.
        Used by ProfileCommandHandler and AliasDiscoveryService.
        """
        if self.log_level >= LOG_DEBUG:
            syslog2(LOG_DEBUG, "bot.complete called", prompt_length=len(prompt), has_system_prompt=system_prompt is not None, system_prompt_length=len(system_prompt) if system_prompt else 0, kwargs=kwargs)
        
        # We run the synchronous LLM call directly.
        # Ideally this should be run_in_executor to avoid blocking the loop, 
        # but for now we keep it simple as the underlying HTTP client might be blocking anyway.
        response = self.llm_client.complete(prompt, system=system_prompt, **kwargs)
        
        # Log messages structure after completion (we can't easily intercept before, but LLMClient logs it)
        if self.log_level >= LOG_DEBUG:
            # Construct what messages would look like
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            syslog2(LOG_DEBUG, "bot.complete messages structure", messages_count=len(messages), system_role_present=system_prompt is not None, user_role_present=True)
        
        return response

    def get_rag_debug_info(self, user_input: str, n_results: int = 5) -> Dict:
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
            user_task=user_input
        )
        
        # Count tokens using helper
        token_usage = self._calculate_token_usage(system_prompt, user_input)
        
        return {
            "chunks": context_chunks,
            "prompt": system_prompt,
            "token_count": token_usage["current_tokens"],
            "chunks_count": len(context_chunks),
            "had_active_context": self.conversation_state.active_context_chunks is not None,
            "active_context_query": self.conversation_state.active_context_query
        }
