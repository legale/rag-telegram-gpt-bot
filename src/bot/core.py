from typing import List, Dict, Optional
from src.storage.db import Database
from src.storage.vector_store import VectorStore
from src.core.embedding import EmbeddingClient
from src.core.retrieval import RetrievalService
from src.core.prompt import PromptEngine
from src.core.llm import LLMClient
import os

class LegaleBot:
    """Main bot class orchestrating the RAG pipeline."""
    
    def __init__(self, db_url: str, vector_db_path: str, model_name: str = "openai/gpt-3.5-turbo", verbosity: int = 0):
        # Initialize components
        if not db_url or not vector_db_path:
            raise ValueError("db_url and vector_db_path must be provided")
            
        self.db = Database(db_url)
        self.vector_store = VectorStore(persist_directory=vector_db_path)
        self.verbosity = verbosity
        
        # Initialize clients with credentials from env
        self.embedding_client = EmbeddingClient()
        self.llm_client = LLMClient(model=model_name, verbosity=verbosity)
        
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
        
        # Model switching support
        self.available_models = self._load_available_models()
        self.current_model_index = 0
        # Find initial model index
        if model_name in self.available_models:
            self.current_model_index = self.available_models.index(model_name)
    
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
            return models if models else ["openai/gpt-3.5-turbo"]
        except FileNotFoundError:
            if self.verbosity >= 1:
                print(f"[Warning] models.txt not found at {models_file}, using default model")
            return ["openai/gpt-3.5-turbo"]
    
    def switch_model(self) -> str:
        """
        Switch to the next model in the list (cyclic).
        
        Returns:
            Message with the new model name.
        """
        if not self.available_models:
            return "❌ Нет доступных моделей для переключения."
        
        # Move to next model (cyclic)
        self.current_model_index = (self.current_model_index + 1) % len(self.available_models)
        new_model = self.available_models[self.current_model_index]
        
        # Recreate LLM client with new model
        self.llm_client = LLMClient(model=new_model, verbosity=self.verbosity)
        
        if self.verbosity >= 1:
            print(f"[Model Switch] Changed to: {new_model}")
        
        return f"✅ Модель переключена на: {new_model}\n({self.current_model_index + 1}/{len(self.available_models)})"
    
    def get_current_model(self) -> str:
        """
        Get current model information.
        
        Returns:
            Current model name and position in the list.
        """
        if not self.available_models:
            return "❌ Нет доступных моделей."
        
        current_model = self.available_models[self.current_model_index]
        return f"🤖 Текущая модель: {current_model}\n({self.current_model_index + 1}/{len(self.available_models)})"
        
    def reset_context(self) -> str:
        """
        Reset the chat history/context.
        
        Returns:
            Confirmation message.
        """
        self.chat_history = []
        return "✅ Контекст сброшен!"
    
    def get_token_usage(self) -> Dict[str, int]:
        """
        Get current token usage statistics.
        
        Returns:
            Dictionary with token usage info.
        """
        # Build current messages to count tokens
        if not self.chat_history:
            return {
                "current_tokens": 0,
                "max_tokens": self.max_context_tokens,
                "percentage": 0.0
            }
        
        # Simulate what would be sent to LLM
        history_for_prompt = []
        for msg in self.chat_history[-5:]:
            sender = "User" if msg['role'] == 'user' else "Bot"
            history_for_prompt.append({"sender": sender, "content": msg['content']})
        
        # Create a sample prompt to estimate tokens
        system_prompt = self.prompt_engine.construct_prompt(
            context_chunks=[],
            chat_history=history_for_prompt,
            user_task="sample"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": system_prompt}
        ]
        
        current_tokens = self.llm_client.count_tokens(messages)
        percentage = (current_tokens / self.max_context_tokens) * 100
        
        return {
            "current_tokens": current_tokens,
            "max_tokens": self.max_context_tokens,
            "percentage": round(percentage, 2)
        }

    def chat(self, user_input: str, n_results: int = 3, respond: bool = True) -> str:
        """
        Process a user message and return the bot's response.
        
        Args:
            user_input: The user's message.
            n_results: Number of context chunks to retrieve.
            respond: Whether to generate a response (default: True).
                     If False, message is only added to history.
            
        Returns:
            The bot's response or empty string if respond=False.
        """
        # Check token usage and auto-reset if needed
        auto_reset_warning = ""
        if self.chat_history:
            token_usage = self.get_token_usage()
            if token_usage["current_tokens"] >= self.max_context_tokens:
                self.reset_context()
                auto_reset_warning = "⚠️ Контекст был автоматически сброшен из-за достижения лимита токенов.\n\n"
                if self.verbosity >= 1:
                    print(f"[Auto-reset] Token limit reached: {token_usage['current_tokens']}/{self.max_context_tokens}")
        
        # If not responding, just add to history and return
        if not respond:
            self.chat_history.append({"role": "user", "content": user_input})
            # We don't add assistant response since there is none
            # But wait, if we don't add response, history will be User, User, User...
            # This is fine for many models, they treat it as monologue or multi-part message.
            return ""
        
        # 1. Retrieve Context
        print(f"Retrieving context ({n_results} chunks)...")
        context_chunks = self.retrieval_service.retrieve(user_input, n_results=n_results)
        
        # 2. Construct Prompt
        # Convert internal history format to what PromptEngine expects if needed
        # PromptEngine expects [{'sender': '...', 'content': '...'}]
        # Our internal history is [{'role': 'user/assistant', 'content': '...'}]
        
        history_for_prompt = []
        for msg in self.chat_history[-5:]: # Last 5 messages
            sender = "User" if msg['role'] == 'user' else "Bot"
            history_for_prompt.append({"sender": sender, "content": msg['content']})
            
        # Add current user input to history for prompt context (optional, or just pass as task)
        # The prompt engine treats user_task as the immediate instruction.
        
        system_prompt = self.prompt_engine.construct_prompt(
            context_chunks=context_chunks,
            chat_history=history_for_prompt,
            user_task=user_input
        )
        
        # 3. Call LLM
        print("Querying LLM...")
        messages = [
            {"role": "system", "content": system_prompt},
            # We don't necessarily need to pass the full history here again if it's already in the system prompt,
            # but standard chat models expect a conversation history.
            # However, our "System Prompt" already includes the "Context" and "History" from the RAG perspective.
            # So we can just send the system prompt and maybe the user message again?
            # Or just the system prompt as a single instruction?
            # Let's try sending just the system prompt as the 'user' message or 'system' message.
            # Actually, usually:
            # System: You are... Here is context...
            # User: <query>
            
            # But our PromptEngine puts the query inside the system prompt template under {task}.
            # So we can just send one message.
            {"role": "user", "content": system_prompt} 
        ]
        
        response = self.llm_client.complete(messages)
        
        # 4. Update History
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": response})
        
        # Prepend auto-reset warning if context was reset
        return auto_reset_warning + response
