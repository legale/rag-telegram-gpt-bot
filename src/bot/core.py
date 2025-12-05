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
    
    def __init__(self, db_url: str = "sqlite:///legale_bot.db", vector_db_path: str = "chroma_db", model_name: str = "openai/gpt-3.5-turbo", verbosity: int = 0):
        # Initialize components
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

    def chat(self, user_input: str, n_results: int = 5) -> str:
        """
        Process a user message and return the bot's response.
        
        Args:
            user_input: The user's message.
            n_results: Number of context chunks to retrieve.
            
        Returns:
            The bot's response.
        """
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
        
        return response
