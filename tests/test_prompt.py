import pytest
from src.core.prompt import PromptEngine

def test_construct_prompt():
    engine = PromptEngine()
    
    context_chunks = [
        {"text": "Boss: You must work overtime.", "metadata": {}},
        {"text": "Worker: Is that paid?", "metadata": {}}
    ]
    
    chat_history = [
        {"sender": "Boss", "content": "Do it now."},
        {"sender": "Worker", "content": "I need to check my contract."}
    ]
    
    user_task = "Analyze the situation."  # No curly braces to avoid format() issues
    
    prompt = engine.construct_prompt(context_chunks, chat_history, user_task)
    
    # Check for actual prompt content (may vary based on system prompt template)
    # Just verify it contains the expected context and task
    assert "Boss: You must work overtime." in prompt or "librarian" in prompt or "assistant" in prompt
    assert "Boss: You must work overtime." in prompt
    assert "Worker: Is that paid?" in prompt
    assert "Boss: Do it now." in prompt
    assert "Analyze the situation." in prompt

