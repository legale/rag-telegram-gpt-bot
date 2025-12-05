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
    
    user_task = "Analyze the situation."
    
    prompt = engine.construct_prompt(context_chunks, chat_history, user_task)
    
    assert "Ты адвокат профсоюза" in prompt
    assert "Boss: You must work overtime." in prompt
    assert "Worker: Is that paid?" in prompt
    assert "Boss: Do it now." in prompt
    assert "Analyze the situation." in prompt
