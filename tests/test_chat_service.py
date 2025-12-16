from src.core.chat_service import chat
from src.core.prompt import PromptEngine


class _Retrieval:
    def __init__(self, chunks):
        self._chunks = chunks
        self.calls = []

    def retrieve(self, query: str, n_results: int = 3, score_threshold=None):
        self.calls.append((query, n_results, score_threshold))
        return list(self._chunks)


class _LLM:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def complete(self, messages, temperature: float = 0.7, max_tokens: int = 1500):
        self.calls.append((messages, temperature, max_tokens))
        return self.response


class _FailRetrieval:
    def retrieve(self, query: str, n_results: int = 3, score_threshold=None):
        raise RuntimeError("boom")


class _FailLLM:
    def complete(self, messages, temperature: float = 0.7, max_tokens: int = 1500):
        raise RuntimeError("boom")


def test_chat_returns_empty_on_blank_input():
    retrieval = _Retrieval([])
    llm = _LLM("x")
    prompt = PromptEngine()
    history = []
    assert chat(retrieval=retrieval, llm=llm, prompt_engine=prompt, user_input="   ", chat_history=history) == ""
    assert history == []


def test_chat_appends_history_and_calls_dependencies():
    retrieval = _Retrieval([{"text": "chunk"}])
    llm = _LLM("answer")
    prompt = PromptEngine()
    history = []
    result = chat(retrieval=retrieval, llm=llm, prompt_engine=prompt, user_input="hi", chat_history=history, n_results=5)
    assert result == "answer"
    assert retrieval.calls == [("hi", 5, None)]
    assert len(llm.calls) == 1
    assert history == [{"sender": "user", "content": "hi"}, {"sender": "assistant", "content": "answer"}]


def test_chat_no_respond_records_user_only():
    retrieval = _Retrieval([{"text": "chunk"}])
    llm = _LLM("answer")
    prompt = PromptEngine()
    history = []
    result = chat(retrieval=retrieval, llm=llm, prompt_engine=prompt, user_input="hi", chat_history=history, respond=False)
    assert result == ""
    assert retrieval.calls == []
    assert llm.calls == []
    assert history == [{"sender": "user", "content": "hi"}]


def test_chat_converts_role_history_for_prompt():
    retrieval = _Retrieval([{"text": "chunk"}])
    llm = _LLM("answer")
    prompt = PromptEngine()
    history = [{"role": "user", "content": "prev"}]
    result = chat(retrieval=retrieval, llm=llm, prompt_engine=prompt, user_input="hi", chat_history=history)
    assert result == "answer"
    assert history[0] == {"role": "user", "content": "prev"}


def test_chat_handles_retrieval_error():
    prompt = PromptEngine()
    history = []
    result = chat(retrieval=_FailRetrieval(), llm=_LLM("x"), prompt_engine=prompt, user_input="hi", chat_history=history)
    assert "ошибка при поиске контекста" in result.lower()
    assert history == []


def test_chat_handles_llm_error():
    prompt = PromptEngine()
    history = []
    result = chat(retrieval=_Retrieval([{"text": "chunk"}]), llm=_FailLLM(), prompt_engine=prompt, user_input="hi", chat_history=history)
    assert "ошибка при обращении к нейросети" in result.lower()
    assert history == []
