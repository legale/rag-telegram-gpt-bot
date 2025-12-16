"""Chat orchestration service (retrieval + prompt + llm)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from src.core.prompt import PromptEngine
from src.lib.syslog2 import LOG_ERR, LOG_WARNING, syslog2


class Retrieval(Protocol):
    def retrieve(self, query: str, n_results: int = 3, score_threshold: Optional[float] = None) -> List[Dict]:
        ...


class LLM(Protocol):
    def complete(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1500) -> str:
        ...


def chat(
    *,
    retrieval: Retrieval,
    llm: LLM,
    prompt_engine: PromptEngine,
    user_input: str,
    chat_history: List[Dict[str, str]],
    n_results: int = 3,
    respond: bool = True,
    system_prompt_template: Optional[str] = None,
    score_threshold: Optional[float] = None,
    log_level: int = LOG_WARNING,
) -> str:
    """
    Execute a single chat turn.

    Keeps transport/file-system concerns outside: depends only on provided objects.
    Mutates `chat_history` in-place by appending the user message and assistant response (if any).
    """
    text = (user_input or "").strip()
    if not text:
        return ""

    if not respond:
        chat_history.append({"sender": "user", "content": text})
        return ""

    try:
        context_chunks = retrieval.retrieve(text, n_results=n_results, score_threshold=score_threshold)
    except Exception as exc:
        syslog2(LOG_ERR, "chat: retrieval failed", error=str(exc))
        return f"Произошла ошибка при поиске контекста: {exc}"

    prompt_history = _to_prompt_history(chat_history)
    system_prompt = prompt_engine.construct_prompt(
        context_chunks=context_chunks,
        chat_history=prompt_history,
        user_task=text,
        custom_template=system_prompt_template,
        log_level=log_level,
    )

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
    try:
        response = llm.complete(messages)
    except Exception as exc:
        syslog2(LOG_ERR, "chat: llm failed", error=str(exc))
        return f"Произошла ошибка при обращении к нейросети: {exc}"

    chat_history.append({"sender": "user", "content": text})
    chat_history.append({"sender": "assistant", "content": response})
    return response


def _to_prompt_history(history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in history:
        sender = item.get("sender")
        content = item.get("content", "")
        if sender is None:
            role = item.get("role")
            if role:
                sender = str(role)
        if sender is None:
            sender = "Unknown"
        normalized.append({"sender": str(sender), "content": str(content)})
    return normalized

