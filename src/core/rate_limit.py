"""Rate limiting / response frequency policy for chat responses."""

from __future__ import annotations

from typing import Dict, Tuple


def allow(
    *,
    chat_id: int,
    frequency: int,
    has_mention: bool,
    is_command: bool,
    is_private: bool,
    chat_counters: Dict[int, int],
) -> Tuple[bool, str]:
    """
    Decide whether to respond based on response frequency.

    Mutates `chat_counters` (per-chat message counter) in-place.
    """
    if is_command or is_private:
        return True, "command_or_private"

    if has_mention:
        return True, "mentioned"

    if frequency < 1:
        return False, "freq_zero_no_mention"

    if frequency == 1:
        return True, "freq_one"

    current = chat_counters.get(chat_id, 0) + 1
    chat_counters[chat_id] = current

    if current % frequency == 0:
        return True, f"freq_match_{current}"
    return False, f"freq_skip_{current}"

