from src.core.rate_limit import allow


def test_allow_command_or_private_always_true():
    counters = {}
    assert allow(chat_id=1, frequency=0, has_mention=False, is_command=True, is_private=False, chat_counters=counters)[0] is True
    assert allow(chat_id=1, frequency=0, has_mention=False, is_command=False, is_private=True, chat_counters=counters)[0] is True
    assert counters == {}


def test_allow_mention_always_true():
    counters = {}
    ok, reason = allow(chat_id=1, frequency=0, has_mention=True, is_command=False, is_private=False, chat_counters=counters)
    assert ok is True
    assert reason == "mentioned"
    assert counters == {}


def test_allow_frequency_zero_skips_without_counter_update():
    counters = {}
    ok, reason = allow(chat_id=1, frequency=0, has_mention=False, is_command=False, is_private=False, chat_counters=counters)
    assert ok is False
    assert reason == "freq_zero_no_mention"
    assert counters == {}


def test_allow_frequency_one_always_responds_without_counter_update():
    counters = {}
    ok, reason = allow(chat_id=1, frequency=1, has_mention=False, is_command=False, is_private=False, chat_counters=counters)
    assert ok is True
    assert reason == "freq_one"
    assert counters == {}


def test_allow_frequency_n_updates_counter_and_matches_every_n():
    counters = {}
    ok, reason = allow(chat_id=1, frequency=2, has_mention=False, is_command=False, is_private=False, chat_counters=counters)
    assert ok is False and reason == "freq_skip_1"
    ok, reason = allow(chat_id=1, frequency=2, has_mention=False, is_command=False, is_private=False, chat_counters=counters)
    assert ok is True and reason == "freq_match_2"
    assert counters[1] == 2

