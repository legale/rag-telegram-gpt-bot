from src.core.access_control import check_access


def test_check_access_allows_admin_everywhere():
    allowed, reason = check_access(
        user_id=1,
        chat_id=2,
        is_private=True,
        is_command=False,
        allowed_chats=[],
        is_admin=lambda _uid: True,
    )
    assert allowed is True
    assert reason is None


def test_check_access_allows_admin_set_in_private():
    allowed, reason = check_access(
        user_id=1,
        chat_id=1,
        is_private=True,
        is_command=True,
        allowed_chats=[],
        is_admin=lambda _uid: False,
        command_text="/admin_set pass",
    )
    assert allowed is True
    assert reason is None


def test_check_access_denies_private_non_admin():
    allowed, reason = check_access(
        user_id=1,
        chat_id=1,
        is_private=True,
        is_command=False,
        allowed_chats=[],
        is_admin=lambda _uid: False,
    )
    assert allowed is False
    assert reason == "private_non_admin"


def test_check_access_allows_group_commands():
    allowed, reason = check_access(
        user_id=1,
        chat_id=10,
        is_private=False,
        is_command=True,
        allowed_chats=[],
        is_admin=lambda _uid: False,
    )
    assert allowed is True
    assert reason is None


def test_check_access_whitelist_required_for_group_messages():
    allowed, reason = check_access(
        user_id=1,
        chat_id=10,
        is_private=False,
        is_command=False,
        allowed_chats=[11],
        is_admin=lambda _uid: False,
    )
    assert allowed is False
    assert reason == "chat_not_whitelisted"

