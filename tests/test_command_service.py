from src.app.types import CommandRequest
from src.core.command_service import CommandService
from src.core.dispatcher import CommandHandler, CommandContext, CommandResult


class _OkHandler(CommandHandler):
    def handle(self, context: CommandContext) -> CommandResult:
        return CommandResult(success=True, message="ok")


def test_dispatch_accepts_command_context():
    service = CommandService()
    service.register("ping", _OkHandler())
    result = service.dispatch(CommandContext(command_name="/ping"))
    assert result.success is True
    assert result.message == "ok"


def test_dispatch_accepts_command_request():
    service = CommandService()
    service.register("ping", _OkHandler())
    result = service.dispatch(CommandRequest(name="/ping", args=[]))
    assert result.success is True
    assert result.message == "ok"


def test_dispatch_command_request_falls_back_to_raw_and_preserves_meta():
    service = CommandService()
    service.register("ping", _OkHandler())
    result = service.dispatch(CommandRequest(name="", raw="/ping arg1", meta={"k": "v"}))
    assert result.success is True
    assert result.message == "ok"


def test_dispatch_rejects_unknown_type():
    service = CommandService()
    try:
        service.dispatch(object())  # type: ignore[arg-type]
    except TypeError as exc:
        assert "dispatch expects" in str(exc)
    else:
        raise AssertionError("expected TypeError")
