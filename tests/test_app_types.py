from src.app.types import AppRequest, AppResponse, CommandRequest, QueryRequest


def test_app_request_defaults():
    req = AppRequest()
    assert req.text == ""
    assert req.transport == "cli"


def test_app_response_defaults():
    resp = AppResponse()
    assert resp.text == ""
    assert resp.actions is None


def test_command_request_defaults():
    cmd = CommandRequest()
    assert cmd.name == ""
    assert cmd.args == []
    cmd.args.append("x")
    assert cmd.args == ["x"]


def test_query_request_defaults():
    q = QueryRequest()
    assert q.text == ""
