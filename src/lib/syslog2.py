import logging
import inspect
import threading
from datetime import datetime
from typing import Optional


LOG_ALERT = 1
LOG_CRIT = 2
LOG_ERR = 3
LOG_WARNING = 4
LOG_NOTICE = 5
LOG_INFO = 6
LOG_DEBUG = 7


def _sys_to_py(level: int) -> int:
    return 80 - level * 10


_log = logging.getLogger("app")
_current_syslog_level = LOG_DEBUG
_setup_done = False
_lock = threading.Lock()
_with_caller = True


def setup_log(syslog_level: int = LOG_DEBUG, with_caller: bool = True) -> None:
    global _current_syslog_level, _setup_done, _with_caller
    _current_syslog_level = syslog_level
    _with_caller = with_caller

    if _setup_done:
        return
    _setup_done = True

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    logging.addLevelName(_sys_to_py(LOG_ALERT), "ALERT")
    logging.addLevelName(_sys_to_py(LOG_CRIT), "CRIT")
    logging.addLevelName(_sys_to_py(LOG_ERR), "ERR")
    logging.addLevelName(_sys_to_py(LOG_WARNING), "WARNING")
    logging.addLevelName(_sys_to_py(LOG_NOTICE), "NOTICE")
    logging.addLevelName(_sys_to_py(LOG_INFO), "INFO")
    logging.addLevelName(_sys_to_py(LOG_DEBUG), "DEBUG")


def _get_caller_info(depth: int = 1) -> tuple[str, int, str]:
    frame = inspect.currentframe()
    try:
        for _ in range(depth):
            if frame is None:
                break
            frame = frame.f_back
        if frame is None:
            return "unknown", 0, "unknown"
        file_name = frame.f_code.co_filename.rsplit("/", 1)[-1]
        return file_name, frame.f_lineno, frame.f_code.co_name
    finally:
        del frame


def _ts_now() -> str:
    return datetime.now().strftime("%d.%m.%y %H:%M:%S.%f")[:-3]


def _lower_first_alpha(s: str) -> str:
    if not s:
        return s
    c = s[0]
    if c.isalpha():
        return c.lower() + s[1:]
    return s


def _format_kv(params: dict) -> str:
    if not params:
        return ""
    parts = []
    for k, v in params.items():
        if isinstance(v, str) and "\n" in v:
            parts.append(f"{k}=\n{v}")
        else:
            parts.append(f"{k}={repr(v)}")
    return " " + " ".join(parts)


# Backwards-compatible alias for tests and external callers
def _format_params(params: dict) -> str:
    return _format_kv(params)


def _prefix() -> str:
    ts = _ts_now()
    if not _with_caller:
        return f"{ts}:"
    file_name, line_no, func_name = _get_caller_info(3)
    return f"{ts} {file_name}:{line_no} {func_name}:"


def syslog2(level: int, msg: str, **params) -> None:
    if level > _current_syslog_level:
        return

    py_level = _sys_to_py(level)
    msg = _lower_first_alpha(msg)
    body = msg + _format_kv(params)

    with _lock:
        _log.log(py_level, f"{_prefix()} {body}")


def syslog2_exc(level: int, msg: str, exc: Optional[BaseException] = None, **params) -> None:
    if level > _current_syslog_level:
        return

    py_level = _sys_to_py(level)
    msg = _lower_first_alpha(msg)

    if exc is None:
        exc_info = True
        body = msg + _format_kv(params)
    else:
        exc_info = exc
        body = msg + _format_kv(params) + f" exc={type(exc).__name__}:{exc}"

    with _lock:
        _log.log(py_level, f"{_prefix()} {body}", exc_info=exc_info)