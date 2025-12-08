# src/core/syslog2.py

import logging
import inspect
from datetime import datetime



class LogLevel:
    """Syslog-like log levels: 1 = highest severity, 7 = lowest"""
    LOG_ALERT   = 1
    LOG_CRIT    = 2
    LOG_ERR     = 3
    LOG_WARNING = 4
    LOG_NOTICE  = 5
    LOG_INFO    = 6
    LOG_DEBUG   = 7



"""global Syslog-like log levels: 1 = highest severity, 7 = lowest"""
LOG_ALERT   = 1
LOG_CRIT    = 2
LOG_ERR     = 3
LOG_WARNING = 4
LOG_NOTICE  = 5
LOG_INFO    = 6
LOG_DEBUG   = 7

# map 1..7 -> unique python logging levels (higher = more severe)
# 1 -> 70, 2 -> 60, 3 -> 50, 4 -> 40, 5 -> 30, 6 -> 20, 7 -> 10
def _sys_to_py(level: int) -> int:
    return 80 - level * 10

_current_syslog_level = LOG_DEBUG
log = logging.getLogger("app")


def setup_log(syslog_level: int = LOG_DEBUG) -> None:
    # init backend and register custom levels
    global _current_syslog_level
    _current_syslog_level = syslog_level

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    logging.addLevelName(_sys_to_py(LOG_ALERT),   "ALERT")
    logging.addLevelName(_sys_to_py(LOG_CRIT),    "CRIT")
    logging.addLevelName(_sys_to_py(LOG_ERR),     "ERR")
    logging.addLevelName(_sys_to_py(LOG_WARNING), "WARNING")
    logging.addLevelName(_sys_to_py(LOG_NOTICE),  "NOTICE")
    logging.addLevelName(_sys_to_py(LOG_INFO),    "INFO")
    logging.addLevelName(_sys_to_py(LOG_DEBUG),   "DEBUG")


def _format_params(params: dict) -> str:
    if not params:
        return ""
    parts = []
    for k, v in params.items():
        if isinstance(v, str) and '\n' in v:
            # Multiline string - format specially
            parts.append(f"{k}=")
            parts.append(v)  # Will be handled in syslog2
        else:
            parts.append(f"{k}={repr(v)}")
    return " " + " ".join(parts) if parts else ""


def _get_caller_info():
    frame = inspect.currentframe()
    # frame is _get_caller_info
    # frame.f_back is syslog2
    # frame.f_back.f_back is the actual caller
    try:
        caller = frame.f_back.f_back
    except AttributeError:
        caller = None
        
    if not caller:
        return "unknown", 0, "unknown"
    file_name = caller.f_code.co_filename.rsplit("/", 1)[-1]
    line_no = caller.f_lineno
    func_name = caller.f_code.co_name
    return file_name, line_no, func_name


def syslog2(level: int, msg: str, **params) -> None:
    # filter by configured 1..7 level
    if level > _current_syslog_level:
        return

    py_level = _sys_to_py(level)
    file_name, line_no, func_name = _get_caller_info()
    ts = datetime.now().strftime("%d.%m.%y %H:%M:%S.%f")[:-3]

    if msg:
        first = msg[0]
        if first.isalpha():
            msg = first.lower() + msg[1:]

    prefix = f"{ts} {file_name}:{line_no} {func_name}:"
    
    # Format parameters, handling multiline values
    params_parts = []
    for k, v in params.items():
        if isinstance(v, str) and '\n' in v:
            params_parts.append((k, v, True))  # (key, value, is_multiline)
        else:
            params_parts.append((k, repr(v), False))
    
    # Build the full message body
    if not params_parts:
        full_body = msg
    else:
        # Start with message and first param
        first_key, first_val, first_multiline = params_parts[0]
        if first_multiline:
            full_body = f"{msg} {first_key}=\n{first_val}"
        else:
            full_body = f"{msg} {first_key}={first_val}"
        
        # Add remaining params
        for k, v, is_multiline in params_parts[1:]:
            if is_multiline:
                full_body += f" {k}=\n{v}"
            else:
                full_body += f" {k}={v}"
    
    # Handle multiline messages
    lines = full_body.split('\n')
    
    if len(lines) == 1:
        # Single line - output as before
        log.log(py_level, f"{prefix} {full_body}")
    else:
        # Multiline - first line with prefix, subsequent lines with indentation
        log.log(py_level, f"{prefix} {lines[0]}")
        indent = " " * len(prefix)
        for line in lines[1:]:
            log.log(py_level, f"{indent} {line}")