# argparse2.py

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Iterable, Sequence, Optional


class CLIError(Exception):
    """Raised when CLI arguments are invalid."""
    pass


class CLIHelp(Exception):
    """Raised to request help output."""
    pass


def matches(prefix: str, string: str) -> bool:
    if not prefix:
        return True
    i = 0
    while i < len(prefix) and i < len(string) and prefix[i] == string[i]:
        i += 1
    return i == len(prefix)


def split_args(s: str) -> list[str]:
    return s.strip().split()


def _find_unique(prefix: str, names: list[str]) -> str:
    hits = [n for n in names if matches(prefix, n)]
    if not hits:
        raise ValueError(f"unknown option: {prefix}")
    if len(hits) > 1:
        raise ValueError(f"ambiguous option: {prefix}")
    return hits[0]


def parse(argv: list[str], opt_table: dict) -> tuple[dict, list[str]]:
    i = 0
    opts = {}
    args = []
    names = list(opt_table.keys())

    while i < len(argv):
        a = argv[i]
        if a == "--":
            i += 1
            break
        if not a.startswith("-") or a == "-":
            break

        opt = a.lstrip("-")
        name = _find_unique(opt, names)
        spec = opt_table[name]

        takes_arg = bool(spec.get("arg"))
        if takes_arg:
            if i + 1 >= len(argv):
                raise ValueError(f"missing arg for {a}")
            opts[name] = argv[i + 1]
            i += 2
        else:
            opts[name] = True
            i += 1

    while i < len(argv):
        args.append(argv[i])
        i += 1

    return opts, args


def gen_help(prog: str, opt_table: dict, cmd_table: dict | None = None) -> str:
    lines = []
    lines.append(f"usage: {prog} [options] <command> [args]")
    lines.append("")
    lines.append("options:")

    names = sorted(opt_table.keys())
    for name in names:
        spec = opt_table[name]
        arg = spec.get("arg")
        meta = spec.get("meta", "ARG") if arg else ""
        desc = spec.get("desc", "")
        opt_str = f"  -{name}"
        if arg:
            opt_str += f" {meta}"
        if desc:
            if len(opt_str) < 26:
                opt_str += " " * (26 - len(opt_str))
            else:
                opt_str += " "
            opt_str += desc
        lines.append(opt_str)

    if cmd_table:
        lines.append("")
        lines.append("commands:")
        for name in sorted(cmd_table.keys()):
            desc = cmd_table[name].get("desc", "")
            s = f"  {name}"
            if desc:
                if len(s) < 26:
                    s += " " * (26 - len(s))
                else:
                    s += " "
                s += desc
            lines.append(s)

    return "\n".join(lines)


def cmd_parse(text_or_argv, opt_table: dict) -> tuple[dict, str, list[str]]:
    if isinstance(text_or_argv, str):
        argv = split_args(text_or_argv)
    else:
        argv = list(text_or_argv)

    opts, args = parse(argv, opt_table)

    if not args:
        return opts, "help", []

    return opts, args[0], args[1:]


# Additional functionality from cli_parser for compatibility

class ArgStream:
    """Utility for consuming command arguments sequentially."""

    def __init__(self, args: Iterable[str]):
        self._args = list(args)
        self._pos = 0

    def has_next(self) -> bool:
        return self._pos < len(self._args)

    def peek(self) -> str | None:
        if not self.has_next():
            return None
        return self._args[self._pos]

    def next(self) -> str:
        if not self.has_next():
            raise CLIError("command line is not complete")
        token = self._args[self._pos]
        self._pos += 1
        return token

    def expect(self, description: str) -> str:
        if not self.has_next():
            raise CLIError(f"expected {description}")
        return self.next()
    
    def find_and_remove(self, token: str) -> bool:
        """Find and remove a token from anywhere in the stream (order-independent)."""
        token_lower = token.lower()
        for i in range(len(self._args)):
            if self._args[i].lower() == token_lower:
                # Remove found token
                self._args.pop(i)
                # Adjust position if needed
                if i < self._pos:
                    self._pos -= 1
                return True
        return False
    
    def find_and_remove_next(self, token: str) -> Optional[str]:
        """Find a token and return the next value, removing both (order-independent)."""
        token_lower = token.lower()
        for i in range(len(self._args)):
            if self._args[i].lower() == token_lower:
                # Remove found token
                self._args.pop(i)
                # Adjust position if needed
                if i < self._pos:
                    self._pos -= 1
                # Get next value if exists
                if i < len(self._args):
                    value = self._args.pop(i)
                    # Adjust position again if needed
                    if i < self._pos:
                        self._pos -= 1
                    return value
                return None
        return None
    
    def reset(self):
        """Reset position to start."""
        self._pos = 0


@dataclass
class CommandSpec:
    name: str
    parser: Callable[[ArgStream], dict]
    help_text: Optional[str] = None


# Utility functions for parsing options
def parse_flag(stream: ArgStream, name: str) -> bool:
    """Parse a flag (boolean option) from anywhere in the stream."""
    return stream.find_and_remove(name)


def parse_option(stream: ArgStream, name: str) -> Optional[str]:
    """Parse an option with value from anywhere in the stream."""
    return stream.find_and_remove_next(name)


def parse_int_option(stream: ArgStream, name: str, default: Optional[int] = None) -> Optional[int]:
    """Parse an integer option from anywhere in the stream."""
    value = stream.find_and_remove_next(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        raise CLIError(f"invalid integer value for {name}: {value}")


def parse_float_option(stream: ArgStream, name: str, default: Optional[float] = None) -> Optional[float]:
    """Parse a float option from anywhere in the stream."""
    value = stream.find_and_remove_next(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        raise CLIError(f"invalid float value for {name}: {value}")


def parse_choice_option(stream: ArgStream, name: str, choices: Sequence[str], default: Optional[str] = None) -> Optional[str]:
    """Parse a choice option from anywhere in the stream."""
    value = stream.find_and_remove_next(name)
    if value is None:
        return default
    value_lower = value.lower()
    for choice in choices:
        if matches(value_lower, (choice,)):
            return choice
    raise CLIError(f"invalid choice for {name}: {value} (choices: {', '.join(choices)})")


class CommandParser:
    """Parses ip-style CLI commands with prefix matching."""

    def __init__(self, commands: Sequence[CommandSpec]) -> None:
        self._commands = list(commands)

    def parse(self, argv: Iterable[str]) -> tuple[str, SimpleNamespace]:
        args = list(argv)
        
        # Handle global flags
        global_opts = {}
        
        # Handle help
        if not args or args[0] in ("-h", "--help", "help"):
            raise CLIHelp()
        
        # Handle version (-v or --version)
        if "-v" in args or "--version" in args:
            # Version info - using syslog2 would require import, but this is just version display
            # Keeping print for version display as it's user-facing CLI output
            print("Legale Bot 1.0.0")
            import sys
            sys.exit(0)
        
        # Handle verbosity (-V <level>)
        verbosity = None
        if "-V" in args:
            idx = args.index("-V")
            if idx + 1 < len(args):
                verbosity = args.pop(idx + 1)
                args.pop(idx)
            else:
                raise CLIError("-V requires a verbosity level (1-7 or LOG_ERR, LOG_WARNING, LOG_INFO, LOG_DEBUG)")
        
        if verbosity:
            # Map verbosity to log level (without LOG_ prefix for CLI compatibility)
            # Convert verbosity to string if it's a number
            verbosity_str = str(verbosity).upper() if not isinstance(verbosity, str) else verbosity.upper()
            verbosity_map = {
                "1": "ALERT",
                "2": "CRIT",
                "3": "ERR",
                "4": "WARNING", 
                "5": "NOTICE",
                "6": "INFO",
                "7": "DEBUG",
                "LOG_ALERT": "ALERT",
                "LOG_CRIT": "CRIT",
                "LOG_ERR": "ERR",
                "LOG_WARNING": "WARNING",
                "LOG_NOTICE": "NOTICE",
                "LOG_INFO": "INFO",
                "LOG_DEBUG": "DEBUG",
                "ALERT": "ALERT",
                "CRIT": "CRIT",
                "ERR": "ERR",
                "WARNING": "WARNING",
                "NOTICE": "NOTICE",
                "INFO": "INFO",
                "DEBUG": "DEBUG",
            }
            log_level = verbosity_map.get(verbosity_str, verbosity_str)
            global_opts["log_level"] = log_level
        
        # Handle log-level (deprecated, use -V instead)
        log_level = None
        if "--log-level" in args:
            idx = args.index("--log-level")
            if idx + 1 < len(args):
                log_level = args.pop(idx + 1)
                args.pop(idx)
        
        if log_level:
            global_opts["log_level"] = log_level
        
        if not args:
            raise CLIHelp()
        
        stream = ArgStream(args)
        cmd_token = stream.next().lower()
        spec = self._match_command(cmd_token)
        options = spec.parser(stream)
        
        # Merge global options
        if isinstance(options, SimpleNamespace):
            for key, value in global_opts.items():
                setattr(options, key, value)
            ns = options
        else:
            options.update(global_opts)
            ns = SimpleNamespace(**options)
        
        if stream.has_next():
            raise CLIError(f"unexpected argument: {stream.peek()}")
        return spec.name, ns

    def _match_command(self, token: str) -> CommandSpec:
        for spec in self._commands:
            if matches(token, (spec.name,)):
                return spec
        raise CLIError(f"unknown command: {token}")
    
    def get_help(self, command: Optional[str] = None) -> str:
        """Generate help text for commands."""
        if command:
            # Find specific command
            for spec in self._commands:
                if matches(command.lower(), (spec.name,)):
                    if spec.help_text:
                        return spec.help_text
                    return f"Command: {spec.name}\n\nNo help available."
            return f"Unknown command: {command}"
        
        # General help
        lines = ["Legale Bot - Librarian chatbot with RAG", ""]
        lines.append("Commands:")
        for spec in self._commands:
            lines.append(f"  {spec.name}")
        lines.append("")
        lines.append("Use 'legale <command>' for command-specific help")
        lines.append("Global flags: -v (version), -V <level> (verbosity), -h (help)")
        return "\n".join(lines)
