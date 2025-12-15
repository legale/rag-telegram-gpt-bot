# argparse2.py

from typing import Any

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


def parse(argv: Any, opt_table: dict) -> tuple[dict, list[str]]:
    """
    Parse argv according to opt_table.

    Args:
        argv: list of arguments or string
        opt_table: dictionary of options

    Returns:
        tuple of options if matched by opt_table and arguments list
        if not matched by opt_table, returns empty options and argv as arguments
        if argv is None, returns empty options and empty arguments
    """
    if argv is None:
        argv = []
    elif isinstance(argv, str):
        argv = split_args(argv)
    else:
        argv = list(argv)

    i = 0
    opts = {}
    args = []

    while i < len(argv):
        tok = argv[i]

        if tok in opt_table:
            spec = opt_table[tok]
            if spec.get("arg"):
                if i + 1 >= len(argv):
                    raise ValueError(f"missing arg for {tok}")
                opts[tok] = argv[i + 1]
                i += 2
            else:
                opts[tok] = True
                i += 1
            continue

        args.append(tok)
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


def cmd_parse(text_or_argv, opt_table: dict, argv_off: int = 0) -> tuple[dict, str, list[str]]:
    if isinstance(text_or_argv, str):
        argv = split_args(text_or_argv)
    else:
        argv = list(text_or_argv)

    if argv_off:
        if argv_off < 0 or argv_off > len(argv):
            raise ValueError("argv_off out of range")
        argv = argv[argv_off:]

    opts, args = parse(argv, opt_table)
    print("cmd_parse_after", opts, argv)
    opts = DotDict(opts)

    if not args:
        return opts, "help", []

    return opts, args[0], args[1:]


class DotDict(dict):
    def __init__(self, initial=None, **kwargs):
        super().__init__()
        data = {}
        if initial:
            data.update(initial)
        data.update(kwargs)
        for key, value in data.items():
            self[key] = self._to_dotdict(value)

    def _to_dotdict(self, value):
        if isinstance(value, dict):
            return DotDict(value)
        return value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)

    def get(self, key, default=None):
        return super().get(key, default)

    def set(self, key, value, default=None):
        if key not in self and default is not None:
            self[key] = default
        self[key] = value
        return self[key]
