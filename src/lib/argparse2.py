# argparse2.py

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