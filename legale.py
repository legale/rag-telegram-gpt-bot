#!/usr/bin/env python3
"""
Legale Bot - Unified CLI Orchestrator

Main entry point for all Legale Bot operations.
Uses src.lib.argparse2 for parsing and help generation.
"""

import os
import sys
import subprocess
import shutil
import warnings
from pathlib import Path
from typing import Optional

from src.lib.syslog2 import *

warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')


def is_in_virtualenv() -> bool:
    return hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )


def _reexec_with_poetry() -> None:
    poetry_path = shutil.which('poetry')
    if not poetry_path:
        syslog2(LOG_ERR, "poetry is not installed")
        syslog2(LOG_NOTICE, "please install poetry first: curl -sSL https://install.python-poetry.org | python3 -")
        syslog2(LOG_NOTICE, "or visit: https://python-poetry.org/docs/#installation")
        sys.exit(1)

    cmd = ['poetry', 'run', 'python'] + sys.argv
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        syslog2(LOG_ERR, "failed to run with poetry", error=str(e))
        sys.exit(1)


if not is_in_virtualenv():
    _reexec_with_poetry()

from dotenv import load_dotenv, set_key
from src.lib.argparse2 import cmd_parse, gen_help, DotDict, parse


current_dir = Path(__file__).parent.absolute()
project_root = current_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _bool_opt(opts: DotDict, name: str) -> bool:
    return bool(opts.get(name, False))


def _parse_log_level(s: Optional[str]) -> int:
    if not s:
        return LOG_NOTICE

    if not isinstance(s, str):
        try:
            return int(s)
        except Exception:
            return LOG_WARNING

    m = {
        "LOG_ALERT": LOG_ALERT,
        "LOG_CRIT": LOG_CRIT,
        "LOG_ERR": LOG_ERR,
        "LOG_WARNING": LOG_WARNING,
        "LOG_NOTICE": LOG_NOTICE,
        "LOG_INFO": LOG_INFO,
        "LOG_DEBUG": LOG_DEBUG,
        "ALERT": LOG_ALERT,
        "CRIT": LOG_CRIT,
        "ERR": LOG_ERR,
        "WARNING": LOG_WARNING,
        "NOTICE": LOG_NOTICE,
        "INFO": LOG_INFO,
        "DEBUG": LOG_DEBUG,
    }
    return m.get(s.upper(), LOG_WARNING)


def _need_help(opts: DotDict, cmd: str) -> bool:
    return _bool_opt(opts, "h") or _bool_opt(opts, "help") or cmd == "help"


def _print_help_and_exit(prog: str, opt_table: dict, cmd_table: dict | None, rc: int = 0) -> None:
    txt = gen_help(prog, opt_table, cmd_table)
    syslog2(LOG_NOTICE, "help", help_text=txt)
    sys.exit(rc)


class ProfileManager:
    ENV_DEFAULTS = {
        "ACTIVE_PROFILE": "default",
        "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
        "MAX_CONTEXT_TOKENS": "14000",
    }

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.profiles_dir = project_root / "profiles"
        self.env_file = project_root / ".env"
        self._ensure_env_defaults()

    def _ensure_env_defaults(self) -> None:
        env_path = str(self.env_file)

        if not self.env_file.exists():
            self.env_file.touch()

        load_dotenv(self.env_file, override=False)

        updated = False
        for key, default_value in self.ENV_DEFAULTS.items():
            if os.getenv(key) is None:
                set_key(env_path, key, default_value)
                updated = True

        if updated:
            load_dotenv(self.env_file, override=True)

    def get_current_profile(self) -> str:
        if self.env_file.exists():
            load_dotenv(self.env_file)
            return os.getenv("ACTIVE_PROFILE", "default")
        return "default"

    def set_current_profile(self, profile_name: str) -> None:
        env_path = str(self.env_file)
        if not self.env_file.exists():
            self.env_file.touch()
        set_key(env_path, "ACTIVE_PROFILE", profile_name)
        syslog2(LOG_NOTICE, "active profile set", profile=profile_name)

    def get_profile_dir(self, profile_name: Optional[str] = None) -> Path:
        if profile_name is None:
            profile_name = self.get_current_profile()
        return self.profiles_dir / profile_name

    def create_profile(self, profile_name: str, set_active: bool = False) -> Path:
        profile_dir = self.get_profile_dir(profile_name)

        if profile_dir.exists():
            syslog2(LOG_WARNING, "profile already exists", profile=profile_name, path=str(profile_dir))
            return profile_dir

        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "chroma_db").mkdir(exist_ok=True)

        syslog2(
            LOG_NOTICE,
            "profile created",
            profile=profile_name,
            path=str(profile_dir),
            database=str(profile_dir / "legale_bot.db"),
            vector_store=str(profile_dir / "chroma_db"),
        )

        if set_active:
            self.set_current_profile(profile_name)

        return profile_dir

    def list_profiles(self) -> None:
        if not self.profiles_dir.exists():
            syslog2(LOG_NOTICE, "no profiles directory found, create your first profile with: legale profile create <name>")
            return

        current = self.get_current_profile()
        profiles = [p.name for p in self.profiles_dir.iterdir() if p.is_dir()]

        if not profiles:
            syslog2(LOG_NOTICE, "no profiles found, create your first profile with: legale profile create <name>")
            return

        syslog2(LOG_NOTICE, "available profiles")
        for profile in sorted(profiles):
            marker = " (active)" if profile == current else ""
            profile_dir = self.profiles_dir / profile
            db_path = profile_dir / "legale_bot.db"
            db_exists = "OK" if db_path.exists() else "MISSING"
            syslog2(LOG_NOTICE, "profile", name=profile, db_exists=db_exists, marker=marker)

        syslog2(LOG_NOTICE, "active profile", profile=current)

    def delete_profile(self, profile_name: str, force: bool = False) -> None:
        if profile_name == self.get_current_profile() and not force:
            syslog2(LOG_WARNING, "cannot delete active profile", profile=profile_name, message="set another profile first or use force flag")
            return

        profile_dir = self.get_profile_dir(profile_name)
        if not profile_dir.exists():
            syslog2(LOG_ERR, "profile does not exist", profile=profile_name)
            return

        if not force:
            response = input(f"Delete profile '{profile_name}' and all its data? [y/N]: ")
            if response.lower() != "y":
                syslog2(LOG_NOTICE, "operation cancelled")
                return

        shutil.rmtree(profile_dir)
        syslog2(LOG_NOTICE, "profile deleted", profile=profile_name)

    def get_profile_paths(self, profile_name: Optional[str] = None) -> dict:
        profile_dir = self.get_profile_dir(profile_name)
        return {
            "profile_dir": profile_dir,
            "db_path": profile_dir / "legale_bot.db",
            "db_url": f"sqlite:///{profile_dir / 'legale_bot.db'}",
            "vector_db_path": profile_dir / "chroma_db",
            "session_file": self.project_root / "telegram_session.session",
        }


def _ensure_default_profile(profile_manager: ProfileManager, cmd_name: str) -> None:
    if cmd_name == "profile":
        return
    default_profile = profile_manager.get_profile_dir("default")
    if not default_profile.exists():
        syslog2(LOG_NOTICE, "creating default profile")
        profile_manager.create_profile("default", set_active=True)


def cmd_profile(argv: list[str], profile_manager: ProfileManager) -> None:
    opt_table = {
        "h": {"desc": "Show help"},
        "help": {"desc": "Show help"},
        "profile": {"arg": True, "desc": "Override active profile", "meta": "NAME"},
        "set_active": {"arg": False, "desc": "Set active profile after create"},
        "force": {"arg": False, "desc": "Force action"},
        "name": {"arg": True, "desc": "Profile name", "meta": "NAME"},
    }
    cmd_table = {
        "list": {"desc": "List profiles"},
        "create": {"desc": "Create profile"},
        "get": {"desc": "Get active profile"},
        "set": {"desc": "Set active profile"},
        "delete": {"desc": "Delete profile"},
        "info": {"desc": "Show profile info"},
        "option": {"desc": "Manage profile options"},
    }

    opts, subcmd, args = cmd_parse(argv, opt_table)
    if _need_help(opts, subcmd):
        _print_help_and_exit("legale profile", opt_table, cmd_table, 0)

    if subcmd == "list":
        profile_manager.list_profiles()
        return

    if subcmd == "create":
        if not args:
            raise ValueError("profile name required")
        name = args[0]
        profile_manager.create_profile(name, set_active=_bool_opt(opts, "set_active"))
        return

    if subcmd == "get":
        syslog2(LOG_NOTICE, "current profile", profile=profile_manager.get_current_profile())
        return

    if subcmd == "set":
        if not args:
            raise ValueError("profile name required")
        name = args[0]
        profile_dir = profile_manager.get_profile_dir(name)
        if not profile_dir.exists():
            syslog2(LOG_ERR, "profile does not exist", profile=name)
            syslog2(LOG_NOTICE, "create it with", command=f"legale profile create {name}")
            sys.exit(1)
        profile_manager.set_current_profile(name)
        return

    if subcmd == "delete":
        if not args:
            raise ValueError("profile name required")
        name = args[0]
        profile_manager.delete_profile(name, force=_bool_opt(opts, "force"))
        return

    if subcmd == "info":
        name = args[0] if args else profile_manager.get_current_profile()
        paths = profile_manager.get_profile_paths(name)

        db_exists = "exists" if paths["db_path"].exists() else "not created"
        vec_exists = "exists" if paths["vector_db_path"].exists() else "not created"
        sess_exists = "exists" if paths["session_file"].exists() else "not created"
        syslog2(
            LOG_NOTICE,
            "profile info",
            profile=name,
            directory=str(paths["profile_dir"]),
            database=str(paths["db_path"]),
            db_status=db_exists,
            vector_db=str(paths["vector_db_path"]),
            vec_status=vec_exists,
            session=str(paths["session_file"]),
            sess_status=sess_exists,
            note="session is shared for all profiles",
        )
        return

    if subcmd == "option":
        cmd_profile_option(argv=args, profile_manager=profile_manager)
        return

    raise ValueError(f"unknown profile subcommand: {subcmd}")


def cmd_ingest(argv: list[str], profile_manager: ProfileManager) -> None:
    opt_table = {
        "h": {"desc": "Show help"},
        "help": {"desc": "Show help"},
        "profile": {"arg": True, "desc": "Override active profile", "meta": "NAME"},
        "model": {"arg": True, "desc": "Embedding model", "meta": "NAME"},
        "batch_size": {"arg": True, "desc": "Batch size", "meta": "N"},
        "batch-size": {"arg": True, "desc": "Batch size (alias)", "meta": "N"},
        "file": {"arg": True, "desc": "Input file path", "meta": "PATH"},
        "force": {"arg": False, "desc": "Force action"},
    }
    cmd_table = {
        "all": {"desc": "Run stages 0-3"},
        "stage0": {"desc": "Parse and store messages"},
        "stage1": {"desc": "Create and store chunks"},
        "stage2": {"desc": "Generate embeddings"},
        "stage3": {"desc": "Sync to vector DB"},
        "clear": {"desc": "Clear stages"},
        "info": {"desc": "Show ingest info"},
    }

    opts, subcmd, args = cmd_parse(argv, opt_table)
    if _need_help(opts, subcmd):
        _print_help_and_exit("legale ingest", opt_table, cmd_table, 0)

    from src.ingestion.pipeline import IngestionPipeline

    profile_name = opts.get("profile") or profile_manager.get_current_profile()
    paths = profile_manager.get_profile_paths(profile_name)

    paths["profile_dir"].mkdir(parents=True, exist_ok=True)
    paths["vector_db_path"].mkdir(parents=True, exist_ok=True)

    syslog2(LOG_NOTICE, "using profile", profile=profile_name, database=str(paths["db_path"]), vector_store=str(paths["vector_db_path"]))

    pipeline = IngestionPipeline(
        db_url=paths["db_url"],
        vector_db_path=str(paths["vector_db_path"]),
        profile_dir=str(paths["profile_dir"]),
    )

    if subcmd == "info":
        info_output = pipeline.get_ingest_info()
        print(info_output)
        return

    if subcmd == "all":
        file_path = opts.get("file") or (args[0] if args else None)
        if not file_path:
            raise ValueError("file path is required for ingest all")
        model = opts.get("model")
        bs = opts.get("batch_size") or opts.get("batch-size")
        batch_size = int(bs) if bs else 128
        pipeline.run_all(file_path, model=model, batch_size=batch_size)
        return

    if subcmd == "stage0":
        file_path = opts.get("file") or (args[0] if args else None)
        if not file_path:
            raise ValueError("file path is required for ingest stage0")
        syslog2(LOG_NOTICE, "running stage0: parse and store")
        pipeline.run_stage0(file_path)
        syslog2(LOG_NOTICE, "stage0 complete")
        return

    if subcmd == "stage1":
        from src.app.bootstrap import create_app
        app = create_app(
            db_url=paths["db_url"],
            vector_db_path=str(paths["vector_db_path"]),
            profile_dir=str(paths["profile_dir"]),
        )
        db = app.get_database()
        if db.count_messages() == 0:
            syslog2(LOG_ERR, "no messages found in database, run ingest stage0 first")
            sys.exit(1)
        syslog2(LOG_NOTICE, "running stage1: create and store chunks")
        pipeline.run_stage1()
        syslog2(LOG_NOTICE, "stage1 complete")
        return

    if subcmd == "stage2":
        from src.app.bootstrap import create_app
        app = create_app(
            db_url=paths["db_url"],
            vector_db_path=str(paths["vector_db_path"]),
            profile_dir=str(paths["profile_dir"]),
        )
        db = app.get_database()
        if db.count_chunks() == 0:
            syslog2(LOG_ERR, "no chunks found in database, run ingest stage1 first")
            sys.exit(1)
        model = opts.get("model")
        bs = opts.get("batch_size") or opts.get("batch-size")
        batch_size = int(bs) if bs else 128
        syslog2(LOG_NOTICE, "running stage2: generate embeddings")
        pipeline.run_stage2(model=model, batch_size=batch_size)
        syslog2(LOG_NOTICE, "stage2 complete")
        return

    if subcmd == "stage3":
        from src.app.bootstrap import create_app
        from src.storage.db import ChunkModel
        app = create_app(
            db_url=paths["db_url"],
            vector_db_path=str(paths["vector_db_path"]),
            profile_dir=str(paths["profile_dir"]),
        )
        db = app.get_database()
        session = db.get_session()
        try:
            cnt = session.query(ChunkModel).filter(ChunkModel.embedding_json.isnot(None)).count()
            if cnt == 0:
                syslog2(LOG_ERR, "no embeddings found in sqlite, run ingest stage2 first")
                sys.exit(1)
        finally:
            session.close()
        syslog2(LOG_NOTICE, "running stage3: sync chunks to vector database")
        pipeline.run_stage3()
        syslog2(LOG_NOTICE, "stage3 complete")
        return

    if subcmd == "clear":
        clear_opt_table = {
            "h": {"desc": "Show help"},
            "help": {"desc": "Show help"},
        }
        clear_cmd_table = {
            "all": {"desc": "Clear all stages"},
            "stage0": {"desc": "Clear messages"},
            "stage1": {"desc": "Clear chunks"},
            "stage2": {"desc": "Clear embeddings in sqlite"},
            "stage3": {"desc": "Clear vectors in vector db"},
        }

        c_opts, c_subcmd, c_args = cmd_parse(args, clear_opt_table)
        if _need_help(c_opts, c_subcmd):
            _print_help_and_exit("legale ingest clear", clear_opt_table, clear_cmd_table, 0)

        if c_subcmd == "all":
            syslog2(LOG_NOTICE, "clearing all stages")
            pipeline.clear_all()
            syslog2(LOG_NOTICE, "all stages cleared")
            return

        if c_subcmd == "stage0":
            syslog2(LOG_NOTICE, "clearing stage0: messages")
            deleted = pipeline.clear_stage0()
            syslog2(LOG_NOTICE, "stage0 cleared", messages_deleted=deleted)
            return

        if c_subcmd == "stage1":
            syslog2(LOG_NOTICE, "clearing stage1: chunks")
            deleted = pipeline.clear_stage1()
            syslog2(LOG_NOTICE, "stage1 cleared", chunks_deleted=deleted)
            return

        if c_subcmd == "stage2":
            syslog2(LOG_NOTICE, "clearing stage2: chunk embeddings (SQLite)")
            updated = pipeline.clear_stage2()
            syslog2(LOG_NOTICE, "stage2 cleared", embeddings_cleared=updated)
            return

        if c_subcmd == "stage3":
            syslog2(LOG_NOTICE, "clearing stage3: vector_db chunks")
            removed = pipeline.clear_stage3()
            syslog2(LOG_NOTICE, "stage3 cleared", vectors_removed=removed)
            return

        raise ValueError(f"unknown clear subcommand: {c_subcmd}")

    raise ValueError(f"unknown ingest subcommand: {subcmd}")


def cmd_telegram(argv: list[str], profile_manager: ProfileManager) -> None:
    opt_table = {
        "h": {"desc": "Show help"},
        "help": {"desc": "Show help"},
        "profile": {"arg": True, "desc": "Override active profile", "meta": "NAME"},
        "limit": {"arg": True, "desc": "Limit messages", "meta": "N"},
        "output": {"arg": True, "desc": "Output file or dir", "meta": "PATH"},
        "model": {"arg": True, "desc": "Embedding model", "meta": "NAME"},
        "batch_size": {"arg": True, "desc": "Batch size", "meta": "N"},
        "batch-size": {"arg": True, "desc": "Batch size (alias)", "meta": "N"},
    }
    cmd_table = {
        "list": {"desc": "List channels"},
        "members": {"desc": "List members"},
        "dump": {"desc": "Dump chat to json"},
        "ingest": {"desc": "Dump and ingest"},
    }

    opts, subcmd, args = cmd_parse(argv, opt_table)
    if _need_help(opts, subcmd):
        _print_help_and_exit("legale telegram", opt_table, cmd_table, 0)

    load_dotenv()
    API_ID = os.getenv("TELEGRAM_API_ID")
    API_HASH = os.getenv("TELEGRAM_API_HASH")

    if not API_ID or not API_HASH:
        syslog2(LOG_ERR, "telegram_api_id and telegram_api_hash must be set in .env file")
        sys.exit(1)

    try:
        API_ID = int(API_ID)
    except ValueError:
        syslog2(LOG_ERR, "telegram_api_id must be an integer")
        sys.exit(1)

    from src.ingestion.telegram import TelegramFetcher

    profile_name = opts.get("profile", profile_manager.get_current_profile()) 
    paths = profile_manager.get_profile_paths(profile_name)
    paths["profile_dir"].mkdir(parents=True, exist_ok=True)
    limit = int(opts.get("limit", 1000))

    session_name = str(paths["session_file"].with_suffix(""))
    syslog2(LOG_NOTICE, "using profile", profile=profile_name, session_file=str(paths["session_file"]), limit=limit, note="shared session for all profiles")

    fetcher = TelegramFetcher(API_ID, API_HASH, session_name=session_name)

    if subcmd == "list":
        fetcher.list_channels()
        return

    if subcmd == "members":
        if not args:
            raise ValueError("target required")
        fetcher.list_members(args[0])
        return

    if subcmd == "dump":
        if not args:
            raise ValueError("target required")
        target = args[0]
        limit = int(opts.get("limit",1000))
        output = opts.get("output")
        if not output:
            output = str(paths["profile_dir"])
        else:
            if not os.path.isabs(output):
                output = str(paths["profile_dir"] / output)
        fetcher.dump_chat(target, limit=limit, output_file=output)
        return

    if subcmd == "ingest":
        ingest_opt_table = {
            "h": {"desc": "Show help"},
            "help": {"desc": "Show help"},
        }
        ingest_cmd_table = {
            "all": {"desc": "Dump and ingest all stages"},
        }

        i_opts, i_subcmd, i_args = cmd_parse(args, ingest_opt_table)
        if _need_help(i_opts, i_subcmd):
            _print_help_and_exit("legale telegram ingest", ingest_opt_table, ingest_cmd_table, 0)

        if i_subcmd != "all":
            raise ValueError(f"unknown telegram ingest subcommand: {i_subcmd}")

        if not i_args:
            raise ValueError("target required")
        target = i_args[0]

        limit = int(opts.get("limit") or 1000)
        model = opts.get("model")
        bs = opts.get("batch_size") or opts.get("batch-size")
        batch_size = int(bs) if bs else 128

        with fetcher.client:
            chat = fetcher._find_chat(target)
            if not chat:
                syslog2(LOG_ERR, "chat not found", target=target)
                sys.exit(1)
            chat_id = abs(chat.id)
            output_dir = str(paths["profile_dir"])
            dump_file = os.path.join(output_dir, f"telegram_dump_{chat_id}.json")

        fetcher.dump_chat(target, limit=limit, output_file=str(paths["profile_dir"]))

        if not os.path.exists(dump_file):
            syslog2(LOG_ERR, "failed to create dump file", file=dump_file)
            sys.exit(1)

        syslog2(LOG_NOTICE, "starting ingestion", dump_file=dump_file)

        from src.ingestion.pipeline import IngestionPipeline

        paths["profile_dir"].mkdir(parents=True, exist_ok=True)
        paths["vector_db_path"].mkdir(parents=True, exist_ok=True)

        pipeline = IngestionPipeline(
            db_url=paths["db_url"],
            vector_db_path=str(paths["vector_db_path"]),
            profile_dir=str(paths["profile_dir"]),
        )
        pipeline.run_all(dump_file, model=model, batch_size=batch_size)
        return

    raise ValueError(f"unknown telegram subcommand: {subcmd}")


def cmd_chat(argv: list[str], profile_manager: ProfileManager, global_log_level: Optional[str]) -> None:
    opt_table = {
        "h": {"desc": "Show help"},
        "help": {"desc": "Show help"},
        "profile": {"arg": True, "desc": "Override active profile", "meta": "NAME"},
        "chunks": {"arg": True, "desc": "Chunks to retrieve", "meta": "N"},
        "debug_rag": {"arg": False, "desc": "Enable RAG debug"},
        "debug-rag": {"arg": False, "desc": "Enable RAG debug (alias)"},
        "retrieval_type": {"arg": True, "desc": "Retrieval type", "meta": "hybrid|fts_only|vector_only"},
        "retrieval-type": {"arg": True, "desc": "Retrieval type (alias)", "meta": "hybrid|fts_only|vector_only"},
    }

    # Use simple option parsing without subcommands for chat
    raw_opts, args = parse(argv, opt_table)
    opts = DotDict(raw_opts)

    if _need_help(opts, "chat"):
        _print_help_and_exit("legale chat", opt_table, None, 0)

    retrieval_type = opts.get("retrieval_type") or opts.get("retrieval-type") or "hybrid"
    if retrieval_type not in ["hybrid", "fts_only", "vector_only"]:
        raise ValueError("invalid retrieval_type, must be one of: hybrid, fts_only, vector_only")

    from src.bot.cli import main as cli_main

    profile_name = opts.get("profile") or profile_manager.get_current_profile()
    paths = profile_manager.get_profile_paths(profile_name)

    if not paths["db_path"].exists():
        syslog2(LOG_ERR, "no database found for profile", profile=profile_name)
        syslog2(LOG_NOTICE, "create one by ingesting data first: legale ingest all -file <dump.json>")
        sys.exit(1)

    syslog2(LOG_NOTICE, "using profile", profile=profile_name, database=str(paths["db_path"]))

    os.environ["DATABASE_URL"] = paths["db_url"]
    os.environ["VECTOR_DB_PATH"] = str(paths["vector_db_path"])
    os.environ["PROFILE_DIR"] = str(paths["profile_dir"])

    cli_args = []
    if global_log_level:
        cli_args.extend(["-V", global_log_level])

    chunks = opts.get("chunks")
    if chunks:
        cli_args.extend(["--chunks", str(int(chunks))])

    # Local debug flag for chat; global debug is handled in main()
    if _bool_opt(opts, "debug_rag") or _bool_opt(opts, "debug-rag"):
        cli_args.append("--debug-rag")

    cli_args.extend(["--retrieval-type", retrieval_type])

    # Pass through remaining args (e.g. -V 7) to the inner CLI
    if args:
        cli_args.extend(args)

    original_argv = sys.argv
    sys.argv = ["cli.py"] + cli_args
    try:
        cli_main()
    finally:
        sys.argv = original_argv


def cmd_bot(argv: list[str], profile_manager: ProfileManager, global_log_level: Optional[str]) -> None:
    opt_table = {
        "h": {"desc": "Show help"},
        "help": {"desc": "Show help"},
        "profile": {"arg": True, "desc": "Override active profile", "meta": "NAME"},
        "host": {"arg": True, "desc": "Bind host", "meta": "IP"},
        "port": {"arg": True, "desc": "Bind port", "meta": "N"},
        "url": {"arg": True, "desc": "Webhook url", "meta": "URL"},
        "token": {"arg": True, "desc": "Bot token", "meta": "TOKEN"},
        "debug_rag": {"arg": False, "desc": "Enable RAG debug"},
        "debug-rag": {"arg": False, "desc": "Enable RAG debug (alias)"},
    }
    cmd_table = {
        "register": {"desc": "Register webhook"},
        "delete": {"desc": "Delete webhook"},
        "run": {"desc": "Run server"},
        "daemon": {"desc": "Run daemon"},
    }

    opts, subcmd, args = cmd_parse(argv, opt_table)
    if _need_help(opts, subcmd):
        _print_help_and_exit("legale bot", opt_table, cmd_table, 0)

    from src.bot.tgbot import register_webhook, delete_webhook, run_server, run_daemon

    load_dotenv()

    profile_name = opts.get("profile") or profile_manager.get_current_profile()
    paths = profile_manager.get_profile_paths(profile_name)

    os.environ["DATABASE_URL"] = paths["db_url"]
    os.environ["VECTOR_DB_PATH"] = str(paths["vector_db_path"])

    syslog2(LOG_NOTICE, "using profile", profile=profile_name)

    if subcmd == "register":
        url = opts.get("url")
        if not url:
            raise ValueError("url required for bot register")
        token = opts.get("token") or os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            syslog2(LOG_ERR, "telegram_bot_token must be set in .env or passed via -token")
            sys.exit(1)
        register_webhook(url, token)
        return

    if subcmd == "delete":
        token = opts.get("token") or os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            syslog2(LOG_ERR, "telegram_bot_token must be set in .env or passed via -token")
            sys.exit(1)
        delete_webhook(token)
        return

    if subcmd == "run":
        host = opts.get("host") or "127.0.0.1"
        port = int(opts.get("port") or 8000)
        debug_rag = _bool_opt(opts, "debug_rag") or _bool_opt(opts, "debug-rag")

        syslog2(LOG_NOTICE, "database", path=str(paths["db_path"]))
        syslog2(LOG_NOTICE, "vector store", path=str(paths["vector_db_path"]))

        ll = global_log_level
        run_server(host, port, log_level=ll, debug_rag=debug_rag, args=DotDict({"log_level": ll, "debug_rag": debug_rag, "profile": profile_name}))
        return

    if subcmd == "daemon":
        host = opts.get("host") or "127.0.0.1"
        port = int(opts.get("port") or 8000)

        syslog2(LOG_NOTICE, "database", path=str(paths["db_path"]))
        syslog2(LOG_NOTICE, "vector store", path=str(paths["vector_db_path"]))

        run_daemon(host, port, args=DotDict({"profile": profile_name}))
        return

    raise ValueError(f"unknown bot subcommand: {subcmd}")


def cmd_config(argv: list[str], profile_manager: ProfileManager) -> None:
    opt_table = {
        "h": {"desc": "Show help"},
        "help": {"desc": "Show help"},
        "profile": {"arg": True, "desc": "Override active profile", "meta": "NAME"},
        "key": {"arg": True, "desc": "Config key", "meta": "KEY"},
        "value": {"arg": True, "desc": "Config value", "meta": "VALUE"},
    }
    cmd_table = {
        "get": {"desc": "Get config value"},
        "set": {"desc": "Set config value"},
    }

    opts, subcmd, args = cmd_parse(argv, opt_table)
    if _need_help(opts, subcmd):
        _print_help_and_exit("legale config", opt_table, cmd_table, 0)

    from src.bot.config import BotConfig

    profile_name = opts.get("profile") or profile_manager.get_current_profile()
    profile_dir = profile_manager.get_profile_dir(profile_name)
    if not profile_dir.exists():
        syslog2(LOG_ERR, "profile does not exist", profile=profile_name)
        sys.exit(1)

    config = BotConfig(profile_dir)

    if subcmd == "get":
        key = opts.get("key") or (args[0] if args else None)
        if not key:
            raise ValueError("key required for config get")
        if key == "system_prompt":
            prompt = config.get_system_prompt()
            if not config.system_prompt:
                prompt = f"(Default)\n{prompt}"
            syslog2(LOG_NOTICE, "system prompt", profile=profile_name, prompt=prompt)
            return
        syslog2(LOG_ERR, "unknown config key", key=key)
        return

    if subcmd == "set":
        key = opts.get("key") or (args[0] if args else None)
        val = opts.get("value") or (args[1] if len(args) > 1 else None)
        if not key or val is None:
            raise ValueError("key and value required for config set")
        if key == "system_prompt":
            config.system_prompt = val
            syslog2(LOG_NOTICE, "system prompt updated", profile=profile_name)
            return
        syslog2(LOG_ERR, "unknown config key", key=key)
        return

    raise ValueError(f"unknown config subcommand: {subcmd}")


def cmd_profile_option(argv: list[str], profile_manager: ProfileManager) -> None:
    from src.bot.config import BotConfig

    if not argv:
        raise ValueError("profile option requires: <option> <action> [value]")

    option = argv[0]
    action = argv[1] if len(argv) > 1 else None
    value = argv[2] if len(argv) > 2 else None
    if not action:
        raise ValueError("profile option requires: <option> <action> [value]")

    profile_name = profile_manager.get_current_profile()
    profile_dir = profile_manager.get_profile_dir(profile_name)
    if not profile_dir.exists():
        syslog2(LOG_ERR, "profile does not exist", profile=profile_name)
        sys.exit(1)

    config = BotConfig(profile_dir)

    available_models = {
        "openrouter": [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ],
        "local": [
            "all-MiniLM-L6-v2",
            "all-MiniLM-L12-v2",
            "all-mpnet-base-v2",
            "paraphrase-multilingual-MiniLM-L12-v2",
            "paraphrase-multilingual-mpnet-base-v2",
        ],
    }
    all_models = available_models["openrouter"] + available_models["local"]
    available_generators = ["openrouter", "openai", "local"]

    if option == "model":
        if action == "list":
            syslog2(LOG_NOTICE, "available embedding models")
            syslog2(LOG_NOTICE, "openrouter/openai models")
            for m in available_models["openrouter"]:
                marker = " (current)" if m == config.embedding_model else ""
                syslog2(LOG_NOTICE, "model", name=m, marker=marker)
            syslog2(LOG_NOTICE, "local models (sentence-transformers)")
            for m in available_models["local"]:
                marker = " (current)" if m == config.embedding_model else ""
                syslog2(LOG_NOTICE, "model", name=m, marker=marker)
            return

        if action == "get":
            syslog2(LOG_NOTICE, "current embedding model", model=config.embedding_model)
            return

        if action == "set":
            if not value:
                syslog2(LOG_ERR, "value is required for set action")
                sys.exit(1)
            if value not in all_models:
                syslog2(LOG_ERR, "unknown model", model=value, available=", ".join(all_models))
                sys.exit(1)
            config.embedding_model = value
            syslog2(LOG_NOTICE, "embedding model set", model=value)
            return

    if option == "generator":
        if action == "list":
            syslog2(LOG_NOTICE, "available embedding generators")
            for g in available_generators:
                marker = " (current)" if g == config.embedding_generator else ""
                syslog2(LOG_NOTICE, "generator", name=g, marker=marker)
            syslog2(LOG_NOTICE, "note", openrouter="Use OpenRouter/OpenAI API", local="Use local sentence-transformers (no API key required)")
            return

        if action == "get":
            syslog2(LOG_NOTICE, "current embedding generator", generator=config.embedding_generator)
            return

        if action == "set":
            if not value:
                syslog2(LOG_ERR, "value is required for set action")
                sys.exit(1)
            if value.lower() not in available_generators:
                syslog2(LOG_ERR, "unknown generator", generator=value, available=", ".join(available_generators))
                sys.exit(1)
            config.embedding_generator = value.lower()
            syslog2(LOG_NOTICE, "embedding generator set", generator=value.lower())
            return

    if option == "frequency":
        if action == "list":
            syslog2(
                LOG_NOTICE,
                "response frequency options",
                option_0="Respond only to mentions",
                option_1="Respond to every message",
                option_n="Respond to every N-th message (N > 1)",
                current=config.response_frequency,
            )
            return

        if action == "get":
            freq = config.response_frequency
            if freq == 0:
                desc = "Respond only to mentions"
            elif freq == 1:
                desc = "Respond to every message"
            else:
                desc = f"Respond to every {freq}-th message"
            syslog2(LOG_NOTICE, "current response frequency", frequency=freq, description=desc)
            return

        if action == "set":
            if not value:
                syslog2(LOG_ERR, "value is required for set action")
                sys.exit(1)
            try:
                freq_value = int(value)
            except ValueError:
                syslog2(LOG_ERR, "frequency must be an integer")
                sys.exit(1)
            if freq_value < 0:
                syslog2(LOG_ERR, "frequency must be >= 0")
                sys.exit(1)
            config.response_frequency = freq_value
            syslog2(LOG_NOTICE, "response frequency set", frequency=freq_value)
            return

    syslog2(LOG_ERR, "unknown profile option", option=option, action=action)
    sys.exit(1)


def cmd_test_embedding(argv: list[str]) -> None:
    opt_table = {
        "h": {"desc": "Show help"},
        "help": {"desc": "Show help"},
        "model": {"arg": True, "desc": "Embedding model", "meta": "NAME"},
        "text": {"arg": True, "desc": "Text", "meta": "TEXT"},
    }
    cmd_table = None

    opts, cmd, args = cmd_parse(argv, opt_table)
    if _need_help(opts, cmd):
        _print_help_and_exit("legale test-embedding", opt_table, cmd_table, 0)

    text = opts.get("text") or (cmd if cmd != "help" else None)
    if not text:
        if args:
            text = args[0]
    if not text:
        raise ValueError("text required for test-embedding")

    model = opts.get("model") or "ai-sage/Giga-Embeddings-instruct"

    from src.core.embedding import LocalEmbeddingClient
    import time

    syslog2(LOG_NOTICE, "testing embedding generation", model=model, text=text)

    try:
        syslog2(LOG_NOTICE, "loading model")
        client = LocalEmbeddingClient(model=model)
        _ = client.get_embedding("warmup")
        syslog2(LOG_NOTICE, "model loaded")

        start_time = time.time()
        emb = client.get_embedding(text)
        duration_ms = (time.time() - start_time) * 1000

        syslog2(LOG_NOTICE, "embedding generation success", duration_ms=int(duration_ms), dimensions=len(emb), first_5_values=emb[:5])
    except Exception as e:
        setup_log(LOG_NOTICE)
        syslog2(LOG_ERR, f"Error: {e}")
        sys.exit(1)


def main() -> None:
    setup_log(LOG_NOTICE)

    global_opt_table = {
        "-V": {"arg": True, "desc": "Set log level", "meta": "LEVEL"},
        "--log-level": {"arg": True, "desc": "Set log level", "meta": "LEVEL"},
        "log-level": {"arg": True, "desc": "Set log level", "meta": "LEVEL"},
        "h": {"desc": "Show help"},
        "help": {"desc": "Show help"},
        "v": {"desc": "Show version"},
        "version": {"desc": "Show version"},
    }

    global_cmd_table = {
        "test-embedding": {"desc": "Test embedding generation"},
        "profile": {"desc": "Profile management"},
        "ingest": {"desc": "Data ingestion"},
        "telegram": {"desc": "Telegram data fetching"},
        "chat": {"desc": "Interactive chat"},
        "bot": {"desc": "Telegram bot webhook"},
        "config": {"desc": "Configuration management"},
    }

    try:
        argv = sys.argv[1:]
        if not argv:
            _print_help_and_exit("legale", global_opt_table, global_cmd_table, 0)

        # Manually split global options (before command) and command + its args
        g_opts_raw: dict[str, object] = {}
        cmd: str | None = None
        cmd_args: list[str] = []

        i = 0
        while i < len(argv):
            tok = argv[i]
            if tok in global_opt_table and cmd is None:
                spec = global_opt_table[tok]
                if spec.get("arg"):
                    if i + 1 >= len(argv):
                        raise ValueError(f"missing arg for {tok}")
                    g_opts_raw[tok] = argv[i + 1]
                    i += 2
                else:
                    g_opts_raw[tok] = True
                    i += 1
                continue

            # First non-global option token is the command
            if cmd is None:
                cmd = tok
                cmd_args = argv[i + 1 :]
                break

        if cmd is None:
            raise ValueError("no command specified")

        g_opts = DotDict(g_opts_raw)

        if _need_help(g_opts, cmd):
            _print_help_and_exit("legale", global_opt_table, global_cmd_table, 0)

        if _bool_opt(g_opts, "v") or _bool_opt(g_opts, "version"):
            print("legale-bot version 1.0")
            sys.exit(0)

        global_log_level = g_opts.get("V") or g_opts.get("log-level") or g_opts.get("--log-level")
        setup_log(_parse_log_level(global_log_level))

        profile_manager = ProfileManager(project_root)
        _ensure_default_profile(profile_manager, cmd)

        if cmd == "test-embedding":
            cmd_test_embedding(cmd_args)
            return

        if cmd == "profile":
            cmd_profile(cmd_args, profile_manager)
            return

        if cmd == "ingest":
            cmd_ingest(cmd_args, profile_manager)
            return

        if cmd == "telegram":
            cmd_telegram(cmd_args, profile_manager)
            return

        if cmd == "chat":
            cmd_chat(cmd_args, profile_manager, global_log_level)
            return

        if cmd == "bot":
            cmd_bot(cmd_args, profile_manager, global_log_level)
            return

        if cmd == "config":
            cmd_config(cmd_args, profile_manager)
            return

        raise ValueError(f"Unknown command: {cmd}")

    except ValueError as e:
        syslog2(LOG_ERR, f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        syslog2(LOG_ERR, "Command execution failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
