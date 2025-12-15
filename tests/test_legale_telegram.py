"""
Tests for telegram-related CLI behavior in legale (cmd_telegram).
We verify that argv is parsed like in legale.py and that
limit/model/batch_size are passed correctly to the underlying services.
"""

import os
import sys
import types

import pytest

import legale


class DummyClient:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class DummyFetcher:
    def __init__(self, api_id, api_hash, session_name):
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.dump_calls = []
        self.list_calls = 0
        self.members_calls = []

    @property
    def client(self):
        return DummyClient()

    def list_channels(self):
        self.list_calls += 1

    def list_members(self, target):
        self.members_calls.append(target)

    def _find_chat(self, target):
        # Minimal stub: pretend chat with fixed id exists
        return types.SimpleNamespace(id=-123456)

    def dump_chat(self, target, limit, output_file):
        self.dump_calls.append((target, int(limit), output_file))
        # If output is a directory, create a dummy dump file
        if os.path.isdir(output_file):
            path = os.path.join(output_file, "telegram_dump_123456.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("[]")


class DummyPipeline:
    def __init__(self):
        self.inits = []
        self.run_all_calls = []

    def __call__(self, db_url: str, vector_db_path: str, profile_dir: str):
        self.inits.append((db_url, vector_db_path, profile_dir))
        return self

    def run_all(self, dump_file: str, model=None, batch_size: int = 128):
        self.run_all_calls.append((dump_file, model, batch_size))


@pytest.fixture
def tmp_profile_manager(tmp_path, monkeypatch):
    # Use isolated project root so tests don't touch real files
    project_root = tmp_path

    # Ensure env for telegram credentials
    monkeypatch.setenv("TELEGRAM_API_ID", "123")
    monkeypatch.setenv("TELEGRAM_API_HASH", "abc")

    pm = legale.ProfileManager(project_root)
    pm.create_profile("default", set_active=True)
    return pm


@pytest.fixture
def patched_telegram(monkeypatch):
    dummy_fetcher = DummyFetcher(123, "abc", "session")

    # Fake module for src.ingestion.telegram that legale.cmd_telegram imports
    telegram_mod = types.SimpleNamespace(TelegramFetcher=lambda *a, **k: dummy_fetcher)
    monkeypatch.setitem(sys.modules, "src.ingestion.telegram", telegram_mod)
    return dummy_fetcher


@pytest.fixture
def patched_pipeline(monkeypatch):
    dummy_pipeline = DummyPipeline()
    pipeline_mod = types.SimpleNamespace(IngestionPipeline=dummy_pipeline)
    monkeypatch.setitem(sys.modules, "src.ingestion.pipeline", pipeline_mod)
    return dummy_pipeline


class TestTelegramDumpCLI:
    def test_dump_default_limit(self, tmp_profile_manager, patched_telegram):
        """If limit is not specified, cmd_telegram uses default 1000."""
        legale.cmd_telegram(["dump", "Chat Name"], tmp_profile_manager)
        assert patched_telegram.dump_calls
        target, limit, output = patched_telegram.dump_calls[-1]
        assert target == "Chat Name"
        assert limit == 1000

    def test_dump_with_limit_option(self, tmp_profile_manager, patched_telegram):
        """New syntax: telegram dump \"Chat\" limit 100000000."""
        legale.cmd_telegram(["dump", "Chat Name", "limit", "100000000"], tmp_profile_manager)
        assert patched_telegram.dump_calls
        target, limit, output = patched_telegram.dump_calls[-1]
        assert target == "Chat Name"
        assert limit == 100000000


class TestTelegramIngestAllCLI:
    def test_ingest_all_with_options(self, tmp_profile_manager, patched_telegram, patched_pipeline):
        """Test telegram ingest all with limit/model/batch_size options."""
        paths = tmp_profile_manager.get_profile_paths("default")
        paths["profile_dir"].mkdir(parents=True, exist_ok=True)
        paths["vector_db_path"].mkdir(parents=True, exist_ok=True)
        dump_path = paths["profile_dir"] / "telegram_dump_123456.json"
        dump_path.write_text("[]", encoding="utf-8")

        argv = [
            "ingest",
            "all",
            "Chat Name",
            "limit",
            "2000",
            "model",
            "my-model",
            "batch_size",
            "256",
        ]
        legale.cmd_telegram(argv, tmp_profile_manager)

        # Last dump_chat call should use limit=2000
        assert patched_telegram.dump_calls
        _, limit, _ = patched_telegram.dump_calls[-1]
        assert limit == 2000

        # Ingestion pipeline should be called with our dump file and options
        assert patched_pipeline.run_all_calls
        dump_used, model_used, batch_size_used = patched_pipeline.run_all_calls[-1]
        assert dump_used.endswith("telegram_dump_123456.json")
        assert model_used == "my-model"
        assert batch_size_used == 256


