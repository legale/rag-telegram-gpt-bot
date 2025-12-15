"""
Tests for telegram-related CLI parsing in legale.
"""

import pytest

import legale


class TestParseTelegramDump:
    """Tests for parse_telegram_dump function."""

    def test_parse_dump_default_limit(self):
        """If limit is not specified, default to 1000."""
        opts = {}
        args = ["Chat Name"]
        result = legale.parse_telegram_dump(opts, args)
        assert result["target"] == "Chat Name"
        assert result["limit"] == 1000
        assert result["output"] is None

    def test_parse_dump_with_limit_kv(self):
        """New syntax: telegram dump \"Chat\" limit 100000000."""
        opts = {}
        args = ["Chat Name", "limit", "100000000"]
        result = legale.parse_telegram_dump(opts, args)
        assert result["target"] == "Chat Name"
        assert result["limit"] == 100000000

    def test_parse_dump_with_output_and_profile(self, tmp_path):
        """Support output and profile as key-value args."""
        opts = {}
        out_file = tmp_path / "dump.json"
        args = ["Chat Name", "limit", "5000", "output", str(out_file), "profile", "mybot"]
        result = legale.parse_telegram_dump(opts, args)
        assert result["target"] == "Chat Name"
        assert result["limit"] == 5000
        assert result["output"] == str(out_file)
        assert result["profile"] == "mybot"

    def test_parse_dump_unknown_arg_raises(self):
        """Unknown trailing tokens should raise ValueError."""
        opts = {}
        args = ["Chat Name", "unknown", "value"]
        with pytest.raises(ValueError):
            legale.parse_telegram_dump(opts, args)


class TestParseTelegramIngestAll:
    """Tests for parse_telegram_ingest_all function."""

    def test_parse_ingest_all_defaults(self):
        """If no kv-args provided, use defaults."""
        opts = {}
        args = ["Chat Name"]
        result = legale.parse_telegram_ingest_all(opts, args)
        assert result["target"] == "Chat Name"
        assert result["limit"] == 1000
        assert result["batch_size"] == 128
        assert result["model"] is None

    def test_parse_ingest_all_with_kv_args(self):
        """New syntax: telegram ingest all \"Chat\" limit 2000 model my-model batch_size 256."""
        opts = {}
        args = ["Chat Name", "limit", "2000", "model", "my-model", "batch_size", "256", "profile", "mybot"]
        result = legale.parse_telegram_ingest_all(opts, args)
        assert result["target"] == "Chat Name"
        assert result["limit"] == 2000
        assert result["model"] == "my-model"
        assert result["batch_size"] == 256
        assert result["profile"] == "mybot"

    def test_parse_ingest_all_unknown_arg_raises(self):
        """Unknown trailing tokens should raise ValueError."""
        opts = {}
        args = ["Chat Name", "foo", "bar"]
        with pytest.raises(ValueError):
            legale.parse_telegram_ingest_all(opts, args)


