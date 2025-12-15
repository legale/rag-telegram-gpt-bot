"""
Tests for argparse2 module.
"""

import pytest
from src.lib.argparse2 import (
    matches, split_args, parse, gen_help, cmd_parse, _find_unique
)


class TestMatches:
    """Tests for matches function."""
    
    def test_matches_exact(self):
        """Test exact match."""
        assert matches("help", "help") is True
        assert matches("register", "register") is True
        assert matches("del", "delete") is True  # "del" is a prefix of "delete"
    
    def test_matches_prefix(self):
        """Test prefix matching."""
        assert matches("reg", "register") is True
        assert matches("del", "delete") is True
        assert matches("he", "help") is True
    
    def test_matches_empty_prefix(self):
        """Test empty prefix matches everything."""
        assert matches("", "help") is True
        assert matches("", "") is True
    
    def test_matches_no_match(self):
        """Test no match returns False."""
        assert matches("xyz", "help") is False
        assert matches("abc", "register") is False


class TestSplitArgs:
    """Tests for split_args function."""
    
    def test_split_args_basic(self):
        """Test basic argument splitting."""
        assert split_args("a b c") == ["a", "b", "c"]
        assert split_args("  a   b  c  ") == ["a", "b", "c"]
    
    def test_split_args_empty(self):
        """Test empty string."""
        assert split_args("") == []
        assert split_args("   ") == []


class TestFindUnique:
    """Tests for _find_unique function."""
    
    def test_find_unique_exact(self):
        """Test finding exact match."""
        names = ["help", "register", "delete"]
        assert _find_unique("help", names) == "help"
        assert _find_unique("register", names) == "register"
    
    def test_find_unique_prefix(self):
        """Test finding by prefix."""
        names = ["help", "register", "delete"]
        assert _find_unique("reg", names) == "register"
        assert _find_unique("del", names) == "delete"
    
    def test_find_unique_not_found(self):
        """Test error when not found."""
        names = ["help", "register"]
        with pytest.raises(ValueError, match="unknown option"):
            _find_unique("xyz", names)
    
    def test_find_unique_ambiguous(self):
        """Test error when ambiguous."""
        names = ["register", "remove"]
        with pytest.raises(ValueError, match="ambiguous option"):
            _find_unique("r", names)


class TestParse:
    """Tests for parse function."""
    
    def test_parse_simple_flag(self):
        """Test parsing simple flag."""
        opt_table = {"verbose": {"desc": "Verbose mode"}}
        # Options are passed as plain names without dashes
        opts, args = parse(["verbose"], opt_table)
        assert opts == {"verbose": True}
        assert args == []
    
    def test_parse_option_with_arg(self):
        """Test parsing option with argument."""
        opt_table = {"port": {"arg": True, "desc": "Port number", "meta": "PORT"}}
        opts, args = parse(["port", "8080"], opt_table)
        assert opts == {"port": "8080"}
        assert args == []
    
    def test_parse_mixed(self):
        """Test parsing mixed options and arguments."""
        opt_table = {
            "verbose": {"desc": "Verbose mode"},
            "port": {"arg": True, "desc": "Port number", "meta": "PORT"}
        }
        opts, args = parse(["verbose", "port", "8080", "file.txt"], opt_table)
        assert opts == {"verbose": True, "port": "8080"}
        assert args == ["file.txt"]
    
    def test_parse_missing_arg(self):
        """Test error when argument is missing."""
        opt_table = {"port": {"arg": True, "desc": "Port number", "meta": "PORT"}}
        with pytest.raises(ValueError, match="missing arg"):
            parse(["port"], opt_table)


class TestGenHelp:
    """Tests for gen_help function."""
    
    def test_gen_help_basic(self):
        """Test basic help generation."""
        opt_table = {
            "h": {"desc": "Show help"},
            "verbose": {"desc": "Verbose mode"}
        }
        help_text = gen_help("test", opt_table)
        assert "usage: test" in help_text
        assert "options:" in help_text
        assert "h" in help_text
        assert "verbose" in help_text
    
    def test_gen_help_with_args(self):
        """Test help with options that take arguments."""
        opt_table = {
            "port": {"arg": True, "desc": "Port number", "meta": "PORT"}
        }
        help_text = gen_help("test", opt_table)
        assert "port PORT" in help_text
    
    def test_gen_help_with_commands(self):
        """Test help with commands."""
        opt_table = {"h": {"desc": "Show help"}}
        cmd_table = {
            "register": {"desc": "Register something"},
            "delete": {"desc": "Delete something"}
        }
        help_text = gen_help("test", opt_table, cmd_table)
        assert "commands:" in help_text
        assert "register" in help_text
        assert "delete" in help_text


class TestCmdParse:
    """Tests for cmd_parse function."""
    
    def test_cmd_parse_string(self):
        """Test parsing from string."""
        opt_table = {"verbose": {"desc": "Verbose mode"}}
        opts, cmd, args = cmd_parse("register file.txt", opt_table)
        assert opts == {}
        assert cmd == "register"
        assert args == ["file.txt"]
    
    def test_cmd_parse_list(self):
        """Test parsing from list."""
        opt_table = {"verbose": {"desc": "Verbose mode"}}
        opts, cmd, args = cmd_parse(["register", "file.txt"], opt_table)
        assert opts == {}
        assert cmd == "register"
        assert args == ["file.txt"]
    
    def test_cmd_parse_with_options(self):
        """Test parsing with options."""
        opt_table = {"verbose": {"desc": "Verbose mode"}}
        # Options are plain names in argv
        opts, cmd, args = cmd_parse(["verbose", "register", "file.txt"], opt_table)
        assert opts == {"verbose": True}
        assert cmd == "register"
        assert args == ["file.txt"]
    
    def test_cmd_parse_no_command(self):
        """Test parsing with no command returns help."""
        opt_table = {"verbose": {"desc": "Verbose mode"}}
        opts, cmd, args = cmd_parse(["verbose"], opt_table)
        assert opts == {"verbose": True}
        assert cmd == "help"
        assert args == []
    
    def test_cmd_parse_empty(self):
        """Test parsing empty input."""
        opt_table = {}
        opts, cmd, args = cmd_parse([], opt_table)
        assert opts == {}
        assert cmd == "help"
        assert args == []

