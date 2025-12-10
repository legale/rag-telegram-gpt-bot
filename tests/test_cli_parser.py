"""
Tests for CLI parser module.
"""

import pytest
from src.core.cli_parser import (
    ArgStream, CommandParser, CommandSpec, CLIError, CLIHelp,
    parse_option, parse_flag, parse_int_option, parse_float_option,
    parse_choice_option, matches
)


class TestMatches:
    """Tests for matches function."""
    
    def test_matches_exact(self):
        """Test exact match."""
        assert matches("help", ["help"]) is True
        assert matches("register", ["register", "delete"]) is True
    
    def test_matches_prefix(self):
        """Test prefix matching."""
        assert matches("reg", ["register"]) is True
        assert matches("del", ["delete", "daemon"]) is True
    
    def test_matches_case_insensitive(self):
        """Test case insensitive matching."""
        assert matches("HELP", ["help"]) is True
        assert matches("Register", ["register"]) is True
    
    def test_matches_empty_token(self):
        """Test empty token returns False."""
        assert matches("", ["help"]) is False
        assert matches("", []) is False
    
    def test_matches_no_match(self):
        """Test no match returns False."""
        assert matches("xyz", ["help", "register"]) is False


class TestArgStream:
    """Tests for ArgStream class."""
    
    def test_basic_operations(self):
        """Test basic stream operations."""
        stream = ArgStream(["a", "b", "c"])
        
        assert stream.has_next() is True
        assert stream.peek() == "a"
        assert stream.next() == "a"
        assert stream.next() == "b"
        assert stream.peek() == "c"
        assert stream.next() == "c"
        assert stream.has_next() is False
    
    def test_expect(self):
        """Test expect method."""
        stream = ArgStream(["test"])
        assert stream.expect("test value") == "test"
        
        stream = ArgStream([])
        with pytest.raises(CLIError, match="expected"):
            stream.expect("value")
    
    def test_find_and_remove(self):
        """Test find_and_remove method."""
        stream = ArgStream(["a", "b", "c", "d"])
        stream.next()  # Move to position 1
        
        assert stream.find_and_remove("c") is True
        assert stream.next() == "b"
        assert stream.next() == "d"
        
        stream = ArgStream(["a", "b"])
        assert stream.find_and_remove("x") is False
    
    def test_find_and_remove_next(self):
        """Test find_and_remove_next method."""
        stream = ArgStream(["a", "url", "http://test.com", "b"])
        
        value = stream.find_and_remove_next("url")
        assert value == "http://test.com"
        
        # Remaining should be ["a", "b"]
        assert stream.next() == "a"
        assert stream.next() == "b"
    
    def test_reset(self):
        """Test reset method."""
        stream = ArgStream(["a", "b", "c"])
        stream.next()
        stream.next()
        
        stream.reset()
        assert stream.next() == "a"


class TestParseFunctions:
    """Tests for parse utility functions."""
    
    def test_parse_flag(self):
        """Test parse_flag function."""
        stream = ArgStream(["--verbose", "command"])
        assert parse_flag(stream, "--verbose") is True
        assert stream.next() == "command"
        
        stream = ArgStream(["command"])
        assert parse_flag(stream, "--verbose") is False
    
    def test_parse_option(self):
        """Test parse_option function."""
        stream = ArgStream(["url", "http://test.com", "other"])
        value = parse_option(stream, "url")
        assert value == "http://test.com"
        assert stream.next() == "other"
        
        stream = ArgStream(["other"])
        value = parse_option(stream, "url")
        assert value is None
    
    def test_parse_int_option(self):
        """Test parse_int_option function."""
        stream = ArgStream(["port", "8080", "other"])
        value = parse_int_option(stream, "port", default=8000)
        assert value == 8080
        
        stream = ArgStream(["other"])
        value = parse_int_option(stream, "port", default=8000)
        assert value == 8000
        
        stream = ArgStream(["port", "invalid"])
        with pytest.raises(CLIError, match="invalid integer"):
            parse_int_option(stream, "port")
    
    def test_parse_float_option(self):
        """Test parse_float_option function."""
        stream = ArgStream(["rate", "0.5", "other"])
        value = parse_float_option(stream, "rate", default=1.0)
        assert value == 0.5
        
        stream = ArgStream(["rate", "invalid"])
        with pytest.raises(CLIError, match="invalid float"):
            parse_float_option(stream, "rate")
    
    def test_parse_choice_option(self):
        """Test parse_choice_option function."""
        stream = ArgStream(["mode", "debug", "other"])
        value = parse_choice_option(stream, "mode", ["debug", "info", "warn"], default="info")
        assert value == "debug"
        
        stream = ArgStream(["mode", "INFO", "other"])  # Case insensitive
        value = parse_choice_option(stream, "mode", ["debug", "info", "warn"])
        assert value == "info"
        
        stream = ArgStream(["mode", "invalid"])
        with pytest.raises(CLIError, match="invalid choice"):
            parse_choice_option(stream, "mode", ["debug", "info"])


class TestCommandParser:
    """Tests for CommandParser class."""
    
    def test_parse_basic_command(self):
        """Test parsing basic command."""
        def parse_test(stream):
            return {"value": stream.next()}
        
        commands = [CommandSpec("test", parse_test)]
        parser = CommandParser(commands)
        
        cmd_name, args = parser.parse(["test", "value"])
        assert cmd_name == "test"
        assert args.value == "value"
    
    def test_parse_with_options(self):
        """Test parsing command with options."""
        def parse_test(stream):
            url = parse_option(stream, "url")
            verbose = parse_flag(stream, "--verbose")
            return {"url": url, "verbose": verbose}
        
        commands = [CommandSpec("register", parse_test)]
        parser = CommandParser(commands)
        
        cmd_name, args = parser.parse(["register", "url", "http://test.com", "--verbose"])
        assert cmd_name == "register"
        assert args.url == "http://test.com"
        assert args.verbose is True
    
    def test_parse_help(self):
        """Test help flag."""
        commands = [CommandSpec("test", lambda s: {})]
        parser = CommandParser(commands)
        
        with pytest.raises(CLIHelp):
            parser.parse(["-h"])
        
        with pytest.raises(CLIHelp):
            parser.parse(["help"])
        
        with pytest.raises(CLIHelp):
            parser.parse([])
    
    def test_parse_unknown_command(self):
        """Test unknown command raises error."""
        commands = [CommandSpec("known", lambda s: {})]
        parser = CommandParser(commands)
        
        with pytest.raises(CLIError):
            parser.parse(["unknown"])
    
    def test_parse_version(self):
        """Test version flag."""
        commands = [CommandSpec("test", lambda s: {})]
        parser = CommandParser(commands)
        
        # Version flag causes sys.exit, which pytest catches
        # We just verify it doesn't raise CLIError
        import sys
        with pytest.raises(SystemExit):
            parser.parse(["-v"])
    
    def test_parse_log_level_option(self):
        """Test -V log level option."""
        def parse_test(stream):
            return {}
        
        commands = [CommandSpec("test", parse_test)]
        parser = CommandParser(commands)
        
        cmd_name, args = parser.parse(["test", "-V", "DEBUG"])
        assert cmd_name == "test"
        assert hasattr(args, 'log_level')
    
    def test_parse_log_level_option_missing_value(self):
        """Test -V without value raises error."""
        commands = [CommandSpec("test", lambda s: {})]
        parser = CommandParser(commands)
        
        with pytest.raises(CLIError, match="requires a log level"):
            parser.parse(["test", "-V"])

