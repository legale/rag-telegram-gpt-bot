import pytest
from src.ingestion.parser import ChatParser, ChatMessage

def test_parser_initialization():
    parser = ChatParser()
    assert parser is not None

def test_parse_file_simple(tmp_path):
    # Create a dummy file
    d = tmp_path / "subdir"
    d.mkdir()
    p = d / "test_chat.txt"
    p.write_text("Hello world\nAnother line")
    
    parser = ChatParser()
    messages = parser.parse_file(str(p))
    
    assert len(messages) == 2
    assert messages[0].content == "Hello world"
    assert messages[1].content == "Another line"
    assert isinstance(messages[0], ChatMessage)
