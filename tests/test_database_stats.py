
import pytest
import sqlite3
import json
from pathlib import Path
from src.bot.utils.database_stats import DatabaseStatsService

class TestDatabaseStatsService:
    @pytest.fixture
    def db_path(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        p = d / "test.db"
        # Create a valid sqlite db
        conn = sqlite3.connect(p)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE chunks (id text, text text, metadata_json text, embedding blob)")
        
        # Insert some data
        meta1 = json.dumps({"date": "2023-01-01T10:00:00"})
        meta2 = json.dumps({"date": "2023-01-05T10:00:00"})
        
        cursor.execute("INSERT INTO chunks VALUES (?, ?, ?, ?)", ("1", "text1", meta1, b""))
        cursor.execute("INSERT INTO chunks VALUES (?, ?, ?, ?)", ("2", "text2", meta2, b""))
        
        conn.commit()
        conn.close()
        return p

    @pytest.fixture
    def vector_path(self, tmp_path):
        d = tmp_path / "vector_store"
        d.mkdir()
        # Create some files
        (d / "file1.bin").write_bytes(b"0" * 1024 * 1024) # 1 MB
        (d / "subdir").mkdir()
        (d / "subdir" / "file2.bin").write_bytes(b"0" * 1024 * 1024) # 1 MB
        return d

    def test_get_chunk_count(self, db_path):
        count = DatabaseStatsService.get_chunk_count(db_path)
        assert count == 2

    def test_get_chunk_count_no_file(self, tmp_path):
        count = DatabaseStatsService.get_chunk_count(tmp_path / "nonexistent.db")
        assert count == 0

    def test_get_chunk_count_invalid_db(self, tmp_path):
        p = tmp_path / "invalid.db"
        p.write_text("not a db")
        count = DatabaseStatsService.get_chunk_count(p)
        assert count == 0

    def test_get_database_size(self, db_path):
        size = DatabaseStatsService.get_database_size(db_path)
        assert size > 0

    def test_get_database_size_no_file(self, tmp_path):
        size = DatabaseStatsService.get_database_size(tmp_path / "nonexistent.db")
        assert size == 0.0

    def test_get_date_range(self, db_path):
        start, end = DatabaseStatsService.get_date_range(db_path)
        assert start == "2023-01-01"
        assert end == "2023-01-05"

    def test_get_date_range_no_file(self, tmp_path):
        result = DatabaseStatsService.get_date_range(tmp_path / "nonexistent.db")
        assert result is None
    
    def test_get_date_range_empty_db(self, tmp_path):
        p = tmp_path / "empty.db"
        conn = sqlite3.connect(p)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE chunks (id text, text text, metadata_json text, embedding blob)")
        conn.commit()
        conn.close()
        
        result = DatabaseStatsService.get_date_range(p)
        assert result is None

    def test_get_vector_store_size(self, vector_path):
        size = DatabaseStatsService.get_vector_store_size(vector_path)
        assert size == 2.0 # 2 files of 1MB each

    def test_get_vector_store_size_no_dir(self, tmp_path):
        size = DatabaseStatsService.get_vector_store_size(tmp_path / "nonexistent")
        assert size == 0.0

    def test_check_database_health(self, db_path):
        assert DatabaseStatsService.check_database_health(db_path) is True

    def test_check_database_health_fail(self, tmp_path):
        assert DatabaseStatsService.check_database_health(tmp_path / "nonexistent.db") is False
        
        p = tmp_path / "corrupt.db"
        p.write_text("not a db")
        assert DatabaseStatsService.check_database_health(p) is False

    def test_get_database_stats(self, db_path):
        stats = DatabaseStatsService.get_database_stats(db_path)
        assert stats["exists"] is True
        assert stats["chunk_count"] == 2
        assert stats["date_range"] == ("2023-01-01", "2023-01-05")
        assert stats["healthy"] is True
        assert stats["size_mb"] > 0

    def test_get_database_stats_no_file(self, tmp_path):
        stats = DatabaseStatsService.get_database_stats(tmp_path / "nonexistent.db")
        assert stats["exists"] is False
        assert stats["chunk_count"] == 0
        assert stats["size_mb"] == 0.0

    def test_get_vector_store_stats(self, vector_path):
        stats = DatabaseStatsService.get_vector_store_stats(vector_path)
        assert stats["exists"] is True
        assert stats["size_mb"] == 2.0
