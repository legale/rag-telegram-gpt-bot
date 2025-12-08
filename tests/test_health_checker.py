"""Tests for health checker utilities."""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.bot.utils.health_checker import HealthChecker

def test_check_database_exists(tmp_path):
    db_path = tmp_path / "test.db"
    db_path.touch()
    name, status = HealthChecker.check_database(db_path)
    assert name == "База данных"
    assert status == "OK"

def test_check_database_not_exists(tmp_path):
    db_path = tmp_path / "nonexistent.db"
    name, status = HealthChecker.check_database(db_path)
    assert status == "Не создана"

def test_check_vector_store(tmp_path):
    vs_path = tmp_path / "vectors"
    vs_path.mkdir()
    name, status = HealthChecker.check_vector_store(vs_path)
    assert status == "OK"
    
    vs_path = tmp_path / "nonexistent"
    name, status = HealthChecker.check_vector_store(vs_path)
    assert status == "Не создано"

@patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test'})
def test_check_llm_api_key():
    name, status = HealthChecker.check_llm_api_key()
    assert status == "Установлен"

@patch.dict('os.environ', {}, clear=True)
def test_check_llm_api_key_missing():
    name, status = HealthChecker.check_llm_api_key()
    assert status == "Не установлен"

@patch.dict('os.environ', {'OPENAI_API_KEY': 'test'})
def test_check_embedding_api_key():
    name, status = HealthChecker.check_embedding_api_key()
    assert status == "Установлен"

def test_check_memory():
    name, status = HealthChecker.check_memory()
    assert name == "Память"

def test_check_disk(tmp_path):
    name, status = HealthChecker.check_disk(tmp_path)
    assert name == "Диск"

def test_run_all_checks(tmp_path):
    db_path = tmp_path / "db"
    vs_path = tmp_path / "vectors"
    vs_path.mkdir()
    checks = HealthChecker.run_all_checks(db_path, vs_path, tmp_path)
    assert len(checks) == 6

