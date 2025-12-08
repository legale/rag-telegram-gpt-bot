"""
Pytest configuration and fixtures.
"""
import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture(scope="session")
def tmp_path_factory():
    """Override tmp_path_factory to use /tmp directory."""
    # Create a base temp directory in /tmp for this test session
    base_tmp = Path("/tmp") / "legale_bot_tests"
    base_tmp.mkdir(exist_ok=True)
    
    # Cleanup function
    def cleanup():
        import shutil
        if base_tmp.exists():
            shutil.rmtree(base_tmp, ignore_errors=True)
    
    yield base_tmp
    cleanup()


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Override tmp_path fixture to use /tmp directory."""
    import tempfile
    import shutil
    
    # Create a unique temporary directory in /tmp
    temp_dir = tempfile.mkdtemp(prefix="pytest_", dir="/tmp")
    path = Path(temp_dir)
    
    yield path
    
    # Cleanup
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)

