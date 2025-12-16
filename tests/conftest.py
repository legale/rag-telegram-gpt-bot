"""
Pytest configuration and fixtures.
"""
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def tmp_path_factory():
    """
    Override tmp_path_factory to use a workspace-local directory.

    This repo runs in sandboxed environments where writing to /tmp may be restricted.
    """
    base_tmp = Path(__file__).resolve().parent.parent / ".pytest_tmp"
    base_tmp.mkdir(parents=True, exist_ok=True)
    
    # Cleanup function
    def cleanup():
        import shutil
        if base_tmp.exists():
            shutil.rmtree(base_tmp, ignore_errors=True)
    
    yield base_tmp
    cleanup()


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Override tmp_path fixture to use workspace-local temp directory."""
    import tempfile
    import shutil
    
    # Create a unique temporary directory under the base temp directory
    temp_dir = tempfile.mkdtemp(prefix="pytest_", dir=str(tmp_path_factory))
    path = Path(temp_dir)
    
    yield path
    
    # Cleanup
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
