"""
Shared pytest configuration and fixtures for the test suite
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from virtual_context_mcp.config.settings import load_config


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Load test configuration"""
    config_path = Path(__file__).parent / "fixtures" / "test_config.yaml"
    return load_config(str(config_path))


@pytest.fixture
def temp_dir():
    """Create and cleanup temporary directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_story_data():
    """Provide sample story data for testing"""
    return {
        "characters": ["Elena", "Magistrate Thorne"],
        "locations": ["Millbrook", "Valorheim", "Family Vault"],
        "plot_threads": ["Shadow Wolf Threat", "Enchanted Arrows", "Magical Bloodline"],
        "sample_text": "Elena drew back her bowstring with practiced ease, her emerald eyes fixed on the distant target."
    }


# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)