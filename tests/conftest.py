"""
Test configuration and fixtures.
Handles imports that may have dependency issues.
"""

import sys
import pytest
from unittest.mock import patch
from prometheus_client import REGISTRY


def pytest_configure(config):
    """Initialize test environment."""
    # Work around macOS lzma issue with pyenv
    # This prevents import-time failures with datasets/sentence-transformers
    pass


@pytest.fixture(autouse=True)
def reset_prometheus():
    """Reset Prometheus metrics between tests to avoid collisions."""
    # Clear the registry before the test
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
    yield
    # Clear after the test
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
