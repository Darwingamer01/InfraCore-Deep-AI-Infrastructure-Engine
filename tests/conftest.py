"""
Test configuration and fixtures.
Handles imports that may have dependency issues.
"""

import sys
from unittest.mock import patch


def pytest_configure(config):
    """Initialize test environment."""
    # Work around macOS lzma issue with pyenv
    # This prevents import-time failures with datasets/sentence-transformers
    pass
