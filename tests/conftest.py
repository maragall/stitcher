"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_tile(rng):
    """Generate a sample tile image."""
    return rng.random((100, 100), dtype=np.float32) * 65535


@pytest.fixture
def sample_multichannel_tile(rng):
    """Generate a sample multi-channel tile."""
    return rng.random((3, 100, 100), dtype=np.float32) * 65535
