"""Pytest configuration and shared fixtures."""

import os

# The GUI is PyQt5-only, but qtpy (pulled in by the napari and pytest-qt plugins)
# resolves to PySide6 when both are installed. Loading two Qt bindings into one
# interpreter segfaults the moment a widget is constructed, which is what
# tests/test_gui_wiring.py does. Pin the binding before anything imports qtpy so
# the suite runs correctly with a bare `pytest`, not only when the caller
# remembers to export QT_API. setdefault() still respects an explicit override.
os.environ.setdefault("QT_API", "pyqt5")
os.environ.setdefault("PYTEST_QT_API", "pyqt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402


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
