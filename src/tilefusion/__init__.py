"""
TileFusion - GPU/CPU-accelerated tile registration and fusion for 2D microscopy.

A Python library for stitching tiled microscopy images with support for
OME-TIFF, individual TIFF folders, and Zarr formats.

Based on the tilefusion module from opm-processing-v2:
https://github.com/QI2lab/opm-processing-v2/blob/tilefusion2D/src/opm_processing/imageprocessing/tilefusion.py

Original author: Doug Shepherd (https://github.com/dpshepherd), QI2lab, Arizona State University
"""

import os

# Pin numba's parallel backend to the always-present "workqueue" layer BEFORE numba is
# imported (transitively, via .core -> .fusion). Left on "default", numba auto-selects
# TBB/OMP/workqueue depending on what each frozen binary happened to bundle, so the same
# fusion could run a different backend on macOS/Windows/Linux (non-deterministic prange
# reduction order -> outputs not bit-identical across OSes, and fragility if a future
# build shifts the selection). workqueue is pure numba, identical everywhere. setdefault
# so an explicit operator override still wins.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

from .core import TileFusion
from .utils import USING_GPU
from .flatfield import (
    calculate_flatfield,
    apply_flatfield,
    apply_flatfield_region,
    save_flatfield,
    load_flatfield,
    HAS_BASICPY,
)

__version__ = "0.1.0"
__all__ = [
    "TileFusion",
    "USING_GPU",
    "__version__",
    "calculate_flatfield",
    "apply_flatfield",
    "apply_flatfield_region",
    "save_flatfield",
    "load_flatfield",
    "HAS_BASICPY",
]
