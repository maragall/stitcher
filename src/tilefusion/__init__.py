"""
TileFusion - GPU/CPU-accelerated tile registration and fusion for 2D microscopy.

A Python library for stitching tiled microscopy images with support for
OME-TIFF, individual TIFF folders, and Zarr formats.

Based on the tilefusion module from opm-processing-v2:
https://github.com/QI2lab/opm-processing-v2/blob/tilefusion2D/src/opm_processing/imageprocessing/tilefusion.py

Original author: Doug Shepherd (https://github.com/dpshepherd), QI2lab, Arizona State University
"""

from .core import TileFusion
from .utils import USING_GPU

__version__ = "0.1.0"
__all__ = ["TileFusion", "USING_GPU", "__version__"]
