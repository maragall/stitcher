"""
OME-TIFF format reader.

Reads tiled OME-TIFF files with stage position metadata.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import xml.etree.ElementTree as ET

import numpy as np
import tifffile


def load_ome_tiff_metadata(tiff_path: Path) -> Dict[str, Any]:
    """
    Load metadata from OME-TIFF file.

    Parameters
    ----------
    tiff_path : Path
        Path to the OME-TIFF file.

    Returns
    -------
    metadata : dict
        Dictionary containing:
        - n_tiles: int
        - n_series: int
        - shape: (Y, X)
        - channels: int
        - pixel_size: (py, px)
        - tile_positions: list of (y, x) tuples
    """
    with tifffile.TiffFile(tiff_path) as tif:
        if not tif.ome_metadata:
            raise ValueError("TIFF file does not contain OME metadata")

        root = ET.fromstring(tif.ome_metadata)
        ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
        images = root.findall(".//ome:Image", ns)

        n_tiles = len(images)
        n_series = len(tif.series)

        first_series = tif.series[0]
        Y, X = first_series.shape[-2:]
        channels = 1
        time_dim = 1
        position_dim = n_tiles

        first_pixels = images[0].find("ome:Pixels", ns)
        px_x = float(first_pixels.get("PhysicalSizeX", 1.0))
        px_y = float(first_pixels.get("PhysicalSizeY", 1.0))
        pixel_size = (px_y, px_x)

        tile_positions = []
        for img in images:
            pixels = img.find("ome:Pixels", ns)
            planes = pixels.findall("ome:Plane", ns)
            if planes:
                p = planes[0]
                x = float(p.get("PositionX", 0))
                y = float(p.get("PositionY", 0))
                tile_positions.append((y, x))
            else:
                tile_positions.append((0.0, 0.0))

    return {
        "n_tiles": n_tiles,
        "n_series": n_series,
        "shape": (Y, X),
        "channels": channels,
        "time_dim": time_dim,
        "position_dim": position_dim,
        "pixel_size": pixel_size,
        "tile_positions": tile_positions,
    }


def read_ome_tiff_tile(tiff_path: Path, tile_idx: int) -> np.ndarray:
    """
    Read a single tile from OME-TIFF (all channels).

    Parameters
    ----------
    tiff_path : Path
        Path to the OME-TIFF file.
    tile_idx : int
        Index of the tile to read.

    Returns
    -------
    arr : ndarray of shape (C, Y, X)
        Tile data as float32.
    """
    with tifffile.TiffFile(tiff_path) as tif:
        arr = tif.series[tile_idx].asarray()
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    # Flip along Y axis to correct orientation
    arr = np.flip(arr, axis=-2)
    return arr.astype(np.float32)


def read_ome_tiff_region(
    tiff_path: Path,
    tile_idx: int,
    y_slice: slice,
    x_slice: slice,
) -> np.ndarray:
    """
    Read a region of a tile from OME-TIFF.

    Parameters
    ----------
    tiff_path : Path
        Path to the OME-TIFF file.
    tile_idx : int
        Index of the tile.
    y_slice, x_slice : slice
        Region to read.

    Returns
    -------
    arr : ndarray of shape (C, h, w)
        Tile region as float32.
    """
    with tifffile.TiffFile(tiff_path) as tif:
        arr = tif.series[tile_idx].asarray()
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    # Flip along Y axis to correct orientation
    arr = np.flip(arr, axis=-2)
    return arr[:, y_slice, x_slice].astype(np.float32)
