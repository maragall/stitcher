"""
Stream a fused mosaic to an OME-TIFF, matching Squid's OME-TIFF conventions.

The stitcher always writes the canonical Zarr tree (the pyramidal OME-Zarr). This
module then builds an OME-TIFF from it for tools that expect Squid-style OME-TIFF
(Squid/software/control/core/utils_ome_tiff_writer.py): a tiled BigTIFF carrying
OME-XML with axes, channel names, and PhysicalSizeX/Y/Z in micrometers.

Mosaics are far too large to hold in RAM (a single 40k x 35k uint16 plane is ~3 GB),
so the write streams TILES straight from the Zarr array -- peak memory is one tile
plus tifffile's small buffers, independent of mosaic size. BigTIFF lifts the 4 GB
classic-TIFF ceiling.
"""

import logging

import numpy as np
import tifffile

logger = logging.getLogger(__name__)

_UM = "µm"  # OME unit string Squid uses for PhysicalSize*Unit


def _as_5d_tczyx(array):
    """Return (array-like, (T, C, Z, Y, X)). Accepts 2D (YX), 3D (CYX), or 5D (TCZYX).
    The array only needs numpy-style slicing that returns ndarrays (zarr arrays do)."""
    shape = tuple(array.shape)
    if len(shape) == 5:
        return array, shape
    if len(shape) == 3:           # (C, Y, X) -> (1, C, 1, Y, X)
        c, y, x = shape
        return array, (1, c, 1, y, x)
    if len(shape) == 2:           # (Y, X) -> (1, 1, 1, Y, X)
        y, x = shape
        return array, (1, 1, 1, y, x)
    raise ValueError(f"Unsupported array shape {shape}; expected 2D, 3D (CYX) or 5D (TCZYX)")


def _read_plane(array, ndim, t, c, z):
    """Read one (Y, X) plane from a 2D/3D/5D array without materializing the rest."""
    if ndim == 5:
        return np.asarray(array[t, c, z])
    if ndim == 3:
        return np.asarray(array[c])
    return np.asarray(array[:])      # 2D


def write_ome_tiff(array, output_path, pixel_size_um=(1.0, 1.0), z_step_um=None,
                   channel_names=None, tile=(1024, 1024), creator="tilefusion"):
    """Stream `array` (zarr-like, 2D/3D-CYX/5D-TCZYX) to a tiled BigTIFF OME-TIFF.

    pixel_size_um : (y_um, x_um) physical pixel size.
    Memory-bounded: reads one (Y, X) plane and emits it tile-by-tile; only one plane
    plus a tile are resident at a time. (One plane is unavoidable for tiled TIFF, but
    it is read once per page, not the whole stack.)
    """
    arr, (T, C, Z, Y, X) = _as_5d_tczyx(array)
    ndim = len(array.shape)
    dtype = np.dtype(array.dtype)
    th, tw = tile
    py_um, px_um = float(pixel_size_um[0]), float(pixel_size_um[1])
    if not channel_names or len(channel_names) != C:
        channel_names = [f"Channel {i}" for i in range(C)]

    # OME metadata. Axis order TZCYX matches Squid's writer (utils_ome_tiff_writer:
    # SHAPE_KEY = (size_t, size_z, size_c, size_y, size_x); stack[t, z, c, :, :]).
    meta = {
        "axes": "TZCYX",
        "Creator": creator,
        "PhysicalSizeX": px_um, "PhysicalSizeXUnit": _UM,
        "PhysicalSizeY": py_um, "PhysicalSizeYUnit": _UM,
        "Channel": {"Name": list(channel_names)},
    }
    if z_step_um is not None:
        meta["PhysicalSizeZ"] = float(z_step_um); meta["PhysicalSizeZUnit"] = _UM

    n_tiles_x = (X + tw - 1) // tw
    n_tiles_y = (Y + th - 1) // th

    def tiles():
        # Pages emitted in TZCYX order (t slowest, then z, then c) to match the OME
        # axes above; the source is read as TCZYX (the fused tensorstore's native
        # order), so we read plane [t, c, z] but emit it in (t, z, c) page order. Edge
        # tiles are zero-padded to the full tile shape.
        for t in range(T):
            for z in range(Z):
                for c in range(C):
                    plane = _read_plane(arr, ndim, t, c, z)
                    for ty in range(n_tiles_y):
                        y0 = ty * th; y1 = min(y0 + th, Y)
                        for tx in range(n_tiles_x):
                            x0 = tx * tw; x1 = min(x0 + tw, X)
                            block = plane[y0:y1, x0:x1]
                            if block.shape != (th, tw):
                                pad = np.zeros((th, tw), dtype=dtype)
                                pad[: block.shape[0], : block.shape[1]] = block
                                block = pad
                            yield block

    with tifffile.TiffWriter(output_path, bigtiff=True, ome=True) as tw_:
        tw_.write(
            tiles(),
            shape=(T, Z, C, Y, X),       # TZCYX (Squid order)
            dtype=dtype,
            tile=(th, tw),
            photometric="minisblack",
            metadata=meta,
        )
    logger.info("Wrote OME-TIFF %s (%dx%d, %d channels)", output_path, Y, X, C)
    return output_path
