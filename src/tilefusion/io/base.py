"""
Reader protocol and factory for tilefusion.io formats.

Provides a uniform ``Reader`` protocol and an ``open_reader()`` factory that
wraps the existing per-format reader functions without reimplementing any logic.
This module is an *additive* layer — it does not modify ``core.py`` or the
underlying reader functions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import tifffile

from tilefusion.io.individual_tiffs import (
    load_individual_tiffs_metadata,
    read_individual_tiffs_region,
    read_individual_tiffs_tile,
)
from tilefusion.io.ome_tiff import (
    load_ome_tiff_metadata,
    read_ome_tiff_region,
    read_ome_tiff_tile,
)
from tilefusion.io.ome_tiff_tiles import (
    load_ome_tiff_tiles_metadata,
    read_ome_tiff_tiles_region,
    read_ome_tiff_tiles_tile,
)
from tilefusion.io.zarr import (
    load_zarr_metadata,
    read_zarr_region,
    read_zarr_tile,
)

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Reader(Protocol):
    """Uniform interface for all tilefusion input formats.

    DTYPE CONVENTION (measured, not aspirational -- pinned by
    tests/test_io_readers.py::TestReaderDtypeConvention):

    * ``read_region`` -- the REGISTRATION path -- returns float32 for every
      reader, without exception. Sub-pixel cross-correlation wants the
      full-precision quotient after flat-field division, so nothing on this
      path re-quantises to integers. SquidXplorer's parity gate pins the
      resulting offsets, so this half of the contract is load-bearing.
    * ``read_tile`` -- the FUSION path -- returns float32 for every reader
      EXCEPT ``_OmeTiffTilesReader``, which returns the file's native dtype
      (uint16 in practice). That is a deliberate documented exception, not an
      accident: it is the one path on which ``flatfield.apply_flatfield``'s
      integer round-and-clip branch actually fires.

    The practical consequence, stated plainly because it used to be implicit:
    the same flat-field applied to the same pixels is rounded to integers on
    the ome_tiff/ folder format and left fractional on every other format.
    Fusion truncates to uint16 at the final write either way
    (fusion.py, ``fused_block.astype(np.uint16)``).
    """

    #: True for formats backed by many files (tiles / individual TIFFs / zarr),
    #: False for a single multi-series OME-TIFF file. Drives parallel-mode
    #: auto-detection and the thread-local handle decision in core.
    is_multi_file: bool

    def load_metadata(self) -> dict:
        """Load and return the metadata dictionary for this dataset."""
        ...

    def read_tile(
        self,
        tile_idx: int,
        z_level: int = 0,
        time_idx: int = 0,
    ) -> np.ndarray:
        """Return all channels of *tile_idx* as an (C, Y, X) array.

        float32 for every reader except ``_OmeTiffTilesReader``, which returns
        the file's native dtype (uint16). See the class docstring -- this is a
        documented exception, and ``flatfield.apply_flatfield`` branches on it.
        """
        ...

    def read_region(
        self,
        tile_idx: int,
        y_slice: slice,
        x_slice: slice,
        channel_idx: int = 0,
        z_level: int = 0,
        time_idx: int = 0,
    ) -> np.ndarray:
        """Return a spatial sub-region of one channel as a float32 array.

        Shape is (1, h, w) for every reader except the ome_tiff_tiles reader,
        which returns a 2D (h, w) array (its tiles are stored single-channel YX).
        core's region consumers tolerate both (flatfield application guards on
        ndim == 3); this is preserved behaviour, not a uniform contract.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete readers — each wraps the existing functions exactly as core.py does
# ---------------------------------------------------------------------------


class _OmeTiffTilesReader:
    """Reader for the ome_tiff/ folder-of-files format."""

    is_multi_file = True

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._meta: Optional[dict] = None

    def load_metadata(self) -> dict:
        if self._meta is None:
            self._meta = load_ome_tiff_tiles_metadata(self._path)
        return self._meta

    def read_tile(
        self,
        tile_idx: int,
        z_level: int = 0,
        time_idx: int = 0,
    ) -> np.ndarray:
        m = self.load_metadata()
        # Mirrors core._read_tile: ome_tiff_tiles branch
        return read_ome_tiff_tiles_tile(
            m["ome_tiff_folder"],
            m["tile_identifiers"],
            m["tile_file_map"],
            tile_idx,
            m["axes"],
            z_level=z_level,
            time_idx=time_idx,
        )

    def read_region(
        self,
        tile_idx: int,
        y_slice: slice,
        x_slice: slice,
        channel_idx: int = 0,
        z_level: int = 0,
        time_idx: int = 0,
    ) -> np.ndarray:
        m = self.load_metadata()
        # Mirrors core._read_tile_region: ome_tiff_tiles branch
        return read_ome_tiff_tiles_region(
            m["ome_tiff_folder"],
            m["tile_identifiers"],
            m["tile_file_map"],
            tile_idx,
            m["axes"],
            y_slice,
            x_slice,
            channel_idx,
            z_level=z_level,
            time_idx=time_idx,
        )


class _IndividualTiffsReader:
    """Reader for the individual per-channel TIFF files format."""

    is_multi_file = True

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._meta: Optional[dict] = None

    def load_metadata(self) -> dict:
        if self._meta is None:
            self._meta = load_individual_tiffs_metadata(self._path)
        return self._meta

    def read_tile(
        self,
        tile_idx: int,
        z_level: int = 0,
        time_idx: int = 0,
    ) -> np.ndarray:
        m = self.load_metadata()
        # Mirrors core._read_tile: individual_tiffs branch
        return read_individual_tiffs_tile(
            m["image_folder"],
            m["channel_names"],
            m["tile_identifiers"],
            tile_idx,
            z_level=z_level,
            time_idx=time_idx,
            time_folders=m.get("time_folders"),
        )

    def read_region(
        self,
        tile_idx: int,
        y_slice: slice,
        x_slice: slice,
        channel_idx: int = 0,
        z_level: int = 0,
        time_idx: int = 0,
    ) -> np.ndarray:
        m = self.load_metadata()
        # Mirrors core._read_tile_region: individual_tiffs branch
        # Note: core passes self.channel_to_use as channel_idx
        return read_individual_tiffs_region(
            m["image_folder"],
            m["channel_names"],
            m["tile_identifiers"],
            tile_idx,
            y_slice,
            x_slice,
            channel_idx,
            z_level=z_level,
            time_idx=time_idx,
            time_folders=m.get("time_folders"),
        )


class _OmeTiffReader:
    """Reader for a single multi-series OME-TIFF file."""

    is_multi_file = False

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._meta: Optional[dict] = None

    def load_metadata(self) -> dict:
        if self._meta is None:
            meta = load_ome_tiff_metadata(self._path)
            # Close the handle immediately, same as core.__init__ does:
            # "Close the metadata handle immediately - we use thread-local
            # handles for thread-safe concurrent reads instead of sharing."
            if "tiff_handle" in meta:
                meta.pop("tiff_handle").close()
            self._meta = meta
        return self._meta

    def read_tile(
        self,
        tile_idx: int,
        z_level: int = 0,
        time_idx: int = 0,
    ) -> np.ndarray:
        self.load_metadata()
        # Mirrors core._read_tile: else branch (single OME-TIFF)
        # core uses a thread-local handle; here we open a fresh handle per
        # call (safe for single-threaded use; the Protocol doc says it is the
        # caller's responsibility for thread safety beyond what core provides).
        return read_ome_tiff_tile(self._path, tile_idx, tiff_handle=None)

    def read_region(
        self,
        tile_idx: int,
        y_slice: slice,
        x_slice: slice,
        channel_idx: int = 0,
        z_level: int = 0,
        time_idx: int = 0,
    ) -> np.ndarray:
        self.load_metadata()
        # Mirrors core._read_tile_region: else branch (single OME-TIFF).
        # Now forwards channel_idx so this returns (1, h, w) like all other readers.
        return read_ome_tiff_region(
            self._path, tile_idx, y_slice, x_slice, channel_idx=channel_idx, tiff_handle=None
        )


class _ZarrReader:
    """Reader for Zarr v3 stores with per_index_metadata."""

    is_multi_file = True

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._meta: Optional[dict] = None

    def load_metadata(self) -> dict:
        if self._meta is None:
            self._meta = load_zarr_metadata(self._path)
        return self._meta

    def read_tile(
        self,
        tile_idx: int,
        z_level: int = 0,
        time_idx: int = 0,
    ) -> np.ndarray:
        m = self.load_metadata()
        # Mirrors core._read_tile: zarr branch
        return read_zarr_tile(
            m["tensorstore"],
            tile_idx,
            m.get("is_3d", False),
        )

    def read_region(
        self,
        tile_idx: int,
        y_slice: slice,
        x_slice: slice,
        channel_idx: int = 0,
        z_level: int = 0,
        time_idx: int = 0,
    ) -> np.ndarray:
        m = self.load_metadata()
        # Mirrors core._read_tile_region: zarr branch
        # core passes: zarr_ts, tile_idx, y_slice, x_slice, self.channel_to_use, is_3d
        return read_zarr_region(
            m["tensorstore"],
            tile_idx,
            y_slice,
            x_slice,
            channel_idx,
            m.get("is_3d", False),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def open_reader(path: Path) -> Reader:
    """
    Detect the format at *path* and return the matching concrete reader.

    Detection order mirrors ``TileFusion.__init__`` (lines 132-163 of core.py):

    1. Directory with ``ome_tiff/`` subfolder containing ``*.ome.tiff`` files
       → ``_OmeTiffTilesReader``
    2. Directory with ``zarr.json`` whose attributes include
       ``per_index_metadata`` → ``_ZarrReader``
    3. Any other directory → ``_IndividualTiffsReader``
    4. Non-directory (file path) → ``_OmeTiffReader``

    The reader is returned *before* metadata is loaded; call
    ``reader.load_metadata()`` to trigger I/O, exactly as core does
    right after detection.
    """
    path = Path(path)

    if path.is_dir():
        # 1. OME-TIFF tiles: ome_tiff/ subfolder with at least one .ome.tiff
        ome_tiff_folder = path / "ome_tiff"
        if ome_tiff_folder.exists() and list(ome_tiff_folder.glob("*.ome.tiff"))[:1]:
            return _OmeTiffTilesReader(path)

        # 2. Zarr: zarr.json exists and attributes contain per_index_metadata
        zarr_json = path / "zarr.json"
        if zarr_json.exists():
            with open(zarr_json) as f:
                meta = json.load(f)
            if "attributes" in meta and "per_index_metadata" in meta.get("attributes", {}):
                return _ZarrReader(path)

        # 3. Everything else in a directory → individual TIFFs
        return _IndividualTiffsReader(path)

    # 4. Not a directory → single OME-TIFF file
    return _OmeTiffReader(path)
