"""
Tests for tilefusion.io.base — Reader protocol + open_reader() factory.

Verifies:
  - open_reader() returns the correct concrete type for each format.
  - load_metadata() output matches the underlying load_*_metadata() functions
    on all shared keys.
  - read_region() output matches read_*_region() byte-for-byte.

Formats covered:
  1. ome_tiff_tiles — committed tests/fixtures/synth_4fov
  2. individual_tiffs — built from scratch with rng seed=42 (same as test_io_readers.py)
  3. zarr — zarr3 store with per_index_metadata (same structure as test_io_readers.py)
  4. single OME-TIFF — multi-series OME-TIFF (same structure as test_io_readers.py)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from tilefusion.io.base import (
    Reader,
    _IndividualTiffsReader,
    _OmeTiffReader,
    _OmeTiffTilesReader,
    _ZarrReader,
    open_reader,
)
from tilefusion.io.individual_tiffs import (
    load_individual_tiffs_metadata,
    read_individual_tiffs_region,
)
from tilefusion.io.ome_tiff import load_ome_tiff_metadata, read_ome_tiff_region
from tilefusion.io.ome_tiff_tiles import (
    load_ome_tiff_tiles_metadata,
    read_ome_tiff_tiles_region,
)
from tilefusion.io.zarr import load_zarr_metadata, read_zarr_region


# ---------------------------------------------------------------------------
# Committed fixture
# ---------------------------------------------------------------------------

SYNTH_4FOV = Path(__file__).parent / "fixtures" / "synth_4fov"


# ---------------------------------------------------------------------------
# Shared per-module fixtures (mirrors test_io_readers.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def individual_tiffs_folder(tmp_path_factory):
    """
    Build a minimal individual-TIFFs dataset:
      tmp/
        0/
          coordinates.csv
          manual_0_0_ch1.tiff  ..  manual_3_0_ch1.tiff
    """
    tmp = tmp_path_factory.mktemp("ind_tiffs_factory")
    img_folder = tmp / "0"
    img_folder.mkdir()

    coords = pd.DataFrame({
        "fov": [0, 1, 2, 3],
        "x (mm)": [0.0, 1.0, 0.0, 1.0],
        "y (mm)": [0.0, 0.0, 1.0, 1.0],
    })
    coords.to_csv(img_folder / "coordinates.csv", index=False)

    rng = np.random.default_rng(42)
    img = rng.integers(0, 65535, (100, 100), dtype=np.uint16)
    for fov in range(4):
        tifffile.imwrite(img_folder / f"manual_{fov}_0_ch1.tiff", img)

    return tmp


@pytest.fixture(scope="module")
def zarr_store_path(tmp_path_factory):
    """
    Build a minimal zarr3 dataset — same structure as test_io_readers.py.
    Shape (T=1, P=2, C=1, Y=64, X=64), is_3d=False.
    """
    import tensorstore as ts

    tmp = tmp_path_factory.mktemp("zarr_factory")
    zarr_path = tmp / "test.zarr"
    zarr_path.mkdir()

    shape = [1, 2, 1, 64, 64]
    chunk_shape = [1, 1, 1, 64, 64]

    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(zarr_path)},
        "metadata": {
            "shape": shape,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunk_shape}},
            "chunk_key_encoding": {"name": "default"},
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "data_type": "uint16",
        },
    }

    store = ts.open(spec, create=True, open=True).result()

    rng = np.random.default_rng(13)
    data = rng.integers(0, 65535, shape, dtype=np.uint16)
    store[...].write(data).result()

    with open(zarr_path / "zarr.json") as f:
        zarr_json = json.load(f)

    zarr_json["attributes"] = {
        "per_index_metadata": {
            "0": {
                "0": {"0": {"stage_position": [0.0, 7.8, 7.8]}},
                "1": {"0": {"stage_position": [0.0, 7.8, 327.6]}},
            }
        },
        "deskewed_voxel_size_um": [1.0, 0.5, 0.5],
        "channels": ["ch0"],
    }

    with open(zarr_path / "zarr.json", "w") as f:
        json.dump(zarr_json, f)

    return zarr_path


@pytest.fixture(scope="module")
def single_ome_tiff_path(tmp_path_factory):
    """Build a minimal 4-tile single OME-TIFF."""
    tmp = tmp_path_factory.mktemp("ome_tiff_factory")
    path = tmp / "test.ome.tiff"

    rng = np.random.default_rng(7)
    tiles = [rng.integers(0, 65535, (100, 100), dtype=np.uint16) for _ in range(4)]

    with tifffile.TiffWriter(path, ome=True) as tif:
        for i, tile in enumerate(tiles):
            tif.write(
                tile,
                metadata={
                    "PhysicalSizeX": 0.5,
                    "PhysicalSizeY": 0.5,
                    "PhysicalSizeXUnit": "um",
                    "PhysicalSizeYUnit": "um",
                    "Plane": [
                        {
                            "PositionX": float((i % 2) * 50),
                            "PositionY": float((i // 2) * 50),
                        }
                    ],
                },
            )

    return path


# ---------------------------------------------------------------------------
# 1. ome_tiff_tiles format — synth_4fov committed fixture
# ---------------------------------------------------------------------------


class TestOmeTiffTilesFactory:
    """Factory picks _OmeTiffTilesReader; metadata + region match raw functions."""

    @pytest.fixture(scope="class")
    def reader(self):
        return open_reader(SYNTH_4FOV)

    # -- type detection

    def test_reader_type(self, reader):
        assert isinstance(reader, _OmeTiffTilesReader)

    def test_protocol_compliance(self, reader):
        assert isinstance(reader, Reader)

    # -- metadata matches underlying function on shared keys

    @pytest.fixture(scope="class")
    def factory_meta(self, reader):
        return reader.load_metadata()

    @pytest.fixture(scope="class")
    def raw_meta(self):
        return load_ome_tiff_tiles_metadata(SYNTH_4FOV)

    _COMMON_KEYS = {
        "n_tiles",
        "n_series",
        "shape",
        "channels",
        "channel_names",
        "n_z",
        "n_t",
        "tile_positions",
        "tile_identifiers",
        "pixel_size",
    }

    def test_metadata_common_keys(self, factory_meta, raw_meta):
        for key in self._COMMON_KEYS:
            assert factory_meta[key] == raw_meta[key], f"mismatch on key={key!r}"

    # -- read_region byte-for-byte match

    @pytest.fixture(scope="class")
    def factory_region(self, reader):
        return reader.read_region(0, slice(0, 64), slice(0, 64), channel_idx=0)

    @pytest.fixture(scope="class")
    def raw_region(self, raw_meta):
        return read_ome_tiff_tiles_region(
            raw_meta["ome_tiff_folder"],
            raw_meta["tile_identifiers"],
            raw_meta["tile_file_map"],
            tile_idx=0,
            axes=raw_meta["axes"],
            y_slice=slice(0, 64),
            x_slice=slice(0, 64),
            channel_idx=0,
        )

    def test_region_shape(self, factory_region, raw_region):
        assert factory_region.shape == raw_region.shape

    def test_region_byte_exact(self, factory_region, raw_region):
        np.testing.assert_array_equal(factory_region, raw_region)

    # -- load_metadata is idempotent (returns cached dict)

    def test_metadata_cached(self, reader):
        m1 = reader.load_metadata()
        m2 = reader.load_metadata()
        assert m1 is m2


# ---------------------------------------------------------------------------
# 2. individual_tiffs format
# ---------------------------------------------------------------------------


class TestIndividualTiffsFactory:
    """Factory picks _IndividualTiffsReader; metadata + region match raw functions."""

    @pytest.fixture(scope="class")
    def reader(self, individual_tiffs_folder):
        return open_reader(individual_tiffs_folder)

    def test_reader_type(self, reader):
        assert isinstance(reader, _IndividualTiffsReader)

    def test_protocol_compliance(self, reader):
        assert isinstance(reader, Reader)

    @pytest.fixture(scope="class")
    def factory_meta(self, reader):
        return reader.load_metadata()

    @pytest.fixture(scope="class")
    def raw_meta(self, individual_tiffs_folder):
        return load_individual_tiffs_metadata(individual_tiffs_folder)

    _COMMON_KEYS = {
        "n_tiles",
        "shape",
        "channels",
        "channel_names",
        "n_z",
        "n_t",
        "tile_positions",
        "tile_identifiers",
        "pixel_size",
    }

    def test_metadata_common_keys(self, factory_meta, raw_meta):
        for key in self._COMMON_KEYS:
            assert factory_meta[key] == raw_meta[key], f"mismatch on key={key!r}"

    @pytest.fixture(scope="class")
    def factory_region(self, reader):
        return reader.read_region(0, slice(0, 32), slice(0, 32), channel_idx=0)

    @pytest.fixture(scope="class")
    def raw_region(self, raw_meta):
        return read_individual_tiffs_region(
            raw_meta["image_folder"],
            raw_meta["channel_names"],
            raw_meta["tile_identifiers"],
            tile_idx=0,
            y_slice=slice(0, 32),
            x_slice=slice(0, 32),
            channel_idx=0,
            time_folders=raw_meta["time_folders"],
        )

    def test_region_shape(self, factory_region, raw_region):
        assert factory_region.shape == raw_region.shape

    def test_region_byte_exact(self, factory_region, raw_region):
        np.testing.assert_array_equal(factory_region, raw_region)

    def test_metadata_cached(self, reader):
        m1 = reader.load_metadata()
        m2 = reader.load_metadata()
        assert m1 is m2


# ---------------------------------------------------------------------------
# 3. zarr format
# ---------------------------------------------------------------------------


class TestZarrFactory:
    """Factory picks _ZarrReader; metadata + region match raw functions."""

    @pytest.fixture(scope="class")
    def reader(self, zarr_store_path):
        return open_reader(zarr_store_path)

    def test_reader_type(self, reader):
        assert isinstance(reader, _ZarrReader)

    def test_protocol_compliance(self, reader):
        assert isinstance(reader, Reader)

    @pytest.fixture(scope="class")
    def factory_meta(self, reader):
        return reader.load_metadata()

    @pytest.fixture(scope="class")
    def raw_meta(self, zarr_store_path):
        return load_zarr_metadata(zarr_store_path)

    _COMMON_KEYS = {
        "n_tiles",
        "n_series",
        "shape",
        "channels",
        "channel_names",
        "pixel_size",
        "tile_positions",
        "is_3d",
    }

    def test_metadata_common_keys(self, factory_meta, raw_meta):
        for key in self._COMMON_KEYS:
            assert factory_meta[key] == raw_meta[key], f"mismatch on key={key!r}"

    @pytest.fixture(scope="class")
    def factory_region(self, reader):
        return reader.read_region(0, slice(0, 32), slice(0, 32), channel_idx=0)

    @pytest.fixture(scope="class")
    def raw_region(self, raw_meta):
        return read_zarr_region(
            raw_meta["tensorstore"],
            tile_idx=0,
            y_slice=slice(0, 32),
            x_slice=slice(0, 32),
            channel_idx=0,
            is_3d=raw_meta["is_3d"],
        )

    def test_region_shape(self, factory_region, raw_region):
        assert factory_region.shape == raw_region.shape

    def test_region_byte_exact(self, factory_region, raw_region):
        np.testing.assert_array_equal(factory_region, raw_region)

    def test_metadata_cached(self, reader):
        m1 = reader.load_metadata()
        m2 = reader.load_metadata()
        assert m1 is m2


# ---------------------------------------------------------------------------
# 4. single OME-TIFF format
# ---------------------------------------------------------------------------


class TestSingleOmeTiffFactory:
    """Factory picks _OmeTiffReader; metadata + region match raw functions."""

    @pytest.fixture(scope="class")
    def reader(self, single_ome_tiff_path):
        return open_reader(single_ome_tiff_path)

    def test_reader_type(self, reader):
        assert isinstance(reader, _OmeTiffReader)

    def test_protocol_compliance(self, reader):
        assert isinstance(reader, Reader)

    @pytest.fixture(scope="class")
    def factory_meta(self, reader):
        return reader.load_metadata()

    _COMMON_KEYS = {
        "n_tiles",
        "n_series",
        "shape",
        "channels",
        "pixel_size",
        "tile_positions",
    }

    def test_metadata_has_no_handle(self, factory_meta):
        # core.py closes and pops the tiff_handle; the factory reader must do the same.
        assert "tiff_handle" not in factory_meta

    def test_metadata_common_keys(self, factory_meta, single_ome_tiff_path):
        raw_meta = load_ome_tiff_metadata(single_ome_tiff_path)
        try:
            for key in self._COMMON_KEYS:
                assert factory_meta[key] == raw_meta[key], f"mismatch on key={key!r}"
        finally:
            raw_meta["tiff_handle"].close()

    @pytest.fixture(scope="class")
    def factory_region(self, reader):
        return reader.read_region(0, slice(0, 32), slice(0, 32), channel_idx=0)

    @pytest.fixture(scope="class")
    def raw_region(self, single_ome_tiff_path):
        return read_ome_tiff_region(
            single_ome_tiff_path,
            tile_idx=0,
            y_slice=slice(0, 32),
            x_slice=slice(0, 32),
        )

    def test_region_shape(self, factory_region, raw_region):
        # Both return (1, h, w) — unified contract matching all other readers.
        assert factory_region.shape == (1, 32, 32)
        assert factory_region.shape == raw_region.shape

    def test_region_byte_exact(self, factory_region, raw_region):
        np.testing.assert_array_equal(factory_region, raw_region)

    def test_region_channel_pixel_preservation(self, reader, single_ome_tiff_path):
        """Behaviour-preservation: reader.read_region channel pixels match the
        old [channel_idx] slice of the former (C,h,w) full-channel result."""
        y_slice, x_slice = slice(0, 32), slice(0, 32)
        with tifffile.TiffFile(single_ome_tiff_path) as tif:
            arr = tif.series[0].asarray()
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        arr = np.flip(arr, axis=-2)
        old_ch0 = arr[0:1, y_slice, x_slice].astype(np.float32)

        result = reader.read_region(0, y_slice, x_slice, channel_idx=0)
        np.testing.assert_array_equal(result, old_ch0)

    def test_metadata_cached(self, reader):
        m1 = reader.load_metadata()
        m2 = reader.load_metadata()
        assert m1 is m2
