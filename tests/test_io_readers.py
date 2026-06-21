"""
Characterization tests for the four tilefusion.io readers.

PURPOSE: Pin each reader's CURRENT output contracts (metadata keys, dtype, shape,
pixel checksum) so that later Slice-1 refactor steps fail loudly if behaviour changes.

DO NOT modify these tests as part of refactoring — update them only when the
behaviour change is intentional and reviewed.

Formats covered:
  1. ome_tiff_tiles  — uses committed tests/fixtures/synth_4fov
  2. individual_tiffs — tiny in-memory fixture (deterministic rng seed=42)
  3. ome_tiff (single) — tiny multi-series OME-TIFF (deterministic rng seed=7)
  4. zarr             — zarr3 store with per_index_metadata (deterministic rng seed=13)
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

# ── Reader imports ────────────────────────────────────────────────────────────
from tilefusion.io.ome_tiff_tiles import (
    load_ome_tiff_tiles_metadata,
    read_ome_tiff_tiles_region,
)
from tilefusion.io.individual_tiffs import (
    load_individual_tiffs_metadata,
    read_individual_tiffs_region,
)
from tilefusion.io.ome_tiff import (
    load_ome_tiff_metadata,
    read_ome_tiff_region,
)
from tilefusion.io.zarr import (
    load_zarr_metadata,
    read_zarr_region,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ome_tiff_tiles — uses committed synth_4fov fixture
# ─────────────────────────────────────────────────────────────────────────────
SYNTH_4FOV = Path(__file__).parent / "fixtures" / "synth_4fov"


class TestOmeTiffTilesReader:
    """Characterize load_ome_tiff_tiles_metadata + read_ome_tiff_tiles_region."""

    @pytest.fixture(scope="class")
    def meta(self):
        return load_ome_tiff_tiles_metadata(SYNTH_4FOV)

    # ── metadata contract ────────────────────────────────────────────────────

    def test_n_tiles(self, meta):
        assert meta["n_tiles"] == 4

    def test_shape(self, meta):
        assert meta["shape"] == (1280, 1280)

    def test_channels(self, meta):
        assert meta["channels"] == 1

    def test_n_z(self, meta):
        assert meta["n_z"] == 1

    def test_n_t(self, meta):
        assert meta["n_t"] == 1

    def test_tile_positions_count(self, meta):
        assert len(meta["tile_positions"]) == 4

    def test_common_keys_present(self, meta):
        required = {
            "n_tiles", "shape", "channels", "channel_names",
            "n_z", "n_t", "tile_positions", "tile_identifiers",
            "pixel_size", "axes", "ome_tiff_folder", "tile_file_map",
        }
        assert required.issubset(meta.keys())

    # ── region contract ──────────────────────────────────────────────────────

    @pytest.fixture(scope="class")
    def region(self, meta):
        return read_ome_tiff_tiles_region(
            meta["ome_tiff_folder"],
            meta["tile_identifiers"],
            meta["tile_file_map"],
            tile_idx=0,
            axes=meta["axes"],
            y_slice=slice(0, 64),
            x_slice=slice(0, 64),
            channel_idx=0,
        )

    def test_region_dtype(self, region):
        assert region.dtype == np.float32

    def test_region_shape(self, region):
        # Current behaviour: YX tiles index [channel_idx, y, x] on a (1,Y,X) array
        # which produces (h, w) — pin this exact shape.
        assert region.shape == (64, 64)

    def test_region_pixel_checksum(self, region):
        assert round(float(region.sum()), 3) == 2931724.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. individual_tiffs — tiny deterministic fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def individual_tiffs_folder(tmp_path_factory):
    """
    Build a minimal individual-TIFFs dataset:
      tmp/
        0/
          coordinates.csv
          manual_0_0_ch1.tiff  ..  manual_3_0_ch1.tiff
    """
    tmp = tmp_path_factory.mktemp("ind_tiffs")
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


class TestIndividualTiffsReader:
    """Characterize load_individual_tiffs_metadata + read_individual_tiffs_region."""

    @pytest.fixture(scope="class")
    def meta(self, individual_tiffs_folder):
        return load_individual_tiffs_metadata(individual_tiffs_folder)

    # ── metadata contract ────────────────────────────────────────────────────

    def test_n_tiles(self, meta):
        assert meta["n_tiles"] == 4

    def test_shape(self, meta):
        assert meta["shape"] == (100, 100)

    def test_channels(self, meta):
        assert meta["channels"] == 1

    def test_channel_names(self, meta):
        assert meta["channel_names"] == ["ch1"]

    def test_n_z(self, meta):
        assert meta["n_z"] == 1

    def test_n_t(self, meta):
        assert meta["n_t"] == 1

    def test_tile_positions_count(self, meta):
        assert len(meta["tile_positions"]) == 4

    def test_pattern(self, meta):
        assert meta["pattern"] == "manual"

    def test_common_keys_present(self, meta):
        required = {
            "n_tiles", "shape", "channels", "channel_names",
            "n_z", "n_t", "tile_positions", "tile_identifiers",
            "pixel_size", "image_folder", "time_folders", "pattern",
        }
        assert required.issubset(meta.keys())

    # ── region contract ──────────────────────────────────────────────────────

    @pytest.fixture(scope="class")
    def region(self, meta):
        return read_individual_tiffs_region(
            meta["image_folder"],
            meta["channel_names"],
            meta["tile_identifiers"],
            tile_idx=0,
            y_slice=slice(0, 32),
            x_slice=slice(0, 32),
            channel_idx=0,
            time_folders=meta["time_folders"],
        )

    def test_region_dtype(self, region):
        assert region.dtype == np.float32

    def test_region_shape(self, region):
        # individual_tiffs returns (1, h, w) — one channel dimension preserved
        assert region.shape == (1, 32, 32)

    def test_region_pixel_checksum(self, region):
        assert round(float(region.sum()), 3) == 34543392.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. single OME-TIFF — tiny deterministic multi-series fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def single_ome_tiff_path(tmp_path_factory):
    """
    Build a minimal 4-tile single OME-TIFF.
    Pixel size set via tifffile metadata dict so load_ome_tiff_metadata
    picks up PhysicalSizeX/Y = 0.5.
    """
    tmp = tmp_path_factory.mktemp("ome_tiff")
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
                    "Plane": [{"PositionX": float((i % 2) * 50), "PositionY": float((i // 2) * 50)}],
                },
            )

    return path


class TestSingleOmeTiffReader:
    """Characterize load_ome_tiff_metadata + read_ome_tiff_region.

    CONTRACT (unified):
      read_ome_tiff_region now accepts ``channel_idx`` and returns shape
      (1, h, w) for the selected channel — matching all other region readers.
      The returned pixels are byte-identical to ``old_result[channel_idx]``
      where ``old_result`` was the former (C, h, w) full-channel array.
    """

    @pytest.fixture(scope="class")
    def meta(self, single_ome_tiff_path):
        meta = load_ome_tiff_metadata(single_ome_tiff_path)
        yield meta
        meta["tiff_handle"].close()

    # ── metadata contract ────────────────────────────────────────────────────

    def test_n_tiles(self, meta):
        assert meta["n_tiles"] == 4

    def test_n_series(self, meta):
        assert meta["n_series"] == 4

    def test_shape(self, meta):
        assert meta["shape"] == (100, 100)

    def test_channels(self, meta):
        assert meta["channels"] == 1

    def test_pixel_size(self, meta):
        assert meta["pixel_size"] == (0.5, 0.5)

    def test_tile_positions_count(self, meta):
        assert len(meta["tile_positions"]) == 4

    def test_common_keys_present(self, meta):
        required = {
            "n_tiles", "n_series", "shape", "channels",
            "pixel_size", "tile_positions", "tiff_handle",
        }
        assert required.issubset(meta.keys())

    # ── region contract ──────────────────────────────────────────────────────

    @pytest.fixture(scope="class")
    def region(self, single_ome_tiff_path):
        return read_ome_tiff_region(
            single_ome_tiff_path,
            tile_idx=0,
            y_slice=slice(0, 32),
            x_slice=slice(0, 32),
        )

    def test_region_dtype(self, region):
        assert region.dtype == np.float32

    def test_region_shape(self, region):
        # Returns (1, h, w) — one selected channel, matching all other readers.
        assert region.shape == (1, 32, 32)

    def test_region_pixel_checksum(self, region):
        assert round(float(region.sum()), 3) == 32910808.0

    def test_region_channel_pixel_preservation(self, single_ome_tiff_path):
        """Behaviour-preservation: channel_idx=0 pixels equal the old [0] slice of (C,h,w).

        The old read_ome_tiff_region returned arr[:, y, x] (all channels).
        The new version returns arr[channel_idx:channel_idx+1, y, x].
        For channel_idx=0 these must be byte-identical.
        """
        import tifffile as _tifffile

        y_slice, x_slice = slice(0, 32), slice(0, 32)
        # Reconstruct "old" full-channel result manually
        with _tifffile.TiffFile(single_ome_tiff_path) as tif:
            arr = tif.series[0].asarray()
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        arr = np.flip(arr, axis=-2)
        old_ch0 = arr[0:1, y_slice, x_slice].astype(np.float32)

        # New result
        new_result = read_ome_tiff_region(
            single_ome_tiff_path,
            tile_idx=0,
            y_slice=y_slice,
            x_slice=x_slice,
            channel_idx=0,
        )
        np.testing.assert_array_equal(new_result, old_ch0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. zarr — minimal zarr3 store with per_index_metadata
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def zarr_store_path(tmp_path_factory):
    """
    Build a minimal zarr3 dataset that load_zarr_metadata can parse.

    Layout: shape (T=1, P=2, C=1, Y=64, X=64)  — 5D, is_3d=False.
    The zarr.json attributes carry the per_index_metadata, channels, and
    deskewed_voxel_size_um that load_zarr_metadata requires.
    """
    import tensorstore as ts

    tmp = tmp_path_factory.mktemp("zarr")
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

    # Inject the attributes load_zarr_metadata reads
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


class TestZarrReader:
    """Characterize load_zarr_metadata + read_zarr_region."""

    @pytest.fixture(scope="class")
    def meta(self, zarr_store_path):
        return load_zarr_metadata(zarr_store_path)

    # ── metadata contract ────────────────────────────────────────────────────

    def test_n_tiles(self, meta):
        assert meta["n_tiles"] == 2

    def test_shape(self, meta):
        assert meta["shape"] == (64, 64)

    def test_channels(self, meta):
        assert meta["channels"] == 1

    def test_channel_names(self, meta):
        assert meta["channel_names"] == ["ch0"]

    def test_is_3d(self, meta):
        assert meta["is_3d"] is False

    def test_tile_positions_count(self, meta):
        assert len(meta["tile_positions"]) == 2

    def test_tile_positions_values(self, meta):
        assert meta["tile_positions"][0] == (7.8, 7.8)
        assert meta["tile_positions"][1] == (7.8, 327.6)

    def test_common_keys_present(self, meta):
        required = {
            "n_tiles", "n_series", "shape", "channels", "channel_names",
            "pixel_size", "tile_positions", "is_3d", "tensorstore",
        }
        assert required.issubset(meta.keys())

    # ── region contract ──────────────────────────────────────────────────────

    @pytest.fixture(scope="class")
    def region(self, meta):
        return read_zarr_region(
            meta["tensorstore"],
            tile_idx=0,
            y_slice=slice(0, 32),
            x_slice=slice(0, 32),
            channel_idx=0,
            is_3d=meta["is_3d"],
        )

    def test_region_dtype(self, region):
        assert region.dtype == np.float32

    def test_region_shape(self, region):
        # zarr returns (1, h, w) — consistent with individual_tiffs
        assert region.shape == (1, 32, 32)

    def test_region_pixel_checksum(self, region):
        assert round(float(region.sum()), 3) == 34054476.0
