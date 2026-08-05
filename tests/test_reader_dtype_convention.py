"""Pin the reader dtype convention documented on ``tilefusion.io.base.Reader``.

The convention is deliberately NOT uniform, and that is exactly what this file
exists to record:

* ``read_region`` -- the REGISTRATION path -- is float32 for every format, with
  no exception. Registration therefore correlates the raw, unrounded flat-field
  quotient. SquidXplorer's parity gate pins the offsets that produces, so this
  half is load-bearing across two repositories.
* ``read_tile`` -- the FUSION path -- is float32 for every format EXCEPT the
  ome_tiff/ folder format, which returns the file's native integer dtype.

That single exception is the only reason ``flatfield.apply_flatfield``'s
round-and-clip branch is reachable at all, while the identical-looking branch in
``apply_flatfield_region`` is dead code kept as a guard. A comment on the latter
used to claim it was "the registration path ... so the bias landed on the pixels
the correlator sees"; it never did.

If a reader is changed to normalise its dtype, these tests fail first, and the
docstrings on io.base.Reader, core._read_tile, core._read_tile_region and
flatfield.apply_flatfield_region must be updated with it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from tilefusion.flatfield import apply_flatfield, apply_flatfield_region
from tilefusion.io.base import open_reader

SYNTH_4FOV = Path(__file__).parent / "fixtures" / "synth_4fov"


@pytest.fixture(scope="module")
def individual_tiffs_folder(tmp_path_factory):
    """Minimal individual-TIFFs dataset (uint16 on disk, like a real acquisition)."""
    tmp = tmp_path_factory.mktemp("dtype_ind_tiffs")
    img_folder = tmp / "0"
    img_folder.mkdir()
    pd.DataFrame(
        {
            "fov": [0, 1, 2, 3],
            "x (mm)": [0.0, 1.0, 0.0, 1.0],
            "y (mm)": [0.0, 0.0, 1.0, 1.0],
        }
    ).to_csv(img_folder / "coordinates.csv", index=False)

    rng = np.random.default_rng(42)
    img = rng.integers(0, 65535, (64, 64), dtype=np.uint16)
    for fov in range(4):
        tifffile.imwrite(img_folder / f"manual_{fov}_0_ch1.tiff", img)
    return tmp


# ── the fusion path: one documented exception ────────────────────────────────


def test_ome_tiff_tiles_read_tile_is_native_integer():
    """The documented exception. Native integer, NOT float32."""
    reader = open_reader(SYNTH_4FOV)
    tile = reader.read_tile(0)
    assert np.issubdtype(tile.dtype, np.integer), (
        f"ome_tiff_tiles read_tile returned {tile.dtype}; it is the documented "
        "integer exception on io.base.Reader. If this is now float32, the "
        "apply_flatfield rounding branch has gone dead and the docs are wrong."
    )


def test_individual_tiffs_read_tile_is_float32(individual_tiffs_folder):
    reader = open_reader(individual_tiffs_folder)
    assert reader.read_tile(0).dtype == np.float32


# ── the registration path: no exception ──────────────────────────────────────


def test_ome_tiff_tiles_read_region_is_float32():
    reader = open_reader(SYNTH_4FOV)
    region = reader.read_region(0, slice(0, 16), slice(0, 16), channel_idx=0)
    assert region.dtype == np.float32


def test_individual_tiffs_read_region_is_float32(individual_tiffs_folder):
    reader = open_reader(individual_tiffs_folder)
    region = reader.read_region(0, slice(0, 16), slice(0, 16), channel_idx=0)
    assert region.dtype == np.float32


# ── what the asymmetry means for flat-field quantisation ─────────────────────


def test_apply_flatfield_region_leaves_float_input_unrounded():
    """The rint in apply_flatfield_region is an unreachable guard for real input.

    Every region reader yields float32, so the integer branch never runs and
    registration sees the fractional quotient. This is intentional: sub-pixel
    cross-correlation wants full precision.
    """
    region = np.full((1, 8, 8), 97.0, dtype=np.float32)
    ff = np.full((1, 8, 8), 0.6, dtype=np.float32)  # 97 / 0.6 = 161.666...
    out = apply_flatfield_region(region, ff, None, slice(0, 8), slice(0, 8))
    assert out.dtype == np.float32
    assert abs(float(out[0, 0, 0]) - 161.6667) < 1e-3, "must stay fractional"


def test_apply_flatfield_rounds_integer_input():
    """The same branch in apply_flatfield IS live, reached via ome_tiff_tiles tiles."""
    tile = np.full((1, 8, 8), 97, dtype=np.uint16)
    ff = np.full((1, 8, 8), 0.6, dtype=np.float32)  # 97 / 0.6 = 161.666...
    out = apply_flatfield(tile, ff, None)
    assert out.dtype == np.uint16
    assert int(out[0, 0, 0]) == 162, "must round to 162, not truncate to 161"


def test_the_two_paths_disagree_by_design():
    """Same pixels, same flat field, two answers — recorded, not accidental."""
    raw = np.full((1, 8, 8), 97, dtype=np.uint16)
    ff = np.full((1, 8, 8), 0.6, dtype=np.float32)
    fused_side = apply_flatfield(raw, ff, None)
    reg_side = apply_flatfield_region(raw.astype(np.float32), ff, None, slice(0, 8), slice(0, 8))
    assert float(fused_side[0, 0, 0]) != float(reg_side[0, 0, 0])
