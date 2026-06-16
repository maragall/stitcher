"""Chunked fuse must equal full-plane fuse (the optimization's correctness guard).

This pins the behaviour the Phase 4 memory optimization must preserve: the
memory-efficient chunked path must produce a byte-identical fused image to the
full-plane path. Registration/optimization are skipped — we set tile positions
directly so the test is deterministic and exercises only the fusion math.
"""

import json

import numpy as np
import pandas as pd
import pytest
import tensorstore as ts
import tifffile

from tilefusion import TileFusion


def _write_dataset(path, tiles, positions):
    """Individual-TIFFs dataset: manual_{fov}_0_ch0.tiff + coordinates.csv."""
    img = path / "0"
    img.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "fov": list(range(len(tiles))),
            "x (mm)": [p[1] / 1000.0 for p in positions],
            "y (mm)": [p[0] / 1000.0 for p in positions],
        }
    ).to_csv(img / "coordinates.csv", index=False)
    for fov, tile in enumerate(tiles):
        tifffile.imwrite(img / f"manual_{fov}_0_ch0.tiff", tile)
    json.dump(
        {"objective": {"magnification": 1.0}, "sensor_pixel_size_um": 1.0},
        open(path / "acquisition parameters.json", "w"),
    )


def _read_scale0(path):
    store = ts.open(
        {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(path / "scale0" / "image")}}
    ).result()
    return np.asarray(store.read().result())


def _prelude(tf):
    """Set up fused space + output store using stage positions (no registration)."""
    tf.chunk_y = tf.chunk_x = 32  # small chunks so a small image still chunks
    tf._compute_fused_image_space()
    tf._pad_to_chunk_multiple()
    scale0 = tf.output_path / "scale0" / "image"
    scale0.parent.mkdir(parents=True, exist_ok=True)
    tf._create_fused_tensorstore(output_path=scale0)


def test_chunked_equals_full_plane(tmp_path):
    # 2x2 grid of 200px tiles, 40px overlap, random with bright features.
    rng = np.random.default_rng(7)
    ts_, ov = 200, 40
    step = ts_ - ov
    tiles = []
    for _ in range(4):
        t = rng.integers(100, 1000, size=(ts_, ts_), dtype=np.uint16)
        t[60:140, 60:140] += 5000
        tiles.append(t)
    positions = [(0, 0), (0, step), (step, 0), (step, step)]

    full_dir = tmp_path / "full_data"
    chunk_dir = tmp_path / "chunk_data"
    _write_dataset(full_dir, tiles, positions)
    _write_dataset(chunk_dir, tiles, positions)

    tf_full = TileFusion(full_dir, output_path=tmp_path / "full.zarr", blend_pixels=(0, 0))
    _prelude(tf_full)
    tf_full._fuse_tiles_full_plane()
    full_out = _read_scale0(tf_full.output_path)

    tf_chunk = TileFusion(chunk_dir, output_path=tmp_path / "chunk.zarr", blend_pixels=(0, 0))
    _prelude(tf_chunk)
    # tiny ram_fraction -> block_size hits its floor (chunk_y*2 = 64) << image -> many blocks
    tf_chunk._fuse_tiles_chunked_plane(ram_fraction=1e-9)
    chunk_out = _read_scale0(tf_chunk.output_path)

    np.testing.assert_array_equal(chunk_out, full_out)
