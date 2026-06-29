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
    # block_size is fixed at chunk_y*2 = 64 (chunk_y overridden to 32 above) << image -> many blocks
    tf_chunk._fuse_tiles_chunked_plane()
    chunk_out = _read_scale0(tf_chunk.output_path)

    np.testing.assert_array_equal(chunk_out, full_out)


def test_chunked_equals_full_plane_feathered(tmp_path):
    # Same as test_chunked_equals_full_plane but with feathered blending (blend_pixels>0),
    # which exercises the block-boundary weight-profile slicing -- the path most at risk in
    # the full/chunked merge. Must be byte-identical.
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

    tf_full = TileFusion(full_dir, output_path=tmp_path / "full.zarr", blend_pixels=(20, 20))
    _prelude(tf_full)
    tf_full._fuse_tiles_full_plane()
    full_out = _read_scale0(tf_full.output_path)

    tf_chunk = TileFusion(chunk_dir, output_path=tmp_path / "chunk.zarr", blend_pixels=(20, 20))
    _prelude(tf_chunk)
    tf_chunk._fuse_tiles_chunked_plane()
    chunk_out = _read_scale0(tf_chunk.output_path)

    np.testing.assert_array_equal(chunk_out, full_out)


def test_direct_placement_streams_correctly(tmp_path):
    # Direct mode = no blending: each FOV is placed (streamed) at its origin,
    # overlaps overwritten (last FOV wins). Pins the streaming placement.
    rng = np.random.default_rng(11)
    ts_, ov = 200, 40
    step = ts_ - ov
    tiles = [rng.integers(100, 1000, size=(ts_, ts_), dtype=np.uint16) for _ in range(4)]
    positions = [(0, 0), (0, step), (step, 0), (step, step)]

    ddir = tmp_path / "direct_data"
    _write_dataset(ddir, tiles, positions)

    tf = TileFusion(ddir, output_path=tmp_path / "direct.zarr", blend_pixels=(0, 0))
    _prelude(tf)
    tf._fuse_tiles_direct_plane()
    out = _read_scale0(tf.output_path)  # (T, C, Z, Y, X)

    # Expected: place each FOV at its pixel origin in order, last writer wins.
    pad_Y, pad_X = tf.padded_shape
    expected = np.zeros((1, 1, 1, pad_Y, pad_X), dtype=np.uint16)
    for (oy_f, ox_f), tile in zip(tf._tile_pixel_origins(), tiles):
        # origins are now sub-pixel floats; direct mode floors to integer placement
        oy, ox = int(oy_f), int(ox_f)
        y_end, x_end = min(oy + tf.Y, pad_Y), min(ox + tf.X, pad_X)
        expected[0, 0, 0, oy:y_end, ox:x_end] = tile[: y_end - oy, : x_end - ox]

    np.testing.assert_array_equal(out, expected)


def test_fuse_plane_honours_subpixel_offset():
    """fuse_plane must place tiles at SUB-PIXEL precision, not truncate to integer.

    A bright single pixel fused at a 0.5 px fractional y-origin must split across two
    rows (bilinear), not land entirely in one -- this guards the sub-pixel placement
    that the old _tile_pixel_origins int-cast destroyed. The existing equivalence tests
    use integer offsets (frac=0), so they never exercise this path.
    """
    from tilefusion.fusion import fuse_plane

    C, Y, X = 1, 8, 8
    tile = np.zeros((C, Y, X), dtype=np.float32)
    tile[0, 4, 4] = 1000.0
    captured = {}

    def read_tile(t, z, ti):
        return tile

    def write_block(y0, y1, x0, x1, arr):
        captured["arr"] = np.asarray(arr)

    fuse_plane(
        read_tile=read_tile,
        write_block=write_block,
        origins=[(0.5, 0.0)],  # 0.5 px fractional y-origin
        padded_shape=(16, 16),
        tile_shape=(Y, X),
        channels=C,
        y_profile=np.ones(Y, dtype=np.float32),
        x_profile=np.ones(X, dtype=np.float32),
        block_size=16,
    )
    col = captured["arr"][0][:, 4]  # column through the bright pixel
    # bright pixel at tile row 4, +0.5 px -> split between output rows 4 and 5
    assert col[4] > 0 and col[5] > 0, f"0.5px shift should split across rows; got {col}"
    assert (
        abs(int(col[4]) - int(col[5])) < 250
    ), f"0.5px split should be ~even; got {col[4]},{col[5]}"
    # integer truncation would have dumped all 1000 into a single row:
    assert col.max() < 900, f"value looks truncated to one row (max={col.max()})"
