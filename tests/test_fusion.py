"""Tests for tilefusion.fusion module."""

import numpy as np
import pytest

from tilefusion.fusion import accumulate_tile_shard, normalize_shard, blend_numba_2d


class TestAccumulateTileShard:
    """Tests for accumulate_tile_shard function."""

    def test_basic_accumulation(self):
        """Test basic weighted accumulation."""
        fused = np.zeros((1, 100, 100), dtype=np.float32)
        weight = np.zeros((1, 100, 100), dtype=np.float32)
        sub = np.ones((1, 20, 20), dtype=np.float32) * 100
        w2d = np.ones((20, 20), dtype=np.float32)

        accumulate_tile_shard(fused, weight, sub, w2d, 10, 10)

        # Check accumulated region
        assert fused[0, 15, 15] == 100.0
        assert weight[0, 15, 15] == 1.0
        # Check outside region
        assert fused[0, 0, 0] == 0.0
        assert weight[0, 0, 0] == 0.0

    def test_weighted_accumulation(self):
        """Test that weights are applied correctly."""
        fused = np.zeros((1, 50, 50), dtype=np.float32)
        weight = np.zeros((1, 50, 50), dtype=np.float32)
        sub = np.ones((1, 10, 10), dtype=np.float32) * 100
        w2d = np.ones((10, 10), dtype=np.float32) * 0.5

        accumulate_tile_shard(fused, weight, sub, w2d, 0, 0)

        assert fused[0, 5, 5] == 50.0  # 100 * 0.5
        assert weight[0, 5, 5] == 0.5

    def test_overlapping_accumulation(self):
        """Test accumulation of overlapping tiles."""
        fused = np.zeros((1, 50, 50), dtype=np.float32)
        weight = np.zeros((1, 50, 50), dtype=np.float32)
        sub1 = np.ones((1, 20, 20), dtype=np.float32) * 100
        sub2 = np.ones((1, 20, 20), dtype=np.float32) * 100
        w2d = np.ones((20, 20), dtype=np.float32)

        accumulate_tile_shard(fused, weight, sub1, w2d, 0, 0)
        accumulate_tile_shard(fused, weight, sub2, w2d, 10, 10)

        # Non-overlapping region
        assert fused[0, 5, 5] == 100.0
        assert weight[0, 5, 5] == 1.0
        # Overlapping region
        assert fused[0, 15, 15] == 200.0
        assert weight[0, 15, 15] == 2.0

    def test_boundary_handling(self):
        """Test that out-of-bounds pixels are ignored."""
        fused = np.zeros((1, 20, 20), dtype=np.float32)
        weight = np.zeros((1, 20, 20), dtype=np.float32)
        sub = np.ones((1, 10, 10), dtype=np.float32) * 100
        w2d = np.ones((10, 10), dtype=np.float32)

        # Place partially outside
        accumulate_tile_shard(fused, weight, sub, w2d, 15, 15)

        # Only 5x5 should be inside
        assert fused[0, 17, 17] == 100.0
        assert fused[0, 10, 10] == 0.0

    def test_multichannel(self):
        """Test accumulation with multiple channels."""
        fused = np.zeros((3, 50, 50), dtype=np.float32)
        weight = np.zeros((3, 50, 50), dtype=np.float32)
        sub = np.ones((3, 10, 10), dtype=np.float32)
        sub[0] *= 100
        sub[1] *= 200
        sub[2] *= 300
        w2d = np.ones((10, 10), dtype=np.float32)

        accumulate_tile_shard(fused, weight, sub, w2d, 0, 0)

        assert fused[0, 5, 5] == 100.0
        assert fused[1, 5, 5] == 200.0
        assert fused[2, 5, 5] == 300.0


class TestNormalizeShard:
    """Tests for normalize_shard function."""

    def test_basic_normalization(self):
        """Test basic weight normalization."""
        fused = np.ones((1, 10, 10), dtype=np.float32) * 200
        weight = np.ones((1, 10, 10), dtype=np.float32) * 2

        normalize_shard(fused, weight)

        assert np.allclose(fused, 100.0)

    def test_zero_weight_handling(self):
        """Test that zero weights result in zero values."""
        fused = np.ones((1, 10, 10), dtype=np.float32) * 100
        weight = np.zeros((1, 10, 10), dtype=np.float32)

        normalize_shard(fused, weight)

        assert np.allclose(fused, 0.0)

    def test_varying_weights(self):
        """Test normalization with varying weights."""
        fused = np.array([[[100, 200], [300, 400]]], dtype=np.float32)
        weight = np.array([[[1, 2], [4, 4]]], dtype=np.float32)

        normalize_shard(fused, weight)

        expected = np.array([[[100, 100], [75, 100]]], dtype=np.float32)
        assert np.allclose(fused, expected)


class TestBlendNumba2D:
    """Tests for blend_numba_2d function."""

    def test_equal_weights(self):
        """Test blending with equal weights."""
        sub_i = np.ones((10, 10), dtype=np.float32) * 100
        sub_j = np.ones((10, 10), dtype=np.float32) * 200
        wy_i = np.ones(10, dtype=np.float32)
        wx_i = np.ones(10, dtype=np.float32)
        wy_j = np.ones(10, dtype=np.float32)
        wx_j = np.ones(10, dtype=np.float32)
        out = np.zeros((10, 10), dtype=np.float32)

        result = blend_numba_2d(sub_i, sub_j, wy_i, wx_i, wy_j, wx_j, out)

        assert np.allclose(result, 150.0)

    def test_weighted_blend(self):
        """Test blending with different weights."""
        sub_i = np.ones((10, 10), dtype=np.float32) * 100
        sub_j = np.ones((10, 10), dtype=np.float32) * 200
        wy_i = np.ones(10, dtype=np.float32) * 0.75
        wx_i = np.ones(10, dtype=np.float32)
        wy_j = np.ones(10, dtype=np.float32) * 0.25
        wx_j = np.ones(10, dtype=np.float32)
        out = np.zeros((10, 10), dtype=np.float32)

        result = blend_numba_2d(sub_i, sub_j, wy_i, wx_i, wy_j, wx_j, out)

        # (0.75 * 100 + 0.25 * 200) / (0.75 + 0.25) = 125
        assert np.allclose(result, 125.0)

    def test_zero_total_weight(self):
        """Test fallback when total weight is zero."""
        sub_i = np.ones((10, 10), dtype=np.float32) * 100
        sub_j = np.ones((10, 10), dtype=np.float32) * 200
        wy_i = np.zeros(10, dtype=np.float32)
        wx_i = np.zeros(10, dtype=np.float32)
        wy_j = np.zeros(10, dtype=np.float32)
        wx_j = np.zeros(10, dtype=np.float32)
        out = np.zeros((10, 10), dtype=np.float32)

        result = blend_numba_2d(sub_i, sub_j, wy_i, wx_i, wy_j, wx_j, out)

        # Should fall back to sub_i
        assert np.allclose(result, 100.0)


class TestFusedWriteQuantisation:
    """fuse_plane must ROUND at the uint16 write, not truncate toward zero.

    The blend accumulates in float32; the write quantises. A bare ``.astype(np.uint16)``
    drops the fractional part of every pixel, which is a systematic -0.5 count bias over
    the whole mosaic rather than a wobble that averages out -- and the bias is absolute,
    so it costs a dim plane proportionally far more than a bright one. Same defect class
    already fixed one layer up in ``flatfield.apply_flatfield``.

    ``read_tile`` returns float32 for every reader except ome_tiff (see io/base.py's
    dtype convention), so fractional fused values are the normal case, not a corner one.
    """

    @staticmethod
    def _fuse_single_tile(tile):
        """Fuse one float32 tile at the origin with flat weights -> fused == tile exactly.

        Isolates the write-side quantisation from the blend: with a single FOV and unit
        feather profiles, normalize_shard divides by a weight of 1, so whatever comes out
        of the cast is purely the quantisation of ``tile``.
        """
        from tilefusion.fusion import fuse_plane

        C, Y, X = tile.shape
        written = {}

        def write_block(y0, y1, x0, x1, arr):
            written["out"] = np.array(arr)

        fuse_plane(
            read_tile=lambda i, z, t: tile,
            write_block=write_block,
            origins=[(0.0, 0.0)],
            padded_shape=(Y, X),
            tile_shape=(Y, X),
            channels=C,
            y_profile=np.ones(Y, np.float32),
            x_profile=np.ones(X, np.float32),
            block_size=max(Y, X),
        )
        return written["out"]

    def test_write_rounds_instead_of_truncating(self):
        # Fractional parts spread evenly over [0, 1) so truncation's bias is measurable.
        rng = np.random.default_rng(3)
        exact = (rng.integers(200, 4000, size=(1, 64, 64)) + rng.random((1, 64, 64))).astype(
            np.float32
        )

        out = self._fuse_single_tile(exact)

        assert out.dtype == np.uint16
        # Exactly the rounded value, not the truncated one.
        np.testing.assert_array_equal(out, np.rint(exact).astype(np.uint16))

        err = out.astype(np.float64) - exact.astype(np.float64)
        # Truncation lands at about -0.5 here; rounding must be unbiased.
        assert abs(err.mean()) < 0.01, f"quantisation is biased: mean error {err.mean():+.4f}"
        # Hard per-pixel bound: quantisation may never move a pixel by a whole count.
        assert np.abs(err).max() <= 0.5 + 1e-6

    def test_dim_signal_keeps_its_relative_accuracy(self):
        # The bias is absolute, so a dim plane is where truncation hurts: at ~5 counts,
        # losing half a count is a 10% error.
        out_hi = self._fuse_single_tile(np.full((1, 32, 32), 5.6, dtype=np.float32))
        assert out_hi.min() == out_hi.max() == 6, "truncation would report 5 for a true 5.6"

        out_lo = self._fuse_single_tile(np.full((1, 32, 32), 5.4, dtype=np.float32))
        assert out_lo.min() == out_lo.max() == 5  # rounding down is correct here

        rel = abs(float(out_hi.mean()) - 5.6) / 5.6
        assert rel < 0.08, f"dim-signal relative error {rel:.3f} too large"

    def test_out_of_range_saturates_instead_of_wrapping(self):
        # A flat-field-corrected blend can exceed 65535; the bare cast wrapped it
        # (70000 -> 4464), turning the brightest pixels of the mosaic into the dimmest.
        out = self._fuse_single_tile(np.full((1, 16, 16), 70000.0, dtype=np.float32))

        assert out.min() == 65535, "uint16 overflow must saturate, not wrap around"
