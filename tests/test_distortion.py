"""Unit tests for per-seam elastic distortion correction (tilefusion.distortion).

These pin the SAFETY + CORRECTNESS contract:
  - CV order selection picks the simplest order the data supports (no overfit);
  - a flat (no-distortion) seam yields the identity field (None) -> no change;
  - a real bow yields a field whose midline matches the measured shift, split
    symmetrically between the two tiles and feathered to zero at the interiors.
"""

import numpy as np
from scipy.ndimage import map_coordinates

from tilefusion.distortion import _loocv_order, materialize_field
from tilefusion.fusion import accumulate_tile_shard, accumulate_tile_shard_distorted


def test_loocv_prefers_linear_for_linear_data():
    pos = np.linspace(0, 1000, 12)
    s = 0.01 * pos - 3.0  # exactly linear
    deg, _ = _loocv_order(pos, s, max_deg=3)
    assert deg == 1


def test_loocv_picks_quadratic_for_a_bow():
    pos = np.linspace(0, 1000, 12)
    s = 2.5e-5 * (pos - 500) ** 2 - 6.0  # a bow (quadratic)
    deg, _ = _loocv_order(pos, s, max_deg=3)
    assert deg == 2


def test_flat_seam_is_identity():
    # A seam with ~zero shift everywhere must produce no field (identity fallback).
    corr = [
        {
            "vert": True,
            "od": 200.0,
            "dy": 100,
            "dx": 0,
            "cy": np.array([0.0, 0.05]),
            "cx": np.array([0.0, 0.0]),  # < _MIN_CORRECTION_PX
            "sign": +0.5,
            "side": "i",
        }
    ]
    assert materialize_field(corr, 300, 300) is None


def test_symmetric_split_and_feather():
    Y = X = 300
    dy = 100  # vertical seam, overlap depth od = 200
    od = Y - dy
    # constant perpendicular shift of +10px along the whole seam
    cy = np.array([10.0])  # poly1d constant
    cx = np.array([0.0])
    base = {"vert": True, "od": float(od), "dy": dy, "dx": 0, "cy": cy, "cx": cx}
    Di = materialize_field([{**base, "sign": +0.5, "side": "i"}], Y, X)
    Dj = materialize_field([{**base, "sign": -0.5, "side": "j"}], Y, X)
    # Each tile carries half the correction at the overlap midline...
    mid_i = dy + od // 2
    assert np.isclose(Di[0, mid_i, X // 2], +5.0, atol=0.2)
    mid_j = od // 2
    assert np.isclose(Dj[0, mid_j, X // 2], -5.0, atol=0.2)
    # ...and zero deep in each interior (feathered out).
    assert np.isclose(Di[0, 0, X // 2], 0.0, atol=1e-3)  # tile i top (single)
    assert np.isclose(Dj[0, Y - 1, X // 2], 0.0, atol=1e-3)  # tile j bottom (single)


def test_folded_kernel_matches_warp_then_blend():
    """The fused distortion fold (accumulate_tile_shard_distorted) must equal the old
    path: warp the whole tile by the field, then blend the warped pixels. This guards
    the perf optimization (no separate warp pass) against silently changing output."""
    rng = np.random.default_rng(0)
    C, tY, tX = 2, 48, 48
    tile = rng.random((C, tY, tX)).astype(np.float32)
    yy, xx = np.mgrid[0:tY, 0:tX].astype(np.float32)
    # smooth small displacement field (a gentle bow), peak ~1.5px
    Dy = (1.5 * np.sin(np.pi * xx / tX)).astype(np.float32)
    Dx = (0.8 * np.cos(np.pi * yy / tY)).astype(np.float32)

    # interior sub-region (away from the tile edge, so no clamp ambiguity)
    sy0, sx0, subY, subX = 8, 8, 32, 32
    w2d = np.ones((subY, subX), np.float32)

    # OLD: warp full tile via map_coordinates, then blend the warped sub-region.
    warped = np.stack(
        [map_coordinates(tile[c], [yy + Dy, xx + Dx], order=1, mode="nearest") for c in range(C)]
    ).astype(np.float32)
    f_old = np.zeros((C, subY, subX), np.float32)
    w_old = np.zeros((C, subY, subX), np.float32)
    accumulate_tile_shard(f_old, w_old, warped[:, sy0 : sy0 + subY, sx0 : sx0 + subX], w2d, 0, 0)

    # NEW: fold the field into the blend sampler, no separate warp.
    f_new = np.zeros((C, subY, subX), np.float32)
    w_new = np.zeros((C, subY, subX), np.float32)
    accumulate_tile_shard_distorted(
        f_new, w_new, tile, w2d, 0, 0, sy0, sx0, subY, subX, 0.0, 0.0, Dy, Dx
    )

    assert np.allclose(f_old, f_new, atol=1e-4)


# --- brightfield reader: color channels must reduce to 2D so they stack ----------
def test_to_grayscale_2d_handles_color_channels():
    from tilefusion.io.individual_tiffs import _to_grayscale_2d
    import numpy as np

    gray = np.ones((2084, 2084), np.uint16)
    rgb = np.ones((2084, 2084, 3), np.uint16)
    assert _to_grayscale_2d(gray).shape == (2084, 2084)  # 2D passes through
    assert _to_grayscale_2d(rgb).shape == (2084, 2084)  # color -> 2D luminance
    # heterogeneous brightfield set (B,G,R grayscale + RGB composite) now stacks
    chans = [_to_grayscale_2d(a) for a in [gray, gray, gray, rgb]]
    assert np.stack(chans, 0).shape == (4, 2084, 2084)
