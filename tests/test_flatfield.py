"""Flatfield estimation tests.

Production estimation is BaSiC (low-rank + sparse, pure numpy port). The tests use
synthetic ground truth with tolerances rather than golden values (the iterative
solver drifts across numpy/scipy versions, so exact snapshots would be brittle).
"""
import numpy as np
import pytest

from tilefusion.flatfield import (
    apply_flatfield,
    calculate_flatfield,
    estimate_flatfield_basic,
    estimate_flatfield_channel,
    estimate_flatfield_median,
)


def _shading(Y, X, amp=0.5):
    """A smooth radial illumination dome, normalized to mean 1.0."""
    yy, xx = np.mgrid[0:Y, 0:X]
    s = 1.0 + amp * np.exp(-(((yy - Y / 2) ** 2 + (xx - X / 2) ** 2) / (2 * (Y / 3.0) ** 2)))
    return (s / s.mean()).astype(np.float32)


def _observed_stack(Y, X, n, amp=0.5, seed=0):
    """n tiles = shading * per-pixel random content (content decorrelated across tiles)."""
    rng = np.random.default_rng(seed)
    shading = _shading(Y, X, amp)
    stack = (shading[None] * rng.uniform(40, 200, (n, Y, X))).astype(np.float32)
    return stack, shading


# --------------------------------------------------------------------------- #
# 1. Recovery -- the cornerstone: BaSiC recovers a known shading field.
# --------------------------------------------------------------------------- #
def test_basic_recovers_known_shading():
    stack, shading = _observed_stack(192, 192, 40, seed=1)
    ff, df = estimate_flatfield_basic(stack, estimate_darkfield=False)
    assert df is None
    assert ff.shape == (192, 192)
    assert np.corrcoef(ff.ravel(), shading.ravel())[0, 1] > 0.95


def test_calculate_flatfield_recovers_shading_multichannel():
    rng = np.random.default_rng(0)
    Y = X = 192
    shading = _shading(Y, X)
    tiles = [((shading * rng.uniform(50, 200, (Y, X))).astype(np.float32))[None] for _ in range(40)]
    ff, df = calculate_flatfield(tiles, use_darkfield=False)
    assert ff.shape == (1, Y, X) and df is None
    assert np.corrcoef(ff[0].ravel(), shading.ravel())[0, 1] > 0.95


# --------------------------------------------------------------------------- #
# 2. Mean-normalization contract.
# --------------------------------------------------------------------------- #
def test_flatfield_normalized_to_mean_one():
    stack, _ = _observed_stack(128, 128, 30, seed=2)
    ff, _ = estimate_flatfield_basic(stack)
    assert abs(float(ff.mean()) - 1.0) < 1e-3


# --------------------------------------------------------------------------- #
# 3. Output contracts: shape, dtype, finite, strictly positive (it is a divisor).
# --------------------------------------------------------------------------- #
def test_output_contracts():
    stack, _ = _observed_stack(96, 128, 24, seed=3)
    ff, _ = estimate_flatfield_basic(stack)
    assert ff.shape == (96, 128)
    assert ff.dtype == np.float32
    assert np.isfinite(ff).all()
    assert (ff > 0).all()


# --------------------------------------------------------------------------- #
# 4. Flat input -> ~flat field. Must NOT invent a vignette where there is none
#    (the brightfield / evenly-illuminated case).
# --------------------------------------------------------------------------- #
def test_uniform_illumination_gives_flat_field():
    rng = np.random.default_rng(4)
    Y = X = 128
    # No shading: tiles are pure (decorrelated) content.
    stack = rng.uniform(40, 200, (30, Y, X)).astype(np.float32)
    ff, _ = estimate_flatfield_basic(stack)
    assert abs(float(ff.mean()) - 1.0) < 1e-3
    assert (ff.max() - ff.min()) < 0.10            # field is essentially flat


# --------------------------------------------------------------------------- #
# 5. Degenerate inputs never raise / never NaN -> safe unit fallback.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("make", [
    lambda: np.zeros((10, 64, 64), np.float32),                       # all zero
    lambda: np.ones((1, 64, 64), np.float32) * 123.0,                 # single tile
    lambda: np.full((8, 64, 64), 50.0, np.float32),                   # constant tiles
    lambda: np.full((5, 64, 64), 7.0, np.float32),                    # few constant tiles
])
def test_degenerate_inputs_are_safe(make):
    stack = make()
    ff, _ = estimate_flatfield_basic(stack, estimate_darkfield=False)
    assert ff.shape == stack.shape[1:]
    assert np.isfinite(ff).all() and (ff > 0).all()


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        estimate_flatfield_basic(np.zeros((64, 64), np.float32))      # 2D, not (n,Y,X)


# --------------------------------------------------------------------------- #
# 6. Determinism: same input -> identical output (no hidden RNG dependence).
# --------------------------------------------------------------------------- #
def test_deterministic():
    stack, _ = _observed_stack(96, 96, 20, seed=5)
    ff1, _ = estimate_flatfield_basic(stack)
    ff2, _ = estimate_flatfield_basic(stack)
    assert np.array_equal(ff1, ff2)


# --------------------------------------------------------------------------- #
# 7. apply_flatfield roundtrip: correcting shading*content recovers flat content.
# --------------------------------------------------------------------------- #
def test_apply_flatfield_roundtrip_flattens_shading():
    Y = X = 160
    shading = _shading(Y, X)
    content = 120.0                                  # uniform content
    tile = (shading * content).astype(np.float32)[None]
    stack = np.repeat(tile, 20, axis=0)              # 20 identical-content tiles + shading
    # vary content per tile so BaSiC sees structure to separate
    rng = np.random.default_rng(6)
    stack = (shading[None] * rng.uniform(80, 160, (20, Y, X))).astype(np.float32)
    ff, _ = estimate_flatfield_basic(stack)
    corrected = apply_flatfield(stack[0][None], ff[None], None)[0].astype(np.float32)
    raw = stack[0]
    # corrected content should be flatter across the FOV than the raw (shaded) tile:
    # compare centre-vs-corner ratio, which the shading otherwise imposes.
    def corner_center(img):
        s = img.shape[0] // 8
        corners = np.mean([img[:s, :s].mean(), img[-s:, -s:].mean()])
        center = img[Y // 2 - s:Y // 2 + s, X // 2 - s:X // 2 + s].mean()
        return corners / center
    assert abs(corner_center(corrected) - 1.0) < abs(corner_center(raw) - 1.0)


# --------------------------------------------------------------------------- #
# 8. Few tiles: BaSiC stays sane (finite, mean 1, bounded range) where the naive
#    median over-states the dome.
# --------------------------------------------------------------------------- #
def test_few_tiles_field_is_bounded():
    stack, _ = _observed_stack(128, 128, 9, amp=0.3, seed=7)
    ff, _ = estimate_flatfield_basic(stack)
    assert np.isfinite(ff).all() and (ff > 0).all()
    assert abs(float(ff.mean()) - 1.0) < 1e-3
    assert (ff.max() - ff.min()) < 1.0               # not the median's blown-up range


# --------------------------------------------------------------------------- #
# Interface parity: the production channel helper returns the right shapes and a
# constant darkfield when requested.
# --------------------------------------------------------------------------- #
def test_channel_helper_and_darkfield_shapes():
    stack, _ = _observed_stack(128, 128, 30, seed=8)
    ff, df = estimate_flatfield_channel(stack, use_darkfield=True, constant_darkfield=True)
    assert ff.shape == (128, 128) and df.shape == (128, 128)
    assert np.allclose(df, df.flat[0])               # constant darkfield
    assert np.isfinite(ff).all() and (ff > 0).all()


def test_median_estimator_still_available():
    """The legacy median estimator is retained for benchmarks/comparison."""
    stack, shading = _observed_stack(128, 128, 40, seed=9)
    ff, _ = estimate_flatfield_median(stack, use_darkfield=False)
    assert ff.shape == (128, 128)
    assert np.corrcoef(ff.ravel(), shading.ravel())[0, 1] > 0.95
