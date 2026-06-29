"""Saturation-aware registration-channel auto-pick (brightfield robustness).

A blown-out channel has high std (clipping widens the histogram) but no usable
gradient texture, so plain std picks it and registration fails. The pick weights std
by (1 - fraction_saturated); for fluorescence (~no clipping) it is a no-op.
"""

import numpy as np

from tilefusion.core import TileFusion


class _FakeTF:
    """Minimal stand-in exposing only what _auto_pick_channel touches."""

    def __init__(self, stack):  # stack: (n_tiles, C, Y, X)
        self._stack = stack
        self.n_tiles, self.channels, self.Y, self.X = stack.shape

    def _read_tile(self, k):
        return self._stack[k]


def _pick(stack):
    return TileFusion._auto_pick_channel(_FakeTF(stack))


def test_no_saturation_picks_highest_std_unchanged():
    """Without clipping the pick is the highest-std channel (legacy behaviour)."""
    rng = np.random.default_rng(0)
    Y = X = 1100
    n, C = 4, 3
    stack = np.zeros((n, C, Y, X), np.uint16)
    stack[:, 0] = rng.integers(0, 5000, (n, Y, X))  # low contrast
    stack[:, 1] = rng.integers(0, 40000, (n, Y, X))  # HIGH contrast, no clipping
    stack[:, 2] = rng.integers(0, 15000, (n, Y, X))  # medium
    assert _pick(stack) == 1


def test_severely_saturated_channel_is_rejected():
    """A channel that is mostly clipped at the sensor max loses to a textured one,
    even though its raw std is the largest."""
    rng = np.random.default_rng(1)
    Y = X = 1100
    n, C = 4, 3
    stack = np.zeros((n, C, Y, X), np.uint16)
    # ch0: genuine moderate texture, no saturation
    stack[:, 0] = rng.integers(0, 20000, (n, Y, X))
    # ch1: ~95% pixels pinned at the uint16 max -> huge std, but useless. The other 5%
    # vary, so raw std is large; saturation fraction ~0.95 must veto it.
    sat = np.full((n, Y, X), 65535, np.uint16)
    mask = rng.random((n, Y, X)) > 0.95
    sat[mask] = rng.integers(0, 65535, mask.sum()).astype(np.uint16)
    stack[:, 1] = sat
    # ch2: low texture
    stack[:, 2] = rng.integers(0, 8000, (n, Y, X))
    # raw-std argmax would be ch1; saturation-weighted must pick ch0
    raw_std = [stack[:, c, 300:800, 300:800].std() for c in range(C)]
    assert int(np.argmax(raw_std)) == 1  # confirm the trap exists
    assert _pick(stack) == 0  # ...and the pick avoids it


def test_mild_saturation_does_not_flip_fluorescence_pick():
    """A few percent of saturated pixels (bright nuclei) must NOT veto the best channel
    -- the penalty is ~1 there, so the pick is unchanged."""
    rng = np.random.default_rng(2)
    Y = X = 1100
    n, C = 3, 2
    stack = np.zeros((n, C, Y, X), np.uint16)
    best = rng.integers(0, 40000, (n, Y, X)).astype(np.uint16)
    # 2% bright saturated nuclei in the best channel
    m = rng.random((n, Y, X)) > 0.98
    best[m] = 65535
    stack[:, 0] = best
    stack[:, 1] = rng.integers(0, 12000, (n, Y, X))  # clearly lower contrast
    assert _pick(stack) == 0
