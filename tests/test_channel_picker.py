"""Tests for the auto registration-channel picker and the rotation-aware max_shift cap.

The picker chooses, per dataset, the channel with the most tissue structure (intensity
contrast / std) -- validated across RNAScope + 10x mouse brain to track registration
quality and reject the worst (signal-poor) channel. An explicit channel_to_use overrides
it. The max_shift cap scales with inter-tile spacing so a real stage->image rotation up
to a few degrees is not clipped (the scientist's 1-2 deg spec).
"""

import numpy as np
import pytest
from types import SimpleNamespace

from tilefusion.registration import rotation_aware_max_shift
from tilefusion.core import TileFusion


class TestRotationAwareMaxShift:
    def test_empty_pairs_floor(self):
        assert rotation_aware_max_shift([]) == (100, 100)

    def test_small_spacing_floored(self):
        # spacing 500 px: tan(3deg)*500 ~ 26 -> floored to 100 (no regression)
        pairs = [(0, 1, 0, 500, 100, 100), (1, 2, 500, 0, 100, 100)]
        assert rotation_aware_max_shift(pairs) == (100, 100)

    def test_large_spacing_scales(self):
        # spacing 2447 px: tan(3deg)*2447 ~ 128 -> above the floor
        pairs = [(0, 1, 0, 2447, 0, 0)] * 3
        cap = rotation_aware_max_shift(pairs)[0]
        assert cap == int(np.tan(np.radians(3.0)) * 2447)
        assert cap > 100

    def test_accommodates_two_degrees(self):
        # the cap MUST exceed the perpendicular residual a 2 deg rotation induces
        spacing = 2447
        pairs = [(0, 1, 0, spacing, 0, 0)] * 3
        cap = rotation_aware_max_shift(pairs)[0]
        assert cap > np.tan(np.radians(2.0)) * spacing


class TestAutoPickChannel:
    @staticmethod
    def _fake(channels, n_tiles, reader):
        return SimpleNamespace(
            channels=channels, n_tiles=n_tiles, Y=128, X=128,
            _read_tile=reader, channel_to_use=None,
        )

    def test_single_channel_returns_zero(self):
        fake = self._fake(1, 4, lambda k: np.zeros((1, 128, 128), np.float32))
        assert TileFusion._auto_pick_channel(fake) == 0

    def test_picks_highest_contrast_channel(self):
        rng = np.random.default_rng(0)

        def read(k):
            return np.stack([
                rng.normal(100, 5, (128, 128)),    # ch0: low contrast (signal-poor)
                rng.normal(100, 50, (128, 128)),   # ch1: HIGH contrast (best)
                rng.normal(100, 20, (128, 128)),   # ch2: medium
            ]).astype(np.float32)

        fake = self._fake(3, 4, read)
        assert TileFusion._auto_pick_channel(fake) == 1

    def test_no_readable_tiles_returns_zero(self):
        def boom(k):
            raise IOError("unreadable")
        fake = self._fake(3, 4, boom)
        assert TileFusion._auto_pick_channel(fake) == 0


class TestResolveRegistrationChannel:
    def test_explicit_channel_is_kept(self):
        fake = SimpleNamespace(channels=4, channel_to_use=2)
        TileFusion._resolve_registration_channel(fake)
        assert fake.channel_to_use == 2

    def test_out_of_range_raises(self):
        fake = SimpleNamespace(channels=3, channel_to_use=5)
        with pytest.raises(ValueError):
            TileFusion._resolve_registration_channel(fake)

    def test_none_triggers_autopick(self):
        fake = SimpleNamespace(
            channels=1, n_tiles=2, Y=64, X=64,
            _read_tile=lambda k: np.zeros((1, 64, 64), np.float32),
            channel_to_use=None,
        )
        fake._auto_pick_channel = lambda: TileFusion._auto_pick_channel(fake)
        TileFusion._resolve_registration_channel(fake)
        assert fake.channel_to_use == 0
