"""Tests for the rotation-aware max_shift cap.

The cap scales with inter-tile spacing so a real stage->image rotation up to a few
degrees is not clipped (the scientist's 1-2 deg spec). Registration-channel selection is
covered in test_registration_channel.py (it is an explicit operator choice -- no
auto-pick).
"""

import numpy as np

from tilefusion.registration import rotation_aware_max_shift


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
