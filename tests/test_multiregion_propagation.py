"""``stitch_all_regions`` must hand each region the parent's FULL configuration.

It builds a fresh ``TileFusion`` per region. That constructor call used to omit
``flatfield``, ``darkfield``, ``registration_z`` and ``registration_t``, and never
copied ``enable_distortion_correction`` -- so asking the parent for a flat-field
corrected, distortion-free stitch on z=7 produced, for every region, an
uncorrected stitch on the middle z with distortion forced on. Nothing raised and
nothing was logged; the settings simply evaporated at the region boundary.

These tests intercept the per-region construction rather than running a fusion,
so they are fast and need no acquisition on disk.
"""

import numpy as np
import pytest

import tilefusion.core as core_mod

# Grab the real unbound method up front: the tests replace core_mod.TileFusion
# with a recorder, so it cannot be reached through the class afterwards.
_STITCH_ALL_REGIONS = core_mod.TileFusion.stitch_all_regions


class _FakeParent:
    """Minimal stand-in carrying exactly the attributes stitch_all_regions reads."""

    def __init__(self, tmp_path, flatfield, darkfield):
        self.tiff_path = tmp_path / "acq.ome.tiff"
        self.tiff_path.write_bytes(b"")
        self._unique_regions = ["A", "B"]
        self._blend_pixels = (32, 32)
        self.downsample_factors = (1, 1)
        self.ssim_window = 15
        self.multiscale_factors = (2, 4)
        self.resolution_multiples = ((1, 1), (2, 2))
        self._max_workers = 3
        self._debug = False
        self.channel_to_use = 2
        self.multiscale_downsample = "stride"
        self._flatfield = flatfield
        self._darkfield = darkfield
        self._registration_z = 7
        self._registration_t = 1
        self.enable_distortion_correction = False


class _RecordingTileFusion:
    """Records the kwargs it was built with; run() is a no-op."""

    calls = []

    def __init__(self, *args, **kwargs):
        _RecordingTileFusion.calls.append(kwargs)
        self.enable_distortion_correction = True

    def run(self):
        pass


@pytest.fixture
def region_calls(tmp_path, monkeypatch):
    ff = np.full((3, 8, 8), 1.1, dtype=np.float32)
    df = np.full((3, 8, 8), 2.0, dtype=np.float32)
    _RecordingTileFusion.calls = []
    monkeypatch.setattr(core_mod, "TileFusion", _RecordingTileFusion)

    parent = _FakeParent(tmp_path, ff, df)
    _STITCH_ALL_REGIONS(parent)
    return _RecordingTileFusion.calls, ff, df


def test_one_tilefusion_per_region(region_calls):
    calls, _, _ = region_calls
    assert len(calls) == 2
    assert [c["region"] for c in calls] == ["A", "B"]


def test_flatfield_and_darkfield_reach_every_region(region_calls):
    calls, ff, df = region_calls
    for call in calls:
        assert call["flatfield"] is ff, "flat-field dropped at the region boundary"
        assert call["darkfield"] is df, "dark-field dropped at the region boundary"


def test_registration_plane_reaches_every_region(region_calls):
    calls, _, _ = region_calls
    for call in calls:
        assert call["registration_z"] == 7, "region fell back to the middle z"
        assert call["registration_t"] == 1


def test_registration_channel_and_geometry_reach_every_region(region_calls):
    calls, _, _ = region_calls
    for call in calls:
        assert call["channel_to_use"] == 2
        assert call["blend_pixels"] == (32, 32)
        assert call["downsample_factors"] == (1, 1)


def test_distortion_switch_is_copied_to_every_region(tmp_path, monkeypatch):
    """The flag is set post-construction, so it needs an explicit copy."""
    built = []

    class _Rec:
        def __init__(self, *a, **kw):
            self.enable_distortion_correction = True
            built.append(self)

        def run(self):
            pass

    monkeypatch.setattr(core_mod, "TileFusion", _Rec)
    parent = _FakeParent(tmp_path, None, None)
    parent.enable_distortion_correction = False

    _STITCH_ALL_REGIONS(parent)

    assert len(built) == 2
    assert all(
        tf.enable_distortion_correction is False for tf in built
    ), "regions ran distortion correction the caller had switched off"
