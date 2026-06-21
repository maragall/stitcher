"""
Unit tests for the shared Squid-format helpers in tilefusion.io._squid.

VALUES ARE PINNED from a live run against:
  tests/fixtures/synth_4fov/acquisition parameters.json
    {"sensor_pixel_size_um": 6.5, "objective": {"magnification": 20.0}, "Nz": 1}
  tests/fixtures/synth_4fov/coordinates.csv
    region,x (mm),y (mm),z (mm)
    synth,0.0078,0.0078,0.0
    synth,0.3276,0.0078,0.0
    synth,0.0078,0.3276,0.0
    synth,0.3276,0.3276,0.0

These tests confirm that helpers reproduce EXACTLY the values formerly inlined
in both loaders.  Do not change them as part of refactoring; update only when a
behaviour change is intentional.
"""

from pathlib import Path

import pytest

from tilefusion.io._squid import channel_names_or_default, load_acquisition_params

SYNTH_4FOV = Path(__file__).parent / "fixtures" / "synth_4fov"


# ─────────────────────────────────────────────────────────────────────────────
# load_acquisition_params — fixture with JSON present
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadAcquisitionParams:
    @pytest.fixture(scope="class")
    def params(self):
        return load_acquisition_params(SYNTH_4FOV)

    def test_pixel_size_um(self, params):
        # 6.5 µm sensor / 20× magnification = 0.325 µm/px
        pixel_size_um, *_ = params
        assert pixel_size_um == pytest.approx(0.325, rel=1e-6)

    def test_n_z(self, params):
        _, n_z, _, _ = params
        assert n_z == 1

    def test_n_t(self, params):
        _, _, n_t, _ = params
        assert n_t == 1

    def test_dz_um(self, params):
        # "dz(um)" key absent from fixture JSON → fallback 1.0
        _, _, _, dz_um = params
        assert dz_um == pytest.approx(1.0)

    def test_return_length(self, params):
        assert len(params) == 4


# ─────────────────────────────────────────────────────────────────────────────
# load_acquisition_params — fallback when JSON is absent
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadAcquisitionParamsFallback:
    @pytest.fixture(scope="class")
    def params(self, tmp_path_factory):
        empty = tmp_path_factory.mktemp("no_params")
        return load_acquisition_params(empty)

    def test_fallback_pixel_size_um(self, params):
        pixel_size_um, *_ = params
        # Default: 7.52 µm / 10× = 0.752 µm/px
        assert pixel_size_um == pytest.approx(0.752, rel=1e-6)

    def test_fallback_n_z(self, params):
        _, n_z, _, _ = params
        assert n_z == 1

    def test_fallback_n_t(self, params):
        _, _, n_t, _ = params
        assert n_t == 1

    def test_fallback_dz_um(self, params):
        _, _, _, dz_um = params
        assert dz_um == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# channel_names_or_default
# ─────────────────────────────────────────────────────────────────────────────

class TestChannelNamesOrDefault:
    def test_non_empty_returns_as_is(self):
        names = ["DAPI", "GFP", "mCherry"]
        assert channel_names_or_default(names, 3) == ["DAPI", "GFP", "mCherry"]

    def test_empty_generates_channel_i(self):
        result = channel_names_or_default([], 3)
        assert result == ["Channel_0", "Channel_1", "Channel_2"]

    def test_empty_single_channel(self):
        result = channel_names_or_default([], 1)
        assert result == ["Channel_0"]

    def test_non_empty_ignores_channels_arg(self):
        # channels arg is irrelevant when names is non-empty
        result = channel_names_or_default(["A", "B"], 99)
        assert result == ["A", "B"]


# ─────────────────────────────────────────────────────────────────────────────
# Integration: helpers reproduce what the ome_tiff_tiles reader returns
# for the synth_4fov fixture (cross-check against the characterization tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpersCrossCheckOmeTiffTiles:
    """
    Confirm that _squid helpers reproduce the exact values load_ome_tiff_tiles_metadata
    returns for synth_4fov, specifically the pixel_size and z/t scalars.
    """

    def test_pixel_size_matches_reader(self):
        from tilefusion.io.ome_tiff_tiles import load_ome_tiff_tiles_metadata

        meta = load_ome_tiff_tiles_metadata(SYNTH_4FOV)
        pixel_size_um, _, _, _ = load_acquisition_params(SYNTH_4FOV)
        assert meta["pixel_size"] == (pixel_size_um, pixel_size_um)

    def test_n_z_matches_reader(self):
        from tilefusion.io.ome_tiff_tiles import load_ome_tiff_tiles_metadata

        meta = load_ome_tiff_tiles_metadata(SYNTH_4FOV)
        _, n_z, _, _ = load_acquisition_params(SYNTH_4FOV)
        assert meta["n_z"] == n_z

    def test_n_t_matches_reader(self):
        from tilefusion.io.ome_tiff_tiles import load_ome_tiff_tiles_metadata

        meta = load_ome_tiff_tiles_metadata(SYNTH_4FOV)
        _, _, n_t, _ = load_acquisition_params(SYNTH_4FOV)
        assert meta["n_t"] == n_t

    def test_tile_positions_are_mm_to_um_converted(self):
        """
        Pinned: coordinates.csv has x/y in mm; reader multiplies by 1000.
        0.0078 mm → 7.8 µm, 0.3276 mm → 327.6 µm.
        """
        from tilefusion.io.ome_tiff_tiles import load_ome_tiff_tiles_metadata

        meta = load_ome_tiff_tiles_metadata(SYNTH_4FOV)
        positions = meta["tile_positions"]
        assert len(positions) == 4
        # Row order from CSV: (y_um, x_um)
        assert positions[0] == pytest.approx((7.8, 7.8), rel=1e-6)
        assert positions[1] == pytest.approx((7.8, 327.6), rel=1e-6)
        assert positions[2] == pytest.approx((327.6, 7.8), rel=1e-6)
        assert positions[3] == pytest.approx((327.6, 327.6), rel=1e-6)

    def test_channel_names_fallback_when_no_ome_metadata(self):
        """
        synth_4fov tiles have no OME Channel names → fallback to Channel_0.
        channel_names_or_default([], 1) == ["Channel_0"].
        """
        result = channel_names_or_default([], 1)
        assert result == ["Channel_0"]

        from tilefusion.io.ome_tiff_tiles import load_ome_tiff_tiles_metadata

        meta = load_ome_tiff_tiles_metadata(SYNTH_4FOV)
        assert meta["channel_names"] == ["Channel_0"]
