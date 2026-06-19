import json
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.generate_synth_4fov import (
    BASE_OFFSETS,
    SOURCE,
    backlash_offsets,
    compute_content_offsets,
    generate_fixture,
)


def test_base_offsets_are_the_raster_grid():
    expected = np.array([(0, 0), (0, 1024), (1024, 0), (1024, 1024)], dtype=np.float64)
    np.testing.assert_array_equal(BASE_OFFSETS, expected)


def test_backlash_matches_spec_table():
    expected = np.array([(0, 0), (0, 0), (0, 3), (0, -3)], dtype=np.float64)
    np.testing.assert_array_equal(backlash_offsets(3.0), expected)


def test_content_offsets_deterministic():
    a = compute_content_offsets(seed=42, sigma_px=1.5, b_px=3.0)
    b = compute_content_offsets(seed=42, sigma_px=1.5, b_px=3.0)
    np.testing.assert_array_equal(a, b)


def test_zero_jitter_reduces_to_base_plus_backlash():
    o = compute_content_offsets(seed=42, sigma_px=0.0, b_px=3.0)
    np.testing.assert_array_equal(o, BASE_OFFSETS + backlash_offsets(3.0))


def test_jitter_is_small_and_present():
    o = compute_content_offsets(seed=42, sigma_px=1.5, b_px=3.0)
    resid = o - (BASE_OFFSETS + backlash_offsets(3.0))
    assert np.any(resid != 0.0)
    assert np.all(np.abs(resid) < 10.0)


requires_source = pytest.mark.skipif(
    not (SOURCE / "ome_tiff" / "current_0.ome.tiff").exists(),
    reason="real source FOV not present; fixture regeneration test skipped",
)


@requires_source
def test_generator_roundtrip_and_determinism(tmp_path):
    from tilefusion.core import TileFusion

    a = tmp_path / "a"
    generate_fixture(a)

    assert (a / "coordinates.csv").exists()
    assert (a / "acquisition parameters.json").exists()
    assert sorted(p.name for p in (a / "ome_tiff").glob("*.ome.tiff")) == [
        f"synth_{k}.ome.tiff" for k in range(4)
    ]

    tf = TileFusion(a, region="synth", channel_to_use=0)
    assert tf.n_tiles == 4
    assert tf.channels == 1
    assert tf.n_z == 1
    assert (tf.Y, tf.X) == (1280, 1280)

    gt = json.loads((a / "ground_truth.json").read_text())
    assert len(gt["tiles"]) == 4
    assert gt["params"]["seed"] == 42

    b = tmp_path / "b"
    generate_fixture(b)
    assert (a / "ground_truth.json").read_bytes() == (b / "ground_truth.json").read_bytes()
    for k in range(4):
        assert (
            (a / "ome_tiff" / f"synth_{k}.ome.tiff").read_bytes()
            == (b / "ome_tiff" / f"synth_{k}.ome.tiff").read_bytes()
        )
