import json
from pathlib import Path

import numpy as np
import pytest

from tilefusion.core import TileFusion

FIXTURE = Path(__file__).parent / "fixtures" / "synth_4fov"


def run_pipeline(fixture_dir=FIXTURE) -> TileFusion:
    tf = TileFusion(fixture_dir, region="synth", channel_to_use=0)
    tf.refine_tile_positions_with_cross_correlation()
    tf.optimize_shifts()
    return tf


def _errors_px():
    gt = json.loads((FIXTURE / "ground_truth.json").read_text())
    e = np.array([t["error"] for t in sorted(gt["tiles"], key=lambda t: t["fov"])])
    return e  # (4,2) (y,x) px


def test_accuracy_global_offsets():
    tf = run_pipeline()
    e = _errors_px()
    expected = e - e[0]  # tile 0 is the fixed anchor
    recovered = np.asarray(tf.global_offsets, dtype=np.float64)
    np.testing.assert_allclose(recovered, expected, atol=0.5)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "register stores int(np.round(shift)) (registration.py:75 and "
        "core.py ~line 822), discarding the upsample_factor=10 sub-pixel "
        "precision; pair (2,3) lands ~0.52 px off the 0.5 px bar. The first "
        "accuracy fix removes the int-cast and flips this to pass (strict xfail "
        "will then fail the suite, forcing this marker's removal)."
    ),
)
def test_accuracy_pairwise():
    tf = run_pipeline()
    e = _errors_px()
    assert len(tf.pairwise_metrics) >= 3  # connected graph over 4 tiles
    for (i, j), (dy, dx, score) in tf.pairwise_metrics.items():
        expected = e[j] - e[i]
        np.testing.assert_allclose([dy, dx], expected, atol=0.5)


def test_regression_matches_committed_golden():
    golden = json.loads((FIXTURE / "golden_metrics.json").read_text())
    tf = run_pipeline()

    np.testing.assert_allclose(
        np.asarray(tf.global_offsets, dtype=np.float64),
        np.asarray(golden["global"], dtype=np.float64),
        atol=1e-6,
    )

    cur = {
        f"{i},{j}": [float(dy), float(dx), float(s)]
        for (i, j), (dy, dx, s) in tf.pairwise_metrics.items()
    }
    assert set(cur) == set(golden["pairwise"])
    for key in cur:
        np.testing.assert_allclose(cur[key], golden["pairwise"][key], atol=1e-6)
