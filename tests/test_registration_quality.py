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
    # Sub-pixel gate. Measured worst global error after the int-cast fix is ~0.065 px;
    # 0.15 px is a sound sub-pixel bound (the upsample_factor=10 grid is 0.1 px) with margin.
    np.testing.assert_allclose(recovered, expected, atol=0.15)


def test_accuracy_pairwise():
    # Sub-pixel pairwise accuracy. Before the fix, register stored int(np.round(shift)),
    # discarding the upsample_factor=10 precision -> pair (2,3) was ~0.52 px off. With the
    # int-cast removed, the measured worst pairwise error is ~0.085 px; 0.15 px is a sound
    # sub-pixel bound (0.1 px grid) with margin. If this regresses, investigate -- do not widen.
    tf = run_pipeline()
    e = _errors_px()
    assert len(tf.pairwise_metrics) >= 3  # connected graph over 4 tiles
    for (i, j), (dy, dx, score) in tf.pairwise_metrics.items():
        expected = e[j] - e[i]
        np.testing.assert_allclose([dy, dx], expected, atol=0.15)


def test_regression_matches_committed_golden():
    golden = json.loads((FIXTURE / "golden_metrics.json").read_text())
    tf = run_pipeline()

    # atol=1e-4 (not 1e-6): pairwise values are now FFT-upsampled floats and the global
    # offsets are lstsq solutions, neither bit-stable across BLAS/scipy versions. 1e-4 px
    # absorbs that last-digit noise while staying ~1000x tighter than the 0.15 px accuracy gate.
    np.testing.assert_allclose(
        np.asarray(tf.global_offsets, dtype=np.float64),
        np.asarray(golden["global"], dtype=np.float64),
        atol=1e-4,
    )

    cur = {
        f"{i},{j}": [float(dy), float(dx), float(s)]
        for (i, j), (dy, dx, s) in tf.pairwise_metrics.items()
    }
    assert set(cur) == set(golden["pairwise"])
    for key in cur:
        np.testing.assert_allclose(cur[key], golden["pairwise"][key], atol=1e-4)
