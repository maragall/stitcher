"""Tests for tilefusion.optimization module."""

import numpy as np
import pytest

from tilefusion.optimization import (
    solve_least_squares,
    two_round_optimization,
    _edges_from_pairwise_metrics,
    fit_stage_to_image_transform,
)


def _grid_with_injected_transform(
    n=6,
    step=795.0,
    ps=(0.325, 0.325),
    scale_err=0.017,
    rot_deg=0.5,
    jitter=3.0,
    seed=0,
    drop_tile=None,
):
    """A grid whose true stage->image map is a KNOWN similarity (scale+rotation)+jitter.

    Returns (pairwise_metrics, positions, ps, M_true, dropped). Pairs touching
    `drop_tile` are omitted, simulating a tile whose overlaps could not register.
    """
    rng = np.random.default_rng(seed)
    pos = [(r * step, c * step) for r in range(n) for c in range(n)]
    P = np.array(pos, dtype=np.float64)
    s = (1.0 + scale_err) / ps[0]
    th = np.radians(rot_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    M_true = s * R
    idx = lambda r, c: r * n + c
    pm = {}
    for r in range(n):
        for c in range(n):
            for dr, dc in [(0, 1), (1, 0)]:
                r2, c2 = r + dr, c + dc
                if r2 < n and c2 < n:
                    i, j = idx(r, c), idx(r2, c2)
                    if drop_tile in (i, j):
                        continue
                    sd = P[j] - P[i]
                    measured = M_true @ sd + rng.normal(0, jitter, 2)
                    res = measured - sd / np.array(ps)
                    pm[(i, j)] = (float(res[0]), float(res[1]), 0.9)
    return pm, pos, ps, M_true, drop_tile


class TestFitStageToImageTransform:
    """Ground-truth test of the global stage->image affine: inject a KNOWN scale +
    rotation + jitter and require the fit to recover it and to place tiles (incl. a
    tile with no registered pairs) back to ground truth."""

    def test_recovers_injected_scale_and_rotation(self):
        pm, pos, ps, M_true, _ = _grid_with_injected_transform(
            scale_err=0.017, rot_deg=0.5, jitter=3.0
        )
        out = fit_stage_to_image_transform(pm, pos, ps)
        true_scale = (1.017) / ps[0]
        np.testing.assert_allclose(out["scale"], true_scale, rtol=1e-2)
        assert abs(out["rotation_deg"] - 0.5) < 0.1, out["rotation_deg"]
        assert abs(out["anisotropy"] - 1.0) < 0.02, out["anisotropy"]
        # residual after the fit should be ~the injected jitter (not the systematic error)
        assert out["residual_rms"] < 6.0, out["residual_rms"]

    def test_places_stranded_tile_to_ground_truth(self):
        # tile 14 has ALL its pairs dropped (cannot register) -- the fit from the OTHER
        # pairs must still place it correctly, because the transform is global.
        stranded = 14
        pm, pos, ps, M_true, _ = _grid_with_injected_transform(drop_tile=stranded)
        assert all(stranded not in k for k in pm), "stranded tile must have no pairs"
        M = fit_stage_to_image_transform(pm, pos, ps)["M"]
        P = np.array(pos, dtype=np.float64)
        predicted = M @ (P[stranded] - P[0])
        truth = M_true @ (P[stranded] - P[0])
        assert np.linalg.norm(predicted - truth) < 8.0, np.linalg.norm(predicted - truth)

    def test_optimize_shifts_places_unconstrained_tile(self):
        # Integration: the optimize_shifts path must place a tile the solve left
        # unconstrained (no registered pairs) via the affine, near its true position.
        from types import SimpleNamespace
        from tilefusion.core import TileFusion

        stranded = 12  # interior tile of a 5x5 grid; dropping its pairs strands only it
        pm, pos, ps, M_true, _ = _grid_with_injected_transform(n=5, drop_tile=stranded, jitter=2.0)
        n = len(pos)
        edges = _edges_from_pairwise_metrics(pm)
        assert all(stranded not in k for k in pm)
        fake = SimpleNamespace(
            pairwise_metrics=pm,
            _tile_positions=pos,
            _pixel_size=ps,
            global_offsets=np.zeros((n, 2)),  # solve leaves the stranded tile at 0 (stage pos)
        )
        TileFusion._place_unconstrained_tiles_with_affine(fake, edges, n)
        P = np.array(pos, dtype=np.float64)
        d = P[stranded] - P[0]
        placed_px = d / np.array(ps) + fake.global_offsets[stranded]  # absolute px (rel. to anchor)
        truth_px = M_true @ d
        assert np.linalg.norm(placed_px - truth_px) < 8.0, np.linalg.norm(placed_px - truth_px)
        # a connected tile's offset must be untouched (still 0 here)
        assert np.allclose(fake.global_offsets[0], 0.0)


class TestSolveLeastSquares:
    """Tests for solve_least_squares function."""

    def test_two_tiles_simple(self):
        """Test optimization with two tiles and known offset."""
        links = [
            {"i": 0, "j": 1, "t": np.array([10.0, 5.0]), "w": 1.0},
        ]
        shifts = solve_least_squares(links, n_tiles=2, fixed_indices=[0])

        assert shifts.shape == (2, 2)
        # Tile 0 should be at origin
        assert np.allclose(shifts[0], [0, 0])
        # Tile 1 should be at the measured offset
        assert np.allclose(shifts[1], [10, 5])

    def test_three_tiles_chain(self):
        """Test optimization with three tiles in a chain."""
        links = [
            {"i": 0, "j": 1, "t": np.array([10.0, 0.0]), "w": 1.0},
            {"i": 1, "j": 2, "t": np.array([10.0, 0.0]), "w": 1.0},
        ]
        shifts = solve_least_squares(links, n_tiles=3, fixed_indices=[0])

        assert np.allclose(shifts[0], [0, 0])
        assert np.allclose(shifts[1], [10, 0])
        assert np.allclose(shifts[2], [20, 0])

    def test_overdetermined_system(self):
        """Test optimization with more links than needed."""
        # Triangle of three tiles with redundant links
        links = [
            {"i": 0, "j": 1, "t": np.array([10.0, 0.0]), "w": 1.0},
            {"i": 1, "j": 2, "t": np.array([0.0, 10.0]), "w": 1.0},
            {"i": 0, "j": 2, "t": np.array([10.0, 10.0]), "w": 1.0},
        ]
        shifts = solve_least_squares(links, n_tiles=3, fixed_indices=[0])

        assert np.allclose(shifts[0], [0, 0])
        assert np.allclose(shifts[1], [10, 0], atol=0.1)
        assert np.allclose(shifts[2], [10, 10], atol=0.1)

    def test_weighted_links(self):
        """Test that weights affect the solution."""
        # Two conflicting measurements with different weights
        links = [
            {"i": 0, "j": 1, "t": np.array([10.0, 0.0]), "w": 1.0},
            {"i": 0, "j": 1, "t": np.array([20.0, 0.0]), "w": 3.0},
        ]
        shifts = solve_least_squares(links, n_tiles=2, fixed_indices=[0])

        # Should be closer to 20 due to higher weight
        assert shifts[1, 0] > 15

    def test_no_links(self):
        """Test with no links (should return zeros)."""
        shifts = solve_least_squares([], n_tiles=3, fixed_indices=[0])
        assert np.allclose(shifts, 0)


class TestTwoRoundOptimization:
    """Tests for two_round_optimization function."""

    def test_no_outliers(self):
        """Test with consistent links (no outliers to remove)."""
        links = [
            {"i": 0, "j": 1, "t": np.array([10.0, 0.0]), "w": 1.0},
            {"i": 1, "j": 2, "t": np.array([10.0, 0.0]), "w": 1.0},
        ]
        shifts = two_round_optimization(
            links, n_tiles=3, fixed_indices=[0], rel_thresh=0.5, abs_thresh=5.0, iterative=False
        )

        assert np.allclose(shifts[0], [0, 0])
        assert np.allclose(shifts[1], [10, 0])
        assert np.allclose(shifts[2], [20, 0])

    def test_outlier_removal(self):
        """Test that outliers are removed."""
        # Use more extreme outlier and tighter chain
        links = [
            {"i": 0, "j": 1, "t": np.array([10.0, 0.0]), "w": 2.0},
            {"i": 1, "j": 2, "t": np.array([10.0, 0.0]), "w": 2.0},
            {
                "i": 0,
                "j": 2,
                "t": np.array([1000.0, 0.0]),
                "w": 0.1,
            },  # Clear outlier with low weight
        ]
        shifts = two_round_optimization(
            links, n_tiles=3, fixed_indices=[0], rel_thresh=0.3, abs_thresh=50.0, iterative=False
        )

        # Result should be closer to 20 than to the outlier value
        assert shifts[2, 0] < 100  # Not dominated by outlier

    def test_iterative_mode(self):
        """Test iterative outlier removal converges."""
        # Simple chain with one clear outlier
        links = [
            {"i": 0, "j": 1, "t": np.array([10.0, 0.0]), "w": 2.0},
            {"i": 1, "j": 2, "t": np.array([10.0, 0.0]), "w": 2.0},
            {"i": 0, "j": 2, "t": np.array([500.0, 0.0]), "w": 0.5},  # Clear outlier
        ]
        shifts = two_round_optimization(
            links, n_tiles=3, fixed_indices=[0], rel_thresh=0.3, abs_thresh=50.0, iterative=True
        )

        # Should converge to something reasonable (not dominated by outlier)
        assert shifts[2, 0] < 100


class TestEdgesFromPairwiseMetrics:
    """Tests for _edges_from_pairwise_metrics function."""

    def test_basic_conversion(self):
        """Test basic conversion from pairwise metrics."""
        metrics = {
            (0, 1): (10, 5, 0.9),
            (1, 2): (10, 0, 0.8),
        }
        links = _edges_from_pairwise_metrics(metrics)

        assert len(links) == 2
        assert links[0]["i"] == 0
        assert links[0]["j"] == 1
        assert np.allclose(links[0]["t"], [10, 5])
        assert links[0]["w"] == pytest.approx(np.sqrt(0.9))

    def test_empty_metrics(self):
        """Test with empty metrics."""
        links = _edges_from_pairwise_metrics({})
        assert links == []

    def test_weight_calculation(self):
        """Test that weights are sqrt of scores."""
        metrics = {
            (0, 1): (0, 0, 0.25),
            (1, 2): (0, 0, 1.0),
        }
        links = _edges_from_pairwise_metrics(metrics)

        assert links[0]["w"] == pytest.approx(0.5)
        assert links[1]["w"] == pytest.approx(1.0)


# --- robustness: degenerate / non-finite inputs must never crash the solve ----------


def test_solve_least_squares_drops_nonfinite_edges():
    """A NaN/inf weight or shift must be dropped, not crash the SVD (the Codex case)."""
    edges = [
        {"i": 0, "j": 1, "t": np.array([10.0, 0.0]), "w": 1.0},
        {"i": 1, "j": 2, "t": np.array([10.0, 0.0]), "w": 1.0},
        {"i": 0, "j": 2, "t": np.array([np.nan, 0.0]), "w": 1.0},  # poison row
        {"i": 0, "j": 1, "t": np.array([10.0, 0.0]), "w": np.inf},  # poison weight
    ]
    sh = solve_least_squares(edges, 3, [0])
    assert np.all(np.isfinite(sh))
    assert abs(sh[2, 0] - 20.0) < 1e-6  # chain 0->1->2 still solved from clean edges


def test_fit_stage_to_image_transform_survives_nonfinite_and_sparse():
    """Non-finite scores must be skipped; <2 usable pairs returns identity, not raise."""
    pm = {
        (0, 1): (0.0, 0.0, 0.9),
        (1, 2): (0.0, 0.0, np.nan),  # non-finite score -> skipped
        (0, 2): (0.0, 0.0, 0.8),
    }
    pos = [(0.0, 0.0), (0.0, 795.0), (0.0, 1590.0)]
    out = fit_stage_to_image_transform(pm, pos, (0.325, 0.325))
    assert np.all(np.isfinite(out["M"]))
    # all-bad -> identity fallback, no exception
    bad = {(0, 1): (np.nan, 0.0, 0.5)}
    out2 = fit_stage_to_image_transform(bad, pos, (0.325, 0.325))
    assert np.allclose(out2["M"], np.eye(2))
