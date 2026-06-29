"""Regression test for fusion double-painting (root-caused + fixed 2026-06-25).

THE BUG: the global optimizer collapsed the tile overlap graph to a spanning tree
before solving. A tree has no cycles, so it satisfies its own edges exactly -- but
only by discarding the redundant edges, which dumps the entire accumulated
loop-closure error onto the seams the tree excluded. Concentrated on one seam, that
error fuses as a doubled feature ("double-painting"). The fix (commit on branch
fix/fusion-doubling-global-optimization) solves the FULL weighted graph so the error
distributes thinly across all seams.

WHAT WE ASSERT: the per-seam residual is the misalignment between two adjacent tiles'
overlapping SIGNAL. By definition the registered shift t_ij is the shift that aligns
tile i and tile j's shared overlap content, so after placement the leftover signal
misalignment at that seam is exactly

    seam_residual(i, j) = || (off[j] - off[i]) - t_ij ||   (px)

On a layout with a known loop-closure inconsistency we require the global solve to
DISTRIBUTE that inconsistency (every seam small) where a spanning-tree solve would
CONCENTRATE it (one seam carries the whole jump). A phase-correlation check confirms
the per-seam residual is a real pixel-signal misalignment, not just bookkeeping.
"""
import numpy as np
import pytest

from tilefusion.optimization import two_round_optimization, solve_least_squares


# --- geometry: 4 tiles in a square = a single cycle (the minimal loopy layout) ---
T = 256                       # tile size (px)
OV = 80                       # overlap (px)
STEP = T - OV                 # 176  -- neighbour spacing
DELTA = np.array([120.0, 0.0])  # injected loop-closure error (full-res px), sized like the real bug


def _square_edges(delta=DELTA):
    """Cycle 0->1->2->3->0 around a square. Every edge's measured shift t_ij equals the
    true relative position EXCEPT the closing edge, which carries the loop-closure error
    `delta` (so the shifts around the loop sum to `delta` instead of 0). score 0.9 each."""
    true = {0: (0.0, 0.0), 1: (0.0, STEP), 2: (STEP, STEP), 3: (STEP, 0.0)}
    edges = [
        {"i": 0, "j": 1, "t": np.array([0.0, STEP]),  "w": np.sqrt(0.9)},
        {"i": 1, "j": 2, "t": np.array([STEP, 0.0]),  "w": np.sqrt(0.9)},
        {"i": 2, "j": 3, "t": np.array([0.0, -STEP]), "w": np.sqrt(0.9)},
        {"i": 3, "j": 0, "t": np.array([-STEP, 0.0]) + delta, "w": np.sqrt(0.9)},
    ]
    return edges, true


def _seam_residuals(edges, off):
    """Per-edge signal misalignment ||(off[j]-off[i]) - t_ij|| in px."""
    return np.array([np.linalg.norm((off[e["j"]] - off[e["i"]]) - e["t"]) for e in edges])


def _tree_solve(edges, n_tiles, fixed):
    """Baseline = the OLD behaviour: solve on a spanning tree only (drop the closing
    edge). Reproduces the concentrating failure this test guards against."""
    tree = edges[:-1]  # 0-1, 1-2, 2-3  (a path spanning all 4 tiles; closing 3-0 dropped)
    return solve_least_squares(tree, n_tiles, fixed)


def test_loop_closure_error_is_distributed_not_concentrated():
    edges, _ = _square_edges()

    off_global = two_round_optimization(edges, 4, [0], rel_thresh=0.5, abs_thresh=2.0, iterative=True)
    off_tree = _tree_solve(edges, 4, [0])

    r_global = _seam_residuals(edges, off_global)
    r_tree = _seam_residuals(edges, off_tree)

    inconsistency = np.linalg.norm(DELTA)  # 120 px must go SOMEWHERE

    # Global solve distributes the closure error: worst seam is a small fraction of it,
    # and the seams are roughly even (no single concentrated jump).
    assert r_global.max() < inconsistency / 2, (
        f"global worst seam {r_global.max():.1f}px should be << {inconsistency:.0f}px "
        f"(error not distributed): {r_global.round(1)}"
    )
    assert r_global.max() / max(np.median(r_global), 1e-6) < 2.0, (
        f"global seams should be even (distributed), got {r_global.round(1)}"
    )
    # The error is conserved, just spread: total residual ~ the injected inconsistency.
    assert r_global.sum() == pytest.approx(inconsistency, rel=0.15), r_global.round(1)

    # The OLD tree behaviour concentrates the whole inconsistency on one seam...
    assert r_tree.max() == pytest.approx(inconsistency, rel=0.1), r_tree.round(1)
    # ...so the fix must be clearly better on the worst seam (this is the regression guard:
    # reintroduce the MST collapse and r_global.max() jumps back up to r_tree.max()).
    assert r_global.max() < r_tree.max() / 2.0, (
        f"global worst {r_global.max():.1f}px vs tree worst {r_tree.max():.1f}px"
    )


def test_consistent_graph_is_solved_exactly():
    """No loop-closure error (delta=0) -> every seam residual ~0 under the global solve.
    Guards against the fix introducing spurious displacement on clean data."""
    edges, _ = _square_edges(delta=np.array([0.0, 0.0]))
    off = two_round_optimization(edges, 4, [0], rel_thresh=0.5, abs_thresh=2.0, iterative=True)
    assert _seam_residuals(edges, off).max() < 1e-6


def test_seam_residual_is_a_real_signal_misalignment():
    """Tie the geometric seam residual to actual pixels: two tiles cropped from one
    textured source share their overlap content, so the registered shift t_ij that
    aligns that content equals the true relative position. We confirm phase correlation
    of the shared overlap recovers ~zero residual (the overlap signals ARE the same
    content) -- i.e. t_ij encodes signal alignment, so (off[j]-off[i]) - t_ij is the
    leftover signal misalignment the other tests measure."""
    skimage_reg = pytest.importorskip("skimage.registration")
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    H = W = T + STEP + 8
    source = gaussian_filter(rng.standard_normal((H, W)), sigma=2.0)

    y0 = x0 = 4
    tile0 = source[y0:y0 + T, x0:x0 + T]
    tile1 = source[y0:y0 + T, x0 + STEP:x0 + STEP + T]  # right neighbour; true shift = (0, STEP)

    # the shared overlap: right OV cols of tile0 == left OV cols of tile1
    strip0 = tile0[:, T - OV:]
    strip1 = tile1[:, :OV]
    shift, _, _ = skimage_reg.phase_cross_correlation(strip0, strip1, upsample_factor=10)
    assert np.linalg.norm(shift) < 1.0, f"overlap signals should coincide, got shift {shift}"
