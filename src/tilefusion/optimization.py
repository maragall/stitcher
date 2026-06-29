"""
Global position optimization.

Least-squares optimization of tile positions from pairwise measurements, modelled
as a translation-only pose graph (tiles = nodes, registered overlaps = edges
carrying a relative-shift measurement).

We solve the FULL weighted overlap graph in one least-squares pass so the
unavoidable loop-closure inconsistency (the relative shifts around any cycle do
not sum to exactly zero, because each registration is noisy) is DISTRIBUTED across
every overlap, then iteratively reject blunder edges by a scale-relative criterion.
This is the globally-optimal-stitching approach (Preibisch et al. 2009, "Globally
optimal stitching of tiled 3D microscopic image acquisitions"; Hoerl et al. 2019,
"BigStitcher").

We deliberately do NOT collapse the graph to a spanning tree before solving. A
spanning tree has no cycles, so it satisfies its own edges exactly -- but only by
discarding the redundant edges, which dumps the entire accumulated loop-closure
error onto the seams the tree excluded. Concentrated on one seam, that error fuses
as a visibly doubled feature ("double-painting"). The global solve spreads the same
error thinly across all seams, below the visible threshold.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _check_connectivity(edges: List[Dict[str, Any]], n_tiles: int) -> List[List[int]]:
    """
    Check graph connectivity and return connected components.

    Parameters
    ----------
    edges : list of dict
        Edges with 'i', 'j' keys.
    n_tiles : int
        Total number of tiles.

    Returns
    -------
    components : list of list of int
        Each inner list is a connected component (list of tile indices).
        If fully connected, returns a single list of all tile indices.
    """
    parent = list(range(n_tiles))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for edge in edges:
        union(edge["i"], edge["j"])

    components = {}
    for i in range(n_tiles):
        root = find(i)
        components.setdefault(root, []).append(i)

    return list(components.values())


def solve_least_squares(
    edges: List[Dict[str, Any]], n_tiles: int, fixed_indices: List[int]
) -> np.ndarray:
    """
    Solve a linear least-squares for all 2 axes at once,
    given weighted pairwise edges and fixed tile indices.

    Parameters
    ----------
    edges : list of dict
        Each dict has keys: 'i', 'j', 't' (2D offset), 'w' (weight).
    n_tiles : int
        Total number of tiles.
    fixed_indices : list of int
        Indices of tiles to fix at origin.

    Returns
    -------
    shifts : ndarray of shape (n_tiles, 2)
        Optimized shifts for each tile.
    """
    # Drop any edge with a non-finite weight or shift before building the system: a
    # single NaN/inf (e.g. a degenerate registration) poisons the whole solve and is
    # the usual cause of "SVD did not converge". Connectivity is re-checked by callers.
    clean = [
        e for e in edges if np.isfinite(e["w"]) and np.all(np.isfinite(np.asarray(e["t"], float)))
    ]
    if len(clean) != len(edges):
        # debug, not warning: this runs once per iterative-rejection round, so warning
        # would flood; the drop is expected and handled.
        logger.debug(
            "solve_least_squares: dropped %d/%d edges with non-finite " "weight/shift",
            len(edges) - len(clean),
            len(edges),
        )
    edges = clean

    shifts = np.zeros((n_tiles, 2), dtype=np.float64)
    for axis in range(2):
        m = len(edges) + len(fixed_indices)
        A = np.zeros((m, n_tiles), dtype=np.float64)
        b = np.zeros(m, dtype=np.float64)
        row = 0
        for edge in edges:
            i, j = edge["i"], edge["j"]
            t, w = edge["t"][axis], edge["w"]
            A[row, j] = w
            A[row, i] = -w
            b[row] = w * t
            row += 1
        for idx in fixed_indices:
            A[row, idx] = 1.0
            b[row] = 0.0
            row += 1
        shifts[:, axis] = _solve_axis(A, b, n_tiles)
    return shifts


def _solve_axis(A: np.ndarray, b: np.ndarray, n_tiles: int) -> np.ndarray:
    """Least-squares solve for one axis, robust to LAPACK SVD non-convergence.

    np.linalg.lstsq uses the gelsd (SVD) driver, which occasionally fails to converge
    on large or ill-conditioned dense systems. On failure we fall back to the normal
    equations with a tiny ridge (an LU/Cholesky path, no SVD); for a well-conditioned
    full-rank system the ridge is negligible and the result matches lstsq, so good data
    is unaffected and only the pathological case takes the fallback.
    """
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        if np.all(np.isfinite(sol)):
            return sol
        logger.warning(
            "solve_least_squares: lstsq returned non-finite solution; "
            "using ridge normal equations"
        )
    except np.linalg.LinAlgError as e:
        logger.warning("solve_least_squares: lstsq failed (%s); using ridge normal " "equations", e)
    AtA = A.T @ A
    ridge = 1e-9 * (np.trace(AtA) / max(n_tiles, 1) + 1.0)
    AtA[np.diag_indices_from(AtA)] += ridge
    try:
        return np.linalg.solve(AtA, A.T @ b)
    except np.linalg.LinAlgError:
        # Last resort: pseudo-inverse of the (small, regularized) normal matrix.
        return np.linalg.pinv(AtA) @ (A.T @ b)


def _num_components(edges: List[Dict[str, Any]], n_tiles: int) -> int:
    """Number of connected components in the tile graph defined by `edges`."""
    return len(_check_connectivity(edges, n_tiles))


# An edge is a candidate blunder only if its post-solve residual exceeds this
# multiple of the MEDIAN residual. It MUST be > 1: the global solve deliberately
# leaves small, balanced residuals on every good edge (that is how loop-closure
# error is distributed), so a cutoff at or below the median would over-reject good
# edges and silently thin the graph back toward a spanning tree -- the exact
# failure this solver removes. (BigStitcher uses the same relative-error rule; a
# fixed absolute threshold tuned for the spanning-tree regime is what over-rejects.)
_REL_OUTLIER_FACTOR = 3.0

# Absolute floor (px) below which a residual is NEVER an outlier, regardless of the
# relative test. This must be set well above the largest residual a *good* edge can
# carry, because on a real (sparse/loopy) graph the interior solves to a sub-pixel
# median, which would drive factor*median (and any small abs_thresh from a caller)
# down to a couple of pixels -- at which point the rejection deletes normal
# distributed residuals and RE-CONCENTRATES loop-closure error onto real tissue seams
# (measured on the full TMA: a 2px floor left a 130px worst seam; this 150px floor
# leaves 44px == the no-rejection solve). 150px sits just above registration's
# max_shift gate (128px), so only an edge the solve cannot satisfy within any
# plausible registered shift -- i.e. a genuine blunder -- is ever dropped. Moderate
# disagreement is KEPT and averaged, never rejected.
_BLUNDER_FLOOR_PX = 150.0


def two_round_optimization(
    edges: List[Dict[str, Any]],
    n_tiles: int,
    fixed_indices: List[int],
    rel_thresh: float,
    abs_thresh: float,
    iterative: bool,
) -> np.ndarray:
    """
    Robust global optimization of tile positions (translation-only pose graph).

    1. Solve ONE weighted least-squares over the FULL overlap graph, so the
       loop-closure inconsistency is distributed across all overlaps rather than
       concentrated on a few edges (no spanning-tree collapse -- see module docstring).
    2. Conservatively reject only genuine BLUNDERS: take the single worst-disagreeing
       edge whose residual exceeds max(abs_thresh, factor*median, _BLUNDER_FLOOR_PX),
       remove it ONLY if doing so does not strand a tile (never drop a bridge), and
       re-solve. Repeat; if iterative=False, at most one removal. The floor is
       deliberately high (_BLUNDER_FLOOR_PX): on a real loopy graph the interior solves
       to a sub-pixel median, so a low threshold would reject normal distributed
       residuals and re-concentrate loop-closure error onto real seams. Moderate
       disagreement is kept and averaged, not rejected.

    An edge that is the sole link to a sub-cluster is never dropped (it is kept,
    down-weighted by its score in the solve). A tile with no registered edge at all
    is left for the caller's affine/stage-model fallback
    (TileFusion._place_unconstrained_tiles_with_affine).

    Parameters
    ----------
    edges : list of dict
        Pairwise edge data with 'i', 'j', 't' (2D offset), 'w' (weight) keys.
    n_tiles : int
        Total number of tiles.
    fixed_indices : list of int
        Tiles fixed at origin (the solve anchor).
    rel_thresh : float
        Relative-outlier multiple of the median residual. Values <= 1 are unsafe on
        the full graph (they over-reject), so they fall back to _REL_OUTLIER_FACTOR.
    abs_thresh : float
        Caller's absolute residual floor (px). The effective floor is
        max(abs_thresh, _BLUNDER_FLOOR_PX), so a small caller value cannot make the
        rejection over-aggressive; only a caller value above the blunder floor tightens it.
    iterative : bool
        If True, repeat reject + re-solve until convergence; else at most one removal.

    Returns
    -------
    shifts : ndarray of shape (n_tiles, 2)
        Optimized shifts.
    """
    work = list(edges)
    if not work:
        return np.zeros((n_tiles, 2), dtype=np.float64)

    # Connectivity is informational only -- tiles with no edges are placed by the
    # caller's affine fallback. Warn so a fragmented acquisition is visible.
    components = _check_connectivity(work, n_tiles)
    if len(components) > 1:
        sizes = sorted((len(c) for c in components), reverse=True)
        logger.warning(
            "Tile graph has %d disconnected components (%d tiles not connected to "
            "the anchor; placed by the affine fallback).",
            len(components),
            sum(sizes[1:]),
        )
        print(
            f"WARNING: {len(components)} disconnected tile groups detected "
            f"({sum(sizes[1:])} tiles placed by the affine fallback). "
            f"Component sizes: {sizes}"
        )

    # rel_thresh below 1 is the unsafe spanning-tree-regime tuning; clamp to a safe factor.
    factor = rel_thresh if (rel_thresh and rel_thresh > 1.0) else _REL_OUTLIER_FACTOR

    def residuals(ls: List[Dict[str, Any]], sh: np.ndarray) -> np.ndarray:
        return np.array([np.linalg.norm(sh[e["j"]] - sh[e["i"]] - e["t"]) for e in ls])

    shifts = solve_least_squares(work, n_tiles, fixed_indices)

    # At most one edge removed per pass; bounded by the edge count.
    for _ in range(len(work)):
        res = residuals(work, shifts)
        if len(res) == 0:
            break
        cutoff = max(abs_thresh, factor * float(np.median(res)), _BLUNDER_FLOOR_PX)
        candidates = [int(k) for k in np.argsort(res)[::-1] if res[k] > cutoff]
        if not candidates:
            break  # converged: no relative outlier remains

        base_n = _num_components(work, n_tiles)
        removed = False
        for k in candidates:  # worst first; skip any whose removal would strand a tile
            trial = work[:k] + work[k + 1 :]
            if _num_components(trial, n_tiles) <= base_n:
                work.pop(k)  # redundant (loop) edge -- safe to drop
                removed = True
                break
        if not removed:
            break  # every remaining outlier is a sole bridge; keep them, do not strand

        shifts = solve_least_squares(work, n_tiles, fixed_indices)
        if not iterative:
            break

    return shifts


def _edges_from_pairwise_metrics(
    pairwise_metrics: Dict[Tuple[int, int], Tuple[int, int, float]],
) -> List[Dict[str, Any]]:
    """
    Convert pairwise_metrics dict to list of edge dicts.

    Parameters
    ----------
    pairwise_metrics : dict
        Keys are (i, j) tuples, values are (dy, dx, score) tuples.

    Returns
    -------
    edges : list of dict
        Each dict has 'i', 'j', 't', 'w' keys.
    """
    edges = []
    for (i, j), v in pairwise_metrics.items():
        edges.append(
            {
                "i": i,
                "j": j,
                "t": np.array(v[:2], dtype=np.float64),
                "w": np.sqrt(v[2]),
            }
        )
    return edges


def fit_stage_to_image_transform(pairwise_metrics, tile_positions, pixel_size):
    """Fit the global stage->image linear map from registered pairs.

    The reported stage positions map to image pixels by a transform set by the
    instrument (camera scale x magnification, and the sensor-vs-stage mounting
    angle) -- a single 2x2 map shared by every tile, not a per-tile offset. The
    pipeline's default model assumes a pure isotropic scale (1/pixel_size); when
    the real map also has a rotation or a slightly different scale, that error is
    systematic and grows with tile separation.

    This fits the actual 2x2 map M (pixels = M @ stage_mm) by weighted least
    squares over the registered pairs' measured relative displacements (reported
    offset + recovered residual). Because displacements are relative, translation
    cancels, so only the linear part is fit. The map then places EVERY tile --
    including ones whose overlaps were too low-texture to register -- because the
    map is a property of the instrument, not of any individual overlap.

    Parameters
    ----------
    pairwise_metrics : dict
        {(i, j): (dy, dx, score)} -- the registered relative residual shifts (px).
    tile_positions : list of (y, x)
        Reported stage positions (physical units, e.g. mm).
    pixel_size : (py, px)
        The current isotropic pixel size used to seed the reported offsets.

    Returns
    -------
    dict with:
        M : ndarray (2, 2)        the fitted stage->pixel linear map
        scale : float             mean singular value (px per stage-unit)
        rotation_deg : float      rotation component of M, in degrees
        anisotropy : float        ratio of singular values (1.0 = isotropic)
        residual_rms : float      RMS of the per-pair residual after the fit (px)
        n_pairs : int
    """
    pos = np.asarray(tile_positions, dtype=np.float64)
    ps = np.asarray(pixel_size, dtype=np.float64)
    S, P, W = [], [], []
    for (i, j), (dy, dx, score) in pairwise_metrics.items():
        stage_disp = pos[j] - pos[i]
        measured = stage_disp / ps + np.array([dy, dx], dtype=np.float64)
        # Skip non-finite rows (zero/NaN pixel size, NaN registration score/shift); a
        # single one makes LAPACK reject the whole lstsq ("DLASCL illegal value").
        if not (
            np.all(np.isfinite(stage_disp)) and np.all(np.isfinite(measured)) and np.isfinite(score)
        ):
            continue
        S.append(stage_disp)
        P.append(measured)
        W.append(np.sqrt(max(float(score), 1e-6)))
    identity = {
        "M": np.eye(2),
        "scale": 1.0,
        "rotation_deg": 0.0,
        "anisotropy": 1.0,
        "residual_rms": float("nan"),
        "n_pairs": len(S),
    }
    if len(S) < 2:
        # Too few usable pairs to fit. Return an identity stage->image map rather than
        # raising: the only caller places DISCONNECTED tiles with this, and a sparse or
        # degenerate graph (e.g. mostly-isolated tiles) must not crash the whole run.
        logger.warning("fit_stage_to_image_transform: <2 finite pairs; using identity map")
        return identity
    S = np.asarray(S)
    P = np.asarray(P)
    w = np.asarray(W)[:, None]
    # Weighted least squares: (w*P) = (w*S) @ Mt
    try:
        Mt, *_ = np.linalg.lstsq(w * S, w * P, rcond=None)
        if not np.all(np.isfinite(Mt)):
            raise np.linalg.LinAlgError("non-finite affine solution")
    except np.linalg.LinAlgError as e:
        logger.warning("fit_stage_to_image_transform: lstsq failed (%s); using identity map", e)
        return identity
    M = Mt.T
    resid = P - S @ Mt
    U, sv, Vt = np.linalg.svd(M)
    R = U @ Vt
    return {
        "M": M,
        "scale": float(sv.mean()),
        "rotation_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
        "anisotropy": float(sv[0] / sv[1]),
        "residual_rms": float(np.sqrt((resid**2).sum(axis=1)).mean()),
        "n_pairs": len(S),
    }
