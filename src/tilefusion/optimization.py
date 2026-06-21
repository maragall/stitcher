"""
Global position optimization.

Least-squares optimization of tile positions from pairwise measurements.
Uses minimum spanning tree (MST) to select the most reliable edges
before optimization, reducing noise from redundant/bad edges.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _build_mst(edges: List[Dict[str, Any]], n_tiles: int) -> List[Dict[str, Any]]:
    """
    Select edges forming a minimum spanning tree (maximum-weight spanning tree,
    since higher SSIM weight = more reliable).

    Uses Kruskal's algorithm on the negated weights.

    Parameters
    ----------
    edges : list of dict
        All available edges with 'i', 'j', 'w' keys.
    n_tiles : int
        Total number of tiles.

    Returns
    -------
    mst_edges : list of dict
        Subset of edges forming the MST.
    """
    if not edges:
        return []

    # Sort by weight descending (we want maximum spanning tree)
    sorted_edges = sorted(edges, key=lambda e: e["w"], reverse=True)

    # Union-Find for Kruskal's
    parent = list(range(n_tiles))
    rank = [0] * n_tiles

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    mst_edges = []
    for edge in sorted_edges:
        if union(edge["i"], edge["j"]):
            mst_edges.append(edge)
        if len(mst_edges) == n_tiles - 1:
            break

    return mst_edges


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


def solve_least_squares(edges: List[Dict[str, Any]], n_tiles: int, fixed_indices: List[int]) -> np.ndarray:
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
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        shifts[:, axis] = sol
    return shifts


def two_round_optimization(
    edges: List[Dict[str, Any]],
    n_tiles: int,
    fixed_indices: List[int],
    rel_thresh: float,
    abs_thresh: float,
    iterative: bool,
) -> np.ndarray:
    """
    Perform two-round (or iterative two-round) robust optimization:
    1. Select MST edges for robustness (fewer, higher-quality edges).
    2. Solve on MST edges.
    3. Remove any edge whose residual > max(abs_thresh, rel_thresh * median(residuals)).
    4. Re-solve on the remaining edges.
    If iterative=True, repeat step 3 + 4 until no more edges are removed.

    Also checks for disconnected components and warns the user.

    Parameters
    ----------
    edges : list of dict
        Pairwise edge data.
    n_tiles : int
        Total number of tiles.
    fixed_indices : list of int
        Tiles to fix at origin.
    rel_thresh : float
        Relative threshold (fraction of median residual).
    abs_thresh : float
        Absolute threshold for residual.
    iterative : bool
        If True, iterate until convergence.

    Returns
    -------
    shifts : ndarray of shape (n_tiles, 2)
        Optimized shifts.
    """
    # Use MST for initial solve — reduces noise from redundant edges
    mst_edges = _build_mst(edges, n_tiles)

    if len(mst_edges) < len(edges):
        logger.info(
            "MST selected %d of %d edges for optimization", len(mst_edges), len(edges)
        )

    # Check connectivity
    components = _check_connectivity(mst_edges, n_tiles)
    if len(components) > 1:
        sizes = sorted([len(c) for c in components], reverse=True)
        disconnected_tiles = sum(sizes[1:])
        logger.warning(
            "Tile graph has %d disconnected components (%d tiles disconnected). "
            "Disconnected tiles will use stage positions.",
            len(components), disconnected_tiles,
        )
        print(
            f"WARNING: {len(components)} disconnected tile groups detected "
            f"({disconnected_tiles} tiles may be misaligned). "
            f"Component sizes: {sizes}"
        )

    # Solve on MST edges
    work = mst_edges.copy()
    shifts = solve_least_squares(work, n_tiles, fixed_indices)

    def compute_res(ls: List[Dict[str, Any]], sh: np.ndarray) -> np.ndarray:
        return np.array([np.linalg.norm(sh[l["j"]] - sh[l["i"]] - l["t"]) for l in ls])

    res = compute_res(work, shifts)
    if len(res) == 0:
        return shifts
    cutoff = max(abs_thresh, rel_thresh * np.median(res))
    outliers = set(np.where(res > cutoff)[0])

    if iterative:
        while outliers:
            for k in sorted(outliers, reverse=True):
                work.pop(k)
            if not work:
                break
            shifts = solve_least_squares(work, n_tiles, fixed_indices)
            res = compute_res(work, shifts)
            if len(res) == 0:
                break
            cutoff = max(abs_thresh, rel_thresh * np.median(res))
            outliers = set(np.where(res > cutoff)[0])
    else:
        for k in sorted(outliers, reverse=True):
            work.pop(k)
        if work:
            shifts = solve_least_squares(work, n_tiles, fixed_indices)

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
        S.append(stage_disp)
        P.append(measured)
        W.append(np.sqrt(max(score, 1e-6)))
    if len(S) < 2:
        raise ValueError("need >=2 registered pairs to fit the stage->image transform")
    S = np.asarray(S)
    P = np.asarray(P)
    w = np.asarray(W)[:, None]
    # Weighted least squares: (w*P) = (w*S) @ Mt
    Mt, *_ = np.linalg.lstsq(w * S, w * P, rcond=None)
    M = Mt.T
    resid = P - S @ Mt
    U, sv, Vt = np.linalg.svd(M)
    R = U @ Vt
    return {
        "M": M,
        "scale": float(sv.mean()),
        "rotation_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
        "anisotropy": float(sv[0] / sv[1]),
        "residual_rms": float(np.sqrt((resid ** 2).sum(axis=1)).mean()),
        "n_pairs": len(S),
    }
