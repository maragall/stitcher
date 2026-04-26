"""
Global position optimization.

Least-squares optimization of tile positions from pairwise measurements.
Uses minimum spanning tree (MST) to select the most reliable links
before optimization, reducing noise from redundant/bad links.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _build_mst_links(links: List[Dict[str, Any]], n_tiles: int) -> List[Dict[str, Any]]:
    """
    Select links forming a minimum spanning tree (maximum-weight spanning tree,
    since higher SSIM weight = more reliable).

    Uses Kruskal's algorithm on the negated weights.

    Parameters
    ----------
    links : list of dict
        All available links with 'i', 'j', 'w' keys.
    n_tiles : int
        Total number of tiles.

    Returns
    -------
    mst_links : list of dict
        Subset of links forming the MST.
    """
    if not links:
        return []

    # Sort by weight descending (we want maximum spanning tree)
    sorted_links = sorted(links, key=lambda l: l["w"], reverse=True)

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

    mst_links = []
    for link in sorted_links:
        if union(link["i"], link["j"]):
            mst_links.append(link)
        if len(mst_links) == n_tiles - 1:
            break

    return mst_links


def _check_connectivity(links: List[Dict[str, Any]], n_tiles: int) -> List[List[int]]:
    """
    Check graph connectivity and return connected components.

    Parameters
    ----------
    links : list of dict
        Links with 'i', 'j' keys.
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

    for link in links:
        union(link["i"], link["j"])

    components = {}
    for i in range(n_tiles):
        root = find(i)
        components.setdefault(root, []).append(i)

    return list(components.values())


def solve_global(links: List[Dict[str, Any]], n_tiles: int, fixed_indices: List[int]) -> np.ndarray:
    """
    Solve a linear least-squares for all 2 axes at once,
    given weighted pairwise links and fixed tile indices.

    Parameters
    ----------
    links : list of dict
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
        m = len(links) + len(fixed_indices)
        A = np.zeros((m, n_tiles), dtype=np.float64)
        b = np.zeros(m, dtype=np.float64)
        row = 0
        for link in links:
            i, j = link["i"], link["j"]
            t, w = link["t"][axis], link["w"]
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
    links: List[Dict[str, Any]],
    n_tiles: int,
    fixed_indices: List[int],
    rel_thresh: float,
    abs_thresh: float,
    iterative: bool,
) -> np.ndarray:
    """
    Perform two-round (or iterative two-round) robust optimization:
    1. Select MST links for robustness (fewer, higher-quality links).
    2. Solve on MST links.
    3. Remove any link whose residual > max(abs_thresh, rel_thresh * median(residuals)).
    4. Re-solve on the remaining links.
    If iterative=True, repeat step 3 + 4 until no more links are removed.

    Also checks for disconnected components and warns the user.

    Parameters
    ----------
    links : list of dict
        Pairwise link data.
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
    # Use MST for initial solve — reduces noise from redundant links
    mst_links = _build_mst_links(links, n_tiles)

    if len(mst_links) < len(links):
        logger.info(
            "MST selected %d of %d links for optimization", len(mst_links), len(links)
        )

    # Check connectivity
    components = _check_connectivity(mst_links, n_tiles)
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

    # Solve on MST links
    work = mst_links.copy()
    shifts = solve_global(work, n_tiles, fixed_indices)

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
            shifts = solve_global(work, n_tiles, fixed_indices)
            res = compute_res(work, shifts)
            if len(res) == 0:
                break
            cutoff = max(abs_thresh, rel_thresh * np.median(res))
            outliers = set(np.where(res > cutoff)[0])
    else:
        for k in sorted(outliers, reverse=True):
            work.pop(k)
        if work:
            shifts = solve_global(work, n_tiles, fixed_indices)

    return shifts


def links_from_pairwise_metrics(
    pairwise_metrics: Dict[Tuple[int, int], Tuple[int, int, float]],
) -> List[Dict[str, Any]]:
    """
    Convert pairwise_metrics dict to list of link dicts.

    Parameters
    ----------
    pairwise_metrics : dict
        Keys are (i, j) tuples, values are (dy, dx, score) tuples.

    Returns
    -------
    links : list of dict
        Each dict has 'i', 'j', 't', 'w' keys.
    """
    links = []
    for (i, j), v in pairwise_metrics.items():
        links.append(
            {
                "i": i,
                "j": j,
                "t": np.array(v[:2], dtype=np.float64),
                "w": np.sqrt(v[2]),
            }
        )
    return links
