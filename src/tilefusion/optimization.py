"""
Global position optimization.

Least-squares optimization of tile positions from pairwise measurements.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

# Threshold for switching between dense and sparse solvers.
# Below this, dense lstsq is faster; above, sparse LSQR wins.
_SPARSE_THRESHOLD = 100


def _solve_dense(links: List[Dict[str, Any]], n_tiles: int, fixed_indices: List[int]) -> np.ndarray:
    """Dense solver using numpy lstsq (better for small problems)."""
    shifts = np.zeros((n_tiles, 2), dtype=np.float64)
    m = len(links) + len(fixed_indices)
    for axis in range(2):
        A = np.zeros((m, n_tiles), dtype=np.float64)
        b = np.zeros(m, dtype=np.float64)
        for row, link in enumerate(links):
            w = link["w"]
            A[row, link["j"]] = w
            A[row, link["i"]] = -w
            b[row] = w * link["t"][axis]
        for k, idx in enumerate(fixed_indices):
            A[len(links) + k, idx] = 1.0
        shifts[:, axis] = np.linalg.lstsq(A, b, rcond=None)[0]
    return shifts


def _solve_sparse(links: List[Dict[str, Any]], n_tiles: int, fixed_indices: List[int]) -> np.ndarray:
    """Sparse solver using scipy LSQR (better for large problems)."""
    n_links = len(links)
    n_fixed = len(fixed_indices)
    m = n_links + n_fixed

    row_idx = np.empty(2 * n_links + n_fixed, dtype=np.int32)
    col_idx = np.empty(2 * n_links + n_fixed, dtype=np.int32)
    data = np.empty(2 * n_links + n_fixed, dtype=np.float64)

    for k, link in enumerate(links):
        w = link["w"]
        row_idx[2 * k] = k
        col_idx[2 * k] = link["j"]
        data[2 * k] = w
        row_idx[2 * k + 1] = k
        col_idx[2 * k + 1] = link["i"]
        data[2 * k + 1] = -w

    base = 2 * n_links
    for k, idx in enumerate(fixed_indices):
        row_idx[base + k] = n_links + k
        col_idx[base + k] = idx
        data[base + k] = 1.0

    A = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(m, n_tiles))

    b = np.zeros((m, 2), dtype=np.float64)
    for k, link in enumerate(links):
        b[k, 0] = link["w"] * link["t"][0]
        b[k, 1] = link["w"] * link["t"][1]

    shifts = np.zeros((n_tiles, 2), dtype=np.float64)
    shifts[:, 0] = lsqr(A, b[:, 0], atol=1e-10, btol=1e-10)[0]
    shifts[:, 1] = lsqr(A, b[:, 1], atol=1e-10, btol=1e-10)[0]
    return shifts


def solve_global(links: List[Dict[str, Any]], n_tiles: int, fixed_indices: List[int]) -> np.ndarray:
    """
    Solve a linear least-squares for all 2 axes at once,
    given weighted pairwise links and fixed tile indices.

    Uses dense solver for small problems (<100 tiles) and sparse LSQR
    for larger problems where memory and compute savings are significant.

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
    if not links:
        return np.zeros((n_tiles, 2), dtype=np.float64)

    if n_tiles < _SPARSE_THRESHOLD:
        return _solve_dense(links, n_tiles, fixed_indices)
    return _solve_sparse(links, n_tiles, fixed_indices)


def two_round_optimization(
    links: List[Dict[str, Any]],
    n_tiles: int,
    fixed_indices: List[int],
    rel_thresh: float,
    abs_thresh: float,
    iterative: bool,
) -> np.ndarray:
    """
    Perform two-round (or iterative two-round) robust optimization.

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
    shifts = solve_global(links, n_tiles, fixed_indices)

    def compute_res(ls: List[Dict[str, Any]], sh: np.ndarray) -> np.ndarray:
        return np.array([np.linalg.norm(sh[l["j"]] - sh[l["i"]] - l["t"]) for l in ls])

    work = links.copy()
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
    """Convert pairwise_metrics dict to list of link dicts."""
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
