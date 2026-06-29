"""
Tile registration algorithms.

Phase cross-correlation based registration with SSIM scoring.
"""

import gc
import logging
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from tqdm import tqdm

from .utils import (
    USING_GPU,
    block_reduce,
    limit_blas_threads,
    match_histograms,
    phase_cross_correlation,
    shift_array,
    to_numpy,
    xp,
    cp,
)

logger = logging.getLogger(__name__)

# Shared phase correlation parameters — used in both parallel and sequential paths
_UPSAMPLE_FACTOR = 20  # 0.05-px subpixel grid. Swept on the golden fixture: 10->20
# nearly halves the worst pairwise error (0.085->0.043 px); beyond 20 hits the noise
# floor with no further gain. Sub-pixel accuracy ALSO requires registering at full
# resolution (downsample_factors=(1,1)); any downsample coarsens the recovered shift.


def register_pair_worker(args: Tuple) -> Tuple:
    """
    Worker function for parallel registration of a tile pair.

    Parameters
    ----------
    args : tuple
        (i_pos, j_pos, patch_i, patch_j, df, sw, max_shift)

    Returns
    -------
    tuple
        (i_pos, j_pos, dy_s, dx_s, score) or (i_pos, j_pos, None, None, None) on failure
    """
    i_pos, j_pos, patch_i, patch_j, df, sw, max_shift = args

    try:
        # Downsample, then run the SHARED kernel (register_and_score) so the batched
        # and read-ahead backends use identical phase-correlation parameters
        # (incl. disambiguate=True). This path previously reimplemented the kernel
        # inline WITHOUT disambiguate -- a silent backend mismatch.
        reduce_block = (1, df[0], df[1]) if patch_i.ndim == 3 else tuple(df)
        g1 = block_reduce(patch_i, reduce_block, np.mean)
        g2 = block_reduce(patch_j, reduce_block, np.mean)

        shift, ssim_val = register_and_score(g1, g2, win_size=sw)
        if shift is None:
            return (i_pos, j_pos, None, None, None)

        # Scale shift back to original resolution (sub-pixel)
        dy_s, dx_s = float(shift[0] * df[0]), float(shift[1] * df[1])

        # Reject shifts exceeding max_shift (likely spurious)
        if abs(dy_s) > max_shift[0] or abs(dx_s) > max_shift[1]:
            logger.debug(
                "Pair (%d, %d): shift (%.2f, %.2f) exceeds max_shift %s — rejected",
                i_pos, j_pos, dy_s, dx_s, max_shift,
            )
            return (i_pos, j_pos, None, None, None)

        # SSIM is the continuous weight used in optimization (no binary gate)
        return (i_pos, j_pos, dy_s, dx_s, round(ssim_val, 3))

    except Exception as e:
        logger.warning("Pair (%d, %d): registration exception — %s", i_pos, j_pos, e)
        return (i_pos, j_pos, None, None, None)


def register_and_score(
    g1: Any,
    g2: Any,
    win_size: int,
    debug: bool = False,
) -> Union[Tuple[Tuple[float, float], float], Tuple[None, None]]:
    """
    Histogram-match g2->g1, compute the subpixel shift, and score the lock by NCC.

    Parameters
    ----------
    g1, g2 : array-like
        Fixed and moving patches (YX).
    win_size : int
        Retained for API compatibility (was the SSIM window); unused now that the
        confidence score is normalized cross-correlation.
    debug : bool
        If True, print intermediate info.

    Returns
    -------
    shift : (dy, dx)
        Subpixel shift.
    score : float
        Normalized cross-correlation of the aligned overlaps (the optimization edge
        weight, via sqrt). High when the lock is real; collapses on a spurious lock.
    """
    arr1 = xp.asarray(g1, dtype=xp.float32)
    arr2 = xp.asarray(g2, dtype=xp.float32)
    while arr1.ndim > 2 and arr1.shape[0] == 1:
        arr1 = arr1[0]
        arr2 = arr2[0]

    arr2 = match_histograms(arr2, arr1)
    shift, _, _ = phase_cross_correlation(
        arr1,
        arr2,
        disambiguate=True,
        normalization="phase",
        upsample_factor=_UPSAMPLE_FACTOR,
    )
    shift_apply = xp.asarray(shift, dtype=xp.float32)
    g2s = shift_array(arr2, shift_vec=shift_apply)
    out_shift = to_numpy(shift_apply)
    # Score (the optimization edge weight) by NORMALIZED CROSS-CORRELATION of the
    # aligned overlaps, NOT SSIM. NCC collapses when the recovered shift is wrong (a
    # spurious low-texture lock), whereas SSIM -- computed on the same strips that
    # produced the shift -- stays high and "blesses" bad locks, letting them pull the
    # global solve out of place (validated: on failed locks NCC~0.16 while SSIM~0.60).
    # Crop a shift-sized margin so the wrapped edge does not bias the correlation; clamp
    # to a tiny positive so the weight (sqrt of the score) is well-defined and a bad
    # lock is heavily down-weighted rather than hard-zeroed.
    a = to_numpy(arr1)
    b = to_numpy(g2s)
    mgn = max(1, min(min(a.shape) // 4,
                     int(np.ceil(abs(out_shift[0]) + abs(out_shift[1]))) + 3))
    if a.shape[0] > 2 * mgn and a.shape[1] > 2 * mgn:
        av = a[mgn:-mgn, mgn:-mgn].ravel()
        bv = b[mgn:-mgn, mgn:-mgn].ravel()
    else:
        av, bv = a.ravel(), b.ravel()
    ncc = 0.0 if (av.std() < 1e-6 or bv.std() < 1e-6) else float(np.corrcoef(av, bv)[0, 1])
    return tuple(float(s) for s in out_shift), max(ncc, 1e-6)


def find_adjacent_pairs(tile_positions, pixel_size, tile_shape, min_overlap=15):
    """
    Find adjacent tile pairs for registration.

    Parameters
    ----------
    tile_positions : list of (y, x) tuples
        Stage positions for each tile.
    pixel_size : tuple of (py, px)
        Pixel size in physical units.
    tile_shape : tuple of (Y, X)
        Tile dimensions in pixels.
    min_overlap : int
        Minimum overlap in pixels.

    Returns
    -------
    adjacent_pairs : list of tuples
        Each tuple: (i_pos, j_pos, dy, dx, overlap_y, overlap_x)
    """
    n_pos = len(tile_positions)
    Y, X = tile_shape
    adjacent_pairs = []

    for i_pos in range(n_pos):
        for j_pos in range(i_pos + 1, n_pos):
            phys = np.array(tile_positions[j_pos]) - np.array(tile_positions[i_pos])
            vox_off = np.round(phys / np.array(pixel_size)).astype(int)
            dy, dx = vox_off

            overlap_y = Y - abs(dy)
            overlap_x = X - abs(dx)

            # Check if tiles are adjacent
            is_horizontal_neighbor = abs(dy) < min_overlap and overlap_x >= min_overlap
            is_vertical_neighbor = abs(dx) < min_overlap and overlap_y >= min_overlap

            if is_horizontal_neighbor or is_vertical_neighbor:
                adjacent_pairs.append((i_pos, j_pos, dy, dx, overlap_y, overlap_x))

    return adjacent_pairs


def rotation_aware_max_shift(adjacent_pairs, floor=100, budget_deg=3.0):
    """Residual-shift rejection cap (px/axis), adaptive to inter-tile spacing.

    A real stage->image rotation theta induces a perpendicular residual of about
    ``tan(theta) * inter_tile_spacing`` at each seam. A fixed cap therefore clips
    legitimate residuals on widely-spaced mosaics once theta grows. Scaling the cap with
    the median spacing lets registration tolerate rotation up to ``budget_deg`` (default
    3 deg, covering the scientist's 1-2 deg spec with margin) without admitting the much
    larger shifts of a genuinely spurious peak. Floored so small-spacing datasets are
    unchanged from the historical 100 px cap.
    """
    if not adjacent_pairs:
        return (floor, floor)
    spacings = [max(abs(p[2]), abs(p[3])) for p in adjacent_pairs]
    cap = max(floor, int(np.tan(np.radians(budget_deg)) * float(np.median(spacings))))
    return (cap, cap)


def compute_pair_bounds(adjacent_pairs, tile_shape):
    """
    Compute overlap bounds for each adjacent pair.

    Parameters
    ----------
    adjacent_pairs : list
        Output from find_adjacent_pairs.
    tile_shape : tuple of (Y, X)
        Tile dimensions.

    Returns
    -------
    pair_bounds : list of tuples
        Each tuple: (i_pos, j_pos, bounds_i_y, bounds_i_x, bounds_j_y, bounds_j_x)
    """
    Y, X = tile_shape
    pair_bounds = []

    for i_pos, j_pos, dy, dx, overlap_y, overlap_x in adjacent_pairs:
        bounds_i_y = (max(0, dy), min(Y, Y + dy))
        bounds_i_x = (max(0, dx), min(X, X + dx))
        bounds_j_y = (max(0, -dy), min(Y, Y - dy))
        bounds_j_x = (max(0, -dx), min(X, X - dx))

        if bounds_i_y[1] > bounds_i_y[0] and bounds_i_x[1] > bounds_i_x[0]:
            pair_bounds.append((i_pos, j_pos, bounds_i_y, bounds_i_x, bounds_j_y, bounds_j_x))

    return pair_bounds


def register_pairs_batched(
    pair_bounds: List[Tuple],
    read_region: Callable,
    df: Tuple[int, int],
    sw: int,
    max_shift: Tuple[int, int],
    max_workers: int,
    *,
    debug: bool = False,
) -> Dict[Tuple[int, int], Tuple[int, int, float]]:
    """Register tile pairs in bounded-memory batches over a CPU compute pool.

    Strips are read and registered in fixed-size batches tied to the compute pool
    (4 * n_workers pairs), so resident strip memory is a constant set by concurrency,
    independent of RAM and of pair count. Pairs are independent, so batching changes
    only when a strip is resident, never the resulting metrics.
    """
    metrics = {}
    n_pairs = len(pair_bounds)
    if n_pairs == 0:
        return metrics
    n_workers = min(cpu_count(), n_pairs, max_workers)
    io_workers = min(n_pairs, max_workers)

    # Resident strips are bounded by the compute pool, not by free RAM or by
    # the dataset size. The factor gives the workers a small read-ahead so
    # they do not stall between batches.
    batch_size = 4 * n_workers
    n_batches = (n_pairs + batch_size - 1) // batch_size
    print(
        f"Parallel registration: {n_pairs} pairs in {n_batches} batches, "
        f"{n_workers} compute workers, {io_workers} I/O workers"
    )

    def read_pair_patches(args):
        i_pos, j_pos, bounds_i_y, bounds_i_x, bounds_j_y, bounds_j_x = args
        try:
            patch_i = read_region(
                i_pos, slice(bounds_i_y[0], bounds_i_y[1]), slice(bounds_i_x[0], bounds_i_x[1])
            )
            patch_j = read_region(
                j_pos, slice(bounds_j_y[0], bounds_j_y[1]), slice(bounds_j_x[0], bounds_j_x[1])
            )
            return (i_pos, j_pos, patch_i, patch_j)
        except Exception as e:
            logger.warning("I/O failed for pair (%d, %d): %s", i_pos, j_pos, e)
            return (i_pos, j_pos, None, None)

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_pairs)
        batch = pair_bounds[start:end]

        with ThreadPoolExecutor(max_workers=io_workers) as io_executor:
            patches = list(io_executor.map(read_pair_patches, batch))

        work_items = [
            (i, j, pi, pj, df, sw, max_shift)
            for i, j, pi, pj in patches
            if pi is not None
        ]

        desc = f"register {batch_idx+1}/{n_batches}"
        # Pin BLAS to 1 thread per worker: each worker calls numpy linalg (corrcoef,
        # etc.), and W workers x ncores BLAS would oversubscribe the CPU. The I/O pool
        # above is pure disk reads (no BLAS), so it is not pinned.
        with limit_blas_threads(1), ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(
                tqdm(
                    executor.map(register_pair_worker, work_items),
                    total=len(work_items),
                    desc=desc,
                    leave=True,
                )
            )

        for i_pos, j_pos, dy_s, dx_s, score in results:
            if dy_s is not None:
                metrics[(i_pos, j_pos)] = (dy_s, dx_s, score)

        del patches, work_items, results
        gc.collect()

    return metrics


def register_pairs_readahead(
    pair_bounds: List[Tuple],
    read_region: Callable,
    df: Tuple[int, int],
    sw: int,
    max_shift: Tuple[int, int],
    *,
    debug: bool = False,
) -> Dict[Tuple[int, int], Tuple[int, int, float]]:
    """Register tile pairs one at a time, reading each pair's two patches concurrently
    via a 2-worker read-ahead pool. Used for the GPU path and small datasets."""
    metrics = {}
    io_executor = ThreadPoolExecutor(max_workers=2)

    for i_pos, j_pos, bounds_i_y, bounds_i_x, bounds_j_y, bounds_j_x in tqdm(
        pair_bounds, desc="register", leave=True
    ):

        def read_patch(idx, y_bounds, x_bounds):
            return read_region(
                idx, slice(y_bounds[0], y_bounds[1]), slice(x_bounds[0], x_bounds[1])
            )

        try:
            future_i = io_executor.submit(read_patch, i_pos, bounds_i_y, bounds_i_x)
            future_j = io_executor.submit(read_patch, j_pos, bounds_j_y, bounds_j_x)
            patch_i = future_i.result()
            patch_j = future_j.result()
        except Exception as e:
            if debug:
                print(f"Error reading patches for ({i_pos}, {j_pos}): {e}")
            continue

        arr_i = xp.asarray(patch_i)
        arr_j = xp.asarray(patch_j)

        reduce_block = (1, df[0], df[1]) if arr_i.ndim == 3 else tuple(df)
        g1 = block_reduce(arr_i, reduce_block, xp.mean)
        g2 = block_reduce(arr_j, reduce_block, xp.mean)

        try:
            shift_ds, ssim_val = register_and_score(g1, g2, win_size=sw, debug=debug)
        except Exception as e:
            logger.warning("Registration failed for (%d, %d): %s", i_pos, j_pos, e)
            continue

        if shift_ds is None:
            continue
        score = float(max(ssim_val, 1e-6))
        # SSIM is used as continuous weight in optimization, not binary gate

        dy_s, dx_s = [float(shift_ds[k] * df[k]) for k in range(2)]

        if abs(dy_s) > max_shift[0] or abs(dx_s) > max_shift[1]:
            if debug:
                print(f"Dropping link {(i_pos, j_pos)} shift=({dy_s}, {dx_s})")
            continue

        metrics[(i_pos, j_pos)] = (dy_s, dx_s, round(score, 3))

    io_executor.shutdown(wait=True)

    return metrics
