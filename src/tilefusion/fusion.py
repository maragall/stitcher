"""
Tile fusion algorithms.

Numba-accelerated weighted blending and accumulation kernels.
"""

import gc
from typing import Callable, Tuple

import numpy as np
from numba import njit, prange
from tqdm import tqdm

from .utils import USING_GPU, cp


@njit(parallel=True)
def accumulate_tile_shard(
    fused: np.ndarray,
    weight: np.ndarray,
    sub: np.ndarray,
    w2d: np.ndarray,
    y_off: int,
    x_off: int,
) -> None:
    """
    Weighted accumulation of a 2D sub-tile into the fused buffer.

    Parameters
    ----------
    fused : float32[C, Y, X]
        Accumulation buffer.
    weight : float32[C, Y, X]
        Weight accumulation buffer.
    sub : float32[C, Y, X]
        Sub-tile to blend.
    w2d : float32[Y, X]
        Weight profile.
    y_off, x_off : int
        Offsets of sub-tile in the fused volume.
    """
    C, Yp, Xp = fused.shape
    _, sub_Y, sub_X = sub.shape
    total = sub_Y * sub_X

    for idx in prange(total):
        y_i = idx // sub_X
        x_i = idx % sub_X
        gy = y_off + y_i
        gx = x_off + x_i
        if gy < 0 or gy >= Yp or gx < 0 or gx >= Xp:
            continue
        w_val = w2d[y_i, x_i]
        for c in range(C):
            fused[c, gy, gx] += sub[c, y_i, x_i] * w_val
            weight[c, gy, gx] += w_val


@njit(parallel=True)
def normalize_shard(fused: np.ndarray, weight: np.ndarray) -> None:
    """
    Normalize the fused buffer by its weight buffer, in-place.

    Parameters
    ----------
    fused : float32[C, Y, X]
        Accumulation buffer to normalize.
    weight : float32[C, Y, X]
        Corresponding weights.
    """
    C, Yp, Xp = fused.shape
    total = C * Yp * Xp

    for idx in prange(total):
        c = idx // (Yp * Xp)
        rem = idx % (Yp * Xp)
        y_i = rem // Xp
        x_i = rem % Xp
        w_val = weight[c, y_i, x_i]
        fused[c, y_i, x_i] = fused[c, y_i, x_i] / w_val if w_val > 0 else 0.0


@njit(parallel=True)
def blend_numba_2d(
    sub_i: np.ndarray,
    sub_j: np.ndarray,
    wy_i: np.ndarray,
    wx_i: np.ndarray,
    wy_j: np.ndarray,
    wx_j: np.ndarray,
    out_f: np.ndarray,
) -> np.ndarray:
    """
    Feather-blend two overlapping 2D sub-tiles.

    Parameters
    ----------
    sub_i, sub_j : (dy, dx) float32
        Input sub-tiles.
    wy_i, wx_i : 1D float32
        Weight profiles for sub_i.
    wy_j, wx_j : 1D float32
        Weight profiles for sub_j.
    out_f : (dy, dx) float32
        Pre-allocated output buffer.

    Returns
    -------
    out_f : (dy, dx) float32
        Blended result.
    """
    dy, dx = sub_i.shape

    for y in prange(dy):
        wi_y = wy_i[y]
        wj_y = wy_j[y]
        for x in range(dx):
            wi = wi_y * wx_i[x]
            wj = wj_y * wx_j[x]
            tot = wi + wj
            if tot > 1e-6:
                out_f[y, x] = (wi * sub_i[y, x] + wj * sub_j[y, x]) / tot
            else:
                out_f[y, x] = sub_i[y, x]
    return out_f


def fuse_plane(
    *,
    read_tile: Callable,
    write_block: Callable,
    origins: list,
    padded_shape: Tuple[int, int],
    tile_shape: Tuple[int, int],
    channels: int,
    y_profile: np.ndarray,
    x_profile: np.ndarray,
    block_size: int,
    z_level: int = 0,
    time_idx: int = 0,
    show_progress: bool = False,
) -> None:
    """Fuse one z/t plane block-by-block at fixed low memory.

    The canonical fusion path. block_size sets the scratchpad size (memory budget):
    a small block_size bounds peak memory; block_size >= max(padded_shape) is the
    whole-plane case. Output is identical regardless of block_size (the full vs.
    chunked equivalence, guarded by test_fuse_equivalence).
    """
    Y, X = tile_shape

    pad_Y, pad_X = padded_shape

    tile_bounds = [
        (oy, oy + Y, ox, ox + X) for (oy, ox) in origins
    ]

    n_blocks_y = (pad_Y + block_size - 1) // block_size
    n_blocks_x = (pad_X + block_size - 1) // block_size
    total_blocks = n_blocks_y * n_blocks_x
    C = channels

    # Only announce block mode when there is actually more than one block; the
    # whole-plane case (block_size >= plane) stays quiet, as it did before the merge.
    if show_progress and total_blocks > 1:
        print(f"Using chunked mode: {block_size}x{block_size} blocks")

    # Reusable per-block accumulators (sized to the largest block); zeroed
    # and sub-viewed per block instead of re-allocated each iteration.
    max_bh = min(block_size, pad_Y)
    max_bw = min(block_size, pad_X)
    fused_buf = np.zeros((C, max_bh, max_bw), dtype=np.float32)
    weight_buf = np.zeros((C, max_bh, max_bw), dtype=np.float32)

    block_idx = 0
    for block_y in range(0, pad_Y, block_size):
        for block_x in range(0, pad_X, block_size):
            block_idx += 1
            by_end = min(block_y + block_size, pad_Y)
            bx_end = min(block_x + block_size, pad_X)
            bh, bw = by_end - block_y, bx_end - block_x

            overlapping = []
            for t_idx, (ty0, ty1, tx0, tx1) in enumerate(tile_bounds):
                if ty1 > block_y and ty0 < by_end and tx1 > block_x and tx0 < bx_end:
                    overlapping.append(t_idx)

            if not overlapping:
                continue

            fused_block = fused_buf[:, :bh, :bw]
            weight_sum = weight_buf[:, :bh, :bw]
            fused_block[...] = 0.0
            weight_sum[...] = 0.0

            desc = f"block {block_idx}/{total_blocks}"
            iterator = (
                tqdm(overlapping, desc=desc, leave=False) if show_progress else overlapping
            )
            for t_idx in iterator:
                tile_all = read_tile(t_idx, z_level, time_idx)

                ty0, ty1, tx0, tx1 = tile_bounds[t_idx]  # this FOV's rectangle on the plane

                # Intersection of FOV and block, expressed two ways:
                # destination box in BLOCK-local coords (where it lands in fused_block)...
                oy0 = max(ty0, block_y) - block_y
                oy1 = min(ty1, by_end) - block_y
                ox0 = max(tx0, block_x) - block_x
                ox1 = min(tx1, bx_end) - block_x

                # ...and the same region in FOV-local coords (which pixels of the FOV to read).
                sy0 = max(block_y - ty0, 0)
                sy1 = sy0 + (oy1 - oy0)
                sx0 = max(block_x - tx0, 0)
                sx1 = sx0 + (ox1 - ox0)

                # Feather weight for exactly this FOV-local sub-region (A1's window, sliced).
                w2d = y_profile[sy0:sy1, None] * x_profile[None, sx0:sx1]

                # Same blend kernel as the whole-plane path: accumulate this FOV's
                # sub-region into the block at its block-local origin (oy0, ox0).
                accumulate_tile_shard(
                    fused_block, weight_sum, tile_all[:, sy0:sy1, sx0:sx1], w2d, oy0, ox0
                )

            # One blend, shared with _fuse_tiles_full_plane: normalize in place,
            # zero where no FOV covered (weight 0). No mask temporary.
            normalize_shard(fused_block, weight_sum)

            # Write to 5D output: (T, C, Z, Y, X)
            write_block(block_y, by_end, block_x, bx_end, fused_block.astype(np.uint16))

    del fused_buf, weight_buf
    gc.collect()
    if USING_GPU and cp is not None:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
