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
def accumulate_tile_shard_shifted(
    fused: np.ndarray,
    weight: np.ndarray,
    tile: np.ndarray,
    w2d: np.ndarray,
    y_off: int,
    x_off: int,
    sy0: int,
    sx0: int,
    sub_Y: int,
    sub_X: int,
    fy: float,
    fx: float,
) -> None:
    """Weighted accumulation with an on-the-fly bilinear sub-pixel shift.

    Algebraically identical to
        shifted = ndi_shift(tile, (0, fy, fx), order=1, mode="nearest")
        accumulate_tile_shard(fused, weight, shifted[:, sy0:sy0+sub_Y, sx0:sx0+sub_X],
                              w2d, y_off, x_off)
    but interpolates ONLY the pixels actually blended (no full-tile resample pass)
    and reads the original, un-shifted tile -- so the same tile is reused across
    every block it overlaps instead of being re-shifted per block.

    scipy's order-1 spline with mode="nearest" is edge-clamped bilinear
    interpolation, and shift convention is output[p] = input[p - shift], so the
    source coordinate for plane position (sy0+y_i, sx0+x_i) is that minus (fy, fx).
    Passing the FULL tile (not a pre-sliced sub-region) lets sub-region-edge
    pixels read their true in-tile neighbours, matching shift-then-slice exactly.

    Parameters
    ----------
    fused, weight : float32[C, Yp, Xp]   accumulation + weight buffers (block-local)
    tile : float32[C, tY, tX]            the ORIGINAL, un-shifted FOV
    w2d : float32[sub_Y, sub_X]          feather weight for this sub-region
    y_off, x_off : int                   block-local destination origin
    sy0, sx0 : int                       sub-region start in tile coords
    sub_Y, sub_X : int                   sub-region size
    fy, fx : float                       sub-pixel remainder in [0, 1)
    """
    C, Yp, Xp = fused.shape
    _, tY, tX = tile.shape
    total = sub_Y * sub_X

    for idx in prange(total):
        y_i = idx // sub_X
        x_i = idx % sub_X
        gy = y_off + y_i
        gx = x_off + x_i
        if gy < 0 or gy >= Yp or gx < 0 or gx >= Xp:
            continue

        # Source coordinate in the original tile (scipy shift convention).
        src_y = (sy0 + y_i) - fy
        src_x = (sx0 + x_i) - fx
        y0 = int(np.floor(src_y))
        x0 = int(np.floor(src_x))
        wy = src_y - y0
        wx = src_x - x0
        y1 = y0 + 1
        x1 = x0 + 1

        # Nearest-edge clamp (mode="nearest"): out-of-range neighbours fold to the
        # edge pixel, so the interpolation weight simply re-weights the same value.
        if y0 < 0:
            y0 = 0
        elif y0 > tY - 1:
            y0 = tY - 1
        if y1 < 0:
            y1 = 0
        elif y1 > tY - 1:
            y1 = tY - 1
        if x0 < 0:
            x0 = 0
        elif x0 > tX - 1:
            x0 = tX - 1
        if x1 < 0:
            x1 = 0
        elif x1 > tX - 1:
            x1 = tX - 1

        w00 = (1.0 - wy) * (1.0 - wx)
        w01 = (1.0 - wy) * wx
        w10 = wy * (1.0 - wx)
        w11 = wy * wx
        w_val = w2d[y_i, x_i]

        for c in range(C):
            v = (
                w00 * tile[c, y0, x0]
                + w01 * tile[c, y0, x1]
                + w10 * tile[c, y1, x0]
                + w11 * tile[c, y1, x1]
            )
            fused[c, gy, gx] += v * w_val
            weight[c, gy, gx] += w_val


@njit(parallel=True)
def accumulate_tile_shard_distorted(
    fused: np.ndarray,
    weight: np.ndarray,
    tile: np.ndarray,
    w2d: np.ndarray,
    y_off: int,
    x_off: int,
    sy0: int,
    sx0: int,
    sub_Y: int,
    sub_X: int,
    fy: float,
    fx: float,
    Dy: np.ndarray,
    Dx: np.ndarray,
) -> None:
    """As accumulate_tile_shard_shifted, but the source coordinate also carries a
    per-pixel elastic displacement (Dy, Dx are float32[tY, tX] in pixels).

    One bilinear sample folds BOTH the sub-pixel registration remainder (fy, fx) and
    the distortion warp, so there is no separate full-tile warp pass and no double
    interpolation. A warped tile is w[p] = tile[p + D[p]]; sampling it at the
    sub-pixel source p - f reads tile[(p - f) + D], so the source coordinate for plane
    position (sy0+y_i, sx0+x_i) is that minus (fy, fx) plus D at the tile-local pixel.
    D is indexed at the integer tile coordinate (sub-pixel error in D is negligible:
    D varies by ~1px over hundreds of px). Dy/Dx are tile-local, so interiors with
    zero displacement reduce exactly to the sub-pixel-shift behaviour.
    """
    C, Yp, Xp = fused.shape
    _, tY, tX = tile.shape
    total = sub_Y * sub_X

    for idx in prange(total):
        y_i = idx // sub_X
        x_i = idx % sub_X
        gy = y_off + y_i
        gx = x_off + x_i
        if gy < 0 or gy >= Yp or gx < 0 or gx >= Xp:
            continue

        ty = sy0 + y_i
        tx = sx0 + x_i
        src_y = ty - fy + Dy[ty, tx]
        src_x = tx - fx + Dx[ty, tx]
        y0 = int(np.floor(src_y))
        x0 = int(np.floor(src_x))
        wy = src_y - y0
        wx = src_x - x0
        y1 = y0 + 1
        x1 = x0 + 1

        if y0 < 0:
            y0 = 0
        elif y0 > tY - 1:
            y0 = tY - 1
        if y1 < 0:
            y1 = 0
        elif y1 > tY - 1:
            y1 = tY - 1
        if x0 < 0:
            x0 = 0
        elif x0 > tX - 1:
            x0 = tX - 1
        if x1 < 0:
            x1 = 0
        elif x1 > tX - 1:
            x1 = tX - 1

        w00 = (1.0 - wy) * (1.0 - wx)
        w01 = (1.0 - wy) * wx
        w10 = wy * (1.0 - wx)
        w11 = wy * wx
        w_val = w2d[y_i, x_i]

        for c in range(C):
            v = (
                w00 * tile[c, y0, x0]
                + w01 * tile[c, y0, x1]
                + w10 * tile[c, y1, x0]
                + w11 * tile[c, y1, x1]
            )
            fused[c, gy, gx] += v * w_val
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
    get_field: Callable = None,
    progress_callback: Callable = None,
) -> None:
    """Fuse one z/t plane block-by-block at fixed low memory.

    The canonical fusion path. block_size sets the scratchpad size (memory budget):
    a small block_size bounds peak memory; block_size >= max(padded_shape) is the
    whole-plane case. Output is identical regardless of block_size (the full vs.
    chunked equivalence, guarded by test_fuse_equivalence).
    """
    Y, X = tile_shape

    pad_Y, pad_X = padded_shape

    # Origins are fractional (the true registered sub-pixel positions). Split each into
    # an integer floor -- used for all block geometry below -- and a sub-pixel remainder
    # used to fractional-shift the tile content. This honours the sub-pixel registration
    # instead of truncating it to whole pixels (the old int-cast that misaligned seams).
    floor_origins = [(int(np.floor(oy)), int(np.floor(ox))) for (oy, ox) in origins]
    fracs = [(oy - foy, ox - fox) for (oy, ox), (foy, fox) in zip(origins, floor_origins)]

    tile_bounds = [
        (foy, foy + Y, fox, fox + X) for (foy, fox) in floor_origins
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

            # Platform-independent progress for GUIs: an explicit callback per block,
            # rather than scraping tqdm's terminal output (whose live updates depend on
            # TTY detection + stream buffering that differ across OSes).
            if progress_callback is not None:
                progress_callback(block_idx, total_blocks)

            desc = f"block {block_idx}/{total_blocks}"
            iterator = (
                tqdm(overlapping, desc=desc, leave=False) if show_progress else overlapping
            )
            for t_idx in iterator:
                tile_all = read_tile(t_idx, z_level, time_idx)

                # Sub-pixel remainder (Y, X only) honouring the registered position.
                fy, fx = fracs[t_idx]

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

                # Accumulate this FOV's sub-region into the block at (oy0, ox0). The
                # sub-pixel shift is folded into the blend (bilinear sample of the
                # original tile) so we never resample the whole tile per block; a
                # zero remainder takes the no-interp fast path (bit-identical).
                field = get_field(t_idx) if get_field is not None else None
                if field is not None:
                    # Per-seam elastic distortion: fold the per-pixel warp into the
                    # same bilinear sample as the sub-pixel shift (one resample, no
                    # separate warp pass). field is (2, tY, tX) float32 (dy, dx).
                    accumulate_tile_shard_distorted(
                        fused_block, weight_sum, tile_all, w2d, oy0, ox0,
                        sy0, sx0, sy1 - sy0, sx1 - sx0, fy, fx, field[0], field[1],
                    )
                elif fy == 0.0 and fx == 0.0:
                    accumulate_tile_shard(
                        fused_block, weight_sum, tile_all[:, sy0:sy1, sx0:sx1], w2d, oy0, ox0
                    )
                else:
                    accumulate_tile_shard_shifted(
                        fused_block, weight_sum, tile_all, w2d, oy0, ox0,
                        sy0, sx0, sy1 - sy0, sx1 - sx0, fy, fx,
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
