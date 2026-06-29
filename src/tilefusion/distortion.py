"""
Per-seam elastic distortion correction (self-calibrated, no machine learning).

Registration + global optimization place tiles by a single translation each. That
leaves a residual that VARIES along each seam -- optical field distortion (a bow),
plus any smooth local rotation/shear -- which one translation per tile cannot
represent. This module measures that residual per seam and corrects it.

For each registered overlap we:
  1. block-register the overlap (phase correlation on sub-blocks along the seam) to
     get the local shift at points ALONG the seam,
  2. choose the polynomial order (1=linear/rotation, 2=quadratic/bow, 3=cubic) by
     leave-one-out cross-validation -- the data picks the shape, no ML, no overfit,
  3. accumulate a per-tile displacement field, split symmetrically between the two
     tiles (each moves half-way) and feathered to be full at the overlap midline and
     zero into each tile interior, so seams correct without disturbing interiors.

Self-calibrating and safe by construction:
  - no distortion -> flat block-shifts -> ~zero field (identity, no change);
  - too few textured blocks / can't fit -> that seam is skipped (no-op);
  - any error -> caught, that seam contributes nothing.
So the worst case for any tile is the identity transform = today's translation-only
result. The correction can only *earn* a warp from the data.

Applied at fusion time (the tile reader warps each tile by its field before blending).
"""

import logging
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import shift as ndi_shift

from .utils import limit_blas_threads, phase_cross_correlation

logger = logging.getLogger(__name__)

# Per-block correspondence trust gate. 0.3-0.5 is the established "weak but real"
# normalized-cross-correlation band (cf. MIST, Chalfoun 2017); deliberately permissive
# because the seam-level leave-one-out CV is the real filter. Exposed as a kwarg.
_NCC_MIN = 0.4
# Max polynomial order. Cubic is the standard ceiling for smooth optical field
# distortion: Brown-Conrady radial (r^2, r^4) projects to low order along a 1-D seam
# cut, and higher orders ring. CV chooses 1.._MAX_DEG.
_MAX_DEG = 3
# Sub-divisions along each seam (~3x oversampling for a cubic). The effective count is
# further gated by block size (_MIN_BLOCK_WIDTH) and texture (_MIN_STD).
_N_BLOCKS = 12
# Minimum good blocks to fit a seam. Tied to _MAX_DEG: leave-one-out CV at the top
# order needs n >= d+3 (fit d+1 points, hold one out, keep a margin). Bumping _MAX_DEG
# without this coupling would make the top order un-evaluable.
_MIN_BLOCKS = _MAX_DEG + 3
# Only escalate to a higher polynomial order if held-out RMSE improves by more than
# this (parsimony, cf. the cross-validation one-standard-error rule). ~3x the sub-pixel
# registration noise floor (phase correlation at _UPSAMPLE=10 gives ~0.1px).
_ORDER_ESCALATION_MARGIN_PX = 0.3
# If the fitted field is everywhere below this, treat the seam as identity (a sub-pixel
# correction is within blend noise -- not worth a warp). The "earn the warp" floor.
_MIN_CORRECTION_PX = 1.0
# Minimum block width in px along the seam (FFT phase correlation needs enough samples).
# Also sets the minimum usable overlap width (_N_BLOCKS * this). One source of truth.
_MIN_BLOCK_WIDTH = 16
# Minimum block area (~17x17 px) for a meaningful phase correlation.
_MIN_BLOCK_PX = 300
# Below this intensity std a block is effectively flat -> no usable correspondence.
_MIN_STD = 1e-3
# Crop this many px off each side before the post-shift NCC check, to avoid the
# edge artifacts ndi_shift(mode="nearest") introduces.
_NCC_EDGE_MARGIN = 6
# skimage sub-pixel upsample factor -> ~0.1px precision (the conventional choice).
_UPSAMPLE = 10


def _loocv_order(pos: np.ndarray, s: np.ndarray, max_deg: int):
    """Pick polynomial order by leave-one-out CV; return (deg, cv_error) or None."""
    n = len(pos)
    best = None
    for d in range(1, max_deg + 1):
        if n < d + 3:  # need enough points to fit AND hold one out
            continue
        errs = []
        for k in range(n):
            tr = [t for t in range(n) if t != k]
            errs.append(np.polyval(np.polyfit(pos[tr], s[tr], d), pos[k]) - s[k])
        cv = float(np.sqrt(np.mean(np.square(errs))))
        # Greedy parsimony: accept a higher order only if it beats the current best by
        # more than the margin (so noise-chasing wiggles don't win).
        if best is None or cv < best[1] - _ORDER_ESCALATION_MARGIN_PX:
            best = (d, cv)
    return best


def _block_shifts(strip_i: np.ndarray, strip_j: np.ndarray, n_blocks: int, ncc_min: float):
    """Sub-block phase-correlation along the long axis of two aligned overlap strips.
    Returns (pos, sy, sx) for blocks that lock above ncc_min."""
    h, w = strip_i.shape
    longx = w >= h
    L = w if longx else h
    B = L // n_blocks
    if B < _MIN_BLOCK_WIDTH:
        return None
    pos, sy, sx = [], [], []
    for k in range(n_blocks):
        if longx:
            a = strip_i[:, k * B : (k + 1) * B]
            b = strip_j[:, k * B : (k + 1) * B]
        else:
            a = strip_i[k * B : (k + 1) * B, :]
            b = strip_j[k * B : (k + 1) * B, :]
        if a.size < _MIN_BLOCK_PX or a.std() < _MIN_STD or b.std() < _MIN_STD:
            continue
        try:
            sh, _, _ = phase_cross_correlation(
                a, b, upsample_factor=_UPSAMPLE, normalization="phase"
            )
        except Exception as e:
            logger.debug("distortion: block %d phase-corr failed: %s", k, e)
            continue
        bb = ndi_shift(b, sh, order=1, mode="nearest")
        m = _NCC_EDGE_MARGIN
        if min(a.shape) <= 2 * m:
            continue
        av = a[m:-m, m:-m].ravel()
        bv = bb[m:-m, m:-m].ravel()
        if av.std() < 1e-6 or bv.std() < 1e-6:  # 1e-6 = divide-by-zero guard for corrcoef
            continue
        corr = np.corrcoef(av, bv)[0, 1]
        if not (corr >= ncc_min):  # NaN-safe: a NaN correlation is rejected
            continue
        pos.append(k * B + B / 2.0)
        sy.append(float(sh[0]))
        sx.append(float(sh[1]))
    if len(pos) < _MIN_BLOCKS:
        return None
    return np.array(pos), np.array(sy), np.array(sx)


def _fit_seam(tf, i, j, ps, pos, Y, X, n_blocks, ncc_min):
    """Fit one seam's elastic correction in isolation.

    Returns (deg_y, deg_x, base) where base is the per-seam geometry + polynomial
    coeffs, or None if the seam can't be measured/fit. Thread-safe: only thread-local
    tile reads and stateless numpy/scipy, so seams fit concurrently. deg_y/deg_x are
    the CV-chosen orders (returned for the field-of-view parameter diagnostic).
    """
    try:
        rel = (pos[j] - pos[i]) / ps
        dy, dx = int(round(rel[0])), int(round(rel[1]))
        vert = abs(dy) >= abs(dx)
        if vert:
            if not (0 < dy < Y):
                return None
            od = Y - dy
            ow = X - abs(dx)
            xa = max(dx, 0)
            xb = max(-dx, 0)
            if ow < n_blocks * _MIN_BLOCK_WIDTH:
                return None
            si = np.asarray(tf._read_tile_region(i, slice(dy, Y), slice(xa, xa + ow)))
            sj = np.asarray(tf._read_tile_region(j, slice(0, od), slice(xb, xb + ow)))
        else:
            if not (0 < dx < X):
                return None
            od = X - dx
            oh = Y - abs(dy)
            ya = max(dy, 0)
            yb = max(-dy, 0)
            if oh < n_blocks * _MIN_BLOCK_WIDTH:
                return None
            si = np.asarray(tf._read_tile_region(i, slice(ya, ya + oh), slice(dx, X)))
            sj = np.asarray(tf._read_tile_region(j, slice(yb, yb + oh), slice(0, od)))
        while si.ndim > 2:
            si = si[0]
        while sj.ndim > 2:
            sj = sj[0]
        si = si.astype(np.float32)
        sj = sj.astype(np.float32)
        mh = min(si.shape[0], sj.shape[0])
        mw = min(si.shape[1], sj.shape[1])
        si = si[:mh, :mw]
        sj = sj[:mh, :mw]
        bs = _block_shifts(si, sj, n_blocks, ncc_min)
        if bs is None:
            return None
        ppos, sY, sX = bs
        by = _loocv_order(ppos, sY, _MAX_DEG)
        bx = _loocv_order(ppos, sX, _MAX_DEG)
        if by is None or bx is None:
            return None
        cy = np.polyfit(ppos, sY, by[0])
        cx = np.polyfit(ppos, sX, bx[0])
        # pmin/pmax bound the along-seam range the fit was sampled over; materialize_field
        # clamps to it so a cubic is never extrapolated across the un-sampled tile width.
        base = {
            "vert": vert,
            "od": float(od),
            "dy": dy,
            "dx": dx,
            "cy": cy,
            "cx": cx,
            "pmin": float(ppos.min()),
            "pmax": float(ppos.max()),
        }
        return (by[0], bx[0], base)
    except Exception as e:
        logger.debug("distortion: seam (%d,%d) skipped: %s", i, j, e)
        return None


def build_seam_corrections(tf, n_blocks=_N_BLOCKS, ncc_min=_NCC_MIN):
    """Build per-tile elastic corrections from the registered seams (parallel).

    Uses the OPTIMIZED tile positions (so it corrects the residual AFTER the global
    solve) and RAW tile reads (so the measurement isn't polluted). Each seam is fit
    independently via _fit_seam, so the fits run concurrently (reads use thread-local
    handles; phase correlation is stateless). Returns {tile_idx: [correction dicts]}
    -- absent tiles get no warp (identity).
    """
    ps = np.asarray(tf._pixel_size, float)
    pos = np.asarray(tf._tile_positions, float)
    Y, X = tf.Y, tf.X
    pairs = list(tf.pairwise_metrics.keys())
    workers = max(1, int(getattr(tf, "max_workers", 8)))
    # Each worker block-registers a seam (phase-correlation) and fits a polynomial
    # (numpy linalg). Pin BLAS to 1 thread per worker so W workers don't each spawn a
    # full BLAS pool and oversubscribe the CPU.
    with limit_blas_threads(1), ThreadPoolExecutor(max_workers=workers) as ex:
        fits = list(
            ex.map(
                lambda p: (p, _fit_seam(tf, p[0], p[1], ps, pos, Y, X, n_blocks, ncc_min)),
                pairs,
            )
        )
    corr = {}
    n_fit = 0
    for (i, j), res in fits:
        if res is None:
            continue
        _, _, base = res
        corr.setdefault(i, []).append({**base, "sign": +0.5, "side": "i"})
        corr.setdefault(j, []).append({**base, "sign": -0.5, "side": "j"})
        n_fit += 1
    print(
        f"Distortion correction: fit {n_fit} seams; {len(corr)} tiles get a field "
        f"(others identity)."
    )
    return corr


def materialize_field(corrs, Y, X):
    """Build the (2, Y, X) displacement field (dy, dx) for one tile from its seam
    corrections. Returns None if the net field is negligible (identity)."""
    D = np.zeros((2, Y, X), np.float32)
    cols = np.arange(X, dtype=np.float32)
    rows = np.arange(Y, dtype=np.float32)
    for c in corrs:
        od = c["od"]
        sgn = c["sign"]
        # Clamp the along-seam eval coordinate to the sampled range so the polynomial is
        # held constant (not extrapolated) over the un-sampled strip edges. pmin/pmax are
        # absent only for hand-built test dicts -> no clamp there.
        pmin = c.get("pmin")
        pmax = c.get("pmax")
        if c["vert"]:
            xoff = max(c["dx"], 0) if c["side"] == "i" else max(-c["dx"], 0)
            p = cols - xoff
            if pmin is not None:
                p = np.clip(p, pmin, pmax)
            sY = np.polyval(c["cy"], p)
            sX = np.polyval(c["cx"], p)  # (X,)
            if c["side"] == "i":
                f = np.clip((rows - c["dy"]) / (od / 2.0), 0, 1)  # (Y,)
            else:
                f = np.clip((od - rows) / (od / 2.0), 0, 1)
            D[0] += (sgn * sY)[None, :] * f[:, None]
            D[1] += (sgn * sX)[None, :] * f[:, None]
        else:
            yoff = max(c["dy"], 0) if c["side"] == "i" else max(-c["dy"], 0)
            p = rows - yoff
            if pmin is not None:
                p = np.clip(p, pmin, pmax)
            sY = np.polyval(c["cy"], p)
            sX = np.polyval(c["cx"], p)  # (Y,)
            if c["side"] == "i":
                f = np.clip((cols - c["dx"]) / (od / 2.0), 0, 1)  # (X,)
            else:
                f = np.clip((od - cols) / (od / 2.0), 0, 1)
            D[0] += (sgn * sY)[:, None] * f[None, :]
            D[1] += (sgn * sX)[:, None] * f[None, :]
    if np.abs(D).max() < _MIN_CORRECTION_PX:
        return None
    return D


class TileWarper:
    """Provides per-tile elastic displacement fields to the fusion sampler.

    The field is GEOMETRIC (independent of z/time/pixel content), so the warp is
    applied inside the fusion blend (accumulate_tile_shard_distorted) rather than by
    resampling the whole tile in a separate pass. We materialize each (2, Y, X) field
    lazily and keep a small LRU cache. A field is (2, Y, X) float32 (~tens to a couple
    hundred MB for large tiles), so cache_size=8 caps the cache at a 2x2-tile
    neighborhood with reuse while bounding peak memory; the lookup is guarded by a lock
    so the cache stays consistent even if the fusion sampler is ever called from
    multiple threads (tile reads already use thread-local handles elsewhere).
    """

    def __init__(self, corrections, Y, X, cache_size=8):
        self.corr = corrections or {}
        self.Y, self.X = Y, X
        self._cache = OrderedDict()
        self._cap = cache_size
        self._lock = threading.Lock()

    def field(self, tile_idx):
        """Return the (2, Y, X) float32 displacement field for this tile, or None if
        it has no correction (identity). Cached by tile index (thread-safe LRU)."""
        if tile_idx not in self.corr:
            return None
        with self._lock:
            if tile_idx in self._cache:
                self._cache.move_to_end(tile_idx)
                return self._cache[tile_idx]
            try:
                D = materialize_field(self.corr[tile_idx], self.Y, self.X)
            except Exception as e:
                logger.debug("distortion: materialize failed for tile %d: %s", tile_idx, e)
                D = None
            self._cache[tile_idx] = D
            if len(self._cache) > self._cap:
                self._cache.popitem(last=False)
            return D
