"""
Flatfield correction module.

Retrospective illumination (flatfield) + optional darkfield estimation from the tiles
themselves, plus apply/save/load helpers.

The gain (flatfield) is estimated with BaSiC (Peng et al., Nat. Commun. 2017): a
low-rank + sparse decomposition that separates the multiplicative illumination shading
(low-rank, smooth) from sample content (sparse residual). This is the established
method and, unlike a naive per-pixel median, it does NOT mistake content that is
systematically brighter at the FOV centre (tiles acquired centred on tissue) for
shading -- so it stays accurate on sparse data and from few tiles. The implementation
is a pure numpy/scipy port of PyBaSiC's inexact augmented-Lagrangian solver -- no torch,
no jax, no heavy external solver (those are only BaSiCPy's GPU/autodiff backend, not the
algorithm). The earlier per-pixel-median estimator is retained as
``estimate_flatfield_median`` for comparison/benchmarks.

For pathological cases (degenerate input) the estimator falls back to a unit field
(no correction); a flatfield computed offline can also be supplied via load_flatfield().
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.fftpack import dctn, idctn
from scipy.ndimage import gaussian_filter, zoom

# Retained for backward compatibility: the GUI gates its "Calculate from tiles" button
# on this flag. Flatfield calculation is now built in, so it is always available.
HAS_BASICPY = True


def _shrink(theta: np.ndarray, eps) -> np.ndarray:
    """Soft-thresholding (shrinkage) operator."""
    return np.sign(theta) * np.maximum(np.abs(theta) - eps, 0.0)


def _resize_stack(stack: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    n = stack.shape[0]
    out = np.zeros((n,) + out_hw, np.float32)
    zy, zx = out_hw[0] / stack.shape[1], out_hw[1] / stack.shape[2]
    for i in range(n):
        out[i] = zoom(stack[i].astype(np.float32), (zy, zx), order=1)
    return out


def _resize2d(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    return zoom(img.astype(np.float32),
                (out_hw[0] / img.shape[0], out_hw[1] / img.shape[1]), order=1)


def _inexact_alm_l1(imgs, l_s, l_d, weight, estimate_darkfield,
                    tol=1e-6, max_iter=500, rho=1.5):
    """Inexact augmented-Lagrangian solver for the low-rank + sparse decomposition
    at the heart of BaSiC. Ported from PyBaSiC (itself from the BaSiC MATLAB / robust
    PCA). Returns (Ib, Ir, darkfield) flattened over pixels.

    The flatfield is updated in the DCT (frequency) domain with a shrinkage that
    enforces smoothness; the sparse residual Ir is shrunk in the spatial domain.
    """
    N, P, Q = imgs.shape
    D = np.reshape(imgs, (N, P * Q))
    d_norm = np.linalg.norm(D, "fro")
    W = np.reshape(weight, D.shape)
    B1_uplimit = D.min()
    B1 = 0.0

    Sf = dctn(np.zeros((P, Q), np.float32), norm="ortho")  # flatfield in DCT domain
    Ir = np.zeros_like(D)
    B = np.ones((N, 1))
    D_field = np.zeros((1, P * Q))

    Y = 0.0
    s0 = np.linalg.svd(D, compute_uv=False)[0]
    mu = 12.5 / max(s0, 1e-9)
    mu_bar = mu * 1e7
    ent2 = 10.0

    it = 0
    while it < max_iter:
        S = np.reshape(idctn(Sf, norm="ortho"), (1, P * Q))
        Ib = S * B + D_field
        dS = np.reshape(D - Ib - Ir + Y / mu, (N, P, Q)).mean(axis=0)
        Sf = Sf + dctn(dS, norm="ortho")
        Sf = _shrink(Sf, l_s / mu)

        S = np.reshape(idctn(Sf, norm="ortho"), (1, P * Q))
        Ib = S * B + D_field
        Ir = Ir + (D - Ib - Ir + Y / mu)
        Ir = _shrink(Ir, W / mu)

        R = D - Ir
        rmean = R.mean()
        B = R.mean(axis=1, keepdims=True) / rmean if abs(rmean) > 1e-12 else B
        B[B < 0] = 0

        # Darkfield branch is only entered when the baseline split is well-posed
        # (some images below the mean, finite residual). Guards the empty-slice /
        # divide-by-zero degeneracies seen on very sparse data.
        if estimate_darkfield and (B < 1).sum() >= 1 and rmean > 1e-9:
            valid = B < 1
            highS = S > (S.mean() - 1e-6)
            lowS = S < (S.mean() + 1e-6)
            R_high = np.mean(R * (highS * valid), axis=1, keepdims=True)
            R_low = np.mean(R * (lowS * valid), axis=1, keepdims=True)
            B1 = (R_high - R_low) / rmean
            k = valid.sum()
            t1 = np.sum(B[valid] ** 2); t2 = B[valid].sum(); t3 = B1.sum()
            t4 = np.sum(B[valid] * B1); t5 = t2 * t3 - k * t4
            B1 = 0.0 if t5 == 0 else (t1 * t3 - t2 * t4) / t5
            B1 = np.maximum(B1, 0); B1 = np.minimum(B1, B1_uplimit / max(S.mean(), 1e-9))
            Z = B1 * (S.mean() - S)
            A1 = np.ma.masked_array(R, np.tile(~valid, (1, P * Q))).mean(axis=0, keepdims=True) \
                - B[valid].mean() * S
            A1 = A1 - A1.mean()
            A_off = A1 - A1.mean() - Z
            Dr_f = _shrink(dctn(np.reshape(A_off, (P, Q)), norm="ortho"), l_d / (ent2 * mu))
            Dr = _shrink(idctn(Dr_f, norm="ortho").reshape((1, P * Q)), l_d / (mu * ent2))
            D_field = Dr + Z

        dY = D - Ib - Ir
        Y = Y + mu * dY
        mu = min(mu * rho, mu_bar)
        it += 1
        if np.linalg.norm(dY, "fro") / max(d_norm, 1e-12) < tol:
            break

    D_field = D_field + B1 * S
    return Ib, Ir, D_field


def estimate_flatfield_basic(
    channel_stack: np.ndarray,
    estimate_darkfield: bool = False,
    working_size: int = 128,
    reweight_tol: float = 1e-3,
    max_reweight: int = 10,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """BaSiC flatfield (+ optional darkfield) for ONE channel.

    channel_stack : (n_tiles, Y, X). Returns (flatfield (Y, X), normalized to mean 1.0;
    darkfield (Y, X) or None). Estimation runs at ``working_size`` (downsampled, since
    the field is low-frequency) and the result is upsampled to the tile shape -- this is
    both faithful to BaSiC and fast.

    NOTE: the darkfield branch is EXPERIMENTAL and off by default; for fluorescence the
    additive pedestal is ~0 and it can be unstable. Prefer ``estimate_flatfield_channel``
    which pairs the BaSiC gain with a simple, robust constant darkfield.
    """
    cs = np.asarray(channel_stack, dtype=np.float32)
    if cs.ndim != 3 or cs.shape[0] < 1:
        raise ValueError(f"channel_stack must be (n_tiles, Y, X); got {cs.shape}")
    Y, X = cs.shape[1:]
    ws = (min(working_size, Y), min(working_size, X))
    rs = _resize_stack(cs, ws)

    mean_v = rs.mean(axis=0)
    mv = mean_v.mean()
    if mv <= 1e-9 or not np.isfinite(mv):
        # Degenerate (all-zero / non-finite) input -> no correction.
        return np.ones((Y, X), np.float32), (np.zeros((Y, X), np.float32) if estimate_darkfield else None)
    mean_v = mean_v / mv
    mdct = float(np.abs(dctn(mean_v, norm="ortho")).sum())
    l_s = mdct / 800.0
    l_d = mdct / 2000.0

    img_sort = np.sort(rs, axis=0)
    W = np.ones_like(img_sort)
    flat = np.ones(ws, np.float32)
    dark = np.zeros(ws, np.float32)
    eps = 0.1
    for _ in range(max_reweight):
        last_f, last_d = flat.copy(), dark.copy()
        Ib, Ir, Dfield = _inexact_alm_l1(img_sort, l_s, l_d, W, estimate_darkfield)
        Ib = Ib.reshape(rs.shape)
        Ir = Ir.reshape(rs.shape)
        Dfield = Dfield.reshape(ws)
        W = 1.0 / (np.abs(Ir / (Ib.mean() + 1e-6)) + eps)
        W = W * W.size / W.sum()
        flat = Ib.mean(axis=0) - Dfield
        if not np.isfinite(flat).all() or abs(flat.mean()) < 1e-9:
            flat, dark = last_f, last_d  # degenerate step; keep last good fields
            break
        flat = flat / flat.mean()
        dark = Dfield
        mad_f = np.abs(flat - last_f).sum() / max(np.abs(last_f).sum(), 1e-6)
        mad_d = np.abs(dark - last_d).sum()
        mad_d = 0.0 if mad_d < 1e-7 else mad_d / max(np.abs(last_d).sum(), 1e-6)
        if max(mad_f, mad_d) <= reweight_tol:
            break

    ff = _resize2d(flat, (Y, X))
    fm = ff.mean()
    ff = ff / fm if abs(fm) > 1e-9 else np.ones((Y, X), np.float32)
    ff = ff.astype(np.float32)
    ff[ff <= 1e-6] = 1.0  # guard divide-by-zero at apply time
    df = _resize2d(dark, (Y, X)).astype(np.float32) if estimate_darkfield else None
    return ff, df


def estimate_flatfield_median(
    channel_stack: np.ndarray,
    use_darkfield: bool = False,
    constant_darkfield: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Per-pixel-median flatfield for ONE channel (the simple legacy estimator).

    Retained for comparison/benchmarks. The illumination shading is taken as the
    per-pixel median over tiles (which cancels tile-varying content when content is
    spatially decorrelated), smoothed to the FOV scale, normalized to mean 1.0. On
    content-correlated or sparse data this conflates illumination with content and
    over-states the dome -- which is why production uses BaSiC instead.
    """
    cs = np.asarray(channel_stack, dtype=np.float32)
    tile_shape = cs.shape[1:]
    sigma = max(tile_shape) / 16.0  # illumination varies over the whole FOV

    ff = gaussian_filter(np.median(cs, axis=0), sigma=sigma)
    mean_ff = float(ff.mean())
    ff = ff / mean_ff if mean_ff > 1e-9 else np.ones(tile_shape, dtype=np.float32)
    ff[ff <= 1e-6] = 1.0  # guard divide-by-zero at apply time
    ff = ff.astype(np.float32)

    df = None
    if use_darkfield:
        d = gaussian_filter(np.percentile(cs, 2, axis=0), sigma=sigma)
        df = (np.full(tile_shape, float(np.median(d)), dtype=np.float32)
              if constant_darkfield else d.astype(np.float32))
    return ff, df


def estimate_flatfield_channel(
    channel_stack: np.ndarray,
    use_darkfield: bool = False,
    constant_darkfield: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Retrospective flatfield (+ optional darkfield) for ONE channel -- production path.

    channel_stack : (n_tiles, Y, X). Returns (flatfield (Y, X), normalized to mean 1.0;
    darkfield (Y, X) or None). Shared by calculate_flatfield and the GUI's per-channel
    (low-memory) path so both produce identical fields.

    The gain field comes from BaSiC (low-rank + sparse; robust to sample content and to
    few tiles). The darkfield, when requested, is the simple robust constant pedestal (a
    low percentile across tiles); BaSiC's own darkfield branch is left experimental and
    is not used here.
    """
    cs = np.asarray(channel_stack, dtype=np.float32)
    tile_shape = cs.shape[1:]
    ff, _ = estimate_flatfield_basic(cs, estimate_darkfield=False)

    df = None
    if use_darkfield:
        # Additive-offset proxy: the dimmest consistent level across tiles (a low
        # percentile is more stable than the min), smoothed. True dark-current estimation
        # needs no-light frames; this is a usable retrospective approximation.
        sigma = max(tile_shape) / 16.0
        d = gaussian_filter(np.percentile(cs, 2, axis=0), sigma=sigma)
        df = (np.full(tile_shape, float(np.median(d)), dtype=np.float32)
              if constant_darkfield else d.astype(np.float32))
    return ff, df


def calculate_flatfield(
    tiles: List[np.ndarray],
    use_darkfield: bool = False,
    constant_darkfield: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Estimate flatfield (and optionally darkfield) from the tiles themselves.

    Retrospective method (numpy/scipy, no external solver): the illumination shading is
    the low-frequency component common across tiles, so per channel we take the
    per-pixel median over the tiles (cancels tile-varying content), smooth it to the
    illumination scale, and normalize to mean 1.0.

    Parameters
    ----------
    tiles : list of ndarray
        List of tile images, each with shape (C, Y, X) or (Y, X) for single-channel.
        2D arrays are automatically converted to 3D with shape (1, Y, X).
    use_darkfield : bool
        Whether to also compute darkfield correction.
    constant_darkfield : bool
        If True, darkfield is reduced to a single constant value (median) per
        channel. This is physically appropriate since dark current is typically
        uniform across the sensor. Default is True.

    Returns
    -------
    flatfield : ndarray
        Flatfield correction array with shape (C, Y, X), float32.
    darkfield : ndarray or None
        Darkfield correction array with shape (C, Y, X), or None if not computed.
        If constant_darkfield=True, each channel slice will be a constant value.

    Raises
    ------
    ValueError
        If tiles list is empty or tiles have inconsistent shapes.
    """
    if not tiles:
        raise ValueError("tiles list is empty")

    # Validate tile dimensionality: only 2D (Y, X) or 3D (C, Y, X) supported
    for i, t in enumerate(tiles):
        if t.ndim not in (2, 3):
            raise ValueError(f"Tile {i} has {t.ndim} dimensions; expected 2 (Y, X) or 3 (C, Y, X)")

    # Support 2D (Y, X) arrays by converting to 3D (1, Y, X)
    tiles = [t[np.newaxis, ...] if t.ndim == 2 else t for t in tiles]

    # Get shape from first tile
    n_channels = tiles[0].shape[0]
    tile_shape = tiles[0].shape[1:]  # (Y, X)

    # Validate all tiles have same shape
    for i, tile in enumerate(tiles):
        if tile.shape[0] != n_channels:
            raise ValueError(f"Tile {i} has {tile.shape[0]} channels, expected {n_channels}")
        if tile.shape[1:] != tile_shape:
            raise ValueError(f"Tile {i} has shape {tile.shape[1:]}, expected {tile_shape}")

    # Calculate flatfield per channel
    flatfield = np.zeros((n_channels,) + tile_shape, dtype=np.float32)
    darkfield = np.zeros((n_channels,) + tile_shape, dtype=np.float32) if use_darkfield else None

    for ch in range(n_channels):
        # Stack channel data from all tiles: shape (n_tiles, Y, X)
        channel_stack = np.stack([tile[ch] for tile in tiles], axis=0)
        ff, df = estimate_flatfield_channel(channel_stack, use_darkfield, constant_darkfield)
        flatfield[ch] = ff
        if use_darkfield:
            darkfield[ch] = df

    return flatfield, darkfield


def apply_flatfield(
    tile: np.ndarray,
    flatfield: np.ndarray,
    darkfield: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply flatfield correction to a tile.

    Formula:
        If darkfield is provided: corrected = (raw - darkfield) / flatfield
        Otherwise: corrected = raw / flatfield

    Parameters
    ----------
    tile : ndarray
        Input tile with shape (C, Y, X).
    flatfield : ndarray
        Flatfield correction array with shape (C, Y, X).
    darkfield : ndarray, optional
        Darkfield correction array with shape (C, Y, X).

    Returns
    -------
    corrected : ndarray
        Corrected tile with shape (C, Y, X), cast back to the input dtype.
        For integer dtypes, values are clipped to the valid range before
        casting (e.g., negative values clipped to 0 for unsigned types).

    Raises
    ------
    ValueError
        If tile and flatfield shapes are incompatible.
    """
    # Validate shapes
    if tile.shape != flatfield.shape:
        raise ValueError(
            f"Tile shape {tile.shape} does not match flatfield shape {flatfield.shape}"
        )
    if darkfield is not None and tile.shape != darkfield.shape:
        raise ValueError(
            f"Tile shape {tile.shape} does not match darkfield shape {darkfield.shape}"
        )

    # Convert to float32 to avoid underflow with unsigned integer types
    tile_f = tile.astype(np.float32)
    # For flatfield values <= 1e-6, use 1.0 to avoid division by zero/near-zero
    flatfield_safe = np.where(flatfield > 1e-6, flatfield, 1.0).astype(np.float32)

    if darkfield is not None:
        corrected = (tile_f - darkfield.astype(np.float32)) / flatfield_safe
    else:
        corrected = tile_f / flatfield_safe

    # Clip to valid range for integer dtypes to avoid wraparound
    if np.issubdtype(tile.dtype, np.integer):
        info = np.iinfo(tile.dtype)
        corrected = np.clip(corrected, info.min, info.max)

    return corrected.astype(tile.dtype)


def apply_flatfield_region(
    region: np.ndarray,
    flatfield: np.ndarray,
    darkfield: Optional[np.ndarray],
    y_slice: slice,
    x_slice: slice,
) -> np.ndarray:
    """
    Apply flatfield correction to a tile region.

    Parameters
    ----------
    region : ndarray
        Input region with shape (C, h, w) or (h, w).
    flatfield : ndarray
        Full flatfield correction array with shape (C, Y, X).
    darkfield : ndarray, optional
        Full darkfield correction array with shape (C, Y, X).
    y_slice, x_slice : slice
        Slices defining the region within the full tile.

    Returns
    -------
    corrected : ndarray
        Corrected region with same shape as input.

    Raises
    ------
    ValueError
        If region and flatfield shapes are incompatible.
    """
    # Validate channel count for 3D regions
    if region.ndim == 3 and region.shape[0] != flatfield.shape[0]:
        raise ValueError(
            f"Region has {region.shape[0]} channels but flatfield has {flatfield.shape[0]} channels"
        )

    # Extract corresponding flatfield/darkfield regions
    if region.ndim == 2:
        ff_region = flatfield[0, y_slice, x_slice]
        df_region = darkfield[0, y_slice, x_slice] if darkfield is not None else None
    else:
        ff_region = flatfield[:, y_slice, x_slice]
        df_region = darkfield[:, y_slice, x_slice] if darkfield is not None else None

    # Convert to float32 to avoid underflow with unsigned integer types
    region_f = region.astype(np.float32)
    # For flatfield values <= 1e-6, use 1.0 to avoid division by zero/near-zero
    ff_safe = np.where(ff_region > 1e-6, ff_region, 1.0).astype(np.float32)

    if df_region is not None:
        corrected = (region_f - df_region.astype(np.float32)) / ff_safe
    else:
        corrected = region_f / ff_safe

    # Clip to valid range for integer dtypes to avoid wraparound
    if np.issubdtype(region.dtype, np.integer):
        info = np.iinfo(region.dtype)
        corrected = np.clip(corrected, info.min, info.max)

    return corrected.astype(region.dtype)


def save_flatfield(
    path: Path,
    flatfield: np.ndarray,
    darkfield: Optional[np.ndarray] = None,
) -> None:
    """
    Save flatfield (and optionally darkfield) to a .npy file.

    Parameters
    ----------
    path : Path
        Output path (should end with .npy).
    flatfield : ndarray
        Flatfield array with shape (C, Y, X).
    darkfield : ndarray, optional
        Darkfield array with shape (C, Y, X).
    """
    data = {
        "flatfield": flatfield.astype(np.float32),
        "darkfield": darkfield.astype(np.float32) if darkfield is not None else None,
        "channels": flatfield.shape[0],
        "shape": flatfield.shape[1:],
    }
    np.save(path, data, allow_pickle=True)


def load_flatfield(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load flatfield (and optionally darkfield) from a .npy file.

    Parameters
    ----------
    path : Path
        Path to .npy file.

    Returns
    -------
    flatfield : ndarray
        Flatfield array with shape (C, Y, X).
    darkfield : ndarray or None
        Darkfield array with shape (C, Y, X), or None if not present.

    Raises
    ------
    OSError
        If the file cannot be read (not found, permission denied, etc.).
    ValueError
        If the file format is invalid (not a dictionary with 'flatfield' key).
    """
    try:
        loaded = np.load(path, allow_pickle=True)
    except OSError as exc:
        raise OSError(f"Cannot read flatfield file '{path}': {exc}") from exc

    try:
        data = loaded.item()
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            f"Invalid flatfield file format at '{path}'. "
            "Expected a NumPy .npy file containing a dictionary as saved by "
            "`save_flatfield` (with keys like 'flatfield' and 'darkfield')."
        ) from exc

    if not isinstance(data, dict) or "flatfield" not in data:
        raise ValueError(
            f"Invalid flatfield file format at '{path}'. "
            "Expected a dictionary with at least a 'flatfield' entry."
        )

    flatfield = data["flatfield"]
    darkfield = data.get("darkfield", None)
    return flatfield, darkfield
