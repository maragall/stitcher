"""
Shared utilities for tilefusion.

GPU/CPU detection, array operations, and helper functions.
"""

import logging
from contextlib import contextmanager

import numpy as np

logger = logging.getLogger(__name__)

# threadpoolctl lets us cap the BLAS (OpenBLAS/MKL) thread pool at RUNTIME -- env vars
# like OPENBLAS_NUM_THREADS only take effect at BLAS init, too late once numpy is loaded.
# Optional import: if it is unavailable (e.g. a frozen build that didn't bundle it) we
# degrade to a no-op rather than crash; the only cost is the un-pinned behaviour.
try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:  # pragma: no cover - depends on environment
    _threadpool_limits = None
_warned_no_threadpoolctl = False


@contextmanager
def limit_blas_threads(n: int = 1):
    """Cap BLAS threads to `n` within this block, then restore.

    Use this around a ThreadPoolExecutor whose workers call numpy linear algebra.
    Without it, each of W worker threads can spin up its own BLAS pool of ~ncores
    threads, so W workers x ncores BLAS = oversubscription: the CPU pegs at 100% but
    a large share of cycles is spent context-switching, not computing. Pinning BLAS to
    1 thread per worker keeps total threads ~= W ~= cores -- full utilisation, no thrash.
    (Leave BLAS multi-threaded for the SEQUENTIAL heavy ops, e.g. BaSiC, which want it.)
    """
    global _warned_no_threadpoolctl
    if _threadpool_limits is None:
        if not _warned_no_threadpoolctl:
            logger.debug("threadpoolctl not available; BLAS threads not pinned in pools")
            _warned_no_threadpoolctl = True
        yield
    else:
        with _threadpool_limits(limits=n, user_api="blas"):
            yield

try:
    import cupy as cp
    from cupyx.scipy.ndimage import shift as cp_shift
    from cucim.skimage.exposure import match_histograms
    from cucim.skimage.measure import block_reduce
    from cucim.skimage.registration import phase_cross_correlation
    from opm_processing.imageprocessing.ssim_cuda import (
        structural_similarity_cupy_sep_shared as ssim_cuda,
    )

    xp = cp
    USING_GPU = True
except Exception:
    cp = None
    cp_shift = None
    from skimage.exposure import match_histograms
    from skimage.measure import block_reduce
    from skimage.registration import phase_cross_correlation
    from scipy.ndimage import shift as _shift_cpu
    from skimage.metrics import structural_similarity as _ssim_cpu

    xp = np
    USING_GPU = False


def shift_array(arr, shift_vec):
    """Shift array using GPU if available, else CPU fallback."""
    if USING_GPU and cp_shift is not None:
        return cp_shift(arr, shift=shift_vec, order=1, prefilter=False)
    return _shift_cpu(arr, shift=shift_vec, order=1, prefilter=False)


def compute_ssim(arr1, arr2, win_size: int) -> float:
    """SSIM wrapper that routes to GPU kernel or CPU skimage."""
    if USING_GPU and "ssim_cuda" in globals():
        return float(ssim_cuda(arr1, arr2, win_size=win_size))
    arr1_np = np.asarray(arr1)
    arr2_np = np.asarray(arr2)
    data_range = float(arr1_np.max() - arr1_np.min())
    if data_range < 1e-3:
        # Low-contrast / uniform region — SSIM is meaningless
        return 0.0
    return float(_ssim_cpu(arr1_np, arr2_np, win_size=win_size, data_range=data_range))


def make_1d_profile(length: int, blend: int) -> np.ndarray:
    """
    Create a Hann (cosine) ramp profile over `blend` pixels at each end.

    Parameters
    ----------
    length : int
        Number of pixels.
    blend : int
        Ramp width.

    Returns
    -------
    prof : (length,) float32
        Cosine profile with zero-derivative at boundaries.
    """
    blend = min(blend, length // 2)
    prof = np.ones(length, dtype=np.float32)
    if blend > 0:
        # Cosine (Hann) window: zero-derivative at boundaries for smoother transitions
        t = np.linspace(0, 1, blend, endpoint=False, dtype=np.float32)
        ramp = np.float32(0.5) * (1 - np.cos(np.pi * t))
        prof[:blend] = ramp
        prof[-blend:] = ramp[::-1]
    return prof


def to_numpy(arr):
    """Convert array to numpy, handling both CPU and GPU arrays."""
    if USING_GPU and cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def to_device(arr):
    """Move array to current device (GPU if available, else CPU)."""
    return xp.asarray(arr)
