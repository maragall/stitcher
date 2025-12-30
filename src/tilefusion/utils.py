"""
Shared utilities for tilefusion.

GPU/CPU detection, array operations, and helper functions.
"""

import numpy as np

# GPU detection - PyTorch based
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

# CPU fallbacks
from scipy.ndimage import shift as _shift_cpu
from skimage.exposure import match_histograms
from skimage.measure import block_reduce
from skimage.metrics import structural_similarity as _ssim_cpu
from skimage.registration import phase_cross_correlation

# Legacy compatibility
USING_GPU = CUDA_AVAILABLE
xp = np
cp = None


def shift_array(arr, shift_vec):
    """
    Shift array by subpixel amounts using GPU (torch) or CPU (scipy).

    Parameters
    ----------
    arr : ndarray
        2D input array.
    shift_vec : array-like
        (dy, dx) shift amounts.

    Returns
    -------
    shifted : ndarray
        Shifted array, same shape as input.
    """
    arr_np = np.asarray(arr)

    if CUDA_AVAILABLE and arr_np.ndim == 2:
        return _shift_array_torch(arr_np, shift_vec)

    return _shift_cpu(arr_np, shift=shift_vec, order=1, prefilter=False)


def _shift_array_torch(arr: np.ndarray, shift_vec) -> np.ndarray:
    """GPU shift using torch.nn.functional.grid_sample."""
    h, w = arr.shape
    dy, dx = float(shift_vec[0]), float(shift_vec[1])

    # Create pixel coordinate grids
    y_coords = torch.arange(h, device="cuda", dtype=torch.float32)
    x_coords = torch.arange(w, device="cuda", dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # Apply shift: to shift output by (dy, dx), sample from (y-dy, x-dx)
    sample_y = grid_y - dy
    sample_x = grid_x - dx

    # Normalize to [-1, 1] for grid_sample (align_corners=True)
    sample_x = 2 * sample_x / (w - 1) - 1
    sample_y = 2 * sample_y / (h - 1) - 1

    # Stack to (H, W, 2) with (x, y) order, add batch dim -> (1, H, W, 2)
    grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)

    # Input: (1, 1, H, W)
    t = torch.from_numpy(arr).float().cuda().unsqueeze(0).unsqueeze(0)

    # grid_sample with bilinear interpolation
    out = F.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    return out.squeeze().cpu().numpy()


def compute_ssim(arr1, arr2, win_size: int) -> float:
    """SSIM using skimage (CPU)."""
    arr1_np = np.asarray(arr1)
    arr2_np = np.asarray(arr2)
    data_range = float(arr1_np.max() - arr1_np.min())
    if data_range == 0:
        data_range = 1.0
    return float(_ssim_cpu(arr1_np, arr2_np, win_size=win_size, data_range=data_range))


def make_1d_profile(length: int, blend: int) -> np.ndarray:
    """Create a linear ramp profile over `blend` pixels at each end."""
    blend = min(blend, length // 2)
    prof = np.ones(length, dtype=np.float32)
    if blend > 0:
        ramp = np.linspace(0, 1, blend, endpoint=False, dtype=np.float32)
        prof[:blend] = ramp
        prof[-blend:] = ramp[::-1]
    return prof


def to_numpy(arr):
    """Convert array to numpy."""
    if TORCH_AVAILABLE and torch is not None and isinstance(arr, torch.Tensor):
        return arr.cpu().numpy()
    return np.asarray(arr)


def to_device(arr):
    """Move array to GPU if available."""
    if CUDA_AVAILABLE:
        return torch.from_numpy(np.asarray(arr)).cuda()
    return np.asarray(arr)
