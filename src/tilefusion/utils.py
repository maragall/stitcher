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


def compute_ssim(arr1, arr2, win_size: int) -> float:
    """
    Compute SSIM using GPU (torch) or CPU (skimage).

    Parameters
    ----------
    arr1, arr2 : ndarray
        Input images (2D).
    win_size : int
        Window size for local statistics.

    Returns
    -------
    ssim : float
        Mean SSIM value.
    """
    arr1_np = np.asarray(arr1, dtype=np.float32)
    arr2_np = np.asarray(arr2, dtype=np.float32)

    if CUDA_AVAILABLE and arr1_np.ndim == 2:
        data_range = float(arr1_np.max() - arr1_np.min())
        if data_range == 0:
            data_range = 1.0
        return _compute_ssim_torch(arr1_np, arr2_np, win_size, data_range)

    data_range = float(arr1_np.max() - arr1_np.min())
    if data_range == 0:
        data_range = 1.0
    return float(_ssim_cpu(arr1_np, arr2_np, win_size=win_size, data_range=data_range))


def _compute_ssim_torch(arr1: np.ndarray, arr2: np.ndarray, win_size: int, data_range: float) -> float:
    """GPU SSIM using torch conv2d for local statistics."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Create uniform window
    window = torch.ones(1, 1, win_size, win_size, device="cuda") / (win_size * win_size)

    # Convert to tensors (1, 1, H, W)
    img1 = torch.from_numpy(arr1).float().cuda().unsqueeze(0).unsqueeze(0)
    img2 = torch.from_numpy(arr2).float().cuda().unsqueeze(0).unsqueeze(0)

    # Compute local means
    mu1 = F.conv2d(img1, window, padding=win_size // 2)
    mu2 = F.conv2d(img2, window, padding=win_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute local variances and covariance
    sigma1_sq = F.conv2d(img1 ** 2, window, padding=win_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=win_size // 2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=win_size // 2) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(ssim_map.mean().cpu())


def shift_array(arr, shift_vec):
    """Shift array using scipy (CPU)."""
    return _shift_cpu(np.asarray(arr), shift=shift_vec, order=1, prefilter=False)


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
