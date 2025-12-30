"""Unit tests for GPU shift_array."""
import numpy as np
import sys
sys.path.insert(0, "src")

from tilefusion.utils import shift_array, CUDA_AVAILABLE
from scipy.ndimage import shift as scipy_shift


def test_integer_shift():
    arr = np.random.rand(256, 256).astype(np.float32)
    cpu = scipy_shift(arr, (3.0, -5.0), order=1, prefilter=False)
    gpu = shift_array(arr, (3.0, -5.0))
    np.testing.assert_allclose(gpu, cpu, rtol=1e-4, atol=1e-4)


def test_subpixel_mean_error():
    arr = np.random.rand(256, 256).astype(np.float32)
    cpu = scipy_shift(arr, (5.5, -3.2), order=1, prefilter=False)
    gpu = shift_array(arr, (5.5, -3.2))
    mean_diff = np.abs(cpu - gpu).mean()
    assert mean_diff < 0.01, f"Mean diff {mean_diff} too high"


def test_zero_shift():
    arr = np.random.rand(256, 256).astype(np.float32)
    result = shift_array(arr, (0.0, 0.0))
    # Allow small tolerance due to grid_sample interpolation
    np.testing.assert_allclose(result, arr, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_integer_shift()
    test_subpixel_mean_error()
    test_zero_shift()
    print("All tests passed")
