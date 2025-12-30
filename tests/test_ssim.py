"""Unit tests for GPU SSIM."""
import numpy as np
import sys
sys.path.insert(0, "src")

from tilefusion.utils import compute_ssim, CUDA_AVAILABLE
from skimage.metrics import structural_similarity as skimage_ssim


def test_ssim_similar_images():
    arr1 = np.random.rand(256, 256).astype(np.float32)
    arr2 = arr1 + np.random.rand(256, 256).astype(np.float32) * 0.1

    data_range = arr1.max() - arr1.min()
    cpu = skimage_ssim(arr1, arr2, win_size=15, data_range=data_range)
    gpu = compute_ssim(arr1, arr2, win_size=15)

    assert abs(cpu - gpu) < 0.01, f"SSIM diff {abs(cpu-gpu)} too high"


def test_ssim_identical_images():
    arr = np.random.rand(256, 256).astype(np.float32)
    ssim = compute_ssim(arr, arr, win_size=15)
    assert ssim > 0.99, f"SSIM of identical images should be ~1.0, got {ssim}"


def test_ssim_different_images():
    arr1 = np.random.rand(256, 256).astype(np.float32)
    arr2 = np.random.rand(256, 256).astype(np.float32)
    ssim = compute_ssim(arr1, arr2, win_size=15)
    assert ssim < 0.5, f"SSIM of random images should be low, got {ssim}"


if __name__ == "__main__":
    test_ssim_similar_images()
    test_ssim_identical_images()
    test_ssim_different_images()
    print("All tests passed")
