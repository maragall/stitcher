#!/usr/bin/env python3
"""
Inline benchmark that directly tests GPU acceleration functions.

This script doesn't require git operations - it directly benchmarks
the individual GPU-accelerated functions against their CPU counterparts.

Usage:
    python scripts/benchmark_inline.py
    python scripts/benchmark_inline.py --data /path/to/tiles
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Check for CUDA
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
    else:
        GPU_NAME = "N/A"
except ImportError:
    CUDA_AVAILABLE = False
    GPU_NAME = "N/A"


def benchmark_block_reduce():
    """Benchmark block_reduce CPU vs GPU."""
    from skimage.measure import block_reduce as cpu_block_reduce

    arr = np.random.rand(2048, 2048).astype(np.float32)
    block_size = (4, 4)

    # CPU
    t0 = time.perf_counter()
    for _ in range(10):
        _ = cpu_block_reduce(arr, block_size, np.mean)
    cpu_time = (time.perf_counter() - t0) / 10 * 1000

    if not CUDA_AVAILABLE:
        return cpu_time, None, None

    # GPU
    def gpu_block_reduce(arr, block_size):
        t = torch.from_numpy(arr).float().cuda().unsqueeze(0).unsqueeze(0)
        out = torch.nn.functional.avg_pool2d(t, block_size, stride=block_size)
        return out.squeeze().cpu().numpy()

    # Warmup
    _ = gpu_block_reduce(arr, block_size)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(10):
        _ = gpu_block_reduce(arr, block_size)
        torch.cuda.synchronize()
    gpu_time = (time.perf_counter() - t0) / 10 * 1000

    return cpu_time, gpu_time, cpu_time / gpu_time


def benchmark_shift_array():
    """Benchmark shift_array CPU vs GPU."""
    from scipy.ndimage import shift as cpu_shift

    arr = np.random.rand(2048, 2048).astype(np.float32)
    shift_vec = (5.5, -3.2)

    # CPU
    t0 = time.perf_counter()
    for _ in range(10):
        _ = cpu_shift(arr, shift_vec, order=1, prefilter=False)
    cpu_time = (time.perf_counter() - t0) / 10 * 1000

    if not CUDA_AVAILABLE:
        return cpu_time, None, None

    # GPU
    def gpu_shift(arr, shift_vec):
        import torch.nn.functional as F
        h, w = arr.shape
        dy, dx = float(shift_vec[0]), float(shift_vec[1])

        y_coords = torch.arange(h, device="cuda", dtype=torch.float32)
        x_coords = torch.arange(w, device="cuda", dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

        sample_y = grid_y - dy
        sample_x = grid_x - dx
        sample_x = 2 * sample_x / (w - 1) - 1
        sample_y = 2 * sample_y / (h - 1) - 1

        grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)
        t = torch.from_numpy(arr).float().cuda().unsqueeze(0).unsqueeze(0)
        out = F.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return out.squeeze().cpu().numpy()

    # Warmup
    _ = gpu_shift(arr, shift_vec)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(10):
        _ = gpu_shift(arr, shift_vec)
        torch.cuda.synchronize()
    gpu_time = (time.perf_counter() - t0) / 10 * 1000

    return cpu_time, gpu_time, cpu_time / gpu_time


def benchmark_histogram_match():
    """Benchmark histogram matching CPU vs GPU."""
    from skimage.exposure import match_histograms as cpu_match

    img = np.random.rand(1024, 1024).astype(np.float32)
    ref = np.random.rand(1024, 1024).astype(np.float32) * 2 + 1

    # CPU
    t0 = time.perf_counter()
    for _ in range(10):
        _ = cpu_match(img, ref)
    cpu_time = (time.perf_counter() - t0) / 10 * 1000

    if not CUDA_AVAILABLE:
        return cpu_time, None, None

    # GPU
    def gpu_match(image, reference):
        img = torch.from_numpy(image.astype(np.float32)).cuda().flatten()
        ref = torch.from_numpy(reference.astype(np.float32)).cuda().flatten()

        _, img_indices = torch.sort(img)
        ref_sorted, _ = torch.sort(ref)

        inv_indices = torch.empty_like(img_indices)
        inv_indices[img_indices] = torch.arange(len(img), device="cuda")

        interp_values = torch.zeros_like(img)
        interp_values[img_indices] = ref_sorted[
            (inv_indices.float() / len(img) * len(ref)).long().clamp(0, len(ref) - 1)
        ]
        return interp_values.reshape(image.shape).cpu().numpy()

    # Warmup
    _ = gpu_match(img, ref)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(10):
        _ = gpu_match(img, ref)
        torch.cuda.synchronize()
    gpu_time = (time.perf_counter() - t0) / 10 * 1000

    return cpu_time, gpu_time, cpu_time / gpu_time


def benchmark_ssim():
    """Benchmark SSIM CPU vs GPU."""
    from skimage.metrics import structural_similarity as cpu_ssim

    arr1 = np.random.rand(1024, 1024).astype(np.float32)
    arr2 = arr1 + np.random.rand(1024, 1024).astype(np.float32) * 0.1
    data_range = arr1.max() - arr1.min()

    # CPU
    t0 = time.perf_counter()
    for _ in range(10):
        _ = cpu_ssim(arr1, arr2, win_size=15, data_range=data_range)
    cpu_time = (time.perf_counter() - t0) / 10 * 1000

    if not CUDA_AVAILABLE:
        return cpu_time, None, None

    # GPU
    def gpu_ssim(arr1, arr2, win_size, data_range):
        import torch.nn.functional as F
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        window = torch.ones(1, 1, win_size, win_size, device="cuda") / (win_size * win_size)
        img1 = torch.from_numpy(arr1).float().cuda().unsqueeze(0).unsqueeze(0)
        img2 = torch.from_numpy(arr2).float().cuda().unsqueeze(0).unsqueeze(0)

        mu1 = F.conv2d(img1, window, padding=win_size // 2)
        mu2 = F.conv2d(img2, window, padding=win_size // 2)

        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sigma1_sq = F.conv2d(img1 ** 2, window, padding=win_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, window, padding=win_size // 2) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=win_size // 2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(ssim_map.mean().cpu())

    # Warmup
    _ = gpu_ssim(arr1, arr2, 15, data_range)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(10):
        _ = gpu_ssim(arr1, arr2, 15, data_range)
        torch.cuda.synchronize()
    gpu_time = (time.perf_counter() - t0) / 10 * 1000

    return cpu_time, gpu_time, cpu_time / gpu_time


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU accelerations inline")
    parser.add_argument("--data", type=str, help="Optional: path to tile data for full pipeline test")
    args = parser.parse_args()

    print("=" * 70)
    print("GPU ACCELERATION BENCHMARK")
    print("=" * 70)
    print(f"CUDA Available: {CUDA_AVAILABLE}")
    print(f"GPU: {GPU_NAME}")
    print()

    results = []

    benchmarks = [
        ("block_reduce (2048x2048)", benchmark_block_reduce),
        ("shift_array (2048x2048)", benchmark_shift_array),
        ("histogram_match (1024x1024)", benchmark_histogram_match),
        ("ssim (1024x1024)", benchmark_ssim),
    ]

    print(f"{'Function':<35} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
    print("-" * 70)

    for name, func in benchmarks:
        cpu_time, gpu_time, speedup = func()
        gpu_str = f"{gpu_time:.2f}" if gpu_time else "N/A"
        speedup_str = f"{speedup:.2f}x" if speedup else "N/A"
        print(f"{name:<35} {cpu_time:<12.2f} {gpu_str:<12} {speedup_str:<10}")
        results.append((name, cpu_time, gpu_time, speedup))

    print("=" * 70)

    if CUDA_AVAILABLE:
        total_cpu = sum(r[1] for r in results)
        total_gpu = sum(r[2] for r in results if r[2])
        print(f"{'TOTAL':<35} {total_cpu:<12.2f} {total_gpu:<12.2f} {total_cpu/total_gpu:.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
