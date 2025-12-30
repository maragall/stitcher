#!/usr/bin/env python3
"""
Benchmark different GPU acceleration branch combinations.

Dynamically cherry-picks commits from acceleration branches to test
different combinations against the CPU baseline (main).

Usage:
    python scripts/benchmark_branches.py --data /path/to/tiles
    python scripts/benchmark_branches.py --data /path/to/tiles --branches gpu-block-reduce,gpu-ssim
    python scripts/benchmark_branches.py --list
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# Available GPU acceleration branches
GPU_BRANCHES = [
    "gpu-block-reduce",
    "gpu-shift-array",
    "gpu-histogram-match",
    "gpu-ssim",
]


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    branches: List[str]
    registration_time_s: float
    total_time_s: float
    n_pairs: int
    speedup_vs_main: float = 1.0


def run_cmd(cmd: List[str], cwd: Path = None) -> subprocess.CompletedProcess:
    """Run shell command."""
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def get_branch_commit(repo: Path, branch: str) -> Optional[str]:
    """Get commit hash for a branch."""
    result = run_cmd(["git", "rev-parse", branch], cwd=repo)
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def create_test_worktree(repo: Path, branches: List[str]) -> Optional[Path]:
    """Create a temporary worktree with cherry-picked commits."""
    import shutil

    # Create temp directory
    tmp_dir = Path(tempfile.mkdtemp(prefix="bench_"))
    worktree = tmp_dir / "repo"

    # Clone repo
    result = run_cmd(["git", "clone", "--local", str(repo), str(worktree)])
    if result.returncode != 0:
        shutil.rmtree(tmp_dir)
        return None

    # Cherry-pick each branch's commit
    for branch in branches:
        commit = get_branch_commit(repo, branch)
        if commit:
            result = run_cmd(["git", "cherry-pick", "--no-commit", commit], cwd=worktree)
            if result.returncode != 0:
                # Try to continue despite conflicts
                run_cmd(["git", "checkout", "--theirs", "."], cwd=worktree)
                run_cmd(["git", "add", "."], cwd=worktree)

    return worktree


def benchmark_registration(data_path: Path, repo_path: Path) -> BenchmarkResult:
    """Benchmark registration phase only."""
    sys.path.insert(0, str(repo_path / "src"))

    # Clear module cache
    mods = [k for k in sys.modules if "tilefusion" in k]
    for m in mods:
        del sys.modules[m]

    from tilefusion import TileFusion

    tf = TileFusion(data_path, output_path="/tmp/bench_out.zarr", blend_pixels=(0, 0))

    t0 = time.perf_counter()
    tf.refine_tile_positions_with_cross_correlation()
    elapsed = time.perf_counter() - t0

    return elapsed, len(tf.pairwise_metrics)


def run_benchmark(
    repo_path: Path,
    data_path: Path,
    branches: List[str],
    name: str,
    baseline_time: float = None,
) -> BenchmarkResult:
    """Run benchmark for a branch combination."""
    import shutil

    if not branches:
        # Main branch (CPU baseline)
        worktree = repo_path
        cleanup = False
    else:
        worktree = create_test_worktree(repo_path, branches)
        cleanup = True

    if worktree is None:
        return BenchmarkResult(
            name=name,
            branches=branches,
            registration_time_s=-1,
            total_time_s=-1,
            n_pairs=0,
        )

    try:
        reg_time, n_pairs = benchmark_registration(data_path, worktree)
        speedup = baseline_time / reg_time if baseline_time else 1.0

        return BenchmarkResult(
            name=name,
            branches=branches,
            registration_time_s=round(reg_time, 3),
            total_time_s=round(reg_time, 3),
            n_pairs=n_pairs,
            speedup_vs_main=round(speedup, 2),
        )
    finally:
        if cleanup and worktree:
            shutil.rmtree(worktree.parent, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU acceleration branches")
    parser.add_argument("--data", type=str, help="Path to tile data")
    parser.add_argument("--branches", type=str, default="", help="Comma-separated branches")
    parser.add_argument("--list", action="store_true", help="List available branches")
    parser.add_argument("--all-combinations", action="store_true", help="Test all combinations")
    args = parser.parse_args()

    repo_path = Path(__file__).parent.parent

    if args.list:
        print("Available GPU acceleration branches:")
        for b in GPU_BRANCHES:
            commit = get_branch_commit(repo_path, b)
            status = f"[{commit[:7]}]" if commit else "[missing]"
            print(f"  {b} {status}")
        return 0

    if not args.data:
        print("Error: --data required")
        return 1

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        return 1

    print("=" * 70)
    print("GPU ACCELERATION BENCHMARK")
    print("=" * 70)
    print(f"Data: {data_path}")
    print(f"Repo: {repo_path}")
    print()

    results = []

    # Baseline (main)
    print("Running baseline (main - CPU)...")
    baseline = run_benchmark(repo_path, data_path, [], "main (CPU)")
    results.append(baseline)
    print(f"  Registration: {baseline.registration_time_s:.2f}s ({baseline.n_pairs} pairs)")

    # Determine which branches to test
    if args.branches:
        test_branches = [b.strip() for b in args.branches.split(",")]
        combinations = [test_branches]
    elif args.all_combinations:
        # Test each branch individually and all together
        combinations = [[b] for b in GPU_BRANCHES] + [GPU_BRANCHES]
    else:
        # Test each branch individually
        combinations = [[b] for b in GPU_BRANCHES]

    # Run benchmarks
    for branches in combinations:
        name = "+".join(branches) if branches else "main"
        print(f"\nRunning {name}...")

        result = run_benchmark(
            repo_path, data_path, branches, name, baseline.registration_time_s
        )
        results.append(result)
        print(f"  Registration: {result.registration_time_s:.2f}s, Speedup: {result.speedup_vs_main:.2f}x")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<40} {'Time(s)':<12} {'Speedup':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r.name:<40} {r.registration_time_s:<12.2f} {r.speedup_vs_main:<10.2f}x")
    print("=" * 70)

    # Save results
    results_path = Path("/tmp/benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
