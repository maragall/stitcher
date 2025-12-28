#!/usr/bin/env python
"""
Benchmark script for TileFusion stitcher.
Tests multiple datasets with/without blending and captures boundary regions.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tensorstore as ts

from tilefusion import TileFusion


# Test datasets
DATASETS = {
    "20x_scan": "/Users/julioamaragall/Downloads/20x_scan_2025-09-05_17-57-50",
    "COLNOR69MW2_c1": "/Users/julioamaragall/Downloads/COLNOR69MW2-cycle-1.ome.tif",
    "COLNOR69MW2_c2": "/Users/julioamaragall/Downloads/COLNOR69MW2-cycle-2.ome.tif",
    "Monkey": "/Users/julioamaragall/Downloads/Monkey",
    "successful_run": "/Users/julioamaragall/Downloads/successful_run",
    "z_stack_10x": "/Users/julioamaragall/Downloads/test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy",
}

OUTPUT_DIR = Path("/Users/julioamaragall/Downloads/benchmark_results")
BLEND_PIXELS = (80, 80)


def run_stitching(
    input_path: str,
    output_path: Path,
    blend: bool,
    skip_if_exists: bool = True,
) -> Dict:
    """Run stitching and return timing info."""

    if skip_if_exists and output_path.exists():
        print(f"  Output exists, skipping: {output_path}")
        return {"skipped": True}

    # Clean output
    if output_path.exists():
        shutil.rmtree(output_path)

    blend_px = BLEND_PIXELS if blend else (0, 0)

    # Create TileFusion
    tf = TileFusion(
        input_path,
        output_path=output_path,
        blend_pixels=blend_px,
        threshold=0.5,
    )

    result = {
        "n_tiles": tf.n_tiles,
        "tile_shape": (tf.Y, tf.X),
        "blend": blend,
    }

    # Clear cached metrics to force fresh registration
    metrics_path = Path(input_path).parent / "metrics.json"
    if metrics_path.exists():
        metrics_path.unlink()

    # Time registration
    t0 = time.time()
    tf.refine_tile_positions_with_cross_correlation()
    result["registration_time"] = time.time() - t0
    result["n_pairs"] = len(tf.pairwise_metrics)

    # Optimize
    tf.optimize_shifts()
    tf._tile_positions = [
        tuple(np.array(pos) + off * np.array(tf.pixel_size))
        for pos, off in zip(tf._tile_positions, tf.global_offsets)
    ]

    # Time fusion
    tf._compute_fused_image_space()
    tf._pad_to_chunk_multiple()
    result["output_shape"] = tf.padded_shape

    scale0 = output_path / "scale0" / "image"
    scale0.parent.mkdir(parents=True, exist_ok=True)
    tf._create_fused_tensorstore(output_path=scale0)

    t0 = time.time()
    tf._fuse_tiles()
    result["fusion_time"] = time.time() - t0

    result["total_time"] = result["registration_time"] + result["fusion_time"]

    return result


def extract_boundary_region(
    zarr_path: Path,
    center_y: int,
    center_x: int,
    size: int = 256,
) -> np.ndarray:
    """Extract a region from the stitched output."""
    scale0 = zarr_path / "scale0" / "image"

    store = ts.open({
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(scale0)},
    }).result()

    # Get shape (T, C, Z, Y, X) or similar
    shape = store.shape
    y_dim = shape[-2]
    x_dim = shape[-1]

    y0 = max(0, center_y - size // 2)
    y1 = min(y_dim, center_y + size // 2)
    x0 = max(0, center_x - size // 2)
    x1 = min(x_dim, center_x + size // 2)

    # Read region - assume 5D (T, C, Z, Y, X)
    if len(shape) == 5:
        data = store[0, 0, 0, y0:y1, x0:x1].read().result()
    elif len(shape) == 4:
        data = store[0, 0, y0:y1, x0:x1].read().result()
    else:
        data = store[y0:y1, x0:x1].read().result()

    return data


def find_tile_boundaries(tf: TileFusion) -> Tuple[List[int], List[int]]:
    """Find approximate tile boundary locations."""
    positions = np.array(tf._tile_positions)
    pixel_size = np.array(tf.pixel_size)

    # Convert to pixel coordinates relative to origin
    min_pos = positions.min(axis=0)
    pixel_coords = (positions - min_pos) / pixel_size

    # Find unique Y and X positions (tile grid)
    y_coords = np.unique(np.round(pixel_coords[:, 0]).astype(int))
    x_coords = np.unique(np.round(pixel_coords[:, 1]).astype(int))

    # Boundaries are at tile edges
    horizontal_boundaries = []  # Horizontal seams (between rows)
    vertical_boundaries = []    # Vertical seams (between columns)

    for y in y_coords[1:]:  # Skip first
        horizontal_boundaries.append(int(y))

    for x in x_coords[1:]:  # Skip first
        vertical_boundaries.append(int(x + tf.X))  # At right edge of tile

    return horizontal_boundaries, vertical_boundaries


def run_benchmark(datasets: Dict[str, str], output_dir: Path) -> Dict:
    """Run full benchmark on all datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for name, path in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"Path: {path}")
        print(f"{'='*60}")

        if not Path(path).exists():
            print(f"  SKIPPED - path does not exist")
            continue

        results[name] = {}

        # Run without blending
        print(f"\n  [No Blending]")
        out_no_blend = output_dir / f"{name}_no_blend.ome.zarr"
        try:
            results[name]["no_blend"] = run_stitching(path, out_no_blend, blend=False)
            print(f"    Registration: {results[name]['no_blend'].get('registration_time', 0):.1f}s")
            print(f"    Fusion: {results[name]['no_blend'].get('fusion_time', 0):.1f}s")
        except Exception as e:
            print(f"    ERROR: {e}")
            results[name]["no_blend"] = {"error": str(e)}

        # Run with blending
        print(f"\n  [With Blending ({BLEND_PIXELS})]")
        out_blend = output_dir / f"{name}_blend.ome.zarr"
        try:
            results[name]["blend"] = run_stitching(path, out_blend, blend=True)
            print(f"    Registration: {results[name]['blend'].get('registration_time', 0):.1f}s")
            print(f"    Fusion: {results[name]['blend'].get('fusion_time', 0):.1f}s")
        except Exception as e:
            print(f"    ERROR: {e}")
            results[name]["blend"] = {"error": str(e)}

    # Save results
    results_file = output_dir / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    return results


def extract_boundaries_for_dataset(
    name: str,
    input_path: str,
    output_dir: Path,
    region_size: int = 512,
) -> Dict[str, np.ndarray]:
    """Extract boundary regions from stitched outputs."""

    # Load TileFusion to get tile positions
    tf = TileFusion(input_path, output_path="/tmp/dummy.zarr")
    h_bounds, v_bounds = find_tile_boundaries(tf)

    regions = {}

    for blend_mode in ["no_blend", "blend"]:
        zarr_path = output_dir / f"{name}_{blend_mode}.ome.zarr"
        if not zarr_path.exists():
            continue

        # Get output shape
        scale0 = zarr_path / "scale0" / "image"
        store = ts.open({
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(scale0)},
        }).result()
        shape = store.shape
        out_y, out_x = shape[-2], shape[-1]

        # Extract horizontal boundary (between rows)
        if h_bounds:
            h_center = h_bounds[len(h_bounds) // 2]
            x_center = out_x // 2
            try:
                regions[f"{blend_mode}_horizontal"] = extract_boundary_region(
                    zarr_path, h_center, x_center, region_size
                )
            except Exception as e:
                print(f"    Error extracting horizontal boundary: {e}")

        # Extract vertical boundary (between columns)
        if v_bounds:
            v_center = v_bounds[len(v_bounds) // 2]
            y_center = out_y // 2
            try:
                regions[f"{blend_mode}_vertical"] = extract_boundary_region(
                    zarr_path, y_center, v_center, region_size
                )
            except Exception as e:
                print(f"    Error extracting vertical boundary: {e}")

    return regions


def create_figure(results: Dict, output_dir: Path) -> None:
    """Create comprehensive figure with timing and boundary comparisons."""

    # Filter successful results
    valid_results = {
        k: v for k, v in results.items()
        if "no_blend" in v and "error" not in v.get("no_blend", {})
    }

    if not valid_results:
        print("No valid results to plot")
        return

    n_datasets = len(valid_results)

    # Create figure
    fig = plt.figure(figsize=(16, 4 + 3 * n_datasets))

    # Top row: Timing bar chart
    ax_timing = fig.add_subplot(n_datasets + 1, 1, 1)

    datasets_names = list(valid_results.keys())
    x = np.arange(len(datasets_names))
    width = 0.35

    reg_times_no_blend = []
    fusion_times_no_blend = []
    reg_times_blend = []
    fusion_times_blend = []

    for name in datasets_names:
        nb = valid_results[name].get("no_blend", {})
        b = valid_results[name].get("blend", {})

        reg_times_no_blend.append(nb.get("registration_time", 0))
        fusion_times_no_blend.append(nb.get("fusion_time", 0))
        reg_times_blend.append(b.get("registration_time", 0))
        fusion_times_blend.append(b.get("fusion_time", 0))

    # Stacked bars for no blend
    bars1 = ax_timing.bar(x - width/2, reg_times_no_blend, width, label='Registration (no blend)', color='steelblue')
    bars2 = ax_timing.bar(x - width/2, fusion_times_no_blend, width, bottom=reg_times_no_blend, label='Fusion (no blend)', color='lightsteelblue')

    # Stacked bars for blend
    bars3 = ax_timing.bar(x + width/2, reg_times_blend, width, label='Registration (blend)', color='darkorange')
    bars4 = ax_timing.bar(x + width/2, fusion_times_blend, width, bottom=reg_times_blend, label='Fusion (blend)', color='moccasin')

    ax_timing.set_ylabel('Time (seconds)')
    ax_timing.set_title('Stitching Performance: Registration + Fusion Time')
    ax_timing.set_xticks(x)
    ax_timing.set_xticklabels(datasets_names, rotation=15, ha='right')
    ax_timing.legend(loc='upper right')
    ax_timing.grid(axis='y', alpha=0.3)

    # Add total time labels
    for i, name in enumerate(datasets_names):
        total_nb = reg_times_no_blend[i] + fusion_times_no_blend[i]
        total_b = reg_times_blend[i] + fusion_times_blend[i]
        ax_timing.annotate(f'{total_nb:.1f}s', (i - width/2, total_nb), ha='center', va='bottom', fontsize=8)
        ax_timing.annotate(f'{total_b:.1f}s', (i + width/2, total_b), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # Save timing figure
    timing_fig_path = output_dir / "timing_comparison.png"
    fig.savefig(timing_fig_path, dpi=150, bbox_inches='tight')
    print(f"\nTiming figure saved to: {timing_fig_path}")
    plt.close(fig)

    # Create boundary comparison figures for each dataset
    for name, path in DATASETS.items():
        if name not in valid_results:
            continue

        if not Path(path).exists():
            continue

        print(f"\nExtracting boundaries for {name}...")

        try:
            regions = extract_boundaries_for_dataset(name, path, output_dir)

            if not regions:
                print(f"  No boundary regions extracted")
                continue

            # Create comparison figure for this dataset
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Tile Boundary Comparison: {name}', fontsize=14)

            titles = [
                ('no_blend_horizontal', 'No Blending - Horizontal Boundary'),
                ('blend_horizontal', 'With Blending - Horizontal Boundary'),
                ('no_blend_vertical', 'No Blending - Vertical Boundary'),
                ('blend_vertical', 'With Blending - Vertical Boundary'),
            ]

            for ax, (key, title) in zip(axes.flat, titles):
                if key in regions:
                    img = regions[key]
                    # Normalize for display
                    vmin, vmax = np.percentile(img, [1, 99])
                    ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
                    ax.set_title(title)
                    ax.axis('off')

                    # Draw center line to highlight boundary
                    if 'horizontal' in key:
                        ax.axhline(img.shape[0]//2, color='red', linewidth=0.5, alpha=0.5)
                    else:
                        ax.axvline(img.shape[1]//2, color='red', linewidth=0.5, alpha=0.5)
                else:
                    ax.text(0.5, 0.5, 'Not available', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(title)
                    ax.axis('off')

            plt.tight_layout()
            boundary_fig_path = output_dir / f"boundaries_{name}.png"
            fig.savefig(boundary_fig_path, dpi=150, bbox_inches='tight')
            print(f"  Boundary figure saved to: {boundary_fig_path}")
            plt.close(fig)

        except Exception as e:
            print(f"  Error creating boundary figure: {e}")


def main():
    """Main entry point."""
    print("TileFusion Benchmark")
    print("="*60)

    # Run benchmarks
    results = run_benchmark(DATASETS, OUTPUT_DIR)

    # Create figures
    create_figure(results, OUTPUT_DIR)

    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Dataset':<20} {'Tiles':>6} {'Reg(s)':>8} {'Fuse(s)':>8} {'Total(s)':>8} {'Blend':>6}")
    print("-"*60)

    for name, data in results.items():
        for mode in ["no_blend", "blend"]:
            if mode in data and "error" not in data[mode]:
                d = data[mode]
                blend_str = "No" if mode == "no_blend" else "Yes"
                print(f"{name:<20} {d.get('n_tiles', '?'):>6} {d.get('registration_time', 0):>8.1f} "
                      f"{d.get('fusion_time', 0):>8.1f} {d.get('total_time', 0):>8.1f} {blend_str:>6}")

    print(f"\nResults directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
