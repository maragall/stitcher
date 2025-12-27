"""
Convert Squid format data to Zarr format.

Squid format:
- Individual TIFF files: manual_{fov}_{z}_{channel}.tiff
- coordinates.csv with fov, x (mm), y (mm), z (um)
- acquisition parameters.json

Zarr v3 output:
- Shape: (T, P, C, Y, X) for 2D
- zarr.json with:
  - per_index_metadata[t][p]["0"]["stage_position"] = [z, y, x] in µm
  - deskewed_voxel_size_um = [z, y, x] pixel sizes
"""

import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorstore as ts
import tifffile
from tqdm import tqdm


def convert_squid_to_zarr(
    squid_path: str,
    output_path: Optional[str] = None,
    max_workers: int = 8,
    compress: bool = True,
) -> Path:
    """
    Convert Squid format microscopy data to Zarr format.

    Parameters
    ----------
    squid_path : str
        Path to Squid data folder (contains 0/, coordinates.csv, etc.)
    output_path : str, optional
        Output path for the Zarr store. If None, creates {name}.zarr
    max_workers : int
        Number of parallel I/O workers
    compress : bool
        If True, use blosc+zstd compression (3.5x smaller, slightly slower reads).
        If False, store uncompressed (faster reads, larger files).

    Returns
    -------
    Path
        Path to the created Zarr store
    """
    squid_path = Path(squid_path)
    if not squid_path.exists():
        raise FileNotFoundError(f"Squid path not found: {squid_path}")

    # Find image folder (usually "0" for single z-level)
    subfolders = [d for d in squid_path.iterdir() if d.is_dir() and d.name.isdigit()]
    if subfolders:
        image_folder = sorted(subfolders, key=lambda x: int(x.name))[0]
    else:
        image_folder = squid_path

    # Load coordinates
    coords_path = image_folder / "coordinates.csv"
    if not coords_path.exists():
        coords_path = squid_path / "coordinates.csv"
    if not coords_path.exists():
        raise FileNotFoundError(f"coordinates.csv not found in {squid_path}")

    coords = pd.read_csv(coords_path)
    n_tiles = len(coords)
    print(f"Found {n_tiles} tiles")

    # Get channel names from TIFF files
    tiff_files = list(image_folder.glob("*.tiff"))
    if not tiff_files:
        tiff_files = list(image_folder.glob("*.tif"))

    channel_names = set()
    for f in tiff_files:
        parts = f.stem.split("_")
        if len(parts) >= 4:
            channel_name = "_".join(parts[3:])
            channel_names.add(channel_name)
    channel_names = sorted(channel_names)
    n_channels = len(channel_names)
    print(f"Found {n_channels} channels: {channel_names}")

    # Read first image to get dimensions
    first_fov = coords["fov"].iloc[0]
    first_channel = channel_names[0]
    first_img_path = image_folder / f"manual_{first_fov}_0_{first_channel}.tiff"
    if not first_img_path.exists():
        first_img_path = image_folder / f"manual_{first_fov}_0_{first_channel}.tif"
    first_img = tifffile.imread(first_img_path)
    Y, X = first_img.shape[-2:]
    print(f"Tile dimensions: {Y} x {X}")

    # Load acquisition parameters
    params_path = squid_path / "acquisition parameters.json"
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
        magnification = params.get("objective", {}).get("magnification", 10.0)
        sensor_pixel_um = params.get("sensor_pixel_size_um", 7.52)
        pixel_size_um = sensor_pixel_um / magnification
    else:
        pixel_size_um = 0.752  # Default for 10x
        magnification = 10.0

    print(f"Pixel size: {pixel_size_um:.4f} µm (magnification: {magnification}x)")

    # Create output path
    if output_path is None:
        output_path = squid_path.parent / f"{squid_path.name}.zarr"
    else:
        output_path = Path(output_path)

    # Remove existing output if present
    if output_path.exists():
        shutil.rmtree(output_path)

    print(f"Creating Zarr store: {output_path}")

    # Build per_index_metadata (stage positions in z, y, x order, in µm)
    per_index_metadata = {"0": {}}  # t=0
    for _, row in coords.iterrows():
        fov = int(row["fov"])
        x_um = row["x (mm)"] * 1000  # mm to µm
        y_um = row["y (mm)"] * 1000
        z_um = row.get("z (um)", 0.0) if "z (um)" in row else 0.0
        per_index_metadata["0"][str(fov)] = {
            "0": {"stage_position": [z_um, y_um, x_um]}  # z=0 for 2D
        }

    # Shape: (T, P, C, Y, X) for 2D data
    full_shape = [1, n_tiles, n_channels, Y, X]
    chunk_shape = [1, 1, 1, min(1024, Y), min(1024, X)]

    # Build codecs based on compression setting
    if compress:
        codec_chunk = [1, 1, 1, min(512, Y), min(512, X)]
        codecs = [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": codec_chunk,
                    "codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {
                            "name": "blosc",
                            "configuration": {
                                "cname": "zstd",
                                "clevel": 5,
                                "shuffle": "bitshuffle",
                            },
                        },
                    ],
                    "index_codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {"name": "crc32c"},
                    ],
                    "index_location": "end",
                },
            }
        ]
    else:
        codecs = [{"name": "bytes", "configuration": {"endian": "little"}}]

    # Create Zarr v3 store
    config = {
        "context": {
            "file_io_concurrency": {"limit": max_workers},
            "data_copy_concurrency": {"limit": max_workers},
        },
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(output_path)},
        "metadata": {
            "shape": full_shape,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunk_shape}},
            "chunk_key_encoding": {"name": "default"},
            "codecs": codecs,
            "data_type": "uint16",
            "dimension_names": ["t", "p", "c", "y", "x"],
            "attributes": {
                "per_index_metadata": per_index_metadata,
                "deskewed_voxel_size_um": [1.0, pixel_size_um, pixel_size_um],  # z, y, x
                "source_format": "squid",
                "channels": channel_names,
                "magnification": magnification,
            },
        },
    }

    store = ts.open(config, create=True, open=True).result()

    # Write tiles
    fov_indices = coords["fov"].tolist()
    for tile_idx, fov in enumerate(tqdm(fov_indices, desc="Converting tiles")):
        for ch_idx, channel_name in enumerate(channel_names):
            img_path = image_folder / f"manual_{fov}_0_{channel_name}.tiff"
            if not img_path.exists():
                img_path = image_folder / f"manual_{fov}_0_{channel_name}.tif"

            if img_path.exists():
                img = tifffile.imread(img_path)
                store[0, tile_idx, ch_idx, :, :].write(img.astype(np.uint16)).result()

    # Write zarr.json with metadata (tensorstore creates a basic one, we enhance it)
    zarr_json_path = output_path / "zarr.json"
    with open(zarr_json_path, "r") as f:
        zarr_meta = json.load(f)

    # Ensure attributes are in the metadata
    if "attributes" not in zarr_meta:
        zarr_meta["attributes"] = {}
    zarr_meta["attributes"]["per_index_metadata"] = per_index_metadata
    zarr_meta["attributes"]["deskewed_voxel_size_um"] = [1.0, pixel_size_um, pixel_size_um]
    zarr_meta["attributes"]["source_format"] = "squid"
    zarr_meta["attributes"]["channels"] = channel_names

    with open(zarr_json_path, "w") as f:
        json.dump(zarr_meta, f, indent=2)

    print(f"Conversion complete: {output_path}")
    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert individual TIFFs folder to Zarr")
    parser.add_argument("input_path", help="Path to data folder with coordinates.csv")
    parser.add_argument("-o", "--output", help="Output Zarr path (default: {name}.zarr)")
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable compression (faster reads, larger files)",
    )
    args = parser.parse_args()

    convert_squid_to_zarr(args.input_path, args.output, compress=not args.no_compress)


if __name__ == "__main__":
    main()
