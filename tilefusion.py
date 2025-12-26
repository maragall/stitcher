"""
2D tile fusion for OME-TIFF microscopy data.

This module implements GPU/CPU-accelerated tile registration and fusion
for 2D tiled microscopy datasets stored as OME-TIFF files.

The final fused volume is written to an OME-NGFF v0.5 Zarr store using tensorstore.
"""

import gc
import json
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Sequence, Tuple, Union, Any, Dict, Optional
import xml.etree.ElementTree as ET

import numpy as np
import tifffile
import tensorstore as ts
from numba import njit, prange
from tqdm import trange, tqdm

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


def _shift_array(arr: Any, shift_vec: Any) -> Any:
    """Shift array using GPU if available, else CPU fallback."""
    if USING_GPU and cp_shift is not None:
        return cp_shift(arr, shift=shift_vec, order=1, prefilter=False)
    return _shift_cpu(arr, shift=shift_vec, order=1, prefilter=False)


def _ssim(arr1: Any, arr2: Any, win_size: int) -> float:
    """SSIM wrapper that routes to GPU kernel or CPU skimage."""
    if USING_GPU and "ssim_cuda" in globals():
        return float(ssim_cuda(arr1, arr2, win_size=win_size))
    arr1_np = np.asarray(arr1)
    arr2_np = np.asarray(arr2)
    data_range = float(arr1_np.max() - arr1_np.min())
    if data_range == 0:
        data_range = 1.0
    return float(_ssim_cpu(arr1_np, arr2_np, win_size=win_size, data_range=data_range))


def _register_pair_worker(args: Tuple) -> Tuple:
    """
    Worker function for parallel registration of a tile pair.

    Parameters
    ----------
    args : tuple
        (i_pos, j_pos, patch_i, patch_j, df, sw, th, max_shift)

    Returns
    -------
    tuple
        (i_pos, j_pos, dy_s, dx_s, score) or (i_pos, j_pos, None, None, None) on failure
    """
    i_pos, j_pos, patch_i, patch_j, df, sw, th, max_shift = args

    try:
        # Downsample
        reduce_block = (1, df[0], df[1]) if patch_i.ndim == 3 else tuple(df)
        g1 = block_reduce(patch_i, reduce_block, np.mean)
        g2 = block_reduce(patch_j, reduce_block, np.mean)

        # Squeeze to 2D if needed
        while g1.ndim > 2 and g1.shape[0] == 1:
            g1 = g1[0]
            g2 = g2[0]

        # Match histograms
        g2 = match_histograms(g2, g1)

        # Phase cross-correlation
        shift, _, _ = phase_cross_correlation(
            g1.astype(np.float32),
            g2.astype(np.float32),
            normalization="phase",
            upsample_factor=10,
        )

        # Apply shift and compute SSIM
        g2s = _shift_array(g2, shift_vec=shift)
        ssim_val = _ssim(g1, g2s, win_size=sw)

        # Scale shift back to original resolution
        dy_s, dx_s = int(np.round(shift[0] * df[0])), int(np.round(shift[1] * df[1]))

        # Check thresholds
        if th != 0.0 and ssim_val < th:
            return (i_pos, j_pos, None, None, None)
        if abs(dy_s) > max_shift[0] or abs(dx_s) > max_shift[1]:
            return (i_pos, j_pos, None, None, None)

        return (i_pos, j_pos, dy_s, dx_s, round(ssim_val, 3))

    except Exception:
        return (i_pos, j_pos, None, None, None)


@njit(parallel=True)
def _accumulate_tile_shard(
    fused: np.ndarray,
    weight: np.ndarray,
    sub: np.ndarray,
    w2d: np.ndarray,
    y_off: int,
    x_off: int,
) -> None:
    """
    Weighted accumulation of a 2D sub-tile into the fused buffer.

    Parameters
    ----------
    fused : float32[C, Y, X]
        Accumulation buffer.
    weight : float32[C, Y, X]
        Weight accumulation buffer.
    sub : float32[C, Y, X]
        Sub-tile to blend.
    w2d : float32[Y, X]
        Weight profile.
    y_off, x_off : int
        Offsets of sub-tile in the fused volume.
    """
    C, Yp, Xp = fused.shape
    _, sub_Y, sub_X = sub.shape
    total = sub_Y * sub_X

    for idx in prange(total):
        y_i = idx // sub_X
        x_i = idx % sub_X
        gy = y_off + y_i
        gx = x_off + x_i
        if gy < 0 or gy >= Yp or gx < 0 or gx >= Xp:
            continue
        w_val = w2d[y_i, x_i]
        for c in range(C):
            fused[c, gy, gx] += sub[c, y_i, x_i] * w_val
            weight[c, gy, gx] += w_val


@njit(parallel=True)
def _normalize_shard(fused: np.ndarray, weight: np.ndarray) -> None:
    """
    Normalize the fused buffer by its weight buffer, in-place.

    Parameters
    ----------
    fused : float32[C, Y, X]
        Accumulation buffer to normalize.
    weight : float32[C, Y, X]
        Corresponding weights.
    """
    C, Yp, Xp = fused.shape
    total = C * Yp * Xp

    for idx in prange(total):
        c = idx // (Yp * Xp)
        rem = idx % (Yp * Xp)
        y_i = rem // Xp
        x_i = rem % Xp
        w_val = weight[c, y_i, x_i]
        fused[c, y_i, x_i] = fused[c, y_i, x_i] / w_val if w_val > 0 else 0.0


@njit(parallel=True)
def _blend_numba_2d(
    sub_i: np.ndarray,
    sub_j: np.ndarray,
    wy_i: np.ndarray,
    wx_i: np.ndarray,
    wy_j: np.ndarray,
    wx_j: np.ndarray,
    out_f: np.ndarray,
) -> np.ndarray:
    """
    Feather-blend two overlapping 2D sub-tiles.

    Parameters
    ----------
    sub_i, sub_j : (dy, dx) float32
        Input sub-tiles.
    wy_i, wx_i : 1D float32
        Weight profiles for sub_i.
    wy_j, wx_j : 1D float32
        Weight profiles for sub_j.
    out_f : (dy, dx) float32
        Pre-allocated output buffer.

    Returns
    -------
    out_f : (dy, dx) float32
        Blended result.
    """
    dy, dx = sub_i.shape

    for y in prange(dy):
        wi_y = wy_i[y]
        wj_y = wy_j[y]
        for x in range(dx):
            wi = wi_y * wx_i[x]
            wj = wj_y * wx_j[x]
            tot = wi + wj
            if tot > 1e-6:
                out_f[y, x] = (wi * sub_i[y, x] + wj * sub_j[y, x]) / tot
            else:
                out_f[y, x] = sub_i[y, x]
    return out_f


class TileFusion:
    """
    GPU/CPU-accelerated tile registration and fusion for 2D OME-TIFF stacks.

    Parameters
    ----------
    tiff_path : str or Path
        Path to the OME-TIFF file containing tiled images with stage positions.
    output_path : str or Path, optional
        Output path for fused Zarr. If None, derived from input path.
    blend_pixels : tuple of int
        Feather widths (by, bx).
    downsample_factors : tuple of int
        Block-reduce factors for registration.
    ssim_window : int
        Window size for SSIM.
    threshold : float
        SSIM acceptance threshold.
    multiscale_factors : sequence of int
        Downsampling factors for multiscale.
    max_workers : int
        Maximum parallel I/O workers.
    debug : bool
        If True, prints debug info.
    metrics_filename : str
        Filename for storing registration metrics.
    channel_to_use : int
        Channel index for registration.
    multiscale_downsample : str
        Either "stride" (default) or "block_mean" to control multiscale reduction.
    """

    def __init__(
        self,
        tiff_path: Union[str, Path],
        output_path: Union[str, Path] = None,
        blend_pixels: Tuple[int, int] = (0, 0),  # No blending by default
        downsample_factors: Tuple[int, int] = (1, 1),
        ssim_window: int = 15,
        threshold: float = 0.5,
        multiscale_factors: Sequence[int] = (2, 4, 8, 16),
        resolution_multiples: Sequence[Union[int, Sequence[int]]] = (
            (1, 1),
            (2, 2),
            (4, 4),
            (8, 8),
            (16, 16),
        ),
        max_workers: int = 8,
        debug: bool = False,
        metrics_filename: str = "metrics.json",
        channel_to_use: int = 0,
        multiscale_downsample: str = "stride",
    ):
        self.tiff_path = Path(tiff_path)
        if not self.tiff_path.exists():
            raise FileNotFoundError(f"Path not found: {self.tiff_path}")

        self.output_path = (
            Path(output_path)
            if output_path
            else self.tiff_path.parent / f"{self.tiff_path.stem}_fused.ome.zarr"
        )

        # Detect format: folder (SQUID) vs file (OME-TIFF)
        self._is_squid_format = self.tiff_path.is_dir()
        if self._is_squid_format:
            self._load_squid_metadata()
        else:
            self._load_ome_tiff_metadata()

        self.downsample_factors = tuple(downsample_factors)
        self.ssim_window = int(ssim_window)
        self.threshold = float(threshold)
        self.multiscale_factors = tuple(multiscale_factors)
        self.resolution_multiples = [
            r if hasattr(r, "__len__") else (r, r) for r in resolution_multiples
        ]
        self._max_workers = int(max_workers)
        self._debug = bool(debug)
        self.metrics_filename = metrics_filename
        self._blend_pixels = tuple(blend_pixels)
        self.channel_to_use = channel_to_use
        if multiscale_downsample not in ("stride", "block_mean"):
            raise ValueError('multiscale_downsample must be "stride" or "block_mean".')
        self.multiscale_downsample = multiscale_downsample

        self._update_profiles()
        self.chunk_shape = (1, 1024, 1024)
        self.chunk_y, self.chunk_x = self.chunk_shape[-2:]

        self.pairwise_metrics: Dict[Tuple[int, int], Tuple[int, int, float]] = {}
        self.global_offsets: Optional[np.ndarray] = None
        self.offset: Optional[Tuple[float, float]] = None
        self.unpadded_shape: Optional[Tuple[int, int]] = None
        self.padded_shape: Optional[Tuple[int, int]] = None
        self.pad = (0, 0)
        self.fused_ts = None

    def _load_ome_tiff_metadata(self) -> None:
        """Load metadata from OME-TIFF file."""
        with tifffile.TiffFile(self.tiff_path) as tif:
            if not tif.ome_metadata:
                raise ValueError("TIFF file does not contain OME metadata")

            root = ET.fromstring(tif.ome_metadata)
            ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
            images = root.findall(".//ome:Image", ns)

            self.n_tiles = len(images)
            self.n_series = len(tif.series)

            first_series = tif.series[0]
            self.Y, self.X = first_series.shape[-2:]
            self.channels = 1
            self.time_dim = 1
            self.position_dim = self.n_tiles

            first_pixels = images[0].find("ome:Pixels", ns)
            px_x = float(first_pixels.get("PhysicalSizeX", 1.0))
            px_y = float(first_pixels.get("PhysicalSizeY", 1.0))
            self._pixel_size = (px_y, px_x)

            self._tile_positions = []
            for img in images:
                pixels = img.find("ome:Pixels", ns)
                planes = pixels.findall("ome:Plane", ns)
                if planes:
                    p = planes[0]
                    x = float(p.get("PositionX", 0))
                    y = float(p.get("PositionY", 0))
                    self._tile_positions.append((y, x))
                else:
                    self._tile_positions.append((0.0, 0.0))

    def _load_squid_metadata(self) -> None:
        """Load metadata from SQUID folder format (individual TIFFs + coordinates.csv)."""
        import pandas as pd
        import json

        # Find the subfolder containing images (usually "0" for single z-level)
        subfolders = [d for d in self.tiff_path.iterdir() if d.is_dir()]
        if subfolders:
            self._squid_image_folder = subfolders[0]
        else:
            self._squid_image_folder = self.tiff_path

        # Load coordinates from the subfolder's coordinates.csv
        coords_path = self._squid_image_folder / "coordinates.csv"
        if not coords_path.exists():
            coords_path = self.tiff_path / "coordinates.csv"
        if not coords_path.exists():
            raise FileNotFoundError(f"coordinates.csv not found in {self.tiff_path}")

        self._squid_coords = pd.read_csv(coords_path)
        self.n_tiles = len(self._squid_coords)
        self.n_series = self.n_tiles

        # Get list of channels from TIFF filenames
        tiff_files = list(self._squid_image_folder.glob("*.tiff"))
        if not tiff_files:
            tiff_files = list(self._squid_image_folder.glob("*.tif"))

        # Extract unique channel names (e.g., "Fluorescence_405_nm_Ex")
        channel_names = set()
        for f in tiff_files:
            # Pattern: manual_{fov}_{z}_Fluorescence_{wavelength}.tiff
            parts = f.stem.split("_")
            if len(parts) >= 4:
                channel_name = "_".join(parts[3:])  # e.g., "Fluorescence_405_nm_Ex"
                channel_names.add(channel_name)

        self._squid_channels = sorted(channel_names)
        self.channels = len(self._squid_channels)
        self.time_dim = 1
        self.position_dim = self.n_tiles

        # Read first image to get dimensions
        first_fov = self._squid_coords["fov"].iloc[0]
        first_channel = self._squid_channels[0]
        first_img_path = self._squid_image_folder / f"manual_{first_fov}_0_{first_channel}.tiff"
        if not first_img_path.exists():
            first_img_path = self._squid_image_folder / f"manual_{first_fov}_0_{first_channel}.tif"

        first_img = tifffile.imread(first_img_path)
        self.Y, self.X = first_img.shape[-2:]

        # Load pixel size from acquisition parameters
        params_path = self.tiff_path / "acquisition parameters.json"
        if params_path.exists():
            with open(params_path) as f:
                params = json.load(f)
            magnification = params.get("objective", {}).get("magnification", 10.0)
            sensor_pixel_um = params.get("sensor_pixel_size_um", 7.52)
            pixel_size_um = sensor_pixel_um / magnification
        else:
            pixel_size_um = 0.752  # Default for 10x

        self._pixel_size = (pixel_size_um, pixel_size_um)

        # Convert mm coordinates to µm and store as (y, x)
        self._tile_positions = []
        for _, row in self._squid_coords.iterrows():
            x_um = row["x (mm)"] * 1000
            y_um = row["y (mm)"] * 1000
            self._tile_positions.append((y_um, x_um))

        # Store FOV indices for reading tiles
        self._squid_fov_indices = self._squid_coords["fov"].tolist()

    @property
    def tile_positions(self) -> List[Tuple[float, float]]:
        """Stage positions for each tile (y, x)."""
        return self._tile_positions

    @tile_positions.setter
    def tile_positions(self, positions: Sequence[Tuple[float, float]]):
        if any(len(p) != 2 for p in positions):
            raise ValueError("Each position must be a 2-tuple.")
        self._tile_positions = [tuple(p) for p in positions]

    @property
    def pixel_size(self) -> Tuple[float, float]:
        """Pixel size in (y, x)."""
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, size: Tuple[float, float]):
        if len(size) != 2:
            raise ValueError("pixel_size must be a 2-tuple.")
        self._pixel_size = tuple(float(x) for x in size)

    @property
    def blend_pixels(self) -> Tuple[int, int]:
        """Feather widths in (by, bx)."""
        return self._blend_pixels

    @blend_pixels.setter
    def blend_pixels(self, bp: Tuple[int, int]):
        if len(bp) != 2:
            raise ValueError("blend_pixels must be a 2-tuple.")
        self._blend_pixels = tuple(bp)
        self._update_profiles()

    @property
    def max_workers(self) -> int:
        """Maximum concurrent I/O workers."""
        return self._max_workers

    @max_workers.setter
    def max_workers(self, mw: int):
        if mw < 1:
            raise ValueError("max_workers must be >= 1.")
        self._max_workers = int(mw)

    @property
    def debug(self) -> bool:
        """Debug flag for verbose logging."""
        return self._debug

    @debug.setter
    def debug(self, flag: bool):
        self._debug = bool(flag)

    def _update_profiles(self) -> None:
        """Recompute 1D feather profiles from blend_pixels."""
        by, bx = self._blend_pixels
        self.y_profile = self._make_1d_profile(self.Y, by)
        self.x_profile = self._make_1d_profile(self.X, bx)

    @staticmethod
    def _make_1d_profile(length: int, blend: int) -> np.ndarray:
        """
        Create a linear ramp profile over `blend` pixels at each end.

        Parameters
        ----------
        length : int
            Number of pixels.
        blend : int
            Ramp width.

        Returns
        -------
        prof : (length,) float32
            Linear profile.
        """
        blend = min(blend, length // 2)
        prof = np.ones(length, dtype=np.float32)
        if blend > 0:
            ramp = np.linspace(0, 1, blend, endpoint=False, dtype=np.float32)
            prof[:blend] = ramp
            prof[-blend:] = ramp[::-1]
        return prof

    def _read_tile(self, tile_idx: int) -> np.ndarray:
        """Read a single tile from the input data (all channels)."""
        if self._is_squid_format:
            return self._read_squid_tile_all_channels(tile_idx)
        else:
            with tifffile.TiffFile(self.tiff_path) as tif:
                arr = tif.series[tile_idx].asarray()
            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]
            # Flip along Y axis to correct orientation
            arr = np.flip(arr, axis=-2)
            return arr.astype(np.float32)

    def _read_squid_tile_all_channels(self, tile_idx: int) -> np.ndarray:
        """Read all channels of a tile from SQUID folder format."""
        fov = self._squid_fov_indices[tile_idx]

        channels = []
        for channel_name in self._squid_channels:
            img_path = self._squid_image_folder / f"manual_{fov}_0_{channel_name}.tiff"
            if not img_path.exists():
                img_path = self._squid_image_folder / f"manual_{fov}_0_{channel_name}.tif"
            arr = tifffile.imread(img_path)
            channels.append(arr)

        # Stack channels: (C, Y, X)
        stacked = np.stack(channels, axis=0)
        return stacked.astype(np.float32)

    def _read_squid_tile(self, tile_idx: int, channel_idx: int = None) -> np.ndarray:
        """Read a single channel of a tile from SQUID folder format."""
        if channel_idx is None:
            channel_idx = self.channel_to_use

        fov = self._squid_fov_indices[tile_idx]
        channel_name = self._squid_channels[channel_idx]

        img_path = self._squid_image_folder / f"manual_{fov}_0_{channel_name}.tiff"
        if not img_path.exists():
            img_path = self._squid_image_folder / f"manual_{fov}_0_{channel_name}.tif"

        arr = tifffile.imread(img_path)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        return arr.astype(np.float32)

    def _read_tile_region(
        self,
        tile_idx: int,
        y_slice: slice,
        x_slice: slice,
    ) -> np.ndarray:
        """Read a region of a tile from the input data."""
        if self._is_squid_format:
            arr = self._read_squid_tile(tile_idx)
            return arr[:, y_slice, x_slice]
        else:
            with tifffile.TiffFile(self.tiff_path) as tif:
                arr = tif.series[tile_idx].asarray()
            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]
            # Flip along Y axis to correct orientation
            arr = np.flip(arr, axis=-2)
            return arr[:, y_slice, x_slice].astype(np.float32)

    @staticmethod
    def register_and_score(
        g1: Any,
        g2: Any,
        win_size: int,
        debug: bool = False,
    ) -> Union[Tuple[Tuple[float, float], float], Tuple[None, None]]:
        """
        Histogram-match g2->g1, compute subpixel shift, and SSIM.

        Parameters
        ----------
        g1, g2 : array-like
            Fixed and moving patches (YX).
        win_size : int
            SSIM window.
        debug : bool
            If True, print intermediate info.

        Returns
        -------
        shift : (dy, dx)
            Subpixel shift.
        ssim_val : float
            SSIM score.
        """
        arr1 = xp.asarray(g1, dtype=xp.float32)
        arr2 = xp.asarray(g2, dtype=xp.float32)
        while arr1.ndim > 2 and arr1.shape[0] == 1:
            arr1 = arr1[0]
            arr2 = arr2[0]

        arr2 = match_histograms(arr2, arr1)
        shift, _, _ = phase_cross_correlation(
            arr1,
            arr2,
            disambiguate=True,
            normalization="phase",
            upsample_factor=10,
            overlap_ratio=0.5,
        )
        shift_apply = xp.asarray(shift, dtype=xp.float32)
        g2s = _shift_array(arr2, shift_vec=shift_apply)
        ssim_val = _ssim(arr1, g2s, win_size=win_size)
        out_shift = cp.asnumpy(shift_apply) if USING_GPU else np.asarray(shift_apply)
        return tuple(float(s) for s in out_shift), float(ssim_val)

    def refine_tile_positions_with_cross_correlation(
        self,
        downsample_factors: Tuple[int, int] = None,
        ssim_window: int = None,
        ch_idx: int = 0,
        threshold: float = None,
        parallel: bool = True,
    ) -> None:
        """
        Detect and score overlaps between neighboring tile pairs via cross-correlation.

        Parameters
        ----------
        downsample_factors : tuple of int, optional
            Block-reduce factors for registration.
        ssim_window : int, optional
            Window size for SSIM.
        ch_idx : int, optional
            Channel to use.
        threshold : float, optional
            SSIM threshold to accept a link.
        parallel : bool, optional
            If True, use multiprocessing for CPU mode. Default True.
            Ignored when using GPU (GPU is already fast).
        """
        df = downsample_factors or self.downsample_factors
        sw = ssim_window or self.ssim_window
        th = threshold if threshold is not None else self.threshold
        self.pairwise_metrics.clear()

        n_pos = self.position_dim
        max_shift = (100, 100)

        # Build list of adjacent tile pairs (horizontal or vertical neighbors only)
        adjacent_pairs = []
        min_overlap = 15  # At least 15 pixels overlap needed
        for i_pos in range(n_pos):
            for j_pos in range(i_pos + 1, n_pos):
                phys = np.array(self._tile_positions[j_pos]) - np.array(self._tile_positions[i_pos])
                vox_off = np.round(phys / np.array(self._pixel_size)).astype(int)
                dy, dx = vox_off

                overlap_y = self.Y - abs(dy)
                overlap_x = self.X - abs(dx)

                # Check if tiles are adjacent (one dimension has full overlap, other has partial)
                is_horizontal_neighbor = abs(dy) < min_overlap and overlap_x >= min_overlap
                is_vertical_neighbor = abs(dx) < min_overlap and overlap_y >= min_overlap

                if is_horizontal_neighbor or is_vertical_neighbor:
                    adjacent_pairs.append((i_pos, j_pos, dy, dx, overlap_y, overlap_x))

        if self._debug:
            print(f"Found {len(adjacent_pairs)} adjacent tile pairs to register")

        # Compute bounds for each pair
        pair_bounds = []
        for i_pos, j_pos, dy, dx, overlap_y, overlap_x in adjacent_pairs:
            bounds_i_y = (max(0, dy), min(self.Y, self.Y + dy))
            bounds_i_x = (max(0, dx), min(self.X, self.X + dx))
            bounds_j_y = (max(0, -dy), min(self.Y, self.Y - dy))
            bounds_j_x = (max(0, -dx), min(self.X, self.X - dx))

            if bounds_i_y[1] > bounds_i_y[0] and bounds_i_x[1] > bounds_i_x[0]:
                pair_bounds.append((i_pos, j_pos, bounds_i_y, bounds_i_x, bounds_j_y, bounds_j_x))

        # Use parallel processing for CPU mode, sequential for GPU
        use_parallel = parallel and not USING_GPU and len(pair_bounds) > 4

        if use_parallel:
            self._register_parallel(pair_bounds, df, sw, th, max_shift)
        else:
            self._register_sequential(pair_bounds, df, sw, th, max_shift)

    def _register_parallel(
        self,
        pair_bounds: List[Tuple],
        df: Tuple[int, int],
        sw: int,
        th: float,
        max_shift: Tuple[int, int],
    ) -> None:
        """Register tile pairs using parallel processing with batching (CPU mode)."""
        import psutil

        # Calculate batch size based on available RAM and tile size
        available_ram = psutil.virtual_memory().available
        patch_size_est = self.Y * self.X * 4 * 2  # float32, 2 patches per pair
        max_pairs_in_memory = int(available_ram * 0.3 / patch_size_est)
        batch_size = max(16, max_pairs_in_memory)  # Minimum 16 to keep threads busy

        n_pairs = len(pair_bounds)
        n_batches = (n_pairs + batch_size - 1) // batch_size
        n_workers = min(cpu_count(), batch_size, 8)

        if n_batches > 1:
            print(f"Processing {n_pairs} pairs in {n_batches} batches (batch_size={batch_size})")

        def read_pair_patches(args):
            i_pos, j_pos, bounds_i_y, bounds_i_x, bounds_j_y, bounds_j_x = args
            try:
                patch_i = self._read_tile_region(
                    i_pos, slice(bounds_i_y[0], bounds_i_y[1]), slice(bounds_i_x[0], bounds_i_x[1])
                )
                patch_j = self._read_tile_region(
                    j_pos, slice(bounds_j_y[0], bounds_j_y[1]), slice(bounds_j_x[0], bounds_j_x[1])
                )
                return (i_pos, j_pos, patch_i, patch_j)
            except Exception:
                return (i_pos, j_pos, None, None)

        # Process in batches to limit memory usage
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_pairs)
            batch = pair_bounds[start:end]

            # Read this batch's patches
            with ThreadPoolExecutor(max_workers=8) as io_executor:
                patches = list(io_executor.map(read_pair_patches, batch))

            # Filter and prepare work items
            work_items = [
                (i, j, pi, pj, df, sw, th, max_shift) for i, j, pi, pj in patches if pi is not None
            ]

            # Register this batch
            desc = f"register {batch_idx+1}/{n_batches}" if n_batches > 1 else "register"
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = list(
                    tqdm(
                        executor.map(_register_pair_worker, work_items),
                        total=len(work_items),
                        desc=desc,
                        leave=True,
                    )
                )

            # Collect results
            for i_pos, j_pos, dy_s, dx_s, score in results:
                if dy_s is not None:
                    self.pairwise_metrics[(i_pos, j_pos)] = (dy_s, dx_s, score)

            # Free memory
            del patches, work_items, results
            gc.collect()

    def _register_sequential(
        self,
        pair_bounds: List[Tuple],
        df: Tuple[int, int],
        sw: int,
        th: float,
        max_shift: Tuple[int, int],
    ) -> None:
        """Register tile pairs sequentially (GPU mode or small datasets)."""
        io_executor = ThreadPoolExecutor(max_workers=2)

        for i_pos, j_pos, bounds_i_y, bounds_i_x, bounds_j_y, bounds_j_x in tqdm(
            pair_bounds, desc="register", leave=True
        ):

            def read_patch(idx, y_bounds, x_bounds):
                return self._read_tile_region(
                    idx, slice(y_bounds[0], y_bounds[1]), slice(x_bounds[0], x_bounds[1])
                )

            try:
                # Submit both reads in parallel, then wait
                future_i = io_executor.submit(read_patch, i_pos, bounds_i_y, bounds_i_x)
                future_j = io_executor.submit(read_patch, j_pos, bounds_j_y, bounds_j_x)
                patch_i = future_i.result()
                patch_j = future_j.result()
            except Exception as e:
                if self._debug:
                    print(f"Error reading patches for ({i_pos}, {j_pos}): {e}")
                continue

            arr_i = xp.asarray(patch_i)
            arr_j = xp.asarray(patch_j)

            reduce_block = (1, df[0], df[1]) if arr_i.ndim == 3 else tuple(df)
            g1 = block_reduce(arr_i, reduce_block, xp.mean)
            g2 = block_reduce(arr_j, reduce_block, xp.mean)

            try:
                shift_ds, ssim_val = self.register_and_score(g1, g2, win_size=sw, debug=self._debug)
            except Exception as e:
                if self._debug:
                    print(f"Registration failed for ({i_pos}, {j_pos}): {e}")
                continue

            if shift_ds is None:
                continue
            score = float(max(ssim_val, 1e-6))
            if th != 0.0 and score < th:
                continue

            dy_s, dx_s = [int(np.round(shift_ds[k] * df[k])) for k in range(2)]

            if abs(dy_s) > max_shift[0] or abs(dx_s) > max_shift[1]:
                if self._debug:
                    print(
                        f"Dropping link {(i_pos, j_pos)} shift=({dy_s}, {dx_s}) - exceeds max {max_shift}"
                    )
                continue

            self.pairwise_metrics[(i_pos, j_pos)] = (dy_s, dx_s, round(score, 3))

        io_executor.shutdown(wait=True)

    @staticmethod
    def _solve_global(
        links: List[Dict[str, Any]], n_tiles: int, fixed_indices: List[int]
    ) -> np.ndarray:
        """
        Solve a linear least-squares for all 2 axes at once,
        given weighted pairwise links and fixed tile indices.
        """
        shifts = np.zeros((n_tiles, 2), dtype=np.float64)
        for axis in range(2):
            m = len(links) + len(fixed_indices)
            A = np.zeros((m, n_tiles), dtype=np.float64)
            b = np.zeros(m, dtype=np.float64)
            row = 0
            for link in links:
                i, j = link["i"], link["j"]
                t, w = link["t"][axis], link["w"]
                A[row, j] = w
                A[row, i] = -w
                b[row] = w * t
                row += 1
            for idx in fixed_indices:
                A[row, idx] = 1.0
                b[row] = 0.0
                row += 1
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            shifts[:, axis] = sol
        return shifts

    def _two_round_opt(
        self,
        links: List[Dict[str, Any]],
        n_tiles: int,
        fixed_indices: List[int],
        rel_thresh: float,
        abs_thresh: float,
        iterative: bool,
    ) -> np.ndarray:
        """
        Perform two-round (or iterative two-round) robust optimization:
        1. Solve on all links.
        2. Remove any link whose residual > max(abs_thresh, rel_thresh * median(residuals)).
        3. Re-solve on the remaining links.
        If iterative=True, repeat step 2 + 3 until no more links are removed.
        """
        shifts = self._solve_global(links, n_tiles, fixed_indices)

        def compute_res(ls: List[Dict[str, Any]], sh: np.ndarray) -> np.ndarray:
            return np.array([np.linalg.norm(sh[l["j"]] - sh[l["i"]] - l["t"]) for l in ls])

        work = links.copy()
        res = compute_res(work, shifts)
        if len(res) == 0:
            return shifts
        cutoff = max(abs_thresh, rel_thresh * np.median(res))
        outliers = set(np.where(res > cutoff)[0])

        if iterative:
            while outliers:
                for k in sorted(outliers, reverse=True):
                    work.pop(k)
                if not work:
                    break
                shifts = self._solve_global(work, n_tiles, fixed_indices)
                res = compute_res(work, shifts)
                if len(res) == 0:
                    break
                cutoff = max(abs_thresh, rel_thresh * np.median(res))
                outliers = set(np.where(res > cutoff)[0])
        else:
            for k in sorted(outliers, reverse=True):
                work.pop(k)
            if work:
                shifts = self._solve_global(work, n_tiles, fixed_indices)

        return shifts

    def optimize_shifts(
        self,
        method: str = "ONE_ROUND",
        rel_thresh: float = 0.3,
        abs_thresh: float = 5.0,
        iterative: bool = False,
    ) -> None:
        """
        Globally optimize tile shifts using either:
          - ONE_ROUND: single least-squares solve, or
          - TWO_ROUND_SIMPLE: remove outliers once then re-solve, or
          - TWO_ROUND_ITERATIVE: remove outliers repeatedly until none remain.

        Parameters
        ----------
        method : {'ONE_ROUND', 'TWO_ROUND_SIMPLE', 'TWO_ROUND_ITERATIVE'}
        rel_thresh : float
            Relative threshold (fraction of median residual) for link removal.
        abs_thresh : float
            Absolute threshold for link removal.
        iterative : bool
            If True, repeat outlier removal until convergence.
        """
        links: List[Dict[str, Any]] = []
        for (i, j), v in self.pairwise_metrics.items():
            links.append(
                {"i": i, "j": j, "t": np.array(v[:2], dtype=np.float64), "w": np.sqrt(v[2])}
            )
        if not links:
            self.global_offsets = np.zeros((self.position_dim, 2), dtype=np.float64)
            return

        n = len(self._tile_positions)
        fixed = [0]

        if method == "ONE_ROUND":
            d_opt = self._solve_global(links, n, fixed)
        elif method.startswith("TWO_ROUND"):
            d_opt = self._two_round_opt(
                links, n, fixed, rel_thresh, abs_thresh, method.endswith("ITERATIVE")
            )
        else:
            raise ValueError(f"Unknown method {method}")

        self.global_offsets = d_opt

    def save_pairwise_metrics(self, filepath: Union[str, Path]) -> None:
        """Save pairwise_metrics to a JSON file."""
        path = Path(filepath)
        out = {f"{i},{j}": list(v) for (i, j), v in self.pairwise_metrics.items()}
        with open(path, "w") as f:
            json.dump(out, f)

    def load_pairwise_metrics(self, filepath: Union[str, Path]) -> None:
        """Load pairwise_metrics from a JSON file."""
        path = Path(filepath)
        with open(path, "r") as f:
            data = json.load(f)
        self.pairwise_metrics = {tuple(map(int, k.split(","))): tuple(v) for k, v in data.items()}

    def _compute_fused_image_space(self) -> None:
        """Compute fused image physical shape and offset based on tile positions."""
        pos = np.array(self._tile_positions)
        min_y, min_x = pos.min(axis=0)
        max_y = pos[:, 0].max() + self.Y * self._pixel_size[0]
        max_x = pos[:, 1].max() + self.X * self._pixel_size[1]

        sy = int(np.ceil((max_y - min_y) / self._pixel_size[0]))
        sx = int(np.ceil((max_x - min_x) / self._pixel_size[1]))

        self.unpadded_shape = (sy, sx)
        self.offset = (min_y, min_x)
        self.center = ((max_x - min_x) / 2, (max_y - min_y) / 2)

    def _pad_to_chunk_multiple(self) -> None:
        """Pad unpadded_shape to exact multiples of chunk shape."""
        ty, tx = self.chunk_y, self.chunk_x
        sy, sx = self.unpadded_shape

        py = (-sy) % ty
        px = (-sx) % tx

        self.pad = (py, px)
        self.padded_shape = (sy + py, sx + px)

    def _create_fused_tensorstore(
        self,
        output_path: Union[str, Path],
    ) -> None:
        """
        Create the output Zarr v3 store for the fused image.

        Parameters
        ----------
        output_path : str or Path
            Path to create fused store.
        """
        out = Path(output_path)
        full_shape = [1, self.channels, *self.padded_shape]
        shard_chunk = [1, 1, self.chunk_y * 2, self.chunk_x * 2]
        codec_chunk = [1, 1, self.chunk_y, self.chunk_x]
        self.shard_chunk = shard_chunk

        config = {
            "context": {
                "file_io_concurrency": {"limit": self.max_workers},
                "data_copy_concurrency": {"limit": self.max_workers},
            },
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(out)},
            "metadata": {
                "shape": full_shape,
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": shard_chunk}},
                "chunk_key_encoding": {"name": "default"},
                "codecs": [
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
                ],
                "data_type": "uint16",
                "dimension_names": ["t", "c", "y", "x"],
            },
        }

        self.fused_ts = ts.open(config, create=True, open=True).result()

    def _fuse_tiles(
        self, mode: str = "blended", chunked: bool = True, ram_fraction: float = 0.4
    ) -> None:
        """Fuse all tiles into output.

        Parameters
        ----------
        mode : str
            Fusion mode: "blended" (weighted averaging, high quality) or
            "direct" (simple placement, ~5-10x faster but visible seams).
        chunked : bool
            If True, use memory-efficient chunked processing. Default True.
        ram_fraction : float
            Fraction of available RAM to use for block processing. Default 0.4.
        """
        if mode == "direct":
            return self._fuse_tiles_direct()
        if chunked:
            return self._fuse_tiles_chunked(ram_fraction)
        return self._fuse_tiles_full()

    def _fuse_tiles_direct(self) -> None:
        """Fuse tiles using direct placement (fast mode, no blending).

        This is ~5-10x faster than blended mode but produces visible seams
        at tile boundaries if there are intensity variations between tiles.
        Tiles are simply placed at their positions, with later tiles
        overwriting earlier ones in overlap regions.
        """
        import psutil

        offsets = [
            (
                int((y - self.offset[0]) / self._pixel_size[0]),
                int((x - self.offset[1]) / self._pixel_size[1]),
            )
            for (y, x) in self._tile_positions
        ]
        pad_Y, pad_X = self.padded_shape

        # Check if we can fit in memory
        available_ram = psutil.virtual_memory().available
        output_bytes = pad_Y * pad_X * self.channels * 2  # uint16
        use_memory = output_bytes < 0.45 * available_ram

        if use_memory:
            print(f"Direct mode: using in-memory buffer ({output_bytes / 1e9:.2f} GB)")
            # Process all channels at once in memory
            output = np.zeros((1, self.channels, pad_Y, pad_X), dtype=np.uint16)

            for t_idx in trange(len(offsets), desc="placing tiles", leave=True):
                oy, ox = offsets[t_idx]
                tile_all = self._read_tile(t_idx)  # (C, Y, X)

                # Calculate valid output region
                y_end = min(oy + self.Y, pad_Y)
                x_end = min(ox + self.X, pad_X)
                tile_h = y_end - oy
                tile_w = x_end - ox

                if tile_h > 0 and tile_w > 0:
                    output[0, :, oy:y_end, ox:x_end] = tile_all[:, :tile_h, :tile_w]

            # Write to TensorStore
            print("Writing to disk...")
            self.fused_ts[:].write(output).result()
            del output
        else:
            print(f"Direct mode: writing directly to disk (output {output_bytes / 1e9:.2f} GB)")
            # Write each tile directly to output (read each tile once)
            for t_idx in trange(len(offsets), desc="placing tiles", leave=True):
                oy, ox = offsets[t_idx]
                tile_all = self._read_tile(t_idx)  # Read once

                # Calculate valid output region
                y_end = min(oy + self.Y, pad_Y)
                x_end = min(ox + self.X, pad_X)
                tile_h = y_end - oy
                tile_w = x_end - ox

                if tile_h > 0 and tile_w > 0:
                    # Write all channels for this tile region
                    tile_region = tile_all[:, :tile_h, :tile_w].astype(np.uint16)
                    self.fused_ts[0:1, :, oy:y_end, ox:x_end].write(
                        tile_region[np.newaxis, ...]
                    ).result()

        gc.collect()

    def _fuse_tiles_full(self) -> None:
        """Fuse all tiles using full-image accumulator (legacy mode)."""
        offsets = [
            (
                int((y - self.offset[0]) / self._pixel_size[0]),
                int((x - self.offset[1]) / self._pixel_size[1]),
            )
            for (y, x) in self._tile_positions
        ]
        pad_Y, pad_X = self.padded_shape
        C = self.channels

        # Allocate multi-channel accumulators
        fused_block = np.zeros((C, pad_Y, pad_X), dtype=np.float32)
        weight_sum = np.zeros_like(fused_block)

        # Pre-compute weight profile
        w2d = self.y_profile[:, None] * self.x_profile[None, :]

        # Process each tile once, accumulate all channels
        for t_idx in trange(len(offsets), desc="fusing", leave=True):
            oy, ox = offsets[t_idx]
            tile_all = self._read_tile(t_idx)  # (C, Y, X)

            # Ensure tile has correct number of channels
            if tile_all.shape[0] == 1 and C > 1:
                tile_all = np.broadcast_to(tile_all, (C, tile_all.shape[1], tile_all.shape[2]))

            _accumulate_tile_shard(fused_block, weight_sum, tile_all, w2d, oy, ox)

        _normalize_shard(fused_block, weight_sum)

        # Write all channels at once
        self.fused_ts[0, :, :pad_Y, :pad_X].write(fused_block.astype(np.uint16)).result()

        del fused_block, weight_sum
        gc.collect()
        if USING_GPU and cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def _fuse_tiles_chunked(self, ram_fraction: float = 0.4) -> None:
        """Fuse tiles using memory-efficient super-chunk processing.

        Parameters
        ----------
        ram_fraction : float
            Fraction of available RAM to use for block processing.
        """
        import psutil

        # Calculate block size from available RAM
        available_ram = psutil.virtual_memory().available
        usable_ram = int(available_ram * ram_fraction)

        # Need 2 float32 arrays per channel (fused_block + weight_sum)
        bytes_per_pixel = 4 * 2 * self.channels  # float32 × 2 arrays × channels
        max_pixels = usable_ram // bytes_per_pixel
        block_size = int(np.sqrt(max_pixels))

        # Round to chunk boundary for efficient TensorStore writes
        block_size = (block_size // self.chunk_y) * self.chunk_y
        block_size = max(block_size, self.chunk_y * 2)  # Minimum 2 chunks

        # Cap block size for optimal performance (memory bandwidth vs tile re-reads)
        # Large blocks (>10K) cause memory bandwidth saturation during accumulation
        # and slower memory allocation. 8K-12K is optimal for most workloads.
        max_block_size = 10240  # ~0.8GB accumulator, good balance
        block_size = min(block_size, max_block_size)

        pad_Y, pad_X = self.padded_shape

        # If block covers entire image, use full mode (more efficient)
        if block_size >= max(pad_Y, pad_X):
            print(f"Image fits in RAM budget ({usable_ram / 1e9:.1f} GB), using full mode")
            return self._fuse_tiles_full()

        print(
            f"Using chunked mode: {block_size}×{block_size} blocks "
            f"({usable_ram / 1e9:.1f} GB RAM budget)"
        )

        # Build spatial index: tile bounds in output coordinates
        tile_bounds = []
        for y, x in self._tile_positions:
            oy = int((y - self.offset[0]) / self._pixel_size[0])
            ox = int((x - self.offset[1]) / self._pixel_size[1])
            tile_bounds.append((oy, oy + self.Y, ox, ox + self.X))

        # Count total blocks for progress
        n_blocks_y = (pad_Y + block_size - 1) // block_size
        n_blocks_x = (pad_X + block_size - 1) // block_size
        total_blocks = n_blocks_y * n_blocks_x
        C = self.channels

        # Process blocks (all channels at once per block)
        block_idx = 0
        for block_y in range(0, pad_Y, block_size):
            for block_x in range(0, pad_X, block_size):
                block_idx += 1
                by_end = min(block_y + block_size, pad_Y)
                bx_end = min(block_x + block_size, pad_X)
                bh, bw = by_end - block_y, bx_end - block_x

                # Find overlapping tiles
                overlapping = []
                for t_idx, (ty0, ty1, tx0, tx1) in enumerate(tile_bounds):
                    if ty1 > block_y and ty0 < by_end and tx1 > block_x and tx0 < bx_end:
                        overlapping.append(t_idx)

                if not overlapping:
                    continue  # Empty block

                # Allocate multi-channel block accumulator
                fused_block = np.zeros((C, bh, bw), dtype=np.float32)
                weight_sum = np.zeros_like(fused_block)

                # Accumulate overlapping tiles (read each tile once for all channels)
                desc = f"block {block_idx}/{total_blocks}"
                for t_idx in tqdm(overlapping, desc=desc, leave=False):
                    tile_all = self._read_tile(t_idx)  # Read once

                    # Ensure tile has correct number of channels
                    if tile_all.shape[0] == 1 and C > 1:
                        tile_all = np.broadcast_to(
                            tile_all, (C, tile_all.shape[1], tile_all.shape[2])
                        )

                    ty0, ty1, tx0, tx1 = tile_bounds[t_idx]

                    # Compute overlap region in block coordinates
                    oy0 = max(ty0, block_y) - block_y
                    oy1 = min(ty1, by_end) - block_y
                    ox0 = max(tx0, block_x) - block_x
                    ox1 = min(tx1, bx_end) - block_x

                    # Source region in tile coordinates
                    sy0 = max(block_y - ty0, 0)
                    sy1 = sy0 + (oy1 - oy0)
                    sx0 = max(block_x - tx0, 0)
                    sx1 = sx0 + (ox1 - ox0)

                    # Get weight for this region
                    w2d = self.y_profile[sy0:sy1, None] * self.x_profile[None, sx0:sx1]

                    # Accumulate all channels at once
                    for c in range(C):
                        fused_block[c, oy0:oy1, ox0:ox1] += tile_all[c, sy0:sy1, sx0:sx1] * w2d
                        weight_sum[c, oy0:oy1, ox0:ox1] += w2d

                # Normalize and write all channels
                mask = weight_sum > 0
                fused_block[mask] /= weight_sum[mask]

                self.fused_ts[0, :, block_y:by_end, block_x:bx_end].write(
                    fused_block.astype(np.uint16)
                ).result()

                del fused_block, weight_sum

        gc.collect()
        if USING_GPU and cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def _create_multiscales(
        self,
        omezarr_path: Path,
        factors: Sequence[int] = (2, 4, 8),
    ) -> None:
        """
        Build NGFF multiscales by downsampling Y/X iteratively.

        Parameters
        ----------
        omezarr_path : Path
            Root of the NGFF group.
        factors : sequence of int
            Downsampling factors per scale.
        """
        inp = None
        for idx, factor in enumerate(factors):
            out_path = omezarr_path / f"scale{idx + 1}" / "image"
            if inp is not None:
                del inp
            prev = omezarr_path / f"scale{idx}" / "image"
            inp = ts.open(
                {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(prev)}}
            ).result()

            factor_to_use = factors[idx] // factors[idx - 1] if idx > 0 else factors[0]
            _, _, Y, X = inp.shape
            new_y, new_x = Y // factor_to_use, X // factor_to_use

            chunk_y = min(1024, new_y)
            chunk_x = min(1024, new_x)

            self.padded_shape = (new_y, new_x)
            self.chunk_y, self.chunk_x = chunk_y, chunk_x

            self._create_fused_tensorstore(output_path=out_path)

            for y0 in trange(0, new_y, chunk_y, desc=f"scale{idx + 1}", leave=True):
                by = min(chunk_y, new_y - y0)
                in_y0 = y0 * factor_to_use
                in_y1 = min(Y, (y0 + by) * factor_to_use)
                for x0 in range(0, new_x, chunk_x):
                    bx = min(chunk_x, new_x - x0)
                    in_x0 = x0 * factor_to_use
                    in_x1 = min(X, (x0 + bx) * factor_to_use)

                    slab = inp[:, :, in_y0:in_y1, in_x0:in_x1].read().result()
                    if self.multiscale_downsample == "stride":
                        down = slab[..., ::factor_to_use, ::factor_to_use]
                    else:
                        arr = xp.asarray(slab)
                        block = (1, 1, factor_to_use, factor_to_use)
                        down_arr = block_reduce(arr, block_size=block, func=xp.mean)
                        down = (
                            cp.asnumpy(down_arr)
                            if USING_GPU and cp is not None
                            else np.asarray(down_arr)
                        )
                    down = down.astype(slab.dtype, copy=False)
                    self.fused_ts[:, :, y0 : y0 + by, x0 : x0 + bx].write(down).result()

            ngff = {
                "attributes": {"_ARRAY_DIMENSIONS": ["t", "c", "y", "x"]},
                "zarr_format": 3,
                "node_type": "group",
            }
            (omezarr_path / f"scale{idx + 1}").mkdir(parents=True, exist_ok=True)
            with open(omezarr_path / f"scale{idx + 1}" / "zarr.json", "w") as f:
                json.dump(ngff, f, indent=2)

    def _generate_ngff_zarr3_json(
        self,
        omezarr_path: Path,
        resolution_multiples: Sequence[Union[int, Sequence[int]]],
        dataset_name: str = "image",
        version: str = "0.5",
    ) -> None:
        """
        Write OME-NGFF v0.5 multiscales JSON for Zarr3.

        Parameters
        ----------
        omezarr_path : Path
            Root path of the NGFF group.
        resolution_multiples : sequence
            Resolution factors per scale.
        dataset_name : str
            Name of the dataset node.
        version : str
            NGFF version.
        """
        axes = [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ]
        norm_res = [tuple(r) if hasattr(r, "__len__") else (r, r) for r in resolution_multiples]
        base_scale = [1.0, 1.0] + [float(s) for s in self._pixel_size]
        trans = [0.0, 0.0] + list(self.center)

        datasets = []
        prev_sp = base_scale[2:]
        for lvl, factors in enumerate(norm_res):
            spatial = [base_scale[i + 2] * factors[i] for i in range(2)]
            scale = [1.0, 1.0] + spatial
            if lvl == 0:
                translation = trans
            else:
                translation = [
                    0.0,
                    0.0,
                    datasets[-1]["coordinateTransformations"][1]["translation"][2]
                    + 0.5 * prev_sp[0],
                    datasets[-1]["coordinateTransformations"][1]["translation"][3]
                    + 0.5 * prev_sp[1],
                ]
            datasets.append(
                {
                    "path": f"scale{lvl}/{dataset_name}",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": scale},
                        {"type": "translation", "translation": translation},
                    ],
                }
            )
            prev_sp = spatial

        mult = {
            "axes": axes,
            "datasets": datasets,
            "name": dataset_name,
            "@type": "ngff:Image",
        }
        metadata = {
            "attributes": {"ome": {"version": version, "multiscales": [mult]}},
            "zarr_format": 3,
            "node_type": "group",
        }
        with open(omezarr_path / "zarr.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def run(self) -> None:
        """Execute the full tile fusion pipeline end-to-end."""
        metrics_path = self.tiff_path.parent / self.metrics_filename

        try:
            self.load_pairwise_metrics(metrics_path)
            print(f"Loaded {len(self.pairwise_metrics)} pairwise metrics from {metrics_path}")
        except FileNotFoundError:
            print("Computing pairwise registration metrics...")
            self.refine_tile_positions_with_cross_correlation(
                downsample_factors=self.downsample_factors,
                ch_idx=self.channel_to_use,
                threshold=self.threshold,
            )
            self.save_pairwise_metrics(metrics_path)
            print(f"Saved {len(self.pairwise_metrics)} pairwise metrics to {metrics_path}")

        if len(self.pairwise_metrics) == 0:
            print(
                "No overlapping tile pairs found for registration. Using stage positions directly."
            )
        else:
            print("Optimizing global tile positions...")
        self.optimize_shifts(
            method="TWO_ROUND_ITERATIVE", rel_thresh=0.5, abs_thresh=2.0, iterative=True
        )
        gc.collect()
        if USING_GPU and cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

        self._tile_positions = [
            tuple(np.array(pos) + off * np.array(self._pixel_size))
            for pos, off in zip(self._tile_positions, self.global_offsets)
        ]

        print("Computing fused image space...")
        self._compute_fused_image_space()
        self._pad_to_chunk_multiple()
        print(f"Fused image size: {self.padded_shape}")

        omezarr = self.output_path
        scale0 = omezarr / "scale0" / "image"
        scale0.parent.mkdir(parents=True, exist_ok=True)

        print("Creating fused tensorstore...")
        self._create_fused_tensorstore(output_path=scale0)

        print("Fusing tiles...")
        self._fuse_tiles()

        ngff = {
            "attributes": {"_ARRAY_DIMENSIONS": ["t", "c", "y", "x"]},
            "zarr_format": 3,
            "node_type": "group",
        }
        with open(omezarr / "scale0" / "zarr.json", "w") as f:
            json.dump(ngff, f, indent=2)

        print("Building multiscale pyramid...")
        self._create_multiscales(omezarr, factors=self.multiscale_factors)
        self._generate_ngff_zarr3_json(omezarr, resolution_multiples=self.resolution_multiples)

        print(f"Fusion complete! Output: {omezarr}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        tiff_path = sys.argv[1]
    else:
        tiff_path = "data/ashlar/COLNOR69MW2-cycle-1.ome.tif"

    fusion = TileFusion(tiff_path)
    fusion.run()
