"""
TileFusion - GPU/CPU-accelerated tile registration and fusion for 2D OME-TIFF stacks.

Main orchestration class that composes registration, fusion, optimization, and I/O modules.
"""

import gc
import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

# Output Zarr codec-chunk side, in pixels. The shard is 2x this per side.
CODEC_CHUNK = 1024

import numpy as np
import tensorstore as ts
import tifffile
from tqdm import trange

from .utils import (
    USING_GPU,
    block_reduce,
    cp,
    make_1d_profile,
    xp,
)
from .registration import (
    compute_pair_bounds,
    find_adjacent_pairs,
    register_pairs_batched,
    register_pairs_readahead,
    rotation_aware_max_shift,
)
from .fusion import fuse_plane
from .optimization import (
    _check_connectivity,
    _edges_from_pairwise_metrics,
    fit_stage_to_image_transform,
    solve_least_squares,
    two_round_optimization,
)
from .flatfield import apply_flatfield, apply_flatfield_region
from .io import (
    open_reader,
    read_ome_tiff_tile,
    read_ome_tiff_region,
    create_zarr_store,
    write_ngff_metadata,
    write_scale_group_metadata,
)


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
    multiscale_factors : sequence of int
        Downsampling factors for multiscale.
    resolution_multiples : sequence
        Resolution multipliers per scale level.
    max_workers : int, optional
        Maximum parallel compute/I/O workers. Defaults to the logical CPU count.
        BLAS is pinned to 1 thread inside the pools to avoid oversubscription.
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
        blend_pixels: Tuple[int, int] = (0, 0),
        downsample_factors: Tuple[int, int] = (1, 1),
        ssim_window: int = 15,
        multiscale_factors: Sequence[int] = (2, 4, 8, 16),
        resolution_multiples: Sequence[Union[int, Sequence[int]]] = (
            (1, 1),
            (2, 2),
            (4, 4),
            (8, 8),
            (16, 16),
        ),
        max_workers: Optional[int] = None,
        debug: bool = False,
        metrics_filename: str = "metrics.json",
        channel_to_use: Optional[int] = None,
        multiscale_downsample: str = "stride",
        region: Optional[str] = None,
        flatfield: Optional[np.ndarray] = None,
        darkfield: Optional[np.ndarray] = None,
        registration_z: Optional[int] = None,
        registration_t: int = 0,
    ):
        self._resolve_paths(tiff_path, output_path)
        self._load_metadata()
        self._apply_region_filter(region)
        self._configure_z_t_planes(registration_z, registration_t)
        self._configure_registration_params(
            downsample_factors,
            ssim_window,
            multiscale_factors,
            resolution_multiples,
            max_workers,
            debug,
            metrics_filename,
            blend_pixels,
            channel_to_use,
            multiscale_downsample,
        )
        self._update_profiles()
        self._init_chunking()
        self._init_pipeline_state()
        self._init_corrections(flatfield, darkfield)
        self._init_handle_storage()
        self._resolve_registration_channel()

    def _resolve_paths(self, tiff_path: Union[str, Path], output_path: Union[str, Path]) -> None:
        """Resolve the input path (must exist) and the fused-output path."""
        self.tiff_path = Path(tiff_path)
        if not self.tiff_path.exists():
            raise FileNotFoundError(f"Path not found: {self.tiff_path}")

        self.output_path = (
            Path(output_path)
            if output_path
            else self.tiff_path.parent / f"{self.tiff_path.stem}_fused.ome.zarr"
        )

    def _load_metadata(self) -> None:
        """Open the unified reader and extract common tile/grid properties."""
        # Detect the input format and load its metadata via the unified reader.
        # The reader encapsulates format detection, the per-format metadata
        # loaders, and (for single OME-TIFF) the immediate handle close that
        # core used to do inline.
        self._reader = open_reader(self.tiff_path)
        self._metadata = self._reader.load_metadata()

        # Extract common properties
        self.n_tiles = self._metadata["n_tiles"]
        self.n_series = self._metadata["n_series"]
        self.Y, self.X = self._metadata["shape"]
        self.channels = self._metadata["channels"]
        self.time_dim = self._metadata.get("time_dim", 1)
        self.position_dim = self._metadata.get("position_dim", self.n_tiles)
        self._pixel_size = self._metadata["pixel_size"]
        self._tile_positions = self._metadata["tile_positions"]
        self._tile_identifiers = self._metadata.get("tile_identifiers", [])
        self._unique_regions = self._metadata.get("unique_regions", [])

    def _apply_region_filter(self, region: Optional[str]) -> None:
        """Filter tiles to a single region, mutating tile/grid state and metadata."""
        self._region = region

        # Filter to specific region if requested
        if region is not None and self._tile_identifiers:
            filtered_positions = []
            filtered_identifiers = []
            for pos, tile_id in zip(self._tile_positions, self._tile_identifiers):
                if len(tile_id) >= 2 and tile_id[0] == region:
                    filtered_positions.append(pos)
                    filtered_identifiers.append(tile_id)
            if not filtered_positions:
                raise ValueError(f"No tiles found for region '{region}'")
            self._tile_positions = filtered_positions
            self._tile_identifiers = filtered_identifiers
            self.n_tiles = len(filtered_positions)
            self.n_series = self.n_tiles
            self.position_dim = self.n_tiles
            # Update metadata for reading tiles
            self._metadata["tile_positions"] = filtered_positions
            self._metadata["tile_identifiers"] = filtered_identifiers
            self._metadata["n_tiles"] = self.n_tiles

    def _configure_z_t_planes(self, registration_z: Optional[int], registration_t: int) -> None:
        """Set z/t plane counts and validate the registration z/t selection."""
        # Z-stack and time series properties
        self.n_z = self._metadata.get("n_z", 1)
        self.n_t = self._metadata.get("n_t", 1)
        self.dz_um = self._metadata.get("dz_um", 1.0)
        self._time_folders = self._metadata.get("time_folders", None)
        self._middle_z = self.n_z // 2  # Use middle z-level for registration

        # Registration z/t selection (validate after n_z/n_t are known)
        if registration_z is None:
            self._registration_z = self._middle_z
        else:
            if registration_z < 0 or registration_z >= self.n_z:
                raise ValueError(f"registration_z={registration_z} out of range [0, {self.n_z})")
            self._registration_z = registration_z

        if registration_t < 0 or registration_t >= self.n_t:
            raise ValueError(f"registration_t={registration_t} out of range [0, {self.n_t})")
        self._registration_t = registration_t

    def _configure_registration_params(
        self,
        downsample_factors: Tuple[int, int],
        ssim_window: int,
        multiscale_factors: Sequence[int],
        resolution_multiples: Sequence[Union[int, Sequence[int]]],
        max_workers: int,
        debug: bool,
        metrics_filename: str,
        blend_pixels: Tuple[int, int],
        channel_to_use: int,
        multiscale_downsample: str,
    ) -> None:
        """Store registration/fusion config scalars and validate downsample mode."""
        # Configuration
        self.downsample_factors = tuple(downsample_factors)
        self.ssim_window = int(ssim_window)
        self.multiscale_factors = tuple(multiscale_factors)
        self.resolution_multiples = [
            r if hasattr(r, "__len__") else (r, r) for r in resolution_multiples
        ]
        # Default to one worker per logical core (BLAS is pinned to 1 thread inside the
        # pools, so workers ~= cores is full utilisation without oversubscription). The
        # pools further cap by the work count, and registration strips are small, so this
        # stays memory-bounded.
        self._max_workers = int(max_workers) if max_workers else (os.cpu_count() or 8)
        self._debug = bool(debug)
        self.metrics_filename = metrics_filename
        self._blend_pixels = tuple(blend_pixels)
        self.channel_to_use = channel_to_use

        if multiscale_downsample not in ("stride", "block_mean"):
            raise ValueError('multiscale_downsample must be "stride" or "block_mean".')
        self.multiscale_downsample = multiscale_downsample

    def _resolve_registration_channel(self) -> None:
        """Validate the registration channel.

        We deliberately do NOT auto-pick a registration channel. Contrast/entropy proxies
        do not reliably identify the best-registering channel across datasets (validated:
        no single metric picks the optimum on Codex; std alone picks a saturated, useless
        channel on brightfield), so an automatic guess can silently misalign the whole
        mosaic. Following ASHLAR's `--align-channel`, the channel is the operator's choice.

        Single-channel datasets resolve trivially to 0. For multi-channel datasets the
        caller must pass channel_to_use; if it is left None the requirement is enforced at
        registration time (refine_tile_positions_with_cross_correlation), so metadata-only
        uses of TileFusion still work without a channel.
        """
        if self.channel_to_use is None:
            if self.channels == 1:
                self.channel_to_use = 0
            return  # multi-channel: defer the requirement to registration time
        if not (0 <= self.channel_to_use < self.channels):
            raise ValueError(
                f"channel_to_use={self.channel_to_use} out of range [0, {self.channels})"
            )

    def _init_chunking(self) -> None:
        """Set the output chunk shape and its derived y/x sizes."""
        self.chunk_shape = (1, CODEC_CHUNK, CODEC_CHUNK)
        self.chunk_y, self.chunk_x = self.chunk_shape[-2:]

    def _init_pipeline_state(self) -> None:
        """Initialize the mutable registration/fusion pipeline state."""
        # State
        self.pairwise_metrics: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
        self.global_offsets: Optional[np.ndarray] = None
        self.offset: Optional[Tuple[float, float]] = None
        self.unpadded_shape: Optional[Tuple[int, int]] = None
        self.padded_shape: Optional[Tuple[int, int]] = None
        self.fused_ts = None
        self.center = None
        # Per-seam elastic distortion correction (built after optimization, applied
        # at fusion). Default on; self-calibrating with an identity fallback, so the
        # worst case is the translation-only result. See tilefusion.distortion.
        self.enable_distortion_correction: bool = True
        self._tile_warper = None

    def _init_corrections(
        self, flatfield: Optional[np.ndarray], darkfield: Optional[np.ndarray]
    ) -> None:
        """Store optional flatfield/darkfield and validate their shapes."""
        # Flatfield correction (optional)
        self._flatfield = flatfield  # Shape (C, Y, X) or None
        self._darkfield = darkfield  # Shape (C, Y, X) or None

        # Validate flatfield/darkfield shapes match tile dimensions
        expected_shape = (self.channels, self.Y, self.X)
        if flatfield is not None and flatfield.shape != expected_shape:
            raise ValueError(
                f"flatfield.shape {flatfield.shape} does not match expected "
                f"tile shape {expected_shape} (channels, Y, X)"
            )
        if darkfield is not None and darkfield.shape != expected_shape:
            raise ValueError(
                f"darkfield.shape {darkfield.shape} does not match expected "
                f"tile shape {expected_shape} (channels, Y, X)"
            )

    def _init_handle_storage(self) -> None:
        """Initialize thread-local storage for TiffFile handles."""
        # Thread-local storage for TiffFile handles (thread-safe concurrent access)
        self._thread_local = threading.local()
        self._handles_lock = threading.Lock()
        self._all_handles: List[tifffile.TiffFile] = []

    def close(self) -> None:
        """
        Close any open file handles to release resources.

        This should be called when finished using a TileFusion instance,
        or use it as a context manager (``with TileFusion(...) as tf:``)
        for automatic cleanup. Important for OME-TIFF inputs where file
        handles are kept open for performance.

        Warning
        -------
        Only call this method when all read operations are complete. Calling
        ``close()`` while other threads are still reading tiles will close
        their handles mid-operation, causing errors.
        """
        # THREAD SAFETY NOTE:
        # This method is NOT safe to call while other threads are actively reading.
        # The design assumes close() is called only after all work is complete.
        #
        # Race condition scenario:
        #   1. Thread A calls _get_thread_local_handle(), gets handle
        #   2. Main thread calls close(), closes all handles
        #   3. Thread A calls handle.series[idx].asarray() -> ERROR (closed file)
        #
        # We chose documentation over a complex fix (reference counting, read-write
        # locks) because:
        #   - The context manager pattern naturally prevents this issue
        #   - Adding synchronization would hurt performance for the common case
        #   - Users explicitly calling close() should know their threads are done
        #
        # Safe usage patterns:
        #   - Use context manager: with TileFusion(...) as tf: ...
        #   - Call close() only after ThreadPoolExecutor.shutdown(wait=True)
        #   - Use single-threaded access when manually managing lifecycle

        # Close all thread-local handles
        with self._handles_lock:
            for handle in self._all_handles:
                try:
                    handle.close()
                except (OSError, AttributeError):
                    pass  # Best-effort cleanup: handle may be invalid or already closed
            self._all_handles.clear()

        # Reset thread-local storage so future calls to _get_thread_local_handle()
        # will create new handles. Note: This only affects threads that access
        # self._thread_local AFTER this point. Threads that cached a handle reference
        # before close() was called will still have stale (closed) handles, but
        # _get_thread_local_handle() now checks for closed handles and creates new ones.
        self._thread_local = threading.local()

    def __enter__(self) -> "TileFusion":
        """Enter the runtime context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the runtime context and close file handles."""
        try:
            self.close()
        except Exception:
            # If there was no exception in the with-block, propagate the close() failure.
            # If there was an original exception, suppress close() errors so we don't mask it.
            if exc_type is None:
                raise

    def __del__(self):
        """
        Destructor to ensure file handles are closed.

        Note: This is a fallback safety net only. Python does not guarantee
        when (or if) __del__ is called. Always prefer using the context
        manager protocol (``with TileFusion(...) as tf:``) or explicitly
        calling ``close()`` for reliable resource cleanup.
        """
        try:
            self.close()
        except (OSError, AttributeError, TypeError):
            pass  # Object may be partially initialized, or close() may fail during shutdown

    def _get_thread_local_handle(self) -> Optional[tifffile.TiffFile]:
        """
        Get or create a thread-local TiffFile handle for the current thread.

        Each thread gets its own file handle to ensure thread-safe concurrent
        reads. This avoids race conditions that can occur when multiple threads
        share a single file descriptor (seek + read is not atomic on Windows).

        Returns
        -------
        tifffile.TiffFile or None
            Thread-local handle for OME-TIFF files, None for other formats.
        """
        # Only applies to single OME-TIFF format (not zarr, individual tiffs, etc.)
        if self._reader.is_multi_file:
            return None

        # Check if this thread already has a valid (open) handle.
        # NOTE: There is a race condition between this check and using the handle -
        # another thread could call close() after validation but before the handle
        # is used. This is documented behavior; callers must ensure close() is only
        # called after all read operations complete.
        if hasattr(self._thread_local, "tiff_handle"):
            handle = self._thread_local.tiff_handle
            # Verify handle exists and is not closed.
            # We check filehandle.closed which is a reliable indicator.
            if (
                handle is not None
                and handle.filehandle is not None
                and not handle.filehandle.closed
            ):
                return handle
            # Handle was closed or invalid - will create a new one below

        # Create a new handle for this thread
        handle = tifffile.TiffFile(self.tiff_path)
        self._thread_local.tiff_handle = handle

        # Track for cleanup
        with self._handles_lock:
            self._all_handles.append(handle)

        return handle

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _update_profiles(self) -> None:
        """Recompute 1D feather profiles from blend_pixels."""
        by, bx = self._blend_pixels
        self.y_profile = make_1d_profile(self.Y, by)
        self.x_profile = make_1d_profile(self.X, bx)

    # -------------------------------------------------------------------------
    # I/O methods (delegate to format-specific loaders)
    # -------------------------------------------------------------------------

    def _read_tile(self, tile_idx: int, z_level: int = None, time_idx: int = None) -> np.ndarray:
        """Read a single tile from the input data (all channels). FUSION path.

        Returns float32 for every format except the ome_tiff/ folder format, which
        returns native uint16 -- so ``apply_flatfield``'s integer round-and-clip
        branch below fires for that format only. This asymmetry is real, measured
        and documented on ``io.base.Reader``; it is NOT the same dtype the
        registration path (``_read_tile_region``) sees.
        """
        if z_level is None:
            z_level = self._registration_z  # Default to registration z-level
        if time_idx is None:
            time_idx = self._registration_t  # Default to registration timepoint

        if self._reader.is_multi_file:
            tile = self._reader.read_tile(tile_idx, z_level=z_level, time_idx=time_idx)
        else:
            # Single OME-TIFF: use core's thread-local handle for thread-safe
            # concurrent reads (the reader cannot manage per-thread handles).
            handle = self._get_thread_local_handle()
            tile = read_ome_tiff_tile(self.tiff_path, tile_idx, handle)

        # Apply flatfield correction if enabled
        if self._flatfield is not None:
            tile = apply_flatfield(tile, self._flatfield, self._darkfield)

        # Every fusion path expects C channels; broadcast a single-channel read up
        # so the placement/blend code never has to special-case channel count.
        if tile.shape[0] == 1 and self.channels > 1:
            tile = np.broadcast_to(tile, (self.channels, tile.shape[1], tile.shape[2]))

        return tile

    def _tile_field(self, tile_idx: int):
        """Per-tile elastic displacement field (2, Y, X) for the fusion sampler, or
        None if this tile has no correction. The fold happens inside the numba blend
        (accumulate_tile_shard_distorted), so there is no separate warp pass and the
        raw reads feed both registration and fusion unchanged."""
        if self._tile_warper is None:
            return None
        return self._tile_warper.field(tile_idx)

    def _build_distortion_correction(self) -> None:
        """Build the per-seam elastic distortion correction (applied at fusion).

        Call AFTER optimization offsets are applied to _tile_positions and BEFORE
        fusion. Measures the residual that varies ALONG each seam (optical field
        distortion / local rotation that one translation per tile can't capture) and
        builds per-tile warp fields. Self-calibrating with an identity fallback at
        every gate, so any failure degrades to the translation-only result rather
        than breaking the run. Shared by run(), the GUI worker, and the CLI path.
        """
        self._tile_warper = None
        if not self.enable_distortion_correction or len(self.pairwise_metrics) == 0:
            return
        try:
            from .distortion import build_seam_corrections, TileWarper

            print("Building per-seam elastic distortion correction...")
            corrections = build_seam_corrections(self)
            if corrections:
                self._tile_warper = TileWarper(corrections, self.Y, self.X)
        except Exception as e:
            print(f"Distortion correction skipped ({e}); using translation-only placement.")
            self._tile_warper = None

    def _read_tile_region(
        self,
        tile_idx: int,
        y_slice: slice,
        x_slice: slice,
        z_level: int = None,
        time_idx: int = None,
    ) -> np.ndarray:
        """Read a region of a tile from the input data. REGISTRATION path.

        Always float32, for every format. Registration therefore correlates the
        raw, unrounded flat-field quotient: ``apply_flatfield_region``'s integer
        branch is unreachable from here by construction. Deliberate -- sub-pixel
        cross-correlation wants full precision, and SquidXplorer's parity gate
        pins the offsets this produces. See ``io.base.Reader``.
        """
        if z_level is None:
            z_level = self._registration_z  # Default to registration z-level
        if time_idx is None:
            time_idx = self._registration_t  # Default to registration timepoint

        if self._reader.is_multi_file:
            region = self._reader.read_region(
                tile_idx,
                y_slice,
                x_slice,
                channel_idx=self.channel_to_use,
                z_level=z_level,
                time_idx=time_idx,
            )
        else:
            # Single OME-TIFF: use core's thread-local handle for thread-safe
            # concurrent reads (the reader cannot manage per-thread handles).
            handle = self._get_thread_local_handle()
            # Pass channel_to_use so the (1, h, w) result already holds the
            # registration channel (this format is single-channel in practice,
            # so it matches the old all-channel return for every real dataset).
            region = read_ome_tiff_region(
                self.tiff_path, tile_idx, y_slice, x_slice, self.channel_to_use, handle
            )

        # Apply flatfield correction if enabled
        if self._flatfield is not None:
            ff = self._flatfield
            df = self._darkfield
            # When reading a single channel for registration, slice the flatfield
            if region.ndim == 3 and ff.shape[0] != region.shape[0]:
                ch = self.channel_to_use if self.channel_to_use < ff.shape[0] else 0
                ff = ff[ch : ch + 1]
                df = df[ch : ch + 1] if df is not None else None
            region = apply_flatfield_region(region, ff, df, y_slice, x_slice)

        return region

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def refine_tile_positions_with_cross_correlation(
        self,
        downsample_factors: Tuple[int, int] = None,
        ssim_window: int = None,
        ch_idx: int = 0,
        parallel: Optional[bool] = None,
    ) -> None:
        """
        Detect and score overlaps between neighboring tile pairs via cross-correlation.

        Parameters
        ----------
        parallel : bool, optional
            If None (default), auto-detects: enabled for multi-file formats
            (Zarr, individual TIFFs, OME-TIFF tiles), disabled for single-file
            OME-TIFF (due to I/O contention).
        """
        if self.channel_to_use is None:
            raise ValueError(
                "No registration channel selected. Pass channel_to_use "
                f"(0..{self.channels - 1}) -- there is no automatic channel selection; "
                "the registration channel is the operator's choice (cf. ASHLAR "
                "--align-channel)."
            )

        df = downsample_factors or self.downsample_factors
        sw = ssim_window or self.ssim_window
        self.pairwise_metrics.clear()

        # Find adjacent pairs
        adjacent_pairs = find_adjacent_pairs(
            self._tile_positions, self._pixel_size, (self.Y, self.X)
        )

        # Residual-shift sanity cap (px/axis), adaptive to inter-tile spacing so it
        # tolerates a real stage->image rotation up to ~3 deg (the scientist's 1-2 deg
        # spec + margin) without clipping legitimate residuals, while still rejecting the
        # far larger shifts of a spurious phase-correlation peak.
        max_shift = rotation_aware_max_shift(adjacent_pairs)

        if self._debug:
            print(f"Found {len(adjacent_pairs)} adjacent tile pairs to register")

        # Compute bounds
        pair_bounds = compute_pair_bounds(adjacent_pairs, (self.Y, self.X))

        # Auto-detect parallel mode if not specified
        if parallel is None:
            # Parallel helps for individual TIFFs (separate files)
            # but hurts for single-file OME-TIFF (I/O contention)
            parallel = self._reader.is_multi_file

        # Use parallel processing for CPU mode with enough pairs
        use_parallel = parallel and not USING_GPU and len(pair_bounds) > 4

        n_attempted = len(pair_bounds)

        if use_parallel:
            results = register_pairs_batched(
                pair_bounds,
                self._read_tile_region,
                df,
                sw,
                max_shift,
                self._max_workers,
                debug=self._debug,
            )
        else:
            results = register_pairs_readahead(
                pair_bounds,
                self._read_tile_region,
                df,
                sw,
                max_shift,
                debug=self._debug,
            )
        self.pairwise_metrics.update(results)

        n_success = len(self.pairwise_metrics)
        n_failed = n_attempted - n_success
        msg = f"Registration: {n_success}/{n_attempted} pairs succeeded"
        if n_failed:
            # Surface WHICH pairs were dropped and why, instead of leaving it to a
            # debug-only log -- a dropped pair removes a constraint and can shift the
            # result, so it must be visible.
            attempted = {(b[0], b[1]) for b in pair_bounds}
            dropped = sorted(attempted - set(self.pairwise_metrics))
            msg += (
                f", {n_failed} dropped (residual > max_shift {max_shift} px, or a "
                f"read/registration error): {dropped}"
            )
        print(msg)

    # -------------------------------------------------------------------------
    # Optimization
    # -------------------------------------------------------------------------

    def optimize_shifts(
        self,
        method: str = "ONE_ROUND",
        rel_thresh: float = 0.3,
        abs_thresh: float = 5.0,
        iterative: bool = False,
    ) -> None:
        """
        Globally optimize tile shifts.

        Parameters
        ----------
        method : {'ONE_ROUND', 'TWO_ROUND_SIMPLE', 'TWO_ROUND_ITERATIVE'}
        rel_thresh : float
            Relative threshold for link removal.
        abs_thresh : float
            Absolute threshold for link removal.
        iterative : bool
            If True, repeat outlier removal until convergence.
        """
        # 1. Derive solver edges from the canonical pairwise_metrics (transient, private form)
        edges = _edges_from_pairwise_metrics(self.pairwise_metrics)
        if not edges:
            self.global_offsets = np.zeros((self.position_dim, 2), dtype=np.float64)
            return

        # 2. Anchor the first tile at the origin
        n_tiles = len(self._tile_positions)
        anchored = [0]

        # 3. Dispatch to the chosen solver
        if method == "ONE_ROUND":
            d_opt = solve_least_squares(edges, n_tiles, anchored)
        elif method.startswith("TWO_ROUND"):
            d_opt = two_round_optimization(
                edges, n_tiles, anchored, rel_thresh, abs_thresh, method.endswith("ITERATIVE")
            )
        else:
            raise ValueError(f"Unknown method {method}")

        # 4. Store the optimized per-tile offsets
        self.global_offsets = d_opt

        # 5. Place tiles the solve left unconstrained (disconnected from the anchor:
        #    singletons + floating components) via the global stage->image affine,
        #    instead of leaving them at raw, miscalibrated stage positions.
        self._place_unconstrained_tiles_with_affine(edges, n_tiles)

    def _place_unconstrained_tiles_with_affine(self, edges, n_tiles: int) -> None:
        """Position tiles not connected to the anchor via the global stage->image
        affine fit from the registered pairs. Connected tiles keep their solved offset.

        A tile whose overlaps were too low-texture to register is left unconstrained
        by the least-squares solve and floats at its raw, miscalibrated stage position
        -- the source of seam misalignment. The affine (a property of the instrument,
        fit from the pairs that DID register) predicts where it belongs. No-op when the
        graph is fully connected to the anchor (so the synthetic fixtures are unchanged)
        or when too few pairs exist to fit a reliable transform.
        """
        components = _check_connectivity(edges, n_tiles)
        anchor_component = next((c for c in components if 0 in c), [])
        unconstrained = [t for t in range(n_tiles) if t not in anchor_component]
        if not unconstrained:
            return  # fully connected to the anchor: the solve already placed everything

        _MIN_PAIRS_FOR_AFFINE = 8  # a 2-3 DOF global transform needs a handful of spread pairs
        if len(self.pairwise_metrics) < _MIN_PAIRS_FOR_AFFINE:
            logger.warning(
                "%d tile(s) unconstrained but only %d registered pairs (< %d); leaving them "
                "at stage positions rather than an unreliable affine fit.",
                len(unconstrained),
                len(self.pairwise_metrics),
                _MIN_PAIRS_FOR_AFFINE,
            )
            return

        cal = fit_stage_to_image_transform(
            self.pairwise_metrics, self._tile_positions, self._pixel_size
        )
        M = cal["M"]
        pos = np.asarray(self._tile_positions, dtype=np.float64)
        ps = np.asarray(self._pixel_size, dtype=np.float64)
        ref = pos[0]
        for k in unconstrained:
            d = pos[k] - ref
            # store as an offset to the isotropic model, consistent with global_offsets
            self.global_offsets[k] = M @ d - d / ps
        print(
            f"Affine calibration: placed {len(unconstrained)} unconstrained tile(s) "
            f"(scale {cal['scale']:.2f} px/unit, rotation {cal['rotation_deg']:+.3f} deg, "
            f"fit residual {cal['residual_rms']:.1f} px over {cal['n_pairs']} pairs)."
        )

    # -------------------------------------------------------------------------
    # Metrics persistence
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Fused image space
    # -------------------------------------------------------------------------

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

        self.padded_shape = (sy + py, sx + px)

    def _create_fused_tensorstore(self, output_path: Union[str, Path]) -> None:
        """Create the output Zarr v3 store for the fused image."""
        out = Path(output_path)
        # 5D shape: (T, C, Z, Y, X)
        full_shape = [self.n_t, self.channels, self.n_z, *self.padded_shape]
        shard_chunk = [1, 1, 1, self.chunk_y * 2, self.chunk_x * 2]
        codec_chunk = [1, 1, 1, self.chunk_y, self.chunk_x]
        self.shard_chunk = shard_chunk

        self.fused_ts = create_zarr_store(
            out, tuple(full_shape), tuple(codec_chunk), tuple(shard_chunk), self.max_workers
        )

    # -------------------------------------------------------------------------
    # Fusion
    # -------------------------------------------------------------------------

    def _tile_pixel_origins(self) -> List[Tuple[float, float]]:
        """Top-left (y, x) pixel position of each FOV on the plane (sub-pixel).

        Returns FRACTIONAL positions so blended fusion can place tiles at sub-pixel
        precision (honouring the registration instead of truncating it). Callers that
        require integer placement (direct mode) floor locally.
        """
        return [
            (
                (y - self.offset[0]) / self._pixel_size[0],
                (x - self.offset[1]) / self._pixel_size[1],
            )
            for (y, x) in self._tile_positions
        ]

    def _fuse_tiles(self, mode: str = "blended", chunked: bool = True) -> None:
        """Fuse all tiles into output, looping over z-levels and time points."""
        total_planes = self.n_t * self.n_z
        plane_idx = 0

        for t in range(self.n_t):
            for z in range(self.n_z):
                plane_idx += 1
                if total_planes > 1:
                    print(f"Fusing plane {plane_idx}/{total_planes} (t={t}, z={z})...")

                if mode == "direct":
                    self._fuse_tiles_direct_plane(z_level=z, time_idx=t)
                elif chunked:
                    self._fuse_tiles_chunked_plane(z_level=z, time_idx=t)
                else:
                    self._fuse_tiles_full_plane(z_level=z, time_idx=t)

    def _fuse_tiles_direct_plane(self, z_level: int = 0, time_idx: int = 0) -> None:
        """Place tiles directly (no blending) for a single z/t plane.

        Each tile is streamed straight to the output store, so memory is bounded by
        one tile regardless of plane size or machine RAM. Overlaps are overwritten
        (last tile wins); use blended/chunked fusion for feathered seams.
        """
        offsets = self._tile_pixel_origins()
        pad_Y, pad_X = self.padded_shape
        show_progress = self.n_t == 1 and self.n_z == 1  # Only show progress for single plane

        iterator = (
            trange(len(offsets), desc="placing tiles", leave=True)
            if show_progress
            else range(len(offsets))
        )
        for t_idx in iterator:
            # Direct mode is the no-blend, last-tile-wins fast path: integer placement
            # (floor the sub-pixel origin). Sub-pixel placement is for blended fusion.
            oy, ox = int(offsets[t_idx][0]), int(offsets[t_idx][1])
            tile_all = self._read_tile(t_idx, z_level=z_level, time_idx=time_idx)

            y_end = min(oy + self.Y, pad_Y)
            x_end = min(ox + self.X, pad_X)
            tile_h = y_end - oy
            tile_w = x_end - ox

            if tile_h > 0 and tile_w > 0:
                # ROUND, then clip, before the integer cast -- the same convention as
                # fusion.fuse_plane's write and flatfield.apply_flatfield. read_tile hands
                # back float32 for every reader but ome_tiff, so a bare `.astype` truncates
                # each flat-field-corrected pixel toward zero: half a count of systematic
                # dimming on the direct-placement path too, blend or no blend.
                tile_region = np.clip(
                    np.rint(tile_all[:, :tile_h, :tile_w]), 0, 65535
                ).astype(np.uint16)
                # Shape: (1, C, 1, h, w)
                self.fused_ts[
                    time_idx : time_idx + 1, :, z_level : z_level + 1, oy:y_end, ox:x_end
                ].write(tile_region[np.newaxis, :, np.newaxis, :, :]).result()

        gc.collect()

    def _fuse_plane(self, z_level: int, time_idx: int, block_size: int) -> None:
        """Assemble explicit inputs + write closure and delegate to fusion.fuse_plane."""

        def write_block(y0, y1, x0, x1, arr_uint16):
            self.fused_ts[time_idx, :, z_level, y0:y1, x0:x1].write(arr_uint16).result()

        fuse_plane(
            read_tile=self._read_tile,
            get_field=self._tile_field,
            write_block=write_block,
            origins=self._tile_pixel_origins(),
            padded_shape=self.padded_shape,
            tile_shape=(self.Y, self.X),
            channels=self.channels,
            y_profile=self.y_profile,
            x_profile=self.x_profile,
            block_size=block_size,
            z_level=z_level,
            time_idx=time_idx,
            show_progress=(self.n_t == 1 and self.n_z == 1),
            progress_callback=getattr(self, "progress_callback", None),
        )

    def _fuse_tiles_chunked_plane(self, z_level: int = 0, time_idx: int = 0) -> None:
        """Block-by-block fusion at fixed low memory (one shard per side)."""
        self._fuse_plane(z_level, time_idx, block_size=self.chunk_y * 2)

    def _fuse_tiles_full_plane(self, z_level: int = 0, time_idx: int = 0) -> None:
        """Whole-plane fusion (one block). Kept for the equivalence test + non-chunked dispatch."""
        pad_Y, pad_X = self.padded_shape
        self._fuse_plane(z_level, time_idx, block_size=max(pad_Y, pad_X))

    # -------------------------------------------------------------------------
    # Multiscale pyramid
    # -------------------------------------------------------------------------

    def _create_multiscales(
        self,
        omezarr_path: Path,
        factors: Sequence[int] = (2, 4, 8),
    ) -> None:
        """Build NGFF multiscales by downsampling Y/X iteratively (not Z or T)."""
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
            # 5D shape: (T, C, Z, Y, X)
            _, _, _, Y, X = inp.shape
            new_y, new_x = Y // factor_to_use, X // factor_to_use

            chunk_y = min(CODEC_CHUNK, new_y)
            chunk_x = min(CODEC_CHUNK, new_x)

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

                    # Read 5D slab: (T, C, Z, h, w)
                    slab = inp[:, :, :, in_y0:in_y1, in_x0:in_x1].read().result()
                    if self.multiscale_downsample == "stride":
                        down = slab[..., ::factor_to_use, ::factor_to_use]
                    else:
                        arr = xp.asarray(slab)
                        # Only downsample Y, X (last 2 dims)
                        block = (1, 1, 1, factor_to_use, factor_to_use)
                        down_arr = block_reduce(arr, block_size=block, func=xp.mean)
                        down = (
                            cp.asnumpy(down_arr)
                            if USING_GPU and cp is not None
                            else np.asarray(down_arr)
                        )
                    down = down.astype(slab.dtype, copy=False)
                    self.fused_ts[:, :, :, y0 : y0 + by, x0 : x0 + bx].write(down).result()

            write_scale_group_metadata(omezarr_path / f"scale{idx + 1}")

    def _generate_ngff_zarr3_json(
        self,
        omezarr_path: Path,
        resolution_multiples: Sequence[Union[int, Sequence[int]]],
        dataset_name: str = "image",
        version: str = "0.5",
    ) -> None:
        """Write OME-NGFF v0.5 multiscales JSON for Zarr3."""
        write_ngff_metadata(
            omezarr_path,
            self._pixel_size,
            self.center,
            resolution_multiples,
            dataset_name,
            version,
        )

    # -------------------------------------------------------------------------
    # Main pipeline
    # -------------------------------------------------------------------------

    def run(self, register: bool = True) -> None:
        """Execute the full tile fusion pipeline end-to-end.

        Parameters
        ----------
        register : bool
            If False, skip registration entirely and fuse at the reported stage
            positions. The existing empty-metrics path in optimize_shifts then
            zero-fills all offsets, so tiles are placed at their raw stage
            coordinates.
        """
        metrics_path = self.tiff_path.parent / self.metrics_filename

        if register:
            try:
                self.load_pairwise_metrics(metrics_path)
                print(f"Loaded {len(self.pairwise_metrics)} pairwise metrics from {metrics_path}")
            except FileNotFoundError:
                print("Computing pairwise registration metrics...")
                self.refine_tile_positions_with_cross_correlation(
                    downsample_factors=self.downsample_factors,
                    ch_idx=self.channel_to_use,
                )
                self.save_pairwise_metrics(metrics_path)
                print(f"Saved {len(self.pairwise_metrics)} pairwise metrics to {metrics_path}")

        if len(self.pairwise_metrics) == 0:
            print("No overlapping tile pairs found. Using stage positions directly.")
        else:
            print("Optimizing global tile positions...")

        self.optimize_shifts(
            method="TWO_ROUND_ITERATIVE", rel_thresh=0.5, abs_thresh=2.0, iterative=True
        )

        # Apply offsets
        self._tile_positions = [
            tuple(np.array(pos) + off * np.array(self.pixel_size))
            for pos, off in zip(self._tile_positions, self.global_offsets)
        ]

        self._build_distortion_correction()

        print("Computing fused image space...")
        self._compute_fused_image_space()
        self._pad_to_chunk_multiple()
        if self.n_t > 1 or self.n_z > 1:
            print(
                f"Output size: {self.n_t}T x {self.channels}C x {self.n_z}Z x "
                f"{self.padded_shape[0]} x {self.padded_shape[1]}"
            )
        else:
            print(f"Output size: {self.padded_shape[0]} x {self.padded_shape[1]}")

        scale0 = self.output_path / "scale0" / "image"
        scale0.parent.mkdir(parents=True, exist_ok=True)
        self._create_fused_tensorstore(output_path=scale0)

        print("Fusing tiles...")
        self._fuse_tiles()

        write_scale_group_metadata(self.output_path / "scale0")

        print("Building multiscale pyramid...")
        self._create_multiscales(self.output_path, factors=self.multiscale_factors)
        self._generate_ngff_zarr3_json(
            self.output_path, resolution_multiples=self.resolution_multiples
        )

        print(f"Done! Output: {self.output_path}")

    def stitch_all_regions(self) -> None:
        """Stitch all regions in the dataset, creating separate outputs per region.

        Creates output folder structure: {input_name}_fused/{region}.ome.zarr
        """
        if not self._unique_regions:
            print("No multiple regions detected. Running standard stitching...")
            self.run()
            return

        if len(self._unique_regions) == 1:
            print(
                f"Only one region ({self._unique_regions[0]}) found. Running standard stitching..."
            )
            self.run()
            return

        # Create output folder
        output_folder = self.tiff_path.parent / f"{self.tiff_path.stem}_fused"
        output_folder.mkdir(parents=True, exist_ok=True)

        print(f"Found {len(self._unique_regions)} regions: {self._unique_regions}")
        print(f"Output folder: {output_folder}")

        for i, region in enumerate(self._unique_regions):
            print(f"\n{'='*60}")
            print(f"Processing region {i+1}/{len(self._unique_regions)}: {region}")
            print(f"{'='*60}")

            region_output = output_folder / f"{region}.ome.zarr"

            # Every per-region TileFusion must inherit the FULL configuration of the
            # parent, not just its geometry. Flat-field/dark-field, the registration
            # z/t plane and the distortion-correction switch used to be dropped here,
            # so a multi-region dataset was silently stitched with no illumination
            # correction, on the middle z, with distortion forced on -- regardless of
            # what the caller asked the parent for.
            tf = TileFusion(
                self.tiff_path,
                output_path=region_output,
                blend_pixels=self._blend_pixels,
                downsample_factors=self.downsample_factors,
                ssim_window=self.ssim_window,
                multiscale_factors=self.multiscale_factors,
                resolution_multiples=self.resolution_multiples,
                max_workers=self._max_workers,
                debug=self._debug,
                metrics_filename=f"metrics_{region}.json",
                channel_to_use=self.channel_to_use,
                multiscale_downsample=self.multiscale_downsample,
                region=region,
                flatfield=self._flatfield,
                darkfield=self._darkfield,
                registration_z=self._registration_z,
                registration_t=self._registration_t,
            )
            tf.enable_distortion_correction = self.enable_distortion_correction
            tf.run()

        print(f"\n{'='*60}")
        print(f"All regions complete! Output: {output_folder}")
