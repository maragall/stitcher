#!/usr/bin/env python3
"""
Stitcher GUI - A simple interface for tile fusion of OME-TIFF files.
"""

import sys
import os
import traceback
from pathlib import Path

import numpy as np
from scipy.ndimage import shift as ndi_shift

# Per-channel display colors, shared by every napari layer path.
CHANNEL_COLORS = ["blue", "green", "yellow", "red", "magenta", "cyan"]

# Fix Qt plugin path for conda environments on macOS
if sys.platform == "darwin" and "CONDA_PREFIX" in os.environ:
    conda_plugins = Path(os.environ["CONDA_PREFIX"]) / "plugins"
    if conda_plugins.exists() and "QT_PLUGIN_PATH" not in os.environ:
        os.environ["QT_PLUGIN_PATH"] = str(conda_plugins)

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QSpinBox,
    QGroupBox,
    QFileDialog,
    QProgressBar,
    QTextEdit,
    QFrame,
    QComboBox,
    QSlider,
    QRadioButton,
    QButtonGroup,
    QMessageBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QIcon, QPixmap, QPainter
from PyQt5.QtSvg import QSvgRenderer

STYLE_SHEET = """
QGroupBox {
    font-weight: bold;
    margin-top: 12px;
    padding-top: 16px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
}
QPushButton#runButton {
    background-color: #0071e3;
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
}
QPushButton#runButton:hover {
    background-color: #0077ed;
}
QPushButton#runButton:disabled {
    background-color: #c7c7cc;
}
QPushButton#napariButton {
    background-color: #34c759;
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
}
QPushButton#napariButton:hover {
    background-color: #30d158;
}
QPushButton#napariButton:disabled {
    background-color: #c7c7cc;
}
QPushButton#previewButton {
    background-color: #ff9500;
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
}
QPushButton#previewButton:hover {
    background-color: #ff9f0a;
}
QPushButton#previewButton:disabled {
    background-color: #c7c7cc;
}
QProgressBar {
    border: none;
    border-radius: 4px;
    height: 6px;
}
QProgressBar::chunk {
    background-color: #0071e3;
    border-radius: 4px;
}
QPushButton#calcFlatfieldButton {
    background-color: #5856d6;
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    min-height: 20px;
}
QPushButton#calcFlatfieldButton:hover {
    background-color: #6866e0;
}
QPushButton#calcFlatfieldButton:disabled {
    background-color: #c7c7cc;
}
"""


class PreviewWorker(QThread):
    """Worker thread for running preview stitching on subset of tiles."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(object, object, object)  # color_before, color_after, fused
    error = pyqtSignal(str)
    resolved_channel = pyqtSignal(int)  # channel auto-pick resolved to (for the 'Auto' label)

    def __init__(
        self,
        tiff_path,
        preview_cols,
        preview_rows,
        downsample_factor,
        flatfield=None,
        darkfield=None,
        registration_z=None,
        registration_t=0,
        registration_channel=None,
        outlier_rel_thresh=0.5,
        outlier_abs_thresh=2.0,
    ):
        super().__init__()
        self.tiff_path = tiff_path
        self.preview_cols = preview_cols
        self.preview_rows = preview_rows
        self.downsample_factor = downsample_factor
        self.flatfield = flatfield
        self.darkfield = darkfield
        self.registration_z = registration_z
        self.registration_t = registration_t
        self.registration_channel = registration_channel
        self.outlier_rel_thresh = outlier_rel_thresh
        self.outlier_abs_thresh = outlier_abs_thresh

    def run(self):
        try:
            from tilefusion import TileFusion

            self.progress.emit("Loading metadata...")

            # Create TileFusion instance - handles both OME-TIFF and SQUID formats
            tf_full = TileFusion(
                self.tiff_path,
                downsample_factors=(self.downsample_factor, self.downsample_factor),
                flatfield=self.flatfield,
                darkfield=self.darkfield,
                registration_z=self.registration_z,
                registration_t=self.registration_t,
                channel_to_use=self.registration_channel,
            )

            # Echo the settings actually in use so toggling a control is verifiable
            # from the log (channel is the resolved value after any 'Auto' pick).
            self.progress.emit(
                f"Preview settings IN USE -> registration channel: {tf_full.channel_to_use}"
                f"  |  downsample: {self.downsample_factor}x"
            )
            # Surface the resolved channel so the 'Auto' dropdown can show the pick.
            self.resolved_channel.emit(int(tf_full.channel_to_use))

            positions = np.array(tf_full._tile_positions)

            # Build proper grid mapping for irregular grids
            unique_y = np.sort(np.unique(np.round(positions[:, 0], 0)))  # Y positions (rows)
            unique_x = np.sort(np.unique(np.round(positions[:, 1], 0)))  # X positions (cols)
            n_rows, n_cols = len(unique_y), len(unique_x)

            y_to_row = {y: i for i, y in enumerate(unique_y)}
            x_to_col = {x: i for i, x in enumerate(unique_x)}

            # Map (row, col) -> tile index
            grid = {}
            for idx, (y, x) in enumerate(positions):
                r = y_to_row[np.round(y, 0)]
                c = x_to_col[np.round(x, 0)]
                grid[(r, c)] = idx

            self.progress.emit(
                f"Grid: {n_cols}x{n_rows}, selecting center {self.preview_cols}x{self.preview_rows}"
            )

            center_row, center_col = n_rows // 2, n_cols // 2
            half_rows, half_cols = self.preview_rows // 2, self.preview_cols // 2

            selected_indices = []
            selected_grid_pos = []  # Track (row, col) for coloring
            for row in range(center_row - half_rows, center_row - half_rows + self.preview_rows):
                for col in range(
                    center_col - half_cols, center_col - half_cols + self.preview_cols
                ):
                    if (row, col) in grid:
                        selected_indices.append(grid[(row, col)])
                        selected_grid_pos.append(
                            (row - (center_row - half_rows), col - (center_col - half_cols))
                        )

            self.progress.emit(f"Selected {len(selected_indices)} tiles")

            original_positions = tf_full._tile_positions.copy()
            selected_positions = [original_positions[i] for i in selected_indices]

            # Create a new TileFusion for the subset
            tf = TileFusion(
                self.tiff_path,
                downsample_factors=(self.downsample_factor, self.downsample_factor),
                registration_z=self.registration_z,
                registration_t=self.registration_t,
                channel_to_use=self.registration_channel,
            )
            tf._tile_positions = selected_positions
            tf.n_tiles = len(selected_indices)
            tf.position_dim = tf.n_tiles
            tf._tile_index_map = selected_indices

            # Store original read methods
            original_read_tile = tf._read_tile
            original_read_tile_region = tf._read_tile_region

            def patched_read_tile(tile_idx):
                real_idx = tf._tile_index_map[tile_idx]
                # Temporarily restore original method to read from full dataset
                return original_read_tile.__func__(tf_full, real_idx)

            def patched_read_tile_region(tile_idx, y_slice, x_slice):
                real_idx = tf._tile_index_map[tile_idx]
                return original_read_tile_region.__func__(tf_full, real_idx, y_slice, x_slice)

            tf._read_tile = patched_read_tile
            tf._read_tile_region = patched_read_tile_region

            self.progress.emit("Running registration...")
            tf.refine_tile_positions_with_cross_correlation()
            self.progress.emit(f"Found {len(tf.pairwise_metrics)} pairs")

            tf.optimize_shifts(
                method="TWO_ROUND_ITERATIVE",
                rel_thresh=self.outlier_rel_thresh,
                abs_thresh=self.outlier_abs_thresh,
                iterative=True,
            )
            global_offsets = tf.global_offsets

            pixel_size = tf._pixel_size
            min_y = min(p[0] for p in selected_positions)
            min_x = min(p[1] for p in selected_positions)
            max_y = max(p[0] for p in selected_positions) + tf.Y * pixel_size[0]
            max_x = max(p[1] for p in selected_positions) + tf.X * pixel_size[1]

            h = int((max_y - min_y) / pixel_size[0]) + 100
            w = int((max_x - min_x) / pixel_size[1]) + 100

            self.progress.emit(f"Creating preview images ({h}x{w})...")

            color_before = np.zeros((h, w, 3), dtype=np.uint8)
            color_after = np.zeros((h, w, 3), dtype=np.uint8)
            fused = np.zeros((h, w), dtype=np.float32)
            weight = np.zeros((h, w), dtype=np.float32)

            checkerboard_colors = [
                (255, 100, 100),
                (100, 255, 100),
                (100, 100, 255),
                (255, 255, 100),
                (255, 100, 255),
                (100, 255, 255),
            ]

            def get_color(row, col):
                return checkerboard_colors[((row % 2) * 3 + (col % 3)) % 6]

            # Read tiles using TileFusion's format-aware methods
            for i, (pos, orig_idx) in enumerate(zip(selected_positions, selected_indices)):
                arr = tf_full._read_tile(orig_idx)
                if arr.ndim == 3:
                    arr = arr[0]  # Take first channel for preview
                arr_raw = arr.astype(np.float32)

                p1, p99 = np.percentile(arr_raw, [2, 98])
                arr_norm = np.clip((arr_raw - p1) / (p99 - p1 + 1e-6), 0, 1)

                grid_row, grid_col = selected_grid_pos[i]
                color = get_color(grid_row, grid_col)

                th, tw = arr_norm.shape

                # BEFORE: raw stage positions (integer placement; this panel shows the
                # uncorrected misalignment, so sub-pixel doesn't apply here).
                oy_before = int(round((pos[0] - min_y) / pixel_size[0]))
                ox_before = int(round((pos[1] - min_x) / pixel_size[1]))
                y1, y2 = max(0, oy_before), min(oy_before + th, h)
                x1, x2 = max(0, ox_before), min(ox_before + tw, w)
                if y2 > y1 and x2 > x1:
                    tile_h, tile_w = y2 - y1, x2 - x1
                    for c in range(3):
                        color_before[y1:y2, x1:x2, c] = (
                            arr_norm[:tile_h, :tile_w] * color[c]
                        ).astype(np.uint8)

                # AFTER: registered position placed at SUB-PIXEL precision. The
                # registered pixel position is fractional; truncating it to int
                # (the old behaviour) reintroduces up to ~1 px of seam misalignment
                # -- the same int-cast the fusion path still has. Place at the integer
                # floor after fractional-shifting the tile so the sub-pixel offset is
                # honoured.
                oy_f = (pos[0] - min_y) / pixel_size[0] + float(global_offsets[i][0])
                ox_f = (pos[1] - min_x) / pixel_size[1] + float(global_offsets[i][1])
                oy0, ox0 = int(np.floor(oy_f)), int(np.floor(ox_f))
                fy, fx = oy_f - oy0, ox_f - ox0
                # mode="nearest" replicates the real edge pixels; "constant"/cval=0 would
                # pad the leading edge with pure black -> a spurious black border on every
                # tile. The fractional shift is sub-pixel, so edge replication is faithful.
                arr_sub = ndi_shift(arr_norm, (fy, fx), order=1, mode="nearest")
                raw_sub = ndi_shift(arr_raw, (fy, fx), order=1, mode="nearest")
                y1, y2 = max(0, oy0), min(oy0 + th, h)
                x1, x2 = max(0, ox0), min(ox0 + tw, w)
                if y2 > y1 and x2 > x1:
                    tile_h, tile_w = y2 - y1, x2 - x1
                    for c in range(3):
                        color_after[y1:y2, x1:x2, c] = (
                            arr_sub[:tile_h, :tile_w] * color[c]
                        ).astype(np.uint8)
                    fused[y1:y2, x1:x2] += raw_sub[:tile_h, :tile_w]
                    weight[y1:y2, x1:x2] += 1.0

            weight = np.maximum(weight, 1.0)
            fused = fused / weight

            self.progress.emit("Preview ready!")
            self.finished.emit(color_before, color_after, fused)

        except Exception as e:

            self.error.emit(f"Error: {str(e)}\n{traceback.format_exc()}")


class _TqdmSignalRedirect:
    """Redirect tqdm output to a Qt signal, updating the last log line in-place."""

    update_last_line = pyqtSignal(str)

    def __init__(self, signal):
        self._signal = signal
        self._last_text = ""

    def write(self, s):
        s = s.strip()
        if s and s != self._last_text:
            self._last_text = s
            # Emit with special prefix so log() can update in-place
            self._signal.emit(f"\x00PROGRESS:{s}")

    def flush(self):
        pass


def _registration_qc(tf) -> str:
    """One-line registration quality summary for the run log: how many candidate
    overlaps locked vs were rejected, and the NCC distribution (the per-seam
    confidence the global solve is weighted by). Lets the user sanity-check a run."""
    from tilefusion.registration import find_adjacent_pairs

    nccs = (
        np.array([m[2] for m in tf.pairwise_metrics.values()])
        if tf.pairwise_metrics
        else np.array([])
    )
    try:
        cand = len(find_adjacent_pairs(tf._tile_positions, tf._pixel_size, (tf.Y, tf.X)))
    except Exception:
        cand = len(tf.pairwise_metrics)
    locked = len(tf.pairwise_metrics)
    rejected = max(0, cand - locked)
    if nccs.size:
        return (
            f"{locked}/{cand} overlaps locked ({rejected} rejected) | "
            f"NCC mean {nccs.mean():.2f} median {np.median(nccs):.2f} "
            f"min {nccs.min():.2f} | weak (<0.4): {int((nccs < 0.4).sum())}"
        )
    return f"{locked} overlaps locked"


def _run_fusion_pipeline(
    tiff_path,
    do_registration,
    blend_pixels,
    downsample_factor,
    fusion_mode,
    flatfield=None,
    darkfield=None,
    registration_z=None,
    registration_t=0,
    registration_channel=None,
    log_fn=None,
):
    """Shared stitching pipeline used by both single and batch workers.

    Returns the output path string. Raises on failure.
    """
    import gc
    import json
    import shutil
    import time

    from tilefusion import TileFusion

    def log(msg):
        if log_fn:
            log_fn(msg)

    p = Path(tiff_path)
    output_path = p.parent / f"{p.stem}_fused.ome.zarr"
    output_folder = p.parent / f"{p.stem}_fused"

    if output_path.exists():
        shutil.rmtree(output_path)
    if output_folder.exists():
        shutil.rmtree(output_folder)

    metrics_path = p.parent / "metrics.json"
    if metrics_path.exists():
        metrics_path.unlink()
    for m in p.parent.glob("metrics_*.json"):
        m.unlink()

    step_start = time.time()
    tf = TileFusion(
        tiff_path,
        output_path=output_path,
        blend_pixels=blend_pixels,
        downsample_factors=(downsample_factor, downsample_factor),
        flatfield=flatfield,
        darkfield=darkfield,
        registration_z=registration_z,
        registration_t=registration_t,
        channel_to_use=registration_channel,
    )
    load_time = time.time() - step_start
    log(f"Loaded {tf.n_tiles} tiles ({tf.Y}x{tf.X}) [{load_time:.1f}s]")

    if len(tf._unique_regions) > 1:
        log(f"Multi-region dataset: {tf._unique_regions}")
        tf.stitch_all_regions()
        return str(output_folder)

    step_start = time.time()
    if do_registration:
        log("Computing registration...")
        tf.refine_tile_positions_with_cross_correlation()
        tf.save_pairwise_metrics(metrics_path)
        reg_time = time.time() - step_start
        log(f"Registration complete [{reg_time:.1f}s]: {_registration_qc(tf)}")
    else:
        log("Using stage positions (no registration)")

    step_start = time.time()
    log("Optimizing positions...")
    tf.optimize_shifts(method="TWO_ROUND_ITERATIVE", rel_thresh=0.5, abs_thresh=2.0, iterative=True)
    gc.collect()

    tf._tile_positions = [
        tuple(np.array(pos) + off * np.array(tf.pixel_size))
        for pos, off in zip(tf._tile_positions, tf.global_offsets)
    ]
    opt_time = time.time() - step_start
    log(f"Positions optimized [{opt_time:.1f}s]")

    # Per-seam elastic distortion correction (applied at fusion).
    log("Building per-seam distortion correction...")
    tf._build_distortion_correction()

    step_start = time.time()
    log("Computing fused image space...")
    tf._compute_fused_image_space()
    tf._pad_to_chunk_multiple()
    log(f"Output size: {tf.padded_shape[0]} x {tf.padded_shape[1]}")

    scale0 = output_path / "scale0" / "image"
    scale0.parent.mkdir(parents=True, exist_ok=True)
    tf._create_fused_tensorstore(output_path=scale0)

    mode_label = "direct placement" if fusion_mode == "direct" else "blended"
    log(f"Fusing tiles ({mode_label})...")
    # Explicit per-block counter (platform-independent), throttled to ~5% steps.
    _last_pct = {"v": -5}

    def _fuse_progress(block_idx, total_blocks):
        pct = int(100 * block_idx / max(total_blocks, 1))
        if pct >= _last_pct["v"] + 5 or block_idx == total_blocks:
            _last_pct["v"] = pct
            log(f"Fusing block {block_idx}/{total_blocks} ({pct}%)")

    tf.progress_callback = _fuse_progress
    tf._fuse_tiles(mode=fusion_mode)
    tf.progress_callback = None
    fuse_time = time.time() - step_start
    log(f"Tiles fused [{fuse_time:.1f}s]")

    ngff = {
        "attributes": {"_ARRAY_DIMENSIONS": ["t", "c", "y", "x"]},
        "zarr_format": 3,
        "node_type": "group",
    }
    with open(output_path / "scale0" / "zarr.json", "w") as f:
        json.dump(ngff, f, indent=2)

    step_start = time.time()
    log("Building multiscale pyramid...")
    tf._create_multiscales(output_path, factors=tf.multiscale_factors)
    tf._generate_ngff_zarr3_json(output_path, resolution_multiples=tf.resolution_multiples)
    pyramid_time = time.time() - step_start
    log(f"Pyramid built [{pyramid_time:.1f}s]")

    return str(output_path)


class FusionWorker(QThread):
    """Worker thread for running tile fusion."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(str, float)  # output_path, elapsed_time
    error = pyqtSignal(str)
    resolved_channel = pyqtSignal(int)  # channel auto-pick resolved to (for the 'Auto' label)

    def __init__(
        self,
        tiff_path,
        do_registration,
        blend_pixels,
        downsample_factor,
        fusion_mode="blended",
        flatfield=None,
        darkfield=None,
        registration_z=None,
        registration_t=0,
        registration_channel=None,
        outlier_rel_thresh=0.5,
        outlier_abs_thresh=2.0,
        enable_distortion=True,
    ):
        super().__init__()
        self.tiff_path = tiff_path
        self.do_registration = do_registration
        self.blend_pixels = blend_pixels
        self.downsample_factor = downsample_factor
        self.fusion_mode = fusion_mode
        self.flatfield = flatfield
        self.darkfield = darkfield
        self.registration_z = registration_z
        self.registration_t = registration_t
        self.registration_channel = registration_channel
        self.outlier_rel_thresh = outlier_rel_thresh
        self.outlier_abs_thresh = outlier_abs_thresh
        self.enable_distortion = enable_distortion
        self.output_path = None

    def _stdout_context(self):
        """Context manager that redirects meaningful print() to the progress signal."""
        import sys

        signal = self.progress
        orig = sys.stdout

        class _Redirect:
            def write(self, s):
                s = s.strip()
                # Skip decorative lines and empty output
                if not s or s.startswith("=") or s.startswith("-"):
                    return
                signal.emit(s)

            def flush(self):
                pass

        class _Ctx:
            def __enter__(ctx):
                sys.stdout = _Redirect()
                return ctx

            def __exit__(ctx, *exc):
                sys.stdout = orig

        return _Ctx()

    def _tqdm_context(self):
        """Context manager that redirects tqdm output to the progress signal."""
        import sys
        import tqdm as _tqdm_mod

        redirect = _TqdmSignalRedirect(self.progress)
        # Monkey-patch tqdm's default file to our redirect
        orig_init = _tqdm_mod.tqdm.__init__

        def patched_init(tqdm_self, *args, **kwargs):
            kwargs.setdefault("file", redirect)
            kwargs["leave"] = False
            kwargs["bar_format"] = "{desc}: {n}/{total} [{elapsed}<{remaining}]"
            orig_init(tqdm_self, *args, **kwargs)

        class _Ctx:
            def __enter__(ctx):
                _tqdm_mod.tqdm.__init__ = patched_init
                return ctx

            def __exit__(ctx, *exc):
                _tqdm_mod.tqdm.__init__ = orig_init

        return _Ctx()

    def run(self):
        try:
            from tilefusion import TileFusion
            import gc
            import json
            import shutil
            import time

            start_time = time.time()
            self.progress.emit(f"Loading {self.tiff_path}...")

            output_path = (
                Path(self.tiff_path).parent / f"{Path(self.tiff_path).stem}_fused.ome.zarr"
            )
            # Multi-region output folder
            output_folder = Path(self.tiff_path).parent / f"{Path(self.tiff_path).stem}_fused"

            # Remove existing outputs if present
            if output_path.exists():
                shutil.rmtree(output_path)
            if output_folder.exists():
                shutil.rmtree(output_folder)

            # Metrics will be saved inside the output .ome.zarr directory
            # Clean up any old metrics from previous runs in the parent dir
            old_metrics = Path(self.tiff_path).parent / "metrics.json"
            if old_metrics.exists():
                old_metrics.unlink()
            for m in Path(self.tiff_path).parent.glob("metrics_*.json"):
                m.unlink()

            step_start = time.time()
            tf = TileFusion(
                self.tiff_path,
                output_path=output_path,
                blend_pixels=self.blend_pixels or (50, 50),
                downsample_factors=(self.downsample_factor, self.downsample_factor),
                flatfield=self.flatfield,
                darkfield=self.darkfield,
                registration_z=self.registration_z,
                registration_t=self.registration_t,
                channel_to_use=self.registration_channel,
            )
            tf.enable_distortion_correction = self.enable_distortion
            load_time = time.time() - step_start
            self.progress.emit(f"Loaded {tf.n_tiles} tiles ({tf.Y}x{tf.X} each) [{load_time:.1f}s]")
            # Surface the resolved channel so the 'Auto' dropdown can show the pick.
            self.resolved_channel.emit(int(tf.channel_to_use))

            # Auto-compute blend_pixels from tile overlap if requested
            if self.blend_pixels is None:
                from tilefusion.registration import find_adjacent_pairs

                pairs = find_adjacent_pairs(tf._tile_positions, tf._pixel_size, (tf.Y, tf.X))
                if pairs:
                    # Feather over ~2x the actual SEAM overlap. Each pair's overlap has one
                    # small dimension (the real seam depth) and one full-tile dimension; take
                    # the small one. A feather >= the overlap makes the inter-tile weight
                    # cross over GRADUALLY (each pixel dominated by its nearer tile) instead
                    # of a flat 50/50 band -- which is what stops a residual seam misalignment
                    # from reading as a doubled feature. overlap/2 (old) left a 50/50 plateau
                    # that ghosted; ~2x overlap is the convergence point where wider stops
                    # helping. Capped to tile/2 by make_1d_profile.
                    seam_overlaps = [min(p[4], p[5]) for p in pairs if min(p[4], p[5]) > 0]
                    seam = int(np.median(seam_overlaps)) if seam_overlaps else 50
                    b = max(seam * 2, 10)
                    blend_pixels = (b, b)
                    tf.blend_pixels = blend_pixels
                    self.progress.emit(
                        f"Auto blend width: {blend_pixels[0]}px (Y), {blend_pixels[1]}px (X)"
                    )
                else:
                    tf.blend_pixels = (50, 50)
                    self.progress.emit("No adjacent pairs detected — using default 50px blend")

            # Determine regions to process
            regions = tf._unique_regions if len(tf._unique_regions) > 1 else [None]
            is_multi_region = regions[0] is not None

            if is_multi_region:
                self.progress.emit(f"Multi-region dataset: {len(regions)} regions")
                output_folder = Path(self.tiff_path).parent / f"{Path(self.tiff_path).stem}_fused"
                output_folder.mkdir(parents=True, exist_ok=True)

            for region_idx, region in enumerate(regions):
                if is_multi_region:
                    region_output = output_folder / f"{region}.ome.zarr"
                    self.progress.emit(f"\nRegion {region_idx + 1}/{len(regions)}: {region}")
                    tf = TileFusion(
                        self.tiff_path,
                        output_path=region_output,
                        blend_pixels=self.blend_pixels or (50, 50),
                        downsample_factors=(self.downsample_factor, self.downsample_factor),
                        flatfield=self.flatfield,
                        darkfield=self.darkfield,
                        registration_z=self.registration_z,
                        registration_t=self.registration_t,
                        channel_to_use=self.registration_channel,
                        region=region,
                    )
                    tf.enable_distortion_correction = self.enable_distortion
                    self.progress.emit(f"Loaded {tf.n_tiles} tiles ({tf.Y}x{tf.X} each)")
                    cur_output = region_output
                else:
                    cur_output = output_path

                # Registration
                step_start = time.time()
                metrics_path = cur_output / "metrics.json"
                if self.do_registration:
                    self.progress.emit("Registering...")
                    with self._tqdm_context():
                        tf.refine_tile_positions_with_cross_correlation()
                    cur_output.mkdir(parents=True, exist_ok=True)
                    tf.save_pairwise_metrics(metrics_path)
                    reg_time = time.time() - step_start
                    self.progress.emit(f"Registration [{reg_time:.1f}s]: {_registration_qc(tf)}")
                else:
                    self.progress.emit("Using stage positions (no registration)")

                # Optimize
                self.progress.emit("Optimizing positions...")
                tf.optimize_shifts(
                    method="TWO_ROUND_ITERATIVE",
                    rel_thresh=self.outlier_rel_thresh,
                    abs_thresh=self.outlier_abs_thresh,
                    iterative=True,
                )
                gc.collect()

                tf._tile_positions = [
                    tuple(np.array(pos) + off * np.array(tf.pixel_size))
                    for pos, off in zip(tf._tile_positions, tf.global_offsets)
                ]
                opt_time = time.time() - step_start
                self.progress.emit(f"Positions optimized [{opt_time:.1f}s]")

                # Per-seam elastic distortion correction (applied at fusion).
                self.progress.emit("Building per-seam distortion correction...")
                tf._build_distortion_correction()

                # Compute fused space
                step_start = time.time()
                self.progress.emit("Computing fused image space...")
                tf._compute_fused_image_space()
                tf._pad_to_chunk_multiple()
                self.progress.emit(f"Output: {tf.padded_shape[0]} x {tf.padded_shape[1]}")

                # Create output store
                scale0 = cur_output / "scale0" / "image"
                scale0.parent.mkdir(parents=True, exist_ok=True)
                tf._create_fused_tensorstore(output_path=scale0)

                # Fuse. Use an explicit per-block callback (not tqdm scraping) so the
                # live counter shows identically on Linux and macOS; throttle to a few
                # percent so large mosaics don't flood the log.
                mode_label = "direct placement" if self.fusion_mode == "direct" else "blended"
                self.progress.emit(f"Fusing tiles ({mode_label})...")
                _last_pct = {"v": -5}

                def _fuse_progress(block_idx, total_blocks):
                    pct = int(100 * block_idx / max(total_blocks, 1))
                    if pct >= _last_pct["v"] + 5 or block_idx == total_blocks:
                        _last_pct["v"] = pct
                        self.progress.emit(f"Fusing block {block_idx}/{total_blocks} ({pct}%)")

                tf.progress_callback = _fuse_progress
                with self._tqdm_context():
                    tf._fuse_tiles(mode=self.fusion_mode)
                tf.progress_callback = None
                fuse_time = time.time() - step_start
                self.progress.emit(f"Tiles fused [{fuse_time:.1f}s]")

                # Metadata
                ngff = {
                    "attributes": {"_ARRAY_DIMENSIONS": ["t", "c", "y", "x"]},
                    "zarr_format": 3,
                    "node_type": "group",
                }
                with open(cur_output / "scale0" / "zarr.json", "w") as f:
                    json.dump(ngff, f, indent=2)

                # Pyramid
                step_start = time.time()
                self.progress.emit("Building pyramid...")
                with self._tqdm_context():
                    tf._create_multiscales(cur_output, factors=tf.multiscale_factors)
                tf._generate_ngff_zarr3_json(
                    cur_output, resolution_multiples=tf.resolution_multiples
                )
                pyramid_time = time.time() - step_start
                self.progress.emit(f"Pyramid built [{pyramid_time:.1f}s]")

            elapsed_time = time.time() - start_time
            if is_multi_region:
                self.output_path = str(output_folder)
                self.finished.emit(str(output_folder), elapsed_time)
                return

            self.output_path = str(output_path)
            self.finished.emit(str(output_path), elapsed_time)

        except Exception as e:

            self.error.emit(f"Error: {str(e)}\n{traceback.format_exc()}")


class BatchFusionWorker(QThread):
    """Worker thread for batch processing multiple folders/files."""

    progress = pyqtSignal(str)
    item_started = pyqtSignal(int, int, str)  # (current_index, total, item_name)
    item_finished = pyqtSignal(int, int)  # (current_index, total) for progress bar
    finished = pyqtSignal(int, int, float)  # (succeeded, failed, total_time)
    error = pyqtSignal(str)

    def __init__(
        self,
        paths,
        do_registration,
        blend_pixels,
        downsample_factor,
        fusion_mode="blended",
        flatfield=None,
        darkfield=None,
    ):
        super().__init__()
        self.paths = paths
        self.do_registration = do_registration
        self.blend_pixels = blend_pixels
        self.downsample_factor = downsample_factor
        self.fusion_mode = fusion_mode
        self.flatfield = flatfield
        self.darkfield = darkfield

    def _log(self, index, total, name, message):
        self.progress.emit(f"[{index + 1}/{total} {name}] {message}")

    def run(self):
        try:
            self._run_batch()
        except Exception as e:

            self.error.emit(f"Batch processing failed: {e}\n{traceback.format_exc()}")
            self.finished.emit(0, len(self.paths), 0.0)

    def _run_batch(self):
        import time

        total = len(self.paths)
        succeeded = 0
        failed = 0
        batch_start = time.time()

        for idx, tiff_path in enumerate(self.paths):
            name = Path(tiff_path).name
            self.item_started.emit(idx, total, name)

            try:

                def log_fn(msg, _idx=idx, _total=total, _name=name):
                    self._log(_idx, _total, _name, msg)

                _run_fusion_pipeline(
                    tiff_path,
                    self.do_registration,
                    self.blend_pixels,
                    self.downsample_factor,
                    self.fusion_mode,
                    flatfield=self.flatfield,
                    darkfield=self.darkfield,
                    log_fn=log_fn,
                )
                succeeded += 1
            except MemoryError:
                failed += 1
                self._log(idx, total, name, "FAILED: Out of memory. Stopping batch.")
                self.item_finished.emit(idx, total)
                break
            except Exception as e:

                failed += 1
                self._log(idx, total, name, f"FAILED: {e}")
                self._log(idx, total, name, traceback.format_exc())

            self.item_finished.emit(idx, total)

        total_time = time.time() - batch_start
        self.finished.emit(succeeded, failed, total_time)


class DropArea(QFrame):
    """Drag and drop area for files or folders. Supports single and multi-drop."""

    fileDropped = pyqtSignal(str)
    filesDropped = pyqtSignal(list)  # list of path strings (directories or .tif/.tiff files)
    _default_style = "border: 2px dashed #888; border-radius: 8px; background: #fafafa;"
    _hover_style = "border: 2px dashed #0071e3; border-radius: 8px; background: #e8f4ff;"
    _active_style = "border: 2px solid #34c759; border-radius: 8px; background: #f0fff4;"
    _warn_style = "border: 2px solid #ff9500; border-radius: 8px; background: #fff8f0;"

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(100)
        self.setStyleSheet(self._default_style)

        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(12, 12, 12, 12)

        layout.addStretch()

        self.label = QLabel("Drop OME-TIFF or SQUID folder here\nor click to browse")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setStyleSheet("border: none; background: transparent;")
        layout.addWidget(self.label)

        self.file_paths = []

    @property
    def file_path(self):
        return self.file_paths[0] if self.file_paths else None

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(self._hover_style)

    def dragLeaveEvent(self, event):
        if self.file_path:
            self.setStyleSheet(self._active_style)
        else:
            self.setStyleSheet(self._default_style)

    def _is_valid_path(self, file_path):
        """Check if a path is a valid folder or TIFF file."""
        path = Path(file_path)
        return path.is_dir() or file_path.endswith((".tif", ".tiff"))

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            self.setStyleSheet(self._default_style)
            return

        valid_paths = []
        invalid_names = []
        for url in urls:
            file_path = url.toLocalFile()
            if self._is_valid_path(file_path):
                valid_paths.append(file_path)
            else:
                invalid_names.append(Path(file_path).name)

        if not valid_paths:
            self.setStyleSheet(self._default_style)
            return

        if len(valid_paths) == 1:
            self.setFile(valid_paths[0])
            self.fileDropped.emit(valid_paths[0])
        else:
            self.setFiles(valid_paths, invalid_names)
            self.filesDropped.emit(valid_paths)

    def mousePressEvent(self, event):
        from PyQt5.QtWidgets import QMenu

        menu = QMenu(self)
        file_action = menu.addAction("Select OME-TIFF file...")
        folder_action = menu.addAction("Select SQUID folder...")

        action = menu.exec_(self.mapToGlobal(event.pos()))

        if action == file_action:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select OME-TIFF file", "", "TIFF files (*.tif *.tiff);;All files (*.*)"
            )
            if file_path:
                self.setFile(file_path)
                self.fileDropped.emit(file_path)
        elif action == folder_action:
            folder_path = QFileDialog.getExistingDirectory(self, "Select SQUID folder")
            if folder_path:
                self.setFile(folder_path)
                self.fileDropped.emit(folder_path)

    def setFile(self, file_path):
        self.file_paths = [file_path]
        path = Path(file_path)
        self.setStyleSheet(self._active_style)
        self.label.setText(path.name)

    def setFiles(self, paths, invalid_names=None):
        """Set multiple paths and update the display for batch mode."""
        self.file_paths = list(paths)
        names = [Path(p).name for p in paths]
        label_lines = f"📦 {len(paths)} items selected:\n" + "\n".join(f"  {n}" for n in names)
        if invalid_names:
            label_lines += f"\n⚠ Skipped: {', '.join(invalid_names)}"
            self.setStyleSheet(self._warn_style)
        else:
            self.setStyleSheet(self._active_style)
        self.icon_label.setText("✅")
        self.label.setText(label_lines)


class FlatfieldDropArea(QFrame):
    """Small drag and drop area for flatfield .npy files."""

    fileDropped = pyqtSignal(str)
    _default_style = "border: 2px dashed #888; border-radius: 8px; background: #fafafa;"
    _hover_style = "border: 2px dashed #5856d6; border-radius: 8px; background: #f0f0ff;"
    _active_style = "border: 2px solid #5856d6; border-radius: 8px; background: #f5f5ff;"

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)
        self.setStyleSheet(self._default_style)

        layout = QHBoxLayout(self)
        layout.setSpacing(8)

        self.label = QLabel("Drop flatfield .npy here or click to browse")
        self.label.setStyleSheet("border: none; background: transparent;")
        layout.addWidget(self.label)
        layout.addStretch()

        self.file_path = None

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(self._hover_style)

    def dragLeaveEvent(self, event):
        if self.file_path:
            self.setStyleSheet(self._active_style)
        else:
            self.setStyleSheet(self._default_style)

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.endswith(".npy"):
                self.setFile(file_path)
                self.fileDropped.emit(file_path)
            else:
                self.setStyleSheet(self._default_style)
        else:
            self.setStyleSheet(self._default_style)

    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select flatfield file", "", "NumPy files (*.npy);;All files (*.*)"
        )
        if file_path:
            self.setFile(file_path)
            self.fileDropped.emit(file_path)

    def setFile(self, file_path):
        self.file_path = file_path
        path = Path(file_path)
        self.setStyleSheet(self._active_style)
        self.label.setText(path.name)

    def clear(self):
        self.file_path = None
        self.setStyleSheet(self._default_style)
        self.label.setText("Drop flatfield .npy here or click to browse")


class FlatfieldWorker(QThread):
    """Worker thread for calculating flatfield (retrospective, internal numpy estimator)."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(object, object)  # flatfield, darkfield (or None)
    error = pyqtSignal(str)

    def __init__(self, file_path, n_samples=50, use_darkfield=False):
        super().__init__()
        self.file_path = file_path
        self.n_samples = n_samples
        self.use_darkfield = use_darkfield

    def run(self):
        try:
            from tilefusion import TileFusion
            from tilefusion.flatfield import estimate_flatfield_channel

            self.progress.emit("Loading metadata...")

            # Create TileFusion instance to read tiles.
            # NOTE: No flatfield/darkfield passed intentionally - flatfield estimation
            # must be performed on raw, uncorrected tiles.
            tf = TileFusion(self.file_path)

            n_tiles = tf.n_tiles
            n_channels = tf.channels
            n_samples = min(self.n_samples, n_tiles)

            rng = np.random.default_rng(42)
            sample_indices = sorted(rng.choice(n_tiles, size=n_samples, replace=False))

            self.progress.emit(f"Computing flatfield: {n_samples} tiles, {n_channels} channels...")

            # Process per-channel to avoid OOM on large multi-channel z-stacks.
            # Only one channel's worth of tiles is in memory at a time.
            tile_shape = (tf.Y, tf.X)
            flatfield = np.zeros((n_channels,) + tile_shape, dtype=np.float32)
            darkfield = (
                np.zeros((n_channels,) + tile_shape, dtype=np.float32)
                if self.use_darkfield
                else None
            )

            for ch in range(n_channels):
                self.progress.emit(f"Channel {ch + 1}/{n_channels}: reading tiles...")
                # Load only this channel per tile to minimize memory.
                # _read_tile_region reads one z-level; we index the channel.
                images = np.empty((n_samples,) + tile_shape, dtype=np.float32)
                for i, tile_idx in enumerate(sample_indices):
                    tile = tf._read_tile(tile_idx)  # (C, Y, X), one z-level
                    images[i] = tile[ch if ch < tile.shape[0] else 0]
                    del tile

                self.progress.emit(f"Channel {ch + 1}/{n_channels}: estimating flatfield...")
                ff, df = estimate_flatfield_channel(images, self.use_darkfield)
                flatfield[ch] = ff
                if self.use_darkfield:
                    darkfield[ch] = df

                del images

            self.progress.emit("Flatfield calculation complete!")
            self.finished.emit(flatfield, darkfield)

        except Exception as e:

            self.error.emit(f"Error: {str(e)}\n{traceback.format_exc()}")


class OmeTiffExportWorker(QThread):
    """Worker thread for the on-demand OME-TIFF export (off the GUI thread).

    Streams the already-written fused OME-Zarr to a Squid-style tiled BigTIFF;
    pixel size is read from the NGFF metadata inside the export function."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(str)  # tif path
    error = pyqtSignal(str)

    def __init__(self, zarr_dir, channel_names=None):
        super().__init__()
        self.zarr_dir = str(zarr_dir)
        self.channel_names = channel_names

    def run(self):
        try:
            from tilefusion.ome_tiff_export import export_zarr_to_ome_tiff

            self.progress.emit(f"Exporting OME-TIFF from {self.zarr_dir} ...")
            tif_path = export_zarr_to_ome_tiff(self.zarr_dir, channel_names=self.channel_names)
            self.finished.emit(str(tif_path))
        except Exception as e:
            self.error.emit(f"Error: {str(e)}\n{traceback.format_exc()}")


class StitcherGUI(QMainWindow):
    """Main GUI window for the stitcher."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cephla Stitcher")
        self.setMinimumSize(580, 850)
        self._set_cephla_icon()

        self.worker = None
        self.output_path = None
        self.regions = []  # List of region names for multi-region outputs
        self.is_multi_region = False

        # Batch processing state
        self.batch_paths = []

        # Flatfield correction state
        self.flatfield = None  # Shape (C, Y, X) or None
        self.darkfield = None  # Shape (C, Y, X) or None
        self.flatfield_worker = None

        # Dataset dimension state (for registration z/t selection)
        self.dataset_n_z = 1
        self.dataset_n_t = 1
        self.dataset_n_channels = 1
        self.dataset_channel_names = []

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Input drop area (no wrapper group to avoid double border)
        self.drop_area = DropArea()
        self.drop_area.fileDropped.connect(self.on_file_dropped)
        self.drop_area.filesDropped.connect(self.on_files_dropped)
        layout.addWidget(self.drop_area)

        # Preview section
        preview_group = QGroupBox("Preview")
        preview_layout = QHBoxLayout(preview_group)

        preview_layout.addWidget(QLabel("Grid:"))

        self.preview_cols_spin = QSpinBox()
        self.preview_cols_spin.setRange(2, 15)
        self.preview_cols_spin.setValue(5)
        self.preview_cols_spin.setFixedWidth(55)
        preview_layout.addWidget(self.preview_cols_spin)

        preview_layout.addWidget(QLabel("x"))

        self.preview_rows_spin = QSpinBox()
        self.preview_rows_spin.setRange(2, 15)
        self.preview_rows_spin.setValue(5)
        self.preview_rows_spin.setFixedWidth(55)
        preview_layout.addWidget(self.preview_rows_spin)

        preview_layout.addStretch()

        self.preview_button = QPushButton("Preview")
        self.preview_button.setObjectName("previewButton")
        self.preview_button.setCursor(Qt.PointingHandCursor)
        self.preview_button.clicked.connect(self.run_preview)
        self.preview_button.setEnabled(False)
        preview_layout.addWidget(self.preview_button)

        layout.addWidget(preview_group)

        # Flatfield correction section
        flatfield_group = QGroupBox("Flatfield Correction")
        flatfield_layout = QVBoxLayout(flatfield_group)
        flatfield_layout.setSpacing(8)

        self.flatfield_checkbox = QCheckBox("Enable flatfield correction")
        self.flatfield_checkbox.setChecked(True)
        self.flatfield_checkbox.toggled.connect(self.on_flatfield_toggled)
        flatfield_layout.addWidget(self.flatfield_checkbox)

        # Container for flatfield options (shown when enabled)
        self.flatfield_options_widget = QWidget()
        flatfield_options_layout = QVBoxLayout(self.flatfield_options_widget)
        flatfield_options_layout.setContentsMargins(20, 0, 0, 0)
        flatfield_options_layout.setSpacing(8)

        # Radio buttons for Calculate vs Load
        self.flatfield_mode_group = QButtonGroup(self)
        radio_layout = QHBoxLayout()

        self.calc_radio = QRadioButton("Calculate from tiles")
        self.calc_radio.setChecked(True)
        self.flatfield_mode_group.addButton(self.calc_radio, 0)
        radio_layout.addWidget(self.calc_radio)

        self.load_radio = QRadioButton("Load from file")
        self.flatfield_mode_group.addButton(self.load_radio, 1)
        radio_layout.addWidget(self.load_radio)

        radio_layout.addStretch()
        flatfield_options_layout.addLayout(radio_layout)

        # Calculate options container
        self.calc_options_widget = QWidget()
        calc_options_layout = QVBoxLayout(self.calc_options_widget)
        calc_options_layout.setContentsMargins(0, 4, 0, 0)
        calc_options_layout.setSpacing(8)

        self.darkfield_checkbox = QCheckBox("Include darkfield correction")
        self.darkfield_checkbox.setChecked(False)
        calc_options_layout.addWidget(self.darkfield_checkbox)

        calc_btn_layout = QHBoxLayout()
        self.calc_flatfield_button = QPushButton("Calculate Flatfield")
        self.calc_flatfield_button.setObjectName("calcFlatfieldButton")
        self.calc_flatfield_button.setCursor(Qt.PointingHandCursor)
        self.calc_flatfield_button.clicked.connect(self.calculate_flatfield)
        self.calc_flatfield_button.setEnabled(False)
        calc_btn_layout.addWidget(self.calc_flatfield_button)

        self.save_flatfield_button = QPushButton("Save")
        self.save_flatfield_button.setCursor(Qt.PointingHandCursor)
        self.save_flatfield_button.clicked.connect(self.save_flatfield)
        self.save_flatfield_button.setEnabled(False)
        self.save_flatfield_button.setToolTip("Save calculated flatfield to .npy file")
        self.save_flatfield_button.setMinimumHeight(36)
        calc_btn_layout.addWidget(self.save_flatfield_button)
        calc_btn_layout.addStretch()
        calc_options_layout.addLayout(calc_btn_layout)

        flatfield_options_layout.addWidget(self.calc_options_widget)

        # Load options container
        self.load_options_widget = QWidget()
        self.load_options_widget.setVisible(False)
        load_options_layout = QVBoxLayout(self.load_options_widget)
        load_options_layout.setContentsMargins(0, 0, 0, 0)

        self.flatfield_drop_area = FlatfieldDropArea()
        self.flatfield_drop_area.fileDropped.connect(self.on_flatfield_dropped)
        load_options_layout.addWidget(self.flatfield_drop_area)

        flatfield_options_layout.addWidget(self.load_options_widget)

        # Flatfield status and view button
        status_layout = QHBoxLayout()
        self.flatfield_status = QLabel("No flatfield")
        self.flatfield_status.setStyleSheet("color: #86868b; font-size: 11px;")
        status_layout.addWidget(self.flatfield_status)

        self.view_flatfield_button = QPushButton("View")
        self.view_flatfield_button.setCursor(Qt.PointingHandCursor)
        self.view_flatfield_button.clicked.connect(self.view_flatfield)
        self.view_flatfield_button.setEnabled(False)
        self.view_flatfield_button.setToolTip("View flatfield and darkfield")
        self.view_flatfield_button.setFixedWidth(60)
        status_layout.addWidget(self.view_flatfield_button)

        self.clear_flatfield_button = QPushButton("Clear")
        self.clear_flatfield_button.setCursor(Qt.PointingHandCursor)
        self.clear_flatfield_button.clicked.connect(self.clear_flatfield)
        self.clear_flatfield_button.setEnabled(False)
        self.clear_flatfield_button.setToolTip("Clear loaded flatfield")
        self.clear_flatfield_button.setFixedWidth(60)
        status_layout.addWidget(self.clear_flatfield_button)
        status_layout.addStretch()

        flatfield_options_layout.addLayout(status_layout)

        flatfield_layout.addWidget(self.flatfield_options_widget)

        self.flatfield_mode_group.buttonClicked.connect(self.on_flatfield_mode_changed)

        layout.addWidget(flatfield_group)

        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(8)

        self.registration_checkbox = QCheckBox("Enable registration refinement")
        self.registration_checkbox.setChecked(True)
        self.registration_checkbox.toggled.connect(self.on_registration_toggled)
        settings_layout.addWidget(self.registration_checkbox)

        # Downsample factor (shown when registration enabled)
        self.downsample_widget = QWidget()
        self.downsample_widget.setVisible(True)
        downsample_layout = QHBoxLayout(self.downsample_widget)
        downsample_layout.setContentsMargins(20, 0, 0, 0)
        downsample_layout.addWidget(QLabel("Downsample:"))
        self.downsample_spin = QSpinBox()
        self.downsample_spin.setRange(1, 8)
        # Default 1 = full-resolution registration = sub-pixel accuracy. Downsampling
        # coarsens the recovered shift by the same factor, so >1 trades accuracy for
        # speed (the registration-only cost; output resolution is unaffected).
        self.downsample_spin.setValue(1)
        self.downsample_spin.setToolTip(
            "Registration downsample. 1 = full-resolution = sub-pixel accurate "
            "(slower); higher = faster but coarser registration."
        )
        downsample_layout.addWidget(self.downsample_spin)
        downsample_layout.addStretch()
        settings_layout.addWidget(self.downsample_widget)

        # Registration z/t selection (shown when registration enabled AND multi-z/t dataset)
        self.reg_zt_widget = QWidget()
        self.reg_zt_widget.setVisible(False)
        reg_zt_layout = QHBoxLayout(self.reg_zt_widget)
        reg_zt_layout.setContentsMargins(20, 0, 0, 0)
        self.reg_z_label = QLabel("Z-level:")
        reg_zt_layout.addWidget(self.reg_z_label)
        self.reg_z_spin = QSpinBox()
        self.reg_z_spin.setRange(0, 0)
        self.reg_z_spin.setValue(0)
        self.reg_z_spin.setToolTip("Z-level to use for registration")
        self.reg_z_spin.setFixedWidth(60)
        reg_zt_layout.addWidget(self.reg_z_spin)
        self.reg_t_label = QLabel("Timepoint:")
        reg_zt_layout.addWidget(self.reg_t_label)
        self.reg_t_spin = QSpinBox()
        self.reg_t_spin.setRange(0, 0)
        self.reg_t_spin.setValue(0)
        self.reg_t_spin.setToolTip("Timepoint to use for registration")
        self.reg_t_spin.setFixedWidth(60)
        reg_zt_layout.addWidget(self.reg_t_spin)
        self.reg_channel_label = QLabel("Channel:")
        reg_zt_layout.addWidget(self.reg_channel_label)
        self.reg_channel_combo = QComboBox()
        self.reg_channel_combo.setToolTip("Channel to use for registration")
        self.reg_channel_combo.setMinimumWidth(120)
        reg_zt_layout.addWidget(self.reg_channel_combo)
        reg_zt_layout.addStretch()
        settings_layout.addWidget(self.reg_zt_widget)

        self.blend_checkbox = QCheckBox("Enable blending")
        self.blend_checkbox.setChecked(True)
        self.blend_checkbox.toggled.connect(self.on_blend_toggled)
        settings_layout.addWidget(self.blend_checkbox)

        # Blend pixels (shown when blending enabled)
        self.blend_value_widget = QWidget()
        self.blend_value_widget.setVisible(True)
        blend_value_layout = QHBoxLayout(self.blend_value_widget)
        blend_value_layout.setContentsMargins(20, 0, 0, 0)
        blend_value_layout.addWidget(QLabel("Blend pixels:"))
        self.blend_auto_checkbox = QCheckBox("Auto")
        self.blend_auto_checkbox.setChecked(True)
        self.blend_auto_checkbox.setToolTip("Auto-compute blend width as ~2x the seam overlap")
        self.blend_auto_checkbox.toggled.connect(
            lambda checked: self.blend_spin.setEnabled(not checked)
        )
        blend_value_layout.addWidget(self.blend_auto_checkbox)
        self.blend_spin = QSpinBox()
        self.blend_spin.setRange(1, 500)
        self.blend_spin.setValue(50)
        self.blend_spin.setEnabled(False)
        blend_value_layout.addWidget(self.blend_spin)
        blend_value_layout.addStretch()
        settings_layout.addWidget(self.blend_value_widget)

        # Per-seam elastic distortion correction (built after optimization, applied
        # at fusion). Self-calibrating with an identity fallback, so leaving it on is
        # safe; the worst case per seam is the translation-only result.
        self.distortion_checkbox = QCheckBox("Correct lens distortion (per-seam elastic)")
        self.distortion_checkbox.setChecked(True)
        self.distortion_checkbox.setToolTip(
            "Measure how each seam's alignment varies along its length (optical field\n"
            "distortion / local rotation a single shift can't capture) and warp it out\n"
            "at fusion. Flat or low-texture seams are left unchanged."
        )
        settings_layout.addWidget(self.distortion_checkbox)

        # Outlier threshold controls (shown when registration enabled)
        self.outlier_widget = QWidget()
        self.outlier_widget.setVisible(False)
        outlier_layout = QHBoxLayout(self.outlier_widget)
        outlier_layout.setContentsMargins(20, 0, 0, 0)
        outlier_layout.addWidget(QLabel("Outlier rel:"))
        self.outlier_rel_spin = QSpinBox()
        self.outlier_rel_spin.setRange(1, 200)
        self.outlier_rel_spin.setValue(50)
        self.outlier_rel_spin.setSuffix("%")
        self.outlier_rel_spin.setToolTip(
            "Relative threshold: reject links with residual > this % of median"
        )
        self.outlier_rel_spin.setFixedWidth(80)
        outlier_layout.addWidget(self.outlier_rel_spin)
        outlier_layout.addWidget(QLabel("abs:"))
        self.outlier_abs_spin = QSpinBox()
        self.outlier_abs_spin.setRange(1, 50)
        self.outlier_abs_spin.setValue(2)
        self.outlier_abs_spin.setSuffix("px")
        self.outlier_abs_spin.setToolTip(
            "Absolute threshold: minimum residual (pixels) to reject a link"
        )
        self.outlier_abs_spin.setFixedWidth(80)
        outlier_layout.addWidget(self.outlier_abs_spin)
        outlier_layout.addStretch()
        settings_layout.addWidget(self.outlier_widget)

        # Show outlier controls when registration is enabled
        self.registration_checkbox.toggled.connect(
            lambda checked: self.outlier_widget.setVisible(checked)
        )

        layout.addWidget(settings_group)

        # Run button
        self.run_button = QPushButton("Run Stitching")
        self.run_button.setObjectName("runButton")
        self.run_button.setMinimumHeight(40)
        self.run_button.setCursor(Qt.PointingHandCursor)
        self.run_button.clicked.connect(self.run_stitching)
        self.run_button.setEnabled(False)
        layout.addWidget(self.run_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(6)
        layout.addWidget(self.progress_bar)

        # Log output
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(100)
        self.log_text.setMaximumHeight(140)
        self.log_text.setPlaceholderText("Log output...")
        layout.addWidget(self.log_text)

        # Region selection (for multi-region outputs)
        self.region_widget = QWidget()
        self.region_widget.setVisible(False)
        region_layout = QHBoxLayout(self.region_widget)
        region_layout.setContentsMargins(0, 0, 0, 0)
        region_layout.addWidget(QLabel("Region:"))
        self.region_combo = QComboBox()
        self.region_combo.setMinimumWidth(100)
        self.region_combo.currentIndexChanged.connect(self._on_region_combo_changed)
        region_layout.addWidget(self.region_combo)
        self.region_slider = QSlider(Qt.Horizontal)
        self.region_slider.setMinimum(0)
        self.region_slider.setMaximum(0)
        self.region_slider.valueChanged.connect(self._on_region_slider_changed)
        region_layout.addWidget(self.region_slider)
        layout.addWidget(self.region_widget)

        # Open in Napari button
        self.napari_button = QPushButton("Open in Napari")
        self.napari_button.setToolTip(
            "Open the stitched OME-Zarr result in Napari for visualization."
        )
        self.napari_button.setObjectName("napariButton")
        self.napari_button.setMinimumHeight(40)
        self.napari_button.setCursor(Qt.PointingHandCursor)
        self.napari_button.clicked.connect(self.open_in_napari)
        self.napari_button.setEnabled(False)
        layout.addWidget(self.napari_button)

        # Additional action buttons row
        actions_layout = QHBoxLayout()

        self.mip_button = QPushButton("Max Projection")
        self.mip_button.setCursor(Qt.PointingHandCursor)
        self.mip_button.setToolTip("Compute max intensity projection of 3D result")
        self.mip_button.clicked.connect(self.compute_mip)
        self.mip_button.setEnabled(False)
        actions_layout.addWidget(self.mip_button)

        # Export the fused Zarr to a Squid-style OME-TIFF, on demand (not automatic).
        self.export_ometiff_button = QPushButton("Export OME-TIFF")
        self.export_ometiff_button.setCursor(Qt.PointingHandCursor)
        self.export_ometiff_button.setToolTip(
            "Write a Squid-style OME-TIFF (tiled BigTIFF) from the fused OME-Zarr result."
        )
        self.export_ometiff_button.clicked.connect(self.export_ome_tiff)
        self.export_ometiff_button.setEnabled(False)
        actions_layout.addWidget(self.export_ometiff_button)

        self.open_existing_button = QPushButton("Open Existing")
        self.open_existing_button.setCursor(Qt.PointingHandCursor)
        self.open_existing_button.setToolTip(
            "Load a previously fused output to view in Napari or compute MIP"
        )
        self.open_existing_button.clicked.connect(self.open_existing_in_napari)
        actions_layout.addWidget(self.open_existing_button)

        layout.addLayout(actions_layout)

        layout.addStretch()

        # Subtle branding — logo + text
        brand_widget = QWidget()
        brand_layout = QHBoxLayout(brand_widget)
        brand_layout.setContentsMargins(0, 4, 0, 6)
        brand_layout.setSpacing(5)
        brand_layout.addStretch()

        logo_path = Path(__file__).parent / "cephla_logo.svg"
        if logo_path.exists():
            logo_label = QLabel()
            renderer = QSvgRenderer(str(logo_path))
            pm = QPixmap(16, 16)
            pm.fill(Qt.transparent)
            p = QPainter(pm)
            renderer.render(p)
            p.end()
            logo_label.setPixmap(pm)
            brand_layout.addWidget(logo_label)

        brand_text = QLabel("cephla")
        brand_text.setStyleSheet("color: #31c4f3; font-size: 10px; letter-spacing: 3px;")
        brand_layout.addWidget(brand_text)
        brand_layout.addStretch()
        layout.addWidget(brand_widget)

    def _set_cephla_icon(self):
        """Set the Cephla logo as window icon."""
        logo_path = Path(__file__).parent / "cephla_logo.svg"
        if logo_path.exists():
            renderer = QSvgRenderer(str(logo_path))
            pixmap = QPixmap(64, 64)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
            self.setWindowIcon(QIcon(pixmap))

    @property
    def is_batch_mode(self):
        return len(self.batch_paths) > 1

    def _update_batch_mode_ui(self):
        """Update UI to reflect batch vs single mode."""
        batch = self.is_batch_mode
        self.preview_button.setEnabled(not batch)
        self.calc_flatfield_button.setEnabled(not batch and self.drop_area.file_path is not None)
        self.reg_zt_widget.setEnabled(not batch)
        if batch:
            self.preview_button.setToolTip("Preview is not available in batch mode")
            self.calc_flatfield_button.setToolTip(
                "Calculate flatfield from a single dataset first, then load it for batch"
            )
            self.reg_zt_widget.setToolTip("Registration z/t/channel uses defaults in batch mode")
        else:
            self.preview_button.setToolTip("")
            self.calc_flatfield_button.setToolTip("")
            self.reg_zt_widget.setToolTip("")
            self.napari_button.setToolTip("")

    def on_file_dropped(self, file_path):
        """Handle single file/folder drop — exits batch mode."""
        self.batch_paths = []
        self._update_batch_mode_ui()

        path = Path(file_path)
        if path.is_dir():
            self.log(f"Selected SQUID folder: {file_path}")
        else:
            self.log(f"Selected OME-TIFF: {file_path}")
        self.run_button.setEnabled(True)
        self.preview_button.setEnabled(True)
        self.calc_flatfield_button.setEnabled(True)

        # Re-assert quality defaults on every new dataset load
        self.blend_checkbox.setChecked(True)
        self.blend_auto_checkbox.setChecked(True)
        self.registration_checkbox.setChecked(True)

        # Clear previous flatfield when new file is selected
        self.flatfield = None
        self.darkfield = None
        self.flatfield_status.setText("No flatfield")
        self.flatfield_status.setStyleSheet("color: #86868b; font-size: 11px;")
        self.flatfield_drop_area.clear()
        self.view_flatfield_button.setEnabled(False)
        self.clear_flatfield_button.setEnabled(False)
        self.save_flatfield_button.setEnabled(False)

        # Load dataset dimensions for registration z/t selection
        try:
            from tilefusion import TileFusion

            tf_temp = TileFusion(file_path)
            self.dataset_n_z = tf_temp.n_z
            self.dataset_n_t = tf_temp.n_t
            self.dataset_n_channels = tf_temp.channels
            if "channel_names" in tf_temp._metadata:
                self.dataset_channel_names = tf_temp._metadata["channel_names"]
            else:
                self.dataset_channel_names = [
                    f"Channel {i}" for i in range(self.dataset_n_channels)
                ]
            tf_temp.close()
            if self.dataset_n_z > 1 or self.dataset_n_t > 1:
                self.log(f"Dataset: {self.dataset_n_z} z-levels, {self.dataset_n_t} timepoints")
            self._update_reg_zt_controls()
        except Exception:
            self.dataset_n_z = 1
            self.dataset_n_t = 1
            self.dataset_n_channels = 1
            self.dataset_channel_names = []
            self._update_reg_zt_controls()

        # Auto-load existing flatfield if present, otherwise disable correction
        # For directories (SQUID folders), also check inside the directory
        if path.is_dir():
            flatfield_path = path / f"{path.name}_flatfield.npy"
            if not flatfield_path.exists():
                # Fallback: check next to the directory
                flatfield_path = path.parent / f"{path.name}_flatfield.npy"
        else:
            flatfield_path = path.parent / f"{path.stem}_flatfield.npy"

        if flatfield_path.exists():
            self.log(f"Found existing flatfield: {flatfield_path.name}")
            self.on_flatfield_dropped(str(flatfield_path))
            self.flatfield_drop_area.setFile(str(flatfield_path))
        else:
            # Auto-calculate flatfield instead of silently disabling correction
            self.flatfield_checkbox.setChecked(True)
            self.log("No existing flatfield found — auto-calculating from tiles...")
            self._auto_calculate_flatfield()

    def on_files_dropped(self, paths):
        """Handle multi-drop — validate each path and enter batch mode."""
        from tilefusion import TileFusion

        self.log_text.clear()
        self.log(f"Validating {len(paths)} dropped items...")

        valid_paths = []
        invalid_names = []
        for p in paths:
            name = Path(p).name
            try:
                with TileFusion(p):
                    pass
                valid_paths.append(p)
                self.log(f"  ✓ {name}")
            except Exception as e:
                invalid_names.append(name)
                self.log(f"  ✗ {name}: {e}")

        if not valid_paths:
            self.log("No valid datasets found.")
            self.run_button.setEnabled(False)
            return

        if invalid_names:
            self.log(
                f"\n{len(valid_paths)} of {len(paths)} valid. "
                f"Skipped: {', '.join(invalid_names)}"
            )

        # Single valid item — fall back to normal single-item flow
        if len(valid_paths) == 1:
            self.log(f"\nOnly 1 valid item — using single mode.")
            self.drop_area.setFile(valid_paths[0])
            self.on_file_dropped(valid_paths[0])
            return

        # Multiple valid items — enter batch mode
        self.drop_area.setFiles(valid_paths, invalid_names)
        self.batch_paths = valid_paths
        self._update_batch_mode_ui()
        self.run_button.setEnabled(True)

        self.dataset_n_z = 1
        self.dataset_n_t = 1
        self.dataset_n_channels = 1
        self.dataset_channel_names = []

        if not invalid_names:
            self.log(f"\nAll {len(valid_paths)} items valid. Ready to run batch.")

    def on_registration_toggled(self, checked):
        self.downsample_widget.setVisible(checked)
        self._update_reg_zt_controls()

    def _update_reg_zt_controls(self):
        """Update visibility and ranges of registration z/t controls."""
        registration_enabled = self.registration_checkbox.isChecked()
        has_multi_z = self.dataset_n_z > 1
        has_multi_t = self.dataset_n_t > 1
        has_multi_channel = self.dataset_n_channels > 1

        # Show z/t widget only when registration is enabled AND dataset has multi-z or multi-t or multi-channel
        show_zt = registration_enabled and (has_multi_z or has_multi_t or has_multi_channel)
        self.reg_zt_widget.setVisible(show_zt)

        if show_zt:
            # Update z spinbox
            self.reg_z_label.setVisible(has_multi_z)
            self.reg_z_spin.setVisible(has_multi_z)
            if has_multi_z:
                self.reg_z_spin.setRange(0, self.dataset_n_z - 1)
                self.reg_z_spin.setValue(self.dataset_n_z // 2)  # Default to middle

            # Update t spinbox
            self.reg_t_label.setVisible(has_multi_t)
            self.reg_t_spin.setVisible(has_multi_t)
            if has_multi_t:
                self.reg_t_spin.setRange(0, self.dataset_n_t - 1)
                self.reg_t_spin.setValue(0)  # Default to first timepoint

            # Update channel combo
            self.reg_channel_label.setVisible(has_multi_channel)
            self.reg_channel_combo.setVisible(has_multi_channel)
            if has_multi_channel:
                self.reg_channel_combo.clear()
                # "Auto" (index 0) lets the pipeline pick the highest-tissue-contrast
                # channel per dataset; the named channels below are manual overrides.
                self.reg_channel_combo.addItems(["Auto"] + list(self.dataset_channel_names))
                self.reg_channel_combo.setCurrentIndex(0)

    def on_blend_toggled(self, checked):
        self.blend_value_widget.setVisible(checked)

    def on_flatfield_toggled(self, checked):
        # Only show/hide flatfield options; preserve any loaded/calculated data
        self.flatfield_options_widget.setVisible(checked)

    def on_flatfield_mode_changed(self, button):
        is_calculate = self.calc_radio.isChecked()
        self.calc_options_widget.setVisible(is_calculate)
        self.load_options_widget.setVisible(not is_calculate)

    def _auto_calculate_flatfield(self):
        """Auto-calculate flatfield when no .npy file exists for the loaded dataset."""
        if not self.drop_area.file_path:
            self.flatfield_checkbox.setChecked(False)
            return

        self.flatfield_status.setText("Auto-calculating flatfield...")
        self.flatfield_status.setStyleSheet("color: #ff9500; font-size: 11px;")
        self.calc_flatfield_button.setEnabled(False)

        self.flatfield_worker = FlatfieldWorker(
            self.drop_area.file_path,
            n_samples=50,
            use_darkfield=self.darkfield_checkbox.isChecked(),
        )
        self.flatfield_worker.progress.connect(self.log)
        self.flatfield_worker.finished.connect(self.on_flatfield_calculated)
        self.flatfield_worker.error.connect(self._on_auto_flatfield_error)
        self.flatfield_worker.start()

    def _on_auto_flatfield_error(self, error_msg):
        """Handle auto-flatfield calculation failure gracefully."""
        self.calc_flatfield_button.setEnabled(True)
        self.flatfield_checkbox.setChecked(False)
        self.flatfield_status.setText("Auto-calculation failed — disabled")
        self.flatfield_status.setStyleSheet("color: #ff3b30; font-size: 11px;")
        self.log(f"Auto-flatfield failed: {error_msg}")

    def calculate_flatfield(self):
        if not self.drop_area.file_path:
            return

        self.calc_flatfield_button.setEnabled(False)
        self.flatfield_status.setText("Calculating flatfield...")
        self.flatfield_status.setStyleSheet("color: #ff9500; font-size: 11px;")

        self.flatfield_worker = FlatfieldWorker(
            self.drop_area.file_path,
            n_samples=50,
            use_darkfield=self.darkfield_checkbox.isChecked(),
        )
        self.flatfield_worker.progress.connect(self.log)
        self.flatfield_worker.finished.connect(self.on_flatfield_calculated)
        self.flatfield_worker.error.connect(self.on_flatfield_error)
        self.flatfield_worker.start()

    def on_flatfield_calculated(self, flatfield, darkfield):
        self.flatfield = flatfield
        self.darkfield = darkfield
        self.calc_flatfield_button.setEnabled(True)
        self.save_flatfield_button.setEnabled(True)
        self.view_flatfield_button.setEnabled(True)
        self.clear_flatfield_button.setEnabled(True)

        n_channels = flatfield.shape[0]
        status = f"Flatfield ready ({n_channels} channels)"
        if darkfield is not None:
            status += " + darkfield"
        self.flatfield_status.setText(status)
        self.flatfield_status.setStyleSheet("color: #34c759; font-size: 11px; font-weight: 600;")
        self.log(f"Flatfield calculation complete: {flatfield.shape}")

        # Auto-save flatfield next to input file
        if self.drop_area.file_path:
            try:
                from tilefusion import save_flatfield as save_ff

                input_path = Path(self.drop_area.file_path)
                # Use path.name for directories, path.stem for files (consistent with auto-load)
                if input_path.is_dir():
                    auto_save_path = input_path / f"{input_path.name}_flatfield.npy"
                else:
                    auto_save_path = input_path.parent / f"{input_path.stem}_flatfield.npy"
                save_ff(auto_save_path, self.flatfield, self.darkfield)
                self.log(f"Auto-saved flatfield to {auto_save_path}")
            except Exception as e:
                self.log(f"Warning: Could not auto-save flatfield: {e}")

    def save_flatfield(self):
        if self.flatfield is None:
            return

        # Default path based on input (consistent with auto-save/auto-load)
        default_path = "flatfield.npy"
        if self.drop_area.file_path:
            input_path = Path(self.drop_area.file_path)
            if input_path.is_dir():
                default_path = str(input_path / f"{input_path.name}_flatfield.npy")
            else:
                default_path = str(input_path.parent / f"{input_path.stem}_flatfield.npy")

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Flatfield",
            default_path,
            "NumPy files (*.npy);;All files (*.*)",
        )
        if file_path:
            try:
                from tilefusion import save_flatfield as save_ff

                save_ff(Path(file_path), self.flatfield, self.darkfield)
                self.log(f"Saved flatfield to {file_path}")
            except Exception as e:
                self.log(f"Error saving flatfield: {e}")

    def on_flatfield_error(self, error_msg):
        self.calc_flatfield_button.setEnabled(True)
        self.flatfield_status.setText("Calculation failed")
        self.flatfield_status.setStyleSheet("color: #ff3b30; font-size: 11px;")
        self.log(error_msg)

    def on_flatfield_dropped(self, file_path):

        try:
            from tilefusion import load_flatfield

            self.flatfield, self.darkfield = load_flatfield(Path(file_path))
            n_channels = self.flatfield.shape[0]
            status = f"Loaded ({n_channels} channels)"
            if self.darkfield is not None:
                status += " + darkfield"
            self.flatfield_status.setText(status)
            self.flatfield_status.setStyleSheet(
                "color: #34c759; font-size: 11px; font-weight: 600;"
            )
            self.view_flatfield_button.setEnabled(True)
            self.clear_flatfield_button.setEnabled(True)
            self.save_flatfield_button.setEnabled(True)
            # Enable flatfield correction when successfully loaded
            self.flatfield_checkbox.setChecked(True)
            self.log(f"Loaded flatfield from {file_path}: {self.flatfield.shape}")
        except Exception as e:
            # Clear any stale flatfield data on load failure
            self.flatfield = None
            self.darkfield = None
            self.flatfield_status.setText(f"Load failed: {e}")
            self.flatfield_status.setStyleSheet("color: #ff3b30; font-size: 11px;")
            self.view_flatfield_button.setEnabled(False)
            self.log(f"Error loading flatfield: {e}")

    def view_flatfield(self):
        if self.flatfield is None:
            return

        try:
            import matplotlib

            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
            import tempfile
            import subprocess

            n_channels = self.flatfield.shape[0]
            has_darkfield = self.darkfield is not None
            n_rows = 2 if has_darkfield else 1

            fig, axes = plt.subplots(n_rows, n_channels, figsize=(4 * n_channels, 4 * n_rows))

            # Handle single channel case (axes not 2D)
            if n_channels == 1 and n_rows == 1:
                axes = [[axes]]
            elif n_channels == 1:
                axes = [[ax] for ax in axes]
            elif n_rows == 1:
                axes = [axes]

            # First row: flatfield
            for ch in range(n_channels):
                ax = axes[0][ch]
                im = ax.imshow(self.flatfield[ch], cmap="viridis", vmin=0)
                ax.set_title(f"Flatfield Ch{ch}")
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Second row: darkfield (if available)
            if has_darkfield:
                for ch in range(n_channels):
                    ax = axes[1][ch]
                    im = ax.imshow(self.darkfield[ch], cmap="magma", vmin=0)
                    # Show constant value in title if darkfield is uniform
                    df_val = self.darkfield[ch].ravel()[0]
                    if np.allclose(self.darkfield[ch], df_val):
                        ax.set_title(f"Darkfield Ch{ch} (={df_val:.1f})")
                    else:
                        ax.set_title(f"Darkfield Ch{ch}")
                    ax.axis("off")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.tight_layout()

            # Save to temp file and open with system viewer
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                fig.savefig(f.name, dpi=150, bbox_inches="tight")
                plt.close(fig)
                # Open with default image viewer
                if sys.platform == "darwin":
                    subprocess.Popen(["open", f.name])
                elif sys.platform == "win32":
                    subprocess.Popen(["cmd", "/c", "start", "", f.name])
                else:
                    subprocess.Popen(["xdg-open", f.name])

            self.log("Opened flatfield viewer")
        except Exception as e:
            self.log(f"Error opening viewer: {e}")

    def clear_flatfield(self):
        """Clear loaded/calculated flatfield."""
        self.flatfield = None
        self.darkfield = None
        self.flatfield_status.setText("No flatfield")
        self.flatfield_status.setStyleSheet("color: #86868b; font-size: 11px;")
        self.view_flatfield_button.setEnabled(False)
        self.clear_flatfield_button.setEnabled(False)
        self.save_flatfield_button.setEnabled(False)
        self.flatfield_drop_area.clear()
        self.log("Flatfield cleared")

    def log(self, message):
        if message.startswith("\x00PROGRESS:"):
            # Update last line in-place for progress bars
            text = message[len("\x00PROGRESS:") :]
            cursor = self.log_text.textCursor()
            cursor.movePosition(cursor.End)
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.insertText(text)
        else:
            self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def _flatfield_in_progress(self):
        """Check if flatfield calculation is still running."""
        return self.flatfield_worker is not None and self.flatfield_worker.isRunning()

    def run_stitching(self):
        if not self.drop_area.file_path:
            return

        if self._flatfield_in_progress():
            self.log("Waiting for flatfield calculation to finish...")
            self.run_button.setEnabled(False)
            self.preview_button.setEnabled(False)

            # One-shot connection — disconnect after firing to prevent stacking
            def _on_ready(*_args):
                try:
                    self.flatfield_worker.finished.disconnect(_on_ready)
                except TypeError:
                    pass
                self.run_stitching()

            self.flatfield_worker.finished.connect(_on_ready)
            return

        self.run_button.setEnabled(False)
        self.napari_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log_text.clear()

        if self.blend_checkbox.isChecked():
            if self.blend_auto_checkbox.isChecked():
                blend_pixels = None  # Auto-compute from overlap in FusionWorker
            else:
                blend_val = self.blend_spin.value()
                blend_pixels = (blend_val, blend_val)
            fusion_mode = "blended"
        else:
            blend_pixels = (0, 0)
            fusion_mode = "direct"

        # Get flatfield if enabled
        flatfield = self.flatfield if self.flatfield_checkbox.isChecked() else None
        darkfield = self.darkfield if self.flatfield_checkbox.isChecked() else None

        if self.is_batch_mode:
            self._run_batch(blend_pixels, fusion_mode, flatfield, darkfield)
        else:
            self._run_single(blend_pixels, fusion_mode, flatfield, darkfield)

    def _selected_registration_channel(self):
        """Registration channel from the combo, or None for 'Auto' (let the pipeline
        auto-pick by tissue contrast). Combo index 0 is 'Auto'; named channels follow.
        Single-channel datasets also return None (auto trivially resolves to channel 0).
        """
        if self.dataset_n_channels <= 1:
            return None
        idx = self.reg_channel_combo.currentIndex()
        return None if idx <= 0 else idx - 1

    def _on_resolved_channel(self, ch):
        """Relabel the combo's 'Auto' entry (index 0) to show which channel the
        pipeline auto-picked, e.g. 'Auto (DAPI)'. Only the display text changes;
        index 0 still maps to None in _selected_registration_channel, so the
        auto-pick behaviour is untouched. Resets to plain 'Auto' on dataset reload
        (the combo is repopulated there).
        """
        if self.reg_channel_combo.count() == 0:
            return
        if 0 <= ch < len(self.dataset_channel_names):
            name = self.dataset_channel_names[ch]
        else:
            name = f"channel {ch}"
        self.reg_channel_combo.setItemText(0, f"Auto ({name})")

    def _run_single(self, blend_pixels, fusion_mode, flatfield, darkfield):
        # Get registration z/t values (None means use default middle z)
        registration_z = self.reg_z_spin.value() if self.dataset_n_z > 1 else None
        registration_t = self.reg_t_spin.value() if self.dataset_n_t > 1 else 0
        registration_channel = self._selected_registration_channel()

        self.worker = FusionWorker(
            self.drop_area.file_path,
            self.registration_checkbox.isChecked(),
            blend_pixels,
            self.downsample_spin.value(),
            fusion_mode,
            flatfield=flatfield,
            darkfield=darkfield,
            registration_z=registration_z,
            registration_t=registration_t,
            registration_channel=registration_channel,
            outlier_rel_thresh=self.outlier_rel_spin.value() / 100.0,
            outlier_abs_thresh=float(self.outlier_abs_spin.value()),
        )
        self.worker.progress.connect(self.log)
        self.worker.finished.connect(self.on_fusion_finished)
        self.worker.error.connect(self.on_fusion_error)
        self.worker.resolved_channel.connect(self._on_resolved_channel)
        self.worker.start()

    def _run_batch(self, blend_pixels, fusion_mode, flatfield, darkfield):
        total = len(self.batch_paths)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.log(f"Starting batch processing: {total} items\n")

        self.worker = BatchFusionWorker(
            self.batch_paths,
            self.registration_checkbox.isChecked(),
            blend_pixels,
            self.downsample_spin.value(),
            fusion_mode,
            flatfield=flatfield,
            darkfield=darkfield,
        )
        self.worker.progress.connect(self.log)
        self.worker.error.connect(self.on_fusion_error)
        self.worker.item_started.connect(self._on_batch_item_started)
        self.worker.item_finished.connect(self._on_batch_item_finished)
        self.worker.finished.connect(self._on_batch_finished)
        self.worker.start()

    def _on_batch_item_started(self, index, total, name):
        self.log(f"\n{'='*40}")
        self.log(f"Processing {index + 1}/{total}: {name}")
        self.log(f"{'='*40}")

    def _on_batch_item_finished(self, index, total):
        self.progress_bar.setValue(index + 1)

    def _on_batch_finished(self, succeeded, failed, total_time):
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # Reset to indeterminate for next run
        self.batch_paths = []
        self.run_button.setEnabled(True)
        self.napari_button.setEnabled(True)
        self.export_ometiff_button.setEnabled(True)
        self._update_batch_mode_ui()

        minutes = int(total_time // 60)
        seconds = total_time % 60
        time_str = f"{minutes}m {seconds:.1f}s" if minutes > 0 else f"{seconds:.1f}s"

        self.log(f"\n{'='*40}")
        self.log(f"Batch complete! {succeeded} succeeded, {failed} failed. Total time: {time_str}")
        self.log(f"{'='*40}")

    def on_fusion_finished(self, output_path, elapsed_time):
        self.output_path = output_path
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.napari_button.setEnabled(True)
        self.mip_button.setEnabled(True)
        self.export_ometiff_button.setEnabled(True)

        # Check if this is a multi-region output folder
        output_dir = Path(output_path)
        zarr_subdirs = sorted(output_dir.glob("*.ome.zarr"))
        if zarr_subdirs:
            # Multi-region output
            self.is_multi_region = True
            self.regions = [d.stem.replace(".ome", "") for d in zarr_subdirs]
            self.region_combo.blockSignals(True)
            self.region_combo.clear()
            self.region_combo.addItems(self.regions)
            self.region_combo.blockSignals(False)
            self.region_slider.setMaximum(len(self.regions) - 1)
            self.region_slider.setValue(0)
            self.region_widget.setVisible(True)
            self.log(f"Found {len(self.regions)} regions: {', '.join(self.regions)}")
        else:
            # Single output
            self.is_multi_region = False
            self.regions = []
            self.region_widget.setVisible(False)

        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        time_str = f"{minutes}m {seconds:.1f}s" if minutes > 0 else f"{seconds:.1f}s"

        self.log(f"\nFusion complete! Time: {time_str}\nOutput: {output_path}")

        self.open_in_napari()

    def on_fusion_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.log(f"\nError: {error_msg}")

    def run_preview(self):
        if not self.drop_area.file_path:
            return

        if self._flatfield_in_progress():
            self.log("Waiting for flatfield calculation to finish...")
            self.run_button.setEnabled(False)
            self.preview_button.setEnabled(False)

            def _on_ready(*_args):
                try:
                    self.flatfield_worker.finished.disconnect(_on_ready)
                except TypeError:
                    pass
                self.run_preview()

            self.flatfield_worker.finished.connect(_on_ready)
            return

        self.preview_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log_text.clear()
        self.log("Starting preview...")

        # Get flatfield if enabled
        flatfield = self.flatfield if self.flatfield_checkbox.isChecked() else None
        darkfield = self.darkfield if self.flatfield_checkbox.isChecked() else None

        # Get registration z/t values (None means use default middle z)
        registration_z = self.reg_z_spin.value() if self.dataset_n_z > 1 else None
        registration_t = self.reg_t_spin.value() if self.dataset_n_t > 1 else 0
        registration_channel = self._selected_registration_channel()

        self.preview_worker = PreviewWorker(
            self.drop_area.file_path,
            self.preview_cols_spin.value(),
            self.preview_rows_spin.value(),
            self.downsample_spin.value(),
            flatfield=flatfield,
            darkfield=darkfield,
            registration_z=registration_z,
            registration_t=registration_t,
            registration_channel=registration_channel,
            outlier_rel_thresh=self.outlier_rel_spin.value() / 100.0,
            outlier_abs_thresh=float(self.outlier_abs_spin.value()),
        )
        self.preview_worker.progress.connect(self.log)
        self.preview_worker.finished.connect(self.on_preview_finished)
        self.preview_worker.error.connect(self.on_preview_error)
        self.preview_worker.resolved_channel.connect(self._on_resolved_channel)
        self.preview_worker.start()

    def on_preview_finished(self, color_before, color_after, fused):
        self.progress_bar.setVisible(False)
        self.preview_button.setEnabled(True)
        self.run_button.setEnabled(True)

        self.log("Opening napari with before/after comparison...")

        try:
            import napari

            viewer = napari.Viewer()
            viewer.add_image(color_before, name="BEFORE registration (colored)", rgb=True)
            viewer.add_image(
                color_after, name="AFTER registration (colored)", rgb=True, visible=False
            )
            if fused is not None:
                # The fused preview is a raw-intensity float image; without explicit
                # contrast limits napari renders it black. Scale to the 1-99 percentile
                # of the non-zero (tile) pixels, ignoring the zero background canvas.
                nz = fused[fused > 0]
                if nz.size:
                    lo, hi = (float(v) for v in np.percentile(nz, [1, 99]))
                else:
                    lo, hi = 0.0, 1.0
                if hi <= lo:
                    hi = lo + 1.0
                viewer.add_image(
                    fused,
                    name="Fused result",
                    colormap="gray",
                    contrast_limits=[lo, hi],
                    visible=False,
                )
            napari.run()
        except Exception as e:
            self.log(f"Error opening Napari: {e}\n{traceback.format_exc()}")

    def on_preview_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.preview_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self.log(f"\nError: {error_msg}")

    def _on_region_combo_changed(self, index):
        """Sync slider when dropdown changes."""
        self.region_slider.blockSignals(True)
        self.region_slider.setValue(index)
        self.region_slider.blockSignals(False)

    def _on_region_slider_changed(self, value):
        """Sync dropdown when slider changes."""
        self.region_combo.blockSignals(True)
        self.region_combo.setCurrentIndex(value)
        self.region_combo.blockSignals(False)

    def _fused_channel_names(self):
        """Channel display names from the input dataset metadata, or None."""
        try:
            from tilefusion import TileFusion

            tf = TileFusion(self.drop_area.file_path)
            if "channel_names" in tf._metadata:
                return [ch.replace("_", " ") for ch in tf._metadata["channel_names"]]
        except Exception:
            pass
        return None

    def open_in_napari(self):
        if not self.output_path:
            try:
                import napari

                napari.Viewer()
                napari.run()
            except Exception as e:
                self.log(f"Error opening Napari: {e}\n{traceback.format_exc()}")
            return

        # Determine the actual zarr path to open
        if self.is_multi_region and self.regions:
            selected_region = self.region_combo.currentText()
            zarr_path = Path(self.output_path) / f"{selected_region}.ome.zarr"
            self.log(f"Opening region '{selected_region}' in Napari: {zarr_path}")
        else:
            zarr_path = Path(self.output_path)
            self.log(f"Opening in Napari: {self.output_path}")

        try:
            import napari

            viewer = napari.Viewer()
            _add_fused_zarr(viewer, zarr_path, self._fused_channel_names(), self.log)
            _snapshot_napari_on_close(viewer, Path(zarr_path).stem.replace(".ome", ""), self.log)
            napari.run()
        except Exception as e:
            self.log(f"Error opening Napari: {e}\n{traceback.format_exc()}")

    def export_ome_tiff(self):
        """Build a Squid-style OME-TIFF from the fused OME-Zarr, on demand."""
        if not self.output_path:
            return

        # Resolve the single zarr dir to export (mirrors open_in_napari's logic).
        if self.is_multi_region and self.regions:
            selected_region = self.region_combo.currentText()
            zarr_path = Path(self.output_path) / f"{selected_region}.ome.zarr"
        else:
            zarr_path = Path(self.output_path)

        if not (zarr_path / "scale0" / "image").exists():
            self.log(f"No fused image data found at {zarr_path}; cannot export OME-TIFF.")
            return

        self.export_ometiff_button.setEnabled(False)
        self.ometiff_worker = OmeTiffExportWorker(zarr_path, self._fused_channel_names())
        self.ometiff_worker.progress.connect(self.log)
        self.ometiff_worker.finished.connect(self._on_ome_tiff_exported)
        self.ometiff_worker.error.connect(self._on_ome_tiff_error)
        self.ometiff_worker.start()

    def _on_ome_tiff_exported(self, tif_path):
        self.export_ometiff_button.setEnabled(True)
        self.log(f"OME-TIFF written: {tif_path}")

    def _on_ome_tiff_error(self, error_msg):
        self.export_ometiff_button.setEnabled(True)
        self.log(f"\nOME-TIFF export error: {error_msg}")

    def compute_mip(self):
        """Compute and display max intensity projection in Napari."""
        if not self.output_path:
            return

        try:
            import napari
            import tensorstore as ts

            zarr_path = Path(self.output_path)
            if self.is_multi_region and self.regions:
                selected_region = self.region_combo.currentText()
                zarr_path = zarr_path / f"{selected_region}.ome.zarr"

            scale0 = zarr_path / "scale0" / "image"
            if not scale0.exists():
                self.log("No image data found for MIP")
                return

            store = ts.open(
                {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(scale0)}}
            ).result()
            shape = store.shape
            is_5d = len(shape) == 5

            if not is_5d or shape[2] <= 1:
                self.log("Dataset is 2D — max projection not applicable")
                return

            n_z = shape[2]
            self.log(f"Computing max intensity projection (Z={n_z} planes)...")
            viewer = napari.Viewer()
            n_channels = shape[1]

            for c in range(n_channels):
                # Compute MIP one z-plane at a time to avoid OOM
                mip = None
                for z in range(n_z):
                    plane = np.asarray(store[0, c, z, :, :].read().result())
                    if mip is None:
                        mip = plane.copy()
                    else:
                        np.maximum(mip, plane, out=mip)
                viewer.add_image(
                    mip,
                    name=f"MIP Ch{c}",
                    colormap=CHANNEL_COLORS[c % len(CHANNEL_COLORS)],
                    blending="additive",
                )
                del mip
            _snapshot_napari_on_close(
                viewer, Path(zarr_path).stem.replace(".ome", "") + "_MIP", self.log
            )
            napari.run()
        except Exception as e:
            self.log(f"Error computing MIP: {e}\n{traceback.format_exc()}")

    def open_existing_in_napari(self):
        """Load a previously stitched output so Napari/MIP buttons work."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select fused output folder", str(Path.home())
        )
        if not folder:
            return

        output_dir = Path(folder)

        # Accept either a .ome.zarr directly or a folder containing .ome.zarr(s)
        if output_dir.suffix == ".zarr" and ".ome." in output_dir.name:
            # Single .ome.zarr selected
            self.output_path = folder
            self.is_multi_region = False
            self.regions = []
            self.region_widget.setVisible(False)
        else:
            zarr_subdirs = sorted(output_dir.glob("*.ome.zarr"))
            if zarr_subdirs:
                # Multi-region folder
                self.output_path = folder
                self.is_multi_region = True
                self.regions = [d.stem.replace(".ome", "") for d in zarr_subdirs]
                self.region_combo.blockSignals(True)
                self.region_combo.clear()
                self.region_combo.addItems(self.regions)
                self.region_combo.blockSignals(False)
                self.region_slider.setMaximum(len(self.regions) - 1)
                self.region_slider.setValue(0)
                self.region_widget.setVisible(True)
            else:
                self.log("No .ome.zarr found in selected folder")
                return

        self.napari_button.setEnabled(True)
        self.mip_button.setEnabled(True)
        self.export_ometiff_button.setEnabled(True)
        self.log(f"Loaded: {folder}")
        if self.is_multi_region:
            self.log(f"Regions: {', '.join(self.regions)}")


def auto_contrast(data, pmax=99.9):
    """Contrast limits for fluorescence: histogram-mode background, percentile top."""
    flat = data.ravel()
    if len(flat) > 100000:
        flat = np.random.choice(flat, 100000, replace=False)
    hist, bin_edges = np.histogram(flat, bins=256)
    mode_idx = np.argmax(hist)
    mode_val = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2
    background_pixels = flat[flat <= np.median(flat)]
    bg_std = np.std(background_pixels) if len(background_pixels) > 0 else mode_val * 0.1
    lo = mode_val + 2 * bg_std
    hi = np.percentile(data, pmax)
    if hi - lo < 10:
        hi = lo + 100
    return [float(lo), float(hi)]


def dtype_range(dtype):
    """Valid display range for a numpy dtype."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return [info.min, info.max]
    elif np.issubdtype(dtype, np.floating):
        return [0.0, 1.0]
    return [0, 65535]


class _TSArray:
    """Minimal lazy array view over a tensorstore handle, for dask.from_array.

    Exposes shape/dtype/ndim and returns numpy only for the slice dask requests,
    so napari pages chunks on demand instead of reading whole volumes.
    """

    def __init__(self, store):
        self._store = store
        self.shape = tuple(store.shape)
        self.dtype = store.dtype.numpy_dtype
        self.ndim = len(self.shape)

    def __getitem__(self, idx):
        return np.asarray(self._store[idx].read().result())


def _snapshot_napari_on_close(viewer, name, log):
    """Save a PNG of the napari window to the Desktop when the user closes it.

    The screenshot must be taken *before* the window is destroyed, so we install a
    Qt event filter that fires on the Close event and captures the whole window
    (canvas + dock widgets). The filter is stashed on the viewer so it isn't garbage
    collected while the window lives. Best-effort: any failure is logged, not raised.
    """
    try:
        from datetime import datetime

        from qtpy.QtCore import QEvent, QObject
    except Exception:
        return

    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name)) or "fused"

    class _SnapshotFilter(QObject):
        def __init__(self):
            super().__init__()
            self._done = False

        def eventFilter(self, obj, event):
            if event.type() == QEvent.Close and not self._done:
                self._done = True
                try:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    desktop = Path.home() / "Desktop"
                    desktop.mkdir(parents=True, exist_ok=True)
                    out = desktop / f"stitcher_napari_{safe}_{ts}.png"
                    viewer.window.screenshot(str(out), canvas_only=False)
                    log(f"Saved napari snapshot: {out}")
                except Exception as e:
                    log(f"Could not save napari snapshot: {e}")
            return False

    try:
        win = viewer.window._qt_window
        filt = _SnapshotFilter()
        win.installEventFilter(filt)
        viewer._tf_snapshot_filter = filt  # keep a reference alive
    except Exception as e:
        log(f"Could not arm napari snapshot: {e}")


def _add_fused_zarr(viewer, zarr_path, channel_names, log):
    """Add a fused OME-Zarr as lazy, multiscale, per-channel layers.

    Opens scale*/image with tensorstore (the same engine that wrote the store, so
    it always reads our Zarr v3 output) and wraps each level in a dask array.
    napari pages only the chunk/level it renders, so RAM stays flat regardless of
    dataset size — no full-volume read into numpy, no ome-zarr version dependency.
    """
    import tensorstore as ts
    import dask.array as da

    levels = []
    for scale_dir in sorted(Path(zarr_path).glob("scale*")):
        image_path = scale_dir / "image"
        if image_path.exists():
            store = ts.open(
                {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(image_path)}}
            ).result()
            shp = tuple(store.shape)
            # Chunk Y/X by one codec chunk, 1 elsewhere — lazy, storage-aligned reads.
            chunks = tuple([1] * (len(shp) - 2)) + (min(1024, shp[-2]), min(1024, shp[-1]))
            levels.append(da.from_array(_TSArray(store), chunks=chunks))

    if not levels:
        log(f"No scale*/image data found in {zarr_path}")
        return

    shape = levels[0].shape  # (T, C, Z, Y, X) or (T, C, Y, X)
    is_5d = len(shape) == 5
    n_channels = shape[1] if len(shape) >= 4 else 1
    for c in range(n_channels):
        pyramid = [lvl[:, c] for lvl in levels]  # lazy slice -> (T, Z, Y, X) per level
        sample = pyramid[-1][pyramid[-1].shape[0] // 2]  # smallest level, middle T
        if is_5d:
            sample = sample[sample.shape[0] // 2]  # middle Z -> single (Y, X) plane
        contrast = auto_contrast(np.asarray(sample))
        name = channel_names[c] if channel_names and c < len(channel_names) else f"Channel {c}"
        layer = viewer.add_image(
            pyramid,
            multiscale=True,
            name=name,
            colormap=CHANNEL_COLORS[c % len(CHANNEL_COLORS)],
            blending="additive",
            contrast_limits=contrast,
        )
        layer.contrast_limits_range = dtype_range(pyramid[-1].dtype)

    viewer.dims.axis_labels = ("t", "z", "y", "x") if is_5d else ("t", "y", "x")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = StitcherGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
