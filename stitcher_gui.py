#!/usr/bin/env python3
"""
Stitcher GUI - A simple interface for tile fusion of OME-TIFF files.
"""

import sys
import os
from pathlib import Path
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
    QGraphicsDropShadowEffect,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMimeData, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QFont, QColor, QPalette, QLinearGradient


STYLE_SHEET = """
QMainWindow {
    background-color: #f5f5f7;
}

QWidget {
    font-family: 'SF Pro Display', -apple-system, 'Segoe UI', sans-serif;
    color: #1d1d1f;
}

QLabel#title {
    font-size: 28px;
    font-weight: 600;
    color: #1d1d1f;
    padding: 10px;
}

QLabel#subtitle {
    font-size: 12px;
    color: #86868b;
    padding-bottom: 10px;
}

QGroupBox {
    background-color: transparent;
    border: none;
    margin-top: 8px;
    padding: 5px;
    font-weight: 500;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0px 5px;
    color: #0071e3;
    font-size: 13px;
    font-weight: 600;
}

QCheckBox {
    spacing: 8px;
    font-size: 13px;
    color: #1d1d1f;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid #c7c7cc;
    background-color: #ffffff;
}

QCheckBox::indicator:checked {
    background-color: #0071e3;
    border-color: #0071e3;
}

QCheckBox::indicator:hover {
    border-color: #0071e3;
}

QSpinBox {
    background-color: #ffffff;
    border: 1px solid #c7c7cc;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 13px;
    min-width: 80px;
    color: #1d1d1f;
}

QSpinBox:focus {
    border-color: #0071e3;
}

QSpinBox::up-button, QSpinBox::down-button {
    background-color: #f5f5f7;
    border: none;
    width: 20px;
}

QSpinBox::up-button:hover, QSpinBox::down-button:hover {
    background-color: #e8e8ed;
}

QPushButton#runButton {
    background-color: #0071e3;
    color: white;
    font-size: 15px;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
}

QPushButton#runButton:hover {
    background-color: #0077ed;
}

QPushButton#runButton:disabled {
    background-color: #c7c7cc;
    color: #8e8e93;
}

QPushButton#napariButton {
    background-color: #34c759;
    color: white;
    font-size: 15px;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
}

QPushButton#napariButton:hover {
    background-color: #30d158;
}

QPushButton#napariButton:disabled {
    background-color: #c7c7cc;
    color: #8e8e93;
}

QPushButton#previewButton {
    background-color: #ff9500;
    color: white;
    font-size: 14px;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
}

QPushButton#previewButton:hover {
    background-color: #ff9f0a;
}

QPushButton#previewButton:disabled {
    background-color: #c7c7cc;
    color: #8e8e93;
}

QProgressBar {
    background-color: #e8e8ed;
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #0071e3;
    border-radius: 4px;
}

QTextEdit {
    background-color: #ffffff;
    border: 1px solid #c7c7cc;
    border-radius: 8px;
    padding: 10px;
    font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
    font-size: 11px;
    color: #1d1d1f;
}

QScrollBar:vertical {
    background-color: #f5f5f7;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background-color: #c7c7cc;
    border-radius: 4px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #8e8e93;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""


class PreviewWorker(QThread):
    """Worker thread for running preview stitching on subset of tiles."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(object, object, object)  # color_before, color_after, fused
    error = pyqtSignal(str)

    def __init__(self, tiff_path, preview_cols, preview_rows, downsample_factor):
        super().__init__()
        self.tiff_path = tiff_path
        self.preview_cols = preview_cols
        self.preview_rows = preview_rows
        self.downsample_factor = downsample_factor

    def run(self):
        try:
            import numpy as np
            from tilefusion import TileFusion

            self.progress.emit("Loading metadata...")

            # Create TileFusion instance - handles both OME-TIFF and SQUID formats
            tf_full = TileFusion(
                self.tiff_path, downsample_factors=(self.downsample_factor, self.downsample_factor)
            )

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
                self.tiff_path, downsample_factors=(self.downsample_factor, self.downsample_factor)
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
                method="TWO_ROUND_ITERATIVE", rel_thresh=0.5, abs_thresh=2.0, iterative=True
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

                oy_before = int(round((pos[0] - min_y) / pixel_size[0]))
                ox_before = int(round((pos[1] - min_x) / pixel_size[1]))
                oy_after = oy_before + int(global_offsets[i][0])
                ox_after = ox_before + int(global_offsets[i][1])

                th, tw = arr_norm.shape

                # BEFORE
                y1, y2 = max(0, oy_before), min(oy_before + th, h)
                x1, x2 = max(0, ox_before), min(ox_before + tw, w)
                if y2 > y1 and x2 > x1:
                    tile_h, tile_w = y2 - y1, x2 - x1
                    for c in range(3):
                        color_before[y1:y2, x1:x2, c] = (
                            arr_norm[:tile_h, :tile_w] * color[c]
                        ).astype(np.uint8)

                # AFTER
                y1, y2 = max(0, oy_after), min(oy_after + th, h)
                x1, x2 = max(0, ox_after), min(ox_after + tw, w)
                if y2 > y1 and x2 > x1:
                    tile_h, tile_w = y2 - y1, x2 - x1
                    for c in range(3):
                        color_after[y1:y2, x1:x2, c] = (
                            arr_norm[:tile_h, :tile_w] * color[c]
                        ).astype(np.uint8)
                    fused[y1:y2, x1:x2] += arr_raw[:tile_h, :tile_w]
                    weight[y1:y2, x1:x2] += 1.0

            weight = np.maximum(weight, 1.0)
            fused = fused / weight

            self.progress.emit("Preview ready!")
            self.finished.emit(color_before, color_after, fused)

        except Exception as e:
            import traceback

            self.error.emit(f"Error: {str(e)}\n{traceback.format_exc()}")


class FusionWorker(QThread):
    """Worker thread for running tile fusion."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(str, float)  # output_path, elapsed_time
    error = pyqtSignal(str)

    def __init__(
        self, tiff_path, do_registration, blend_pixels, downsample_factor, fusion_mode="blended"
    ):
        super().__init__()
        self.tiff_path = tiff_path
        self.do_registration = do_registration
        self.blend_pixels = blend_pixels
        self.downsample_factor = downsample_factor
        self.fusion_mode = fusion_mode
        self.output_path = None

    def run(self):
        try:
            from tilefusion import TileFusion
            import shutil
            import time
            import json
            import gc

            start_time = time.time()

            self.progress.emit(f"Loading {self.tiff_path}...")

            output_path = (
                Path(self.tiff_path).parent / f"{Path(self.tiff_path).stem}_fused.ome.zarr"
            )

            # Remove existing output if present
            if output_path.exists():
                shutil.rmtree(output_path)

            # Also remove metrics if not doing registration
            metrics_path = Path(self.tiff_path).parent / "metrics.json"
            if metrics_path.exists():
                metrics_path.unlink()

            step_start = time.time()
            tf = TileFusion(
                self.tiff_path,
                output_path=output_path,
                blend_pixels=self.blend_pixels,
                downsample_factors=(self.downsample_factor, self.downsample_factor),
            )
            load_time = time.time() - step_start
            self.progress.emit(f"Loaded {tf.n_tiles} tiles ({tf.Y}x{tf.X} each) [{load_time:.1f}s]")

            # Registration step
            step_start = time.time()
            if self.do_registration:
                self.progress.emit("Computing registration...")
                tf.refine_tile_positions_with_cross_correlation()
                tf.save_pairwise_metrics(metrics_path)
                reg_time = time.time() - step_start
                self.progress.emit(
                    f"Registration complete: {len(tf.pairwise_metrics)} pairs [{reg_time:.1f}s]"
                )
            else:
                tf.threshold = 1.0  # Skip registration
                self.progress.emit("Using stage positions (no registration)")

            # Optimize shifts
            step_start = time.time()
            self.progress.emit("Optimizing positions...")
            tf.optimize_shifts(
                method="TWO_ROUND_ITERATIVE", rel_thresh=0.5, abs_thresh=2.0, iterative=True
            )
            gc.collect()

            import numpy as np

            tf._tile_positions = [
                tuple(np.array(pos) + off * np.array(tf.pixel_size))
                for pos, off in zip(tf._tile_positions, tf.global_offsets)
            ]
            opt_time = time.time() - step_start
            self.progress.emit(f"Positions optimized [{opt_time:.1f}s]")

            # Compute fused space
            step_start = time.time()
            self.progress.emit("Computing fused image space...")
            tf._compute_fused_image_space()
            tf._pad_to_chunk_multiple()
            self.progress.emit(f"Output size: {tf.padded_shape[0]} x {tf.padded_shape[1]}")

            # Create output store
            scale0 = output_path / "scale0" / "image"
            scale0.parent.mkdir(parents=True, exist_ok=True)
            tf._create_fused_tensorstore(output_path=scale0)

            # Fuse tiles
            mode_label = "direct placement" if self.fusion_mode == "direct" else "blended"
            self.progress.emit(f"Fusing tiles ({mode_label})...")
            tf._fuse_tiles(mode=self.fusion_mode)
            fuse_time = time.time() - step_start
            self.progress.emit(f"Tiles fused [{fuse_time:.1f}s]")

            # Write metadata
            ngff = {
                "attributes": {"_ARRAY_DIMENSIONS": ["t", "c", "y", "x"]},
                "zarr_format": 3,
                "node_type": "group",
            }
            with open(output_path / "scale0" / "zarr.json", "w") as f:
                json.dump(ngff, f, indent=2)

            # Build multiscales
            step_start = time.time()
            self.progress.emit("Building multiscale pyramid...")
            tf._create_multiscales(output_path, factors=tf.multiscale_factors)
            tf._generate_ngff_zarr3_json(output_path, resolution_multiples=tf.resolution_multiples)
            pyramid_time = time.time() - step_start
            self.progress.emit(f"Pyramid built [{pyramid_time:.1f}s]")

            elapsed_time = time.time() - start_time
            self.output_path = str(output_path)
            self.finished.emit(str(output_path), elapsed_time)

        except Exception as e:
            import traceback

            self.error.emit(f"Error: {str(e)}\n{traceback.format_exc()}")


class DropArea(QFrame):
    """Drag and drop area for files or folders."""

    fileDropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.setMinimumHeight(120)
        self.setStyleSheet(
            """
            QFrame {
                border: 2px dashed #c7c7cc;
                border-radius: 12px;
                background-color: #ffffff;
            }
            QFrame:hover {
                border-color: #0071e3;
                background-color: #f0f7ff;
            }
        """
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self.icon_label = QLabel("📂")
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setStyleSheet("font-size: 32px; border: none; background: transparent;")
        layout.addWidget(self.icon_label)

        self.label = QLabel("Drag & Drop OME-TIFF or SQUID folder\nor click to browse")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(
            "color: #86868b; font-size: 13px; border: none; background: transparent;"
        )
        layout.addWidget(self.label)

        self.file_path = None

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(
                """
                QFrame {
                    border: 2px dashed #0071e3;
                    border-radius: 12px;
                    background-color: #e5f1ff;
                }
            """
            )

    def dragLeaveEvent(self, event):
        self.setStyleSheet(
            """
            QFrame {
                border: 2px dashed #c7c7cc;
                border-radius: 12px;
                background-color: #ffffff;
            }
        """
        )

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet(
            """
            QFrame {
                border: 2px dashed #c7c7cc;
                border-radius: 12px;
                background-color: #ffffff;
            }
        """
        )

        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            path = Path(file_path)
            # Accept TIFF files or folders (SQUID format)
            if path.is_dir() or file_path.endswith((".tif", ".tiff")):
                self.setFile(file_path)
                self.fileDropped.emit(file_path)

    def mousePressEvent(self, event):
        from PyQt5.QtWidgets import QMenu, QAction

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
        self.file_path = file_path
        path = Path(file_path)
        self.icon_label.setText("✅")
        if path.is_dir():
            self.label.setText(f"📁 {path.name}")
        else:
            self.label.setText(path.name)
        self.label.setStyleSheet(
            "color: #34c759; font-size: 13px; font-weight: 600; border: none; background: transparent;"
        )


class StitcherGUI(QMainWindow):
    """Main GUI window for the stitcher."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tile Stitcher")
        self.setMinimumSize(500, 600)

        self.worker = None
        self.output_path = None

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Title
        title = QLabel("Tile Stitcher")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("GPU-accelerated tile fusion for OME-TIFF microscopy data")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        # Drop area
        self.drop_area = DropArea()
        self.drop_area.fileDropped.connect(self.on_file_dropped)
        layout.addWidget(self.drop_area)

        # Preview section
        preview_group = QGroupBox("Preview")
        preview_layout = QHBoxLayout(preview_group)

        preview_layout.addWidget(QLabel("Grid size:"))

        self.preview_cols_spin = QSpinBox()
        self.preview_cols_spin.setRange(2, 15)
        self.preview_cols_spin.setValue(5)
        self.preview_cols_spin.setFixedWidth(60)
        preview_layout.addWidget(self.preview_cols_spin)

        preview_layout.addWidget(QLabel("x"))

        self.preview_rows_spin = QSpinBox()
        self.preview_rows_spin.setRange(2, 15)
        self.preview_rows_spin.setValue(5)
        self.preview_rows_spin.setFixedWidth(60)
        preview_layout.addWidget(self.preview_rows_spin)

        preview_layout.addStretch()

        self.preview_button = QPushButton("👁 Preview")
        self.preview_button.setObjectName("previewButton")
        self.preview_button.setCursor(Qt.PointingHandCursor)
        self.preview_button.clicked.connect(self.run_preview)
        self.preview_button.setEnabled(False)
        preview_layout.addWidget(self.preview_button)

        layout.addWidget(preview_group)

        # Registration settings
        reg_group = QGroupBox("Settings")
        reg_layout = QVBoxLayout(reg_group)
        reg_layout.setSpacing(10)

        self.registration_checkbox = QCheckBox("Enable registration refinement")
        self.registration_checkbox.setChecked(False)
        self.registration_checkbox.setMinimumHeight(32)
        self.registration_checkbox.toggled.connect(self.on_registration_toggled)
        reg_layout.addWidget(self.registration_checkbox)

        # Downsample factor (only shown when registration enabled)
        self.downsample_widget = QWidget()
        self.downsample_widget.setMinimumHeight(36)
        self.downsample_widget.setVisible(False)
        downsample_layout = QHBoxLayout(self.downsample_widget)
        downsample_layout.setContentsMargins(24, 0, 0, 0)

        downsample_label = QLabel("Downsample factor:")
        downsample_layout.addWidget(downsample_label)

        self.downsample_spin = QSpinBox()
        self.downsample_spin.setRange(1, 8)
        self.downsample_spin.setValue(2)
        self.downsample_spin.setToolTip("Lower values = slower but more accurate registration")
        downsample_layout.addWidget(self.downsample_spin)

        downsample_layout.addStretch()
        reg_layout.addWidget(self.downsample_widget)

        # Blending checkbox (same level as registration)
        self.blend_checkbox = QCheckBox("Enable blending")
        self.blend_checkbox.setChecked(False)
        self.blend_checkbox.setMinimumHeight(32)
        self.blend_checkbox.toggled.connect(self.on_blend_toggled)
        reg_layout.addWidget(self.blend_checkbox)

        # Blend pixels value (indented under blending)
        self.blend_value_widget = QWidget()
        self.blend_value_widget.setMinimumHeight(36)
        self.blend_value_widget.setVisible(False)
        blend_value_layout = QHBoxLayout(self.blend_value_widget)
        blend_value_layout.setContentsMargins(24, 0, 0, 0)

        blend_label = QLabel("Blend pixels:")
        blend_value_layout.addWidget(blend_label)

        self.blend_spin = QSpinBox()
        self.blend_spin.setRange(1, 500)
        self.blend_spin.setValue(50)
        blend_value_layout.addWidget(self.blend_spin)

        blend_value_layout.addStretch()
        reg_layout.addWidget(self.blend_value_widget)

        layout.addWidget(reg_group)

        # Run button
        self.run_button = QPushButton("▶  Run Stitching")
        self.run_button.setObjectName("runButton")
        self.run_button.setMinimumHeight(48)
        self.run_button.setCursor(Qt.PointingHandCursor)
        self.run_button.clicked.connect(self.run_stitching)
        self.run_button.setEnabled(False)
        layout.addWidget(self.run_button)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(8)
        self.progress_bar.setMaximumHeight(8)
        layout.addWidget(self.progress_bar)

        # Log output
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(120)
        self.log_text.setMaximumHeight(150)
        self.log_text.setPlaceholderText("Log output will appear here...")
        layout.addWidget(self.log_text)

        # Open in Napari button
        self.napari_button = QPushButton("🔬  Open in Napari")
        self.napari_button.setObjectName("napariButton")
        self.napari_button.setMinimumHeight(48)
        self.napari_button.setCursor(Qt.PointingHandCursor)
        self.napari_button.clicked.connect(self.open_in_napari)
        self.napari_button.setEnabled(False)
        layout.addWidget(self.napari_button)

        layout.addStretch()

    def on_file_dropped(self, file_path):
        path = Path(file_path)
        if path.is_dir():
            self.log(f"Selected SQUID folder: {file_path}")
        else:
            self.log(f"Selected OME-TIFF: {file_path}")
        self.run_button.setEnabled(True)
        self.preview_button.setEnabled(True)

    def on_registration_toggled(self, checked):
        self.downsample_widget.setVisible(checked)

    def on_blend_toggled(self, checked):
        self.blend_value_widget.setVisible(checked)

    def log(self, message):
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def run_stitching(self):
        if not self.drop_area.file_path:
            return

        self.run_button.setEnabled(False)
        self.napari_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log_text.clear()

        if self.blend_checkbox.isChecked():
            blend_val = self.blend_spin.value()
            blend_pixels = (blend_val, blend_val)
            fusion_mode = "blended"
        else:
            blend_pixels = (0, 0)
            fusion_mode = "direct"

        self.worker = FusionWorker(
            self.drop_area.file_path,
            self.registration_checkbox.isChecked(),
            blend_pixels,
            self.downsample_spin.value(),
            fusion_mode,
        )
        self.worker.progress.connect(self.log)
        self.worker.finished.connect(self.on_fusion_finished)
        self.worker.error.connect(self.on_fusion_error)
        self.worker.start()

    def on_fusion_finished(self, output_path, elapsed_time):
        self.output_path = output_path
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.napari_button.setEnabled(True)

        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        time_str = f"{minutes}m {seconds:.1f}s" if minutes > 0 else f"{seconds:.1f}s"

        self.log(f"\n✓ Fusion complete! Time: {time_str}\nOutput: {output_path}")

    def on_fusion_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.log(f"\n✗ {error_msg}")

    def run_preview(self):
        if not self.drop_area.file_path:
            return

        self.preview_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log_text.clear()
        self.log("Starting preview...")

        self.preview_worker = PreviewWorker(
            self.drop_area.file_path,
            self.preview_cols_spin.value(),
            self.preview_rows_spin.value(),
            self.downsample_spin.value(),
        )
        self.preview_worker.progress.connect(self.log)
        self.preview_worker.finished.connect(self.on_preview_finished)
        self.preview_worker.error.connect(self.on_preview_error)
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
                viewer.add_image(fused, name="Fused result", colormap="gray", visible=False)
            napari.run()
        except Exception as e:
            self.log(f"Error opening Napari: {e}")

    def on_preview_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.preview_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self.log(f"\n✗ {error_msg}")

    def open_in_napari(self):
        if not self.output_path:
            return

        self.log(f"Opening in Napari: {self.output_path}")

        try:
            import napari
            from ome_zarr.io import parse_url
            from ome_zarr.reader import Reader

            reader = Reader(parse_url(self.output_path))
            nodes = list(reader())

            if nodes:
                data = nodes[0].data
                viewer = napari.Viewer()

                # Check number of channels from shape (t, c, y, x)
                n_channels = data[0].shape[1] if len(data[0].shape) >= 4 else 1

                if n_channels > 1:
                    # Get channel names if available (from SQUID format)
                    channel_names = None
                    try:
                        from tilefusion import TileFusion

                        tf = TileFusion(self.drop_area.file_path)
                        if hasattr(tf, "_squid_channels"):
                            channel_names = [ch.replace("_", " ") for ch in tf._squid_channels]
                    except:
                        pass

                    # Add each channel as separate layer
                    channel_colors = ["blue", "green", "yellow", "red", "magenta", "cyan"]
                    for c in range(n_channels):
                        # Extract channel from each pyramid level
                        channel_data = [d[:, c : c + 1, :, :] for d in data]
                        name = (
                            channel_names[c]
                            if channel_names and c < len(channel_names)
                            else f"Channel {c}"
                        )
                        viewer.add_image(
                            channel_data,
                            multiscale=True,
                            name=name,
                            colormap=channel_colors[c % len(channel_colors)],
                            blending="additive",
                        )
                else:
                    viewer.add_image(
                        data,
                        multiscale=True,
                        name=Path(self.output_path).stem,
                        contrast_limits=[0, 65535],
                    )
                napari.run()
        except Exception as e:
            self.log(f"Error opening Napari: {e}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = StitcherGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
