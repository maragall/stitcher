# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Cephla Stitcher — Windows single-exe build."""

import os

block_cipher = None

a = Analysis(
    ["entry.py"],
    pathex=[os.path.join("..", "src"), os.path.join("..")],
    binaries=[],
    datas=[
        (os.path.join("..", "gui", "cephla_logo.svg"), "gui"),
    ],
    hiddenimports=[
        "tilefusion",
        "tilefusion.core",
        "tilefusion.fusion",
        "tilefusion.registration",
        "tilefusion.optimization",
        "tilefusion.flatfield",
        "tilefusion.utils",
        "tilefusion.io",
        "tilefusion.io.ome_tiff",
        "tilefusion.io.ome_tiff_tiles",
        "tilefusion.io.individual_tiffs",
        "tilefusion.io.zarr",
        "gui",
        "gui.app",
        "installer",
        "installer.smoke_test",
        "scripts",
        "scripts.view_in_napari",
        "scripts.convert_to_zarr",
        "PyQt5",
        "PyQt5.QtWidgets",
        "PyQt5.QtCore",
        "PyQt5.QtGui",
        "PyQt5.QtSvg",
        "numpy",
        "scipy",
        "scipy.ndimage",
        "scipy.optimize",
        "scipy.signal",
        "skimage",
        "skimage.registration",
        "skimage.metrics",
        "numba",
        "pandas",
        "tifffile",
        "tensorstore",
        "psutil",
        "tqdm",
        "PIL",
    ],
    excludes=["tkinter", "matplotlib", "IPython", "pytest"],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="stitcher-gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=None,
)
