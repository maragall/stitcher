# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Cephla Stitcher — Linux AppImage build (onedir)."""

import os
import glob

block_cipher = None

# Collect XCB platform libraries not shipped with PyQt5 wheel
xcb_libs = []
xcb_names = [
    "libxcb-icccm*",
    "libxcb-image*",
    "libxcb-keysyms*",
    "libxcb-randr*",
    "libxcb-render-util*",
    "libxcb-xinerama*",
    "libxcb-xfixes*",
    "libxcb-shape*",
    "libxkbcommon-x11*",
    "libxkbcommon.so*",
]
for pattern in xcb_names:
    for lib_dir in ["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib"]:
        for path in glob.glob(os.path.join(lib_dir, pattern)):
            if os.path.isfile(path):
                xcb_libs.append((path, "."))

a = Analysis(
    ["entry.py"],
    pathex=[os.path.join("..", "src"), os.path.join("..")],
    binaries=xcb_libs,
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
    [],
    exclude_binaries=True,
    name="stitcher-gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,  # strip=True corrupts scipy OpenBLAS .so
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=["libssl*", "libcrypto*"],
    name="stitcher-gui",
)
