# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Cephla Stitcher — Windows single-exe build.

Run from the installer/ directory:
  cd installer && python -m PyInstaller stitcher_windows.spec --noconfirm
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all submodules so PyInstaller bundles entire packages
tilefusion_imports = collect_submodules("tilefusion")
napari_imports = collect_submodules("napari")
skimage_imports = collect_submodules("skimage")

# Collect data files needed at runtime
napari_datas = collect_data_files("napari")

a = Analysis(
    ["entry.py"],
    pathex=[os.path.join("..", "src"), os.path.abspath("..")],
    binaries=[],
    datas=napari_datas + [
        (os.path.join("..", "gui", "cephla_logo.svg"), "gui"),
    ],
    hiddenimports=tilefusion_imports + napari_imports + skimage_imports + [
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
        "numpy.core._methods",
        "numpy.lib.format",
        "scipy",
        "scipy.ndimage",
        "scipy.optimize",
        "scipy.signal",
        "numba",
        "numba.core",
        "pandas",
        "tifffile",
        "tensorstore",
        "psutil",
        "tqdm",
        "PIL",
        "ome_zarr",
        "zarr",
        "zarr.storage",
        "dask",
        "dask.array",
        "vispy",
        "vispy.app",
        "vispy.app.backends._pyqt5",
        "OpenGL",
        "OpenGL.GL",
        "OpenGL.platform.win32",
        "xml.etree.ElementTree",
        "importlib.metadata",
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
