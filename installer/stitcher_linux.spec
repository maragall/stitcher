# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Cephla Stitcher — Linux AppImage build (onedir).

Run from the installer/ directory:
  cd installer && python -m PyInstaller stitcher_linux.spec --noconfirm
"""

import os
import subprocess
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all submodules so PyInstaller bundles entire packages
tilefusion_imports = collect_submodules("tilefusion")
napari_imports = collect_submodules("napari")
skimage_imports = collect_submodules("skimage")

# Collect data files needed at runtime
napari_datas = collect_data_files("napari")

# Collect XCB platform libraries not shipped with PyQt5 wheel
xcb_libs = []
for lib_name in [
    "libxcb-icccm", "libxcb-image", "libxcb-keysyms",
    "libxcb-randr", "libxcb-render-util", "libxcb-xinerama",
    "libxcb-xfixes", "libxcb-shape", "libxkbcommon-x11", "libxkbcommon",
]:
    result = subprocess.run(
        ["find", "/usr/lib", "-name", f"{lib_name}*.so*", "-type", "f"],
        capture_output=True, text=True,
    )
    for line in result.stdout.strip().split("\n"):
        if line and os.path.isfile(line):
            xcb_libs.append((line, "."))

a = Analysis(
    ["entry.py"],
    pathex=[os.path.join("..", "src"), os.path.abspath("..")],
    binaries=xcb_libs,
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
        "OpenGL.platform.glx",
        "OpenGL.platform.egl",
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
    [],
    exclude_binaries=True,
    name="stitcher-gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,  # strip=True corrupts scipy OpenBLAS .so
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[
        "libscipy_openblas*.so*",
        "libopenblas*.so*",
    ],
    name="stitcher-gui",
)
