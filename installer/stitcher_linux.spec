# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for TileFusion Stitcher (Linux / AppImage).

IMPORTANT: Run from the installer/ directory:
  cd installer && python -m PyInstaller stitcher_linux.spec --noconfirm
"""

import os
import subprocess
from PyInstaller.utils.hooks import collect_submodules, copy_metadata

block_cipher = None

# Collect ALL submodules for packages without PyInstaller hooks
tilefusion_imports = collect_submodules("tilefusion")
skimage_imports = collect_submodules("skimage")

# napari chain requires package metadata for version checks
napari_metadata = copy_metadata("imageio") + copy_metadata("napari")
# scikit-learn was only ever present transitively via the old BaSiCPy stack (now
# removed); include its metadata only if it actually happens to be installed.
for _opt in ("scikit-learn", "napari-svg"):
    try:
        napari_metadata += copy_metadata(_opt)
    except Exception:
        pass

# Bundle xcb platform plugin dependencies from the system
xcb_libs = []
for lib_name in [
    "libxcb-icccm",
    "libxcb-image",
    "libxcb-keysyms",
    "libxcb-randr",
    "libxcb-render-util",
    "libxcb-xinerama",
    "libxcb-xfixes",
    "libxcb-shape",
    "libxkbcommon-x11",
    "libxkbcommon",
]:
    result = subprocess.run(
        ["find", "/usr/lib", "-name", f"{lib_name}*.so*", "-type", "f"],
        capture_output=True,
        text=True,
    )
    for line in result.stdout.strip().split("\n"):
        if line and os.path.isfile(line):
            xcb_libs.append((line, "."))

a = Analysis(
    ["entry.py"],
    pathex=[os.path.abspath("..")],
    binaries=xcb_libs,
    datas=napari_metadata
    + [
        (os.path.join("..", "gui", "cephla_logo.svg"), "gui"),
    ],
    hiddenimports=tilefusion_imports
    + skimage_imports
    + [
        "numpy",
        "numpy.core._methods",
        "numpy.lib.format",
        # numpy<2 pickle compat: bundle the numpy.core stub so old flatfield .npy
        # (numpy.core.multiarray) files unpickle in the frozen app.
        "numpy.core",
        "numpy.core.multiarray",
        "numpy.core._multiarray_umath",
        "scipy",
        "scipy.ndimage",
        "scipy.optimize",
        "scipy.sparse",
        "tifffile",
        "tensorstore",
        "ml_dtypes",
        "numba",
        "numba.core",
        "threadpoolctl",
        "pandas",
        "tqdm",
        "psutil",
        "qtpy",
        "qtpy.QtCore",
        "qtpy.QtGui",
        "qtpy.QtWidgets",
        "PyQt5",
        "PyQt5.QtCore",
        "PyQt5.QtGui",
        "PyQt5.QtWidgets",
        "PyQt5.QtSvg",
        "gui",
        "gui.app",
        "installer",
        "installer.smoke_test",
        "napari_ome_zarr",
        "xml.etree.ElementTree",
        "json",
        "gc",
        "shutil",
        "importlib.metadata",
        "matplotlib",
        "matplotlib.pyplot",
        "matplotlib.backends.backend_agg",
        "OpenGL",
        "OpenGL.GL",
        "OpenGL.platform.glx",
        "OpenGL.platform.egl",
    ],
    hookspath=["hooks"],
    excludes=["tkinter", "IPython", "pytest", "cupy", "cupyx", "cucim"],
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
    strip=True,
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
    upx_exclude=["libscipy_openblas*.so*", "libopenblas*.so*"],
    name="stitcher-gui",
)
