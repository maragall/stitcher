# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for TileFusion Stitcher (Windows).

IMPORTANT: Run from the installer/ directory:
  cd installer && python -m PyInstaller stitcher_windows.spec --noconfirm
"""

import os
import glob as _glob
from PyInstaller.utils.hooks import collect_submodules, copy_metadata

block_cipher = None

tilefusion_imports = collect_submodules("tilefusion")
skimage_imports = collect_submodules("skimage")

# sklearn DLLs: sklearn bundles msvcp140.dll in .libs/
sklearn_binaries = []
try:
    import sklearn

    _sklearn_libs = os.path.join(os.path.dirname(sklearn.__file__), ".libs")
    if os.path.isdir(_sklearn_libs):
        for dll in _glob.glob(os.path.join(_sklearn_libs, "*.dll")):
            sklearn_binaries.append((dll, "."))
except ImportError:
    pass

napari_metadata = copy_metadata("imageio") + copy_metadata("napari")
# scikit-learn was only ever present transitively via the old BaSiCPy stack (now
# removed); include its metadata only if it actually happens to be installed.
for _opt in ("scikit-learn", "napari-svg"):
    try:
        napari_metadata += copy_metadata(_opt)
    except Exception:
        pass

a = Analysis(
    ["entry.py"],
    pathex=[os.path.abspath("..")],
    binaries=sklearn_binaries,
    datas=napari_metadata
    + (
        [
            (os.path.join("..", "gui", "cephla_logo.svg"), "gui"),
        ]
        if os.path.exists(os.path.join("..", "gui", "cephla_logo.svg"))
        else []
    ),
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
    strip=False,
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
    name="stitcher-gui",
)
