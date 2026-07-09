# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for TileFusion Stitcher (macOS .app bundle).

IMPORTANT: Run from the installer/ directory:
  cd installer && python -m PyInstaller stitcher_mac.spec --noconfirm
"""

import os
from PyInstaller.utils.hooks import collect_submodules, copy_metadata

block_cipher = None

tilefusion_imports = collect_submodules("tilefusion")
skimage_imports = collect_submodules("skimage")

# napari chain requires package metadata for version checks
napari_metadata = copy_metadata("imageio") + copy_metadata("napari")
# scikit-learn was only ever present transitively via the old BaSiCPy stack (now
# removed); include its metadata only if it actually happens to be installed.
for _opt in ("scikit-learn", "napari-svg", "napari-ome-zarr"):
    try:
        napari_metadata += copy_metadata(_opt)
    except Exception:
        pass

a = Analysis(
    ["entry.py"],
    pathex=[os.path.abspath("..")],
    binaries=[],
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
        # macOS OpenGL platform (the hooks/ folder also covers this)
        "OpenGL",
        "OpenGL.GL",
        "OpenGL.platform.darwin",
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
    upx=False,
    console=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="stitcher-gui",
)

app = BUNDLE(
    coll,
    name="CephlaStitcher.app",
    icon=None,
    bundle_identifier="com.cephla.stitcher-gui",
    info_plist={
        "CFBundleName": "Cephla Stitcher",
        "CFBundleDisplayName": "Cephla Stitcher",
        "CFBundleExecutable": "stitcher-gui",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "0.1.0",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "11.0",
        "NSPrincipalClass": "NSApplication",
        "NSRequiresAquaSystemAppearance": False,
    },
)
