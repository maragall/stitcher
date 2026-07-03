"""Post-freeze smoke tests for the bundled TileFusion Stitcher application."""

import os
import sys
import tempfile


def _test(name, fn):
    """Run a single test, print PASS/FAIL, return success bool."""
    try:
        fn()
        print(f"PASS: {name}")
        return True
    except Exception as e:
        print(f"FAIL: {name} -- {e}")
        return False


def run():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    results = []

    def t_numpy():
        import numpy as np

        arr = np.arange(12).reshape(3, 4)
        assert arr.sum() == 66

    def t_scipy():
        from scipy.ndimage import zoom
        import numpy as np

        result = zoom(np.ones((10, 10)), 0.5)
        assert result.shape == (5, 5)

    def t_tifffile():
        import numpy as np
        import tifffile

        arr = np.zeros((10, 10), dtype=np.uint16)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = f.name
        try:
            tifffile.imwrite(path, arr)
            data = tifffile.imread(path)
            assert data.shape == (10, 10)
        finally:
            os.unlink(path)

    def t_tensorstore():
        import tensorstore  # noqa: F401

    def t_numba():
        from numba import njit

        @njit
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def t_numba_parallel():
        # Fusion uses @njit(parallel=True)+prange; its threading layer (TBB/OpenMP) is
        # the most platform-fragile piece, so exercise it explicitly -- not just @njit.
        import numpy as np
        from numba import njit, prange

        @njit(parallel=True)
        def psum(a):
            t = 0.0
            for i in prange(a.shape[0]):
                t += a[i]
            return t

        assert psum(np.ones(256, dtype=np.float64)) == 256.0

    def t_tensorstore_zarr3():
        # The actual output driver -- write + read a tiny zarr3 array, not just import.
        import numpy as np
        import tensorstore as ts

        d = tempfile.mkdtemp()
        arr = ts.open(
            {
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": d},
                "metadata": {
                    "shape": [8, 8],
                    "data_type": "uint16",
                    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [8, 8]}},
                },
                "create": True,
            }
        ).result()
        arr.write(np.ones((8, 8), dtype=np.uint16)).result()
        assert int(np.asarray(arr.read().result()).sum()) == 64

    def t_tilefusion():
        # The shipped package itself + its key modules must import in the frozen app.
        import tilefusion  # noqa: F401
        from tilefusion import core, flatfield, fusion, registration  # noqa: F401

    def t_threadpoolctl():
        # BLAS-pinning dep added for the CPU audit; must be bundled for pinning to work.
        from tilefusion.utils import limit_blas_threads

        with limit_blas_threads(1):
            pass

    def t_skimage():
        from skimage.registration import phase_cross_correlation  # noqa: F401

    def t_pandas():
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3]})
        assert len(df) == 3

    def t_pyqt5():
        from PyQt5.QtWidgets import QApplication  # noqa: F401

    def t_napari():
        import napari  # noqa: F401
        from napari.viewer import Viewer  # noqa: F401

    def t_napari_frozen_safe_version():
        # Guards the reported crash: napari.Viewer() dying with "'CanvasBackendDesktop'
        # object has no attribute 'resized'". The binaries FORCE vispy's legacy QGLWidget
        # on Linux/Windows (VISPY_USE_LEGACY_QGLWIDGET=1 for the Blackwell-GPU workaround),
        # and that widget has no 'resized' signal. napari 0.7.x connects to
        # canvas.native.resized and crashes; napari 0.6.x uses its own
        # _welcome_widget.resized and is fine. So the shipped napari MUST be <0.7. Assert
        # it -- GL-free (napari metadata is bundled; constructing a Viewer hangs
        # headlessly). Belt-and-suspenders with the pyproject <0.7 cap.
        from importlib.metadata import version

        from packaging.version import Version

        v = version("napari")
        assert Version(v) < Version("0.7"), (
            f"napari {v} >= 0.7 uses vispy canvas.native.resized, which the forced legacy "
            "QGLWidget lacks -> napari.Viewer() crashes in the frozen binary; pin napari <0.7"
        )

    tests = [
        ("import numpy", t_numpy),
        ("scipy.ndimage zoom", t_scipy),
        ("tifffile read/write", t_tifffile),
        ("import tensorstore", t_tensorstore),
        ("tensorstore zarr3 write/read", t_tensorstore_zarr3),
        ("numba jit", t_numba),
        ("numba parallel (prange)", t_numba_parallel),
        ("skimage registration", t_skimage),
        ("import pandas", t_pandas),
        ("PyQt5 QApplication", t_pyqt5),
        ("napari viewer import", t_napari),
        ("napari <0.7 (works with forced legacy QGLWidget)", t_napari_frozen_safe_version),
        ("import tilefusion + modules", t_tilefusion),
        ("threadpoolctl BLAS limiter", t_threadpoolctl),
    ]

    for name, fn in tests:
        results.append(_test(name, fn))

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} smoke tests passed.")
    sys.exit(0 if all(results) else 1)
