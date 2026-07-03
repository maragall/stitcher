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

    def t_napari_vispy_canvas_resized():
        # Guards the reported crash: napari.Viewer() dying with "'CanvasBackendDesktop'
        # object has no attribute 'resized'". napari connects to vispy's
        # canvas.native.resized (a QOpenGLWidget signal); if the frozen binary's vispy
        # falls back to the legacy PyQt5.QtOpenGL.QGLWidget (as it did on Python 3.12 /
        # vispy 0.16.2) that signal is absent and Viewer construction dies. Assert vispy's
        # pyqt5 backend resolves the modern QOpenGLWidget -- GL-free (no rendering context;
        # constructing a Viewer hangs headlessly).
        import sys

        import vispy.app

        # --- diagnostics: WHY does the frozen binary pick the legacy widget? ---
        try:
            from PyQt5.QtCore import QT_VERSION_STR, PYQT_VERSION_STR

            print(f"    [diag] QT_VERSION_STR={QT_VERSION_STR!r} PYQT={PYQT_VERSION_STR!r}")
        except Exception as e:
            print(f"    [diag] PyQt5.QtCore version import FAILED: {e!r}")
        try:
            from packaging.version import Version

            print(f"    [diag] Version(QT)>=5.4.0 -> {Version(QT_VERSION_STR) >= Version('5.4.0')}")
        except Exception as e:
            print(f"    [diag] packaging Version check FAILED: {e!r}")
        try:
            from PyQt5.QtWidgets import QOpenGLWidget  # noqa: F401

            print("    [diag] PyQt5.QtWidgets.QOpenGLWidget import: OK")
        except Exception as e:
            print(f"    [diag] PyQt5.QtWidgets.QOpenGLWidget import FAILED: {e!r}")
        try:
            from vispy.util.config import config

            print(f"    [diag] vispy gl_backend={config.get('gl_backend')!r}")
        except Exception as e:
            print(f"    [diag] vispy config read FAILED: {e!r}")

        vispy.app.use_app("pyqt5")
        qtmod = sys.modules["vispy.app.backends._qt"]
        base = qtmod.QGLWidget
        print(
            f"    [diag] vispy QGLWidget={base!r} USE_EGL={getattr(qtmod, 'USE_EGL', '?')} "
            f"QT5_NEW_API={getattr(qtmod, 'QT5_NEW_API', '?')}"
        )
        assert base is not object and hasattr(base, "resized"), (
            f"vispy Qt canvas base {base!r} lacks the 'resized' signal "
            "(legacy QGLWidget / EGL fallback) -- napari.Viewer() would crash"
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
        ("vispy Qt canvas has resized (QOpenGLWidget)", t_napari_vispy_canvas_resized),
        ("import tilefusion + modules", t_tilefusion),
        ("threadpoolctl BLAS limiter", t_threadpoolctl),
    ]

    for name, fn in tests:
        results.append(_test(name, fn))

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} smoke tests passed.")
    sys.exit(0 if all(results) else 1)
