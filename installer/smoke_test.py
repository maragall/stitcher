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

    tests = [
        ("import numpy", t_numpy),
        ("scipy.ndimage zoom", t_scipy),
        ("tifffile read/write", t_tifffile),
        ("import tensorstore", t_tensorstore),
        ("numba jit", t_numba),
        ("skimage registration", t_skimage),
        ("import pandas", t_pandas),
        ("PyQt5 QApplication", t_pyqt5),
        ("napari viewer", t_napari),
    ]

    for name, fn in tests:
        results.append(_test(name, fn))

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} smoke tests passed.")
    sys.exit(0 if all(results) else 1)
