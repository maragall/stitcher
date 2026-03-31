"""Post-freeze smoke tests for Cephla Stitcher."""
import sys


def run_smoke_tests() -> bool:
    tests = [
        ("numpy", lambda: __import__("numpy")),
        ("scipy", lambda: __import__("scipy.ndimage")),
        ("tifffile", lambda: __import__("tifffile")),
        ("numba", lambda: __import__("numba")),
        ("scikit-image", lambda: __import__("skimage")),
        ("PyQt5", lambda: __import__("PyQt5.QtWidgets")),
        ("PyQt5.QtSvg", lambda: __import__("PyQt5.QtSvg")),
        ("pandas", lambda: __import__("pandas")),
        ("psutil", lambda: __import__("psutil")),
        ("tensorstore", lambda: __import__("tensorstore")),
        ("tilefusion", lambda: __import__("tilefusion")),
    ]

    passed = 0
    failed = 0
    for name, test in tests:
        try:
            test()
            print(f"  OK   {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} smoke tests passed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_smoke_tests() else 1)
