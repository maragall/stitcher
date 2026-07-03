"""Frozen entry point for PyInstaller-built TileFusion Stitcher."""

# MUST be the first import + call: in a frozen app, child processes spawned
# by multiprocessing relaunch the bundle binary. freeze_support() short-
# circuits the relaunch in the child so it does NOT re-run main() and pop
# another GUI window. No-op in the parent.
import multiprocessing

multiprocessing.freeze_support()

import os

# Pin BLAS thread count to 1 BEFORE numpy is imported. macOS gives
# secondary threads only 512 KB of stack by default, but OpenBLAS's
# parallel LU path (dgetrf_parallel, used by np.linalg.solve / inv)
# allocates large work arrays on the stack and blows past it, killing
# the process with SIGBUS / KERN_PROTECTION_FAILURE the moment a
# tensorstore worker (or any non-main QThread) calls into linalg.
# Forcing single-threaded BLAS keeps the math on the calling thread,
# which has plenty of stack. Linux/Windows defaults are large enough
# that this is a no-op there, so we set it unconditionally and call
# it a day. setdefault() respects user overrides.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys

# Force VisPy's legacy QGLWidget on Linux/Windows so the bundled (and
# patched) vispy avoids the QOpenGLWidget FBO corruption on NVIDIA
# Blackwell GPUs. SKIP on macOS — Apple deprecated the legacy QGLWidget
# years ago and the modern macOS Qt build renders it to a context that
# gives a blank white canvas. Macs don't ship Blackwell GPUs anyway.
# Must be set BEFORE vispy imports.
#
# COUPLING: the legacy QGLWidget has NO 'resized' signal. napari 0.7.x connects to
# canvas.native.resized and crashes here ("'CanvasBackendDesktop' has no attribute
# 'resized'"); napari 0.6.x uses its own _welcome_widget.resized and is fine. So while
# this workaround is on, napari MUST stay <0.7 (capped in pyproject; asserted in
# installer/smoke_test.py). Don't bump napari past 0.7 without also removing this.
if sys.platform != "darwin":
    os.environ.setdefault("VISPY_USE_LEGACY_QGLWIDGET", "1")

import traceback
import threading

if getattr(sys, "frozen", False):
    _meipass = sys._MEIPASS
    if sys.platform == "win32":
        os.environ["QT_PLUGIN_PATH"] = os.path.join(_meipass, "PyQt5", "Qt5", "plugins")
    elif sys.platform == "darwin":
        # macOS: PyInstaller's @loader_path handles dylib resolution. Don't
        # touch DYLD_LIBRARY_PATH (SIP strips it for child procs and it can
        # break framework loading). Just point Qt at its plugin tree.
        qt_plugins = os.path.join(_meipass, "PyQt5", "Qt5", "plugins")
        os.environ["QT_PLUGIN_PATH"] = qt_plugins
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(qt_plugins, "platforms")
    else:
        qt_plugins = os.path.join(_meipass, "PyQt5", "Qt5", "plugins")
        os.environ["QT_PLUGIN_PATH"] = qt_plugins
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(qt_plugins, "platforms")
        qt_lib = os.path.join(_meipass, "PyQt5", "Qt5", "lib")
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{_meipass}:{qt_lib}:{existing_ld}"
    # Tell qtpy (napari's Qt abstraction) to use PyQt5
    os.environ["QT_API"] = "pyqt5"
    if sys.platform == "darwin":
        # Inside a .app bundle, sys.executable lives at Contents/MacOS/ which
        # is read-only after Gatekeeper translocation. Log to ~/Library.
        _log_dir = os.path.expanduser("~/Library/Logs/CephlaStitcher")
        os.makedirs(_log_dir, exist_ok=True)
        _log_path = os.path.join(_log_dir, "crash.log")
    else:
        _log_path = os.path.join(os.path.dirname(sys.executable), "crash.log")

    def _write_crash(tb_text):
        with open(_log_path, "a") as f:
            f.write(tb_text + "\n")

    _orig_excepthook = threading.excepthook

    def _thread_excepthook(args):
        tb_text = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_tb))
        _write_crash(f"Thread crash ({args.thread}):\n{tb_text}")
        if _orig_excepthook:
            _orig_excepthook(args)

    threading.excepthook = _thread_excepthook

    _orig_sys_excepthook = sys.excepthook

    def _sys_excepthook(exc_type, exc_value, exc_tb):
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        _write_crash(f"Main thread crash:\n{tb_text}")
        _orig_sys_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _sys_excepthook

if "--smoke-test" in sys.argv:
    from installer.smoke_test import run

    run()
else:
    try:
        from gui.app import main

        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        if getattr(sys, "frozen", False):
            _write_crash(f"Entry point crash:\n{tb}")
        raise
