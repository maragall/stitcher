"""Frozen entry point for PyInstaller-built TileFusion Stitcher."""
import os
import sys
import traceback
import threading

if getattr(sys, "frozen", False):
    _meipass = sys._MEIPASS
    if sys.platform == "win32":
        os.environ["QT_PLUGIN_PATH"] = os.path.join(
            _meipass, "PyQt5", "Qt5", "plugins"
        )
    else:
        qt_plugins = os.path.join(_meipass, "PyQt5", "Qt5", "plugins")
        os.environ["QT_PLUGIN_PATH"] = qt_plugins
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
            qt_plugins, "platforms"
        )
        qt_lib = os.path.join(_meipass, "PyQt5", "Qt5", "lib")
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{_meipass}:{qt_lib}:{existing_ld}"
    # Tell qtpy (napari's Qt abstraction) to use PyQt5
    os.environ["QT_API"] = "pyqt5"
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
