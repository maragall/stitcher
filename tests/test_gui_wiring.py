"""Regression tests: every GUI control must actually reach the pipeline.

These pin the four end-to-end breaks found in the standalone audit:

1. The "Correct lens distortion" checkbox was built, checked and tooltipped, but
   ``.isChecked()`` was never read — unchecking it did nothing and per-seam
   elastic correction always ran.
2. The outlier rel/abs thresholds were wired to the worker but their container
   was never shown: ``registration_checkbox`` was ``setChecked(True)`` before the
   visibility slot was connected, so no ``toggled`` signal ever fired and the two
   spin boxes stayed invisible unless the user unchecked and rechecked
   registration.
3. ``DropArea.setFiles`` touched a non-existent ``self.icon_label``. The
   AttributeError fired inside ``dropEvent``, which Qt swallows, so multi-drop
   silently did nothing and batch mode was unreachable.
4. ``BatchFusionWorker`` accepted none of registration z/t/channel, the outlier
   thresholds or the distortion switch, so batch quietly ran on defaults — and on
   a multi-channel dataset failed outright, since the pipeline refuses to guess a
   registration channel.

The workers are stubbed: these tests assert on what the GUI *hands to* the
pipeline, never on pipeline output, so they are fast and need no data on disk.
"""

import inspect
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

import gui.app as app_mod  # noqa: E402

# Deliberately NOT using pytest-qt's qtbot: that plugin instantiates its own
# QApplication under whichever binding qtpy resolves to (PySide6 in this env),
# and two Qt bindings live in one process is an immediate segfault. The GUI is
# PyQt5-only, so we own the PyQt5 QApplication here and never let qtbot near it.


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _Sig:
    """Stand-in for a pyqtSignal on a stubbed worker."""

    def connect(self, *a, **kw):
        pass

    def disconnect(self, *a, **kw):
        pass

    def emit(self, *a, **kw):
        pass


def _make_stub(sink, name):
    class _Stub:
        def __init__(self, *args, **kwargs):
            sink[name] = {"args": args, "kwargs": kwargs}
            for sig in (
                "progress",
                "finished",
                "error",
                "item_started",
                "item_finished",
            ):
                setattr(self, sig, _Sig())

        def start(self):
            pass

        def isRunning(self):
            return False

    return _Stub


@pytest.fixture
def gui(qapp, monkeypatch):
    """A live StitcherGUI with all four workers stubbed out."""
    sink = {}
    for name in ("FusionWorker", "BatchFusionWorker", "PreviewWorker", "FlatfieldWorker"):
        monkeypatch.setattr(app_mod, name, _make_stub(sink, name))

    w = app_mod.StitcherGUI()
    w.sink = sink
    yield w
    w.close()
    w.deleteLater()


# ── 1. the distortion checkbox reaches the worker ────────────────────────────


@pytest.mark.parametrize("checked", [True, False])
def test_distortion_checkbox_reaches_fusion_worker(gui, checked):
    gui.drop_area.file_paths = ["/nonexistent/dataset"]
    gui.dataset_n_channels = 1  # skip the explicit-channel gate
    gui.distortion_checkbox.setChecked(checked)

    gui.run_stitching()

    kwargs = gui.sink["FusionWorker"]["kwargs"]
    assert "enable_distortion" in kwargs, "distortion checkbox never reaches the pipeline"
    assert kwargs["enable_distortion"] is checked


@pytest.mark.parametrize("checked", [True, False])
def test_distortion_checkbox_reaches_batch_worker(gui, checked):
    gui.drop_area.file_paths = ["/a", "/b"]
    gui.batch_paths = ["/a", "/b"]
    gui.dataset_n_channels = 1
    gui.distortion_checkbox.setChecked(checked)

    gui.run_stitching()

    assert gui.sink["BatchFusionWorker"]["kwargs"]["enable_distortion"] is checked


def test_fusion_pipeline_honours_enable_distortion_argument():
    """The batch/shared pipeline must accept the switch, not hardcode it."""
    params = inspect.signature(app_mod._run_fusion_pipeline).parameters
    assert "enable_distortion" in params
    assert "outlier_rel_thresh" in params
    assert "outlier_abs_thresh" in params


# ── 2. the outlier controls are reachable ────────────────────────────────────


def test_outlier_controls_visible_at_startup_with_registration_on(gui):
    gui.show()
    assert gui.registration_checkbox.isChecked()
    assert gui.outlier_widget.isVisible(), (
        "outlier thresholds are wired but invisible: the user cannot reach them "
        "without toggling registration off and on again"
    )


def test_outlier_controls_follow_registration_checkbox(gui):
    gui.show()
    gui.registration_checkbox.setChecked(False)
    assert not gui.outlier_widget.isVisible()
    gui.registration_checkbox.setChecked(True)
    assert gui.outlier_widget.isVisible()


def test_outlier_values_reach_the_worker(gui):
    gui.drop_area.file_paths = ["/nonexistent/dataset"]
    gui.dataset_n_channels = 1
    gui.outlier_rel_spin.setValue(77)
    gui.outlier_abs_spin.setValue(9)

    gui.run_stitching()

    kwargs = gui.sink["FusionWorker"]["kwargs"]
    assert kwargs["outlier_rel_thresh"] == pytest.approx(0.77)
    assert kwargs["outlier_abs_thresh"] == pytest.approx(9.0)


# ── 3. multi-drop does not crash ─────────────────────────────────────────────


def test_drop_area_set_files_does_not_raise(qapp):
    area = app_mod.DropArea()
    area.setFiles(["/one", "/two"], ["skipped.txt"])  # used to raise AttributeError
    assert area.file_paths == ["/one", "/two"]
    assert "2 items selected" in area.label.text()


def test_drop_area_has_no_phantom_icon_label(qapp):
    """setFiles must only touch widgets that exist on the frame."""
    area = app_mod.DropArea()
    assert not hasattr(area, "icon_label")


# ── 4. batch carries every setting ───────────────────────────────────────────


def test_batch_worker_accepts_all_single_run_settings():
    single = set(inspect.signature(app_mod.FusionWorker.__init__).parameters)
    batch = set(inspect.signature(app_mod.BatchFusionWorker.__init__).parameters)
    # Everything the single-run worker takes, minus the two names that are
    # genuinely single-item concepts.
    expected = single - {"self", "tiff_path"}
    missing = expected - batch
    assert not missing, f"batch mode silently drops these settings: {sorted(missing)}"


def test_batch_run_forwards_registration_channel(gui):
    gui.drop_area.file_paths = ["/a", "/b"]
    gui.batch_paths = ["/a", "/b"]
    gui.dataset_n_channels = 3
    gui.dataset_channel_names = ["c0", "c1", "c2"]
    gui._update_reg_zt_controls()
    gui.reg_channel_combo.setCurrentIndex(2)  # -> channel 1

    gui.run_stitching()

    kwargs = gui.sink["BatchFusionWorker"]["kwargs"]
    assert kwargs["registration_channel"] == 1, (
        "batch must forward the chosen registration channel; the pipeline raises "
        "rather than guessing one"
    )


def test_batch_run_is_gated_on_an_explicit_channel_choice(gui):
    """With no channel chosen, a multi-channel batch must not start at all."""
    gui.drop_area.file_paths = ["/a", "/b"]
    gui.batch_paths = ["/a", "/b"]
    gui.dataset_n_channels = 3
    gui.dataset_channel_names = ["c0", "c1", "c2"]
    gui._update_reg_zt_controls()  # leaves the "— Select channel —" prompt selected

    gui.run_stitching()

    assert "BatchFusionWorker" not in gui.sink
