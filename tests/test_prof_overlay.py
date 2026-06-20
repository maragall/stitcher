from profiling.plots import plot_timeline_overlay


def test_plot_timeline_overlay_writes_file(tmp_path):
    before = [(0.0, 1000.0), (1.0, 3971.0), (2.0, 1200.0)]
    after = [(0.0, 800.0), (1.0, 2200.0), (2.0, 900.0)]
    out = tmp_path / "overlay.png"
    plot_timeline_overlay(before, after, str(out), before_peak=3971.0, after_peak=2200.0)
    assert out.exists() and out.stat().st_size > 0
