from profiling.perpair import PairRecord
from profiling.plots import plot_pair_variability, plot_scan_pattern


def test_plot_pair_variability_writes_file(tmp_path):
    records = [PairRecord(i, i, i + 1, 50, 50, 100 + i, 5.0 + i) for i in range(8)]
    out = tmp_path / "variability.png"
    plot_pair_variability(records, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_scan_pattern_writes_file(tmp_path):
    # 2x3 grid in raster index order
    grid = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1), 5: (1, 2)}
    out = tmp_path / "scan.png"
    plot_scan_pattern(grid, str(out), pattern="raster")
    assert out.exists() and out.stat().st_size > 0
