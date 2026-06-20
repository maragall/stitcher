import csv
from profiling.conclude import _read_timeline, _metrics, build_parser


def _write_timeline(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ms", "rss_mb", "stage"])
        for t_ms, rss in rows:
            w.writerow([t_ms, rss, "Fuse"])


def test_read_timeline_and_metrics(tmp_path):
    p = tmp_path / "timeline.csv"
    _write_timeline(p, [(0, 1000.0), (1000, 4000.0), (2000, 1000.0)])
    series = _read_timeline(str(p))
    assert series[1] == (1.0, 4000.0)  # (t_s, rss_mb)

    m = _metrics([(0.0, 1000.0), (1.0, 4000.0)], [(0.0, 800.0), (1.0, 2000.0)])
    assert m["before_peak"] == 4000.0
    assert m["after_peak"] == 2000.0
    assert abs(m["peak_pct_improvement"] - 50.0) < 1e-9


def test_parser_requires_before_and_after():
    args = build_parser().parse_args(["--before", "b/timeline.csv", "--after", "a/timeline.csv"])
    assert args.before.endswith("timeline.csv")
    assert args.after.endswith("timeline.csv")
