from profiling.sampler import Sample
from profiling.attribution import AllocRecord
from profiling.ranking import compute_ranking
from profiling.plots import plot_timeline, plot_function_lines, plot_pareto


def _records():
    return [
        AllocRecord(0.0, "m:funcA", 100.0),
        AllocRecord(1000.0, "m:funcA", 100.0),
        AllocRecord(0.0, "m:funcB", 50.0),
        AllocRecord(1000.0, "m:funcB", 50.0),
    ]


def test_plots_write_nonempty_files(tmp_path):
    samples = [Sample(0.0, 100.0), Sample(500.0, 180.0), Sample(1000.0, 120.0)]
    spans = [("Register", 0.0, 500.0), ("Fuse", 500.0, 1000.0)]
    records = _records()
    ranking = compute_ranking(records)

    p1 = tmp_path / "timeline.png"
    p2 = tmp_path / "functions.png"
    p3 = tmp_path / "pareto.png"
    plot_timeline(samples, spans, str(p1))
    plot_function_lines(samples, records, ranking, str(p2), top_k=2)
    plot_pareto(ranking, str(p3))

    for p in (p1, p2, p3):
        assert p.exists() and p.stat().st_size > 0
