import csv
from profiling.sampler import Sample
from profiling.record import write_timeline_csv, write_functions_csv


def test_write_timeline_csv(tmp_path):
    samples = [Sample(0.0, 100.0), Sample(50.0, 120.0)]
    spans = [("Register", 0.0, 100.0)]
    path = tmp_path / "timeline.csv"
    write_timeline_csv(str(path), samples, spans)

    rows = list(csv.DictReader(path.open()))
    assert rows[0]["stage"] == "Register"
    assert float(rows[1]["rss_mb"]) == 120.0


def test_write_functions_csv(tmp_path):
    ranking = [
        {"function": "m:funcA", "peak_mb": 100.0, "integrated_mb_s": 100.0, "pct_of_total": 66.67}
    ]
    path = tmp_path / "functions.csv"
    write_functions_csv(str(path), ranking)

    rows = list(csv.DictReader(path.open()))
    assert rows[0]["function"] == "m:funcA"
    assert float(rows[0]["pct_of_total"]) == 66.67
