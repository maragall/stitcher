from profiling.attribution import AllocRecord
from profiling.ranking import compute_ranking


def test_compute_ranking_integrates_and_ranks():
    # funcA: flat 100 MB from t=0..1000ms -> 100 MB * 1 s = 100 MB·s
    # funcB: flat 50 MB from t=0..1000ms -> 50 MB·s
    records = [
        AllocRecord(0.0, "m:funcA", 100.0),
        AllocRecord(1000.0, "m:funcA", 100.0),
        AllocRecord(0.0, "m:funcB", 50.0),
        AllocRecord(1000.0, "m:funcB", 50.0),
    ]
    ranking = compute_ranking(records)

    assert [r["function"] for r in ranking] == ["m:funcA", "m:funcB"]
    assert abs(ranking[0]["integrated_mb_s"] - 100.0) < 1e-6
    assert abs(ranking[1]["integrated_mb_s"] - 50.0) < 1e-6
    assert abs(ranking[0]["peak_mb"] - 100.0) < 1e-6
    assert abs(ranking[0]["pct_of_total"] - (100.0 / 150.0 * 100.0)) < 1e-6
