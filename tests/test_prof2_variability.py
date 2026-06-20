from profiling.perpair import PairRecord
from profiling.variability import compute_pair_stats


def _rec(pid, dur, total):
    return PairRecord(pid, pid, pid + 1, total // 2, total - total // 2, total, dur)


def test_compute_pair_stats_mean_std_cv():
    records = [_rec(0, 10.0, 100), _rec(1, 20.0, 100), _rec(2, 30.0, 100)]
    stats = compute_pair_stats(records)

    assert stats["n_pairs"] == 3
    assert abs(stats["duration_ms"]["mean"] - 20.0) < 1e-9
    # population std of [10,20,30] = sqrt(200/3) ~= 8.16497
    assert abs(stats["duration_ms"]["std"] - 8.16496580927726) < 1e-6
    assert abs(stats["duration_ms"]["cv"] - (8.16496580927726 / 20.0)) < 1e-9
    # patch_bytes_total all equal -> cv 0
    assert abs(stats["patch_bytes_total"]["cv"] - 0.0) < 1e-12


def test_compute_pair_stats_empty():
    stats = compute_pair_stats([])
    assert stats["n_pairs"] == 0
