from profiling.perpair import PairRecord
from profiling.swimlanes import schedule_lanes
from profiling.plots import plot_swimlanes


def _rec(pid, dur):
    return PairRecord(pid, pid, pid + 1, 50, 50, 100, dur)


def test_schedule_lanes_packs_into_n_lanes_greedily():
    # 4 pairs, 2 lanes: durations 10, 20, 5, 5
    records = [_rec(0, 10.0), _rec(1, 20.0), _rec(2, 5.0), _rec(3, 5.0)]
    placed = schedule_lanes(records, n_lanes=2)

    assert len(placed) == 4
    starts = {p["pair_id"]: p["start_ms"] for p in placed}
    assert starts[0] == 0.0
    assert starts[1] == 0.0
    # pair 2 goes to the lane that frees first (lane of pair 0, free at 10)
    assert starts[2] == 10.0
    # pair 3 goes to next-free lane (pair 2 ends at 15 vs lane1 free at 20) -> 15
    assert starts[3] == 15.0
    assert all(0 <= p["lane"] < 2 for p in placed)


def test_plot_swimlanes_writes_file(tmp_path):
    records = [_rec(i, 5.0 + i) for i in range(6)]
    out = tmp_path / "swimlanes.png"
    plot_swimlanes(records, str(out), n_lanes=3)
    assert out.exists() and out.stat().st_size > 0
