import time
from profiling.sampler import Sample
from profiling.stages import StageTimer, assign_stages


def test_stage_timer_records_spans_in_order():
    t0 = time.perf_counter()
    timer = StageTimer(t0)
    with timer.stage("Register"):
        time.sleep(0.02)
    with timer.stage("Fuse"):
        time.sleep(0.02)

    names = [s[0] for s in timer.spans]
    assert names == ["Register", "Fuse"]
    for _name, start, end in timer.spans:
        assert end > start


def test_assign_stages_labels_samples_by_time():
    spans = [("Register", 0.0, 100.0), ("Fuse", 100.0, 200.0)]
    samples = [Sample(50.0, 10.0), Sample(150.0, 20.0), Sample(500.0, 5.0)]
    rows = assign_stages(samples, spans)
    assert rows == [
        (50.0, 10.0, "Register"),
        (150.0, 20.0, "Fuse"),
        (500.0, 5.0, "(other)"),
    ]
