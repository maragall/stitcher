import os
import time
import tracemalloc
import pytest
from profiling.harness import _wrap_stage, _collect, profile_dataset, ProfileResult
from profiling.stages import StageTimer

DATASET = os.path.expanduser(
    "~/Cephla/Data/test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy"
)


class _Fake:
    def step(self, x):
        return x * 2


def test_wrap_stage_records_span_and_preserves_return():
    t0 = time.perf_counter()
    timer = StageTimer(t0)
    obj = _Fake()
    _wrap_stage(obj, "step", "Register", timer)

    assert obj.step(21) == 42  # behavior preserved
    assert [s[0] for s in timer.spans] == ["Register"]


@pytest.mark.skipif(not os.path.isdir(DATASET), reason="profiling dataset not present")
def test_profile_dataset_smoke(tmp_path):
    result = profile_dataset(DATASET, region="manual0")
    assert len(result.samples) > 0
    assert len(result.stage_spans) >= 1
    assert any(s[0] == "Register" for s in result.stage_spans)


def test_collect_returns_partial_result_and_cleans_up_on_error():
    t0 = time.perf_counter()
    timer = StageTimer(t0)

    def boom():
        # allocate something, then crash mid-run
        _blob = bytearray(5 * 1024 * 1024)  # noqa: F841
        time.sleep(0.05)
        raise RuntimeError("simulated pipeline failure")

    result = _collect(boom, t0, timer, rss_interval=0.01, alloc_interval=0.02)

    assert isinstance(result, ProfileResult)
    assert result.error is not None
    assert "simulated pipeline failure" in result.error
    assert len(result.samples) > 0  # partial RSS data retained
    assert not tracemalloc.is_tracing()  # tracemalloc cleaned up


def test_collect_success_has_no_error():
    t0 = time.perf_counter()
    timer = StageTimer(t0)

    def ok():
        with timer.stage("Register"):
            time.sleep(0.03)

    result = _collect(ok, t0, timer, rss_interval=0.01, alloc_interval=0.02)
    assert result.error is None
    assert any(s[0] == "Register" for s in result.stage_spans)
    assert not tracemalloc.is_tracing()
