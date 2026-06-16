import os
import time
import pytest
from profiling.harness import _wrap_stage, profile_dataset
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
