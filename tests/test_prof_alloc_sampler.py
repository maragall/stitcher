import time
from profiling.attribution import AllocationSampler, AllocRecord


def _make_blob():
    # Pure-Python allocation tracemalloc can see (not numpy/native).
    return bytearray(30 * 1024 * 1024)  # 30 MB


def test_alloc_sampler_attributes_to_calling_function():
    t0 = time.perf_counter()
    s = AllocationSampler(t0, interval_s=0.02)
    s.start_tracing()
    s.start()
    blob = _make_blob()
    time.sleep(0.1)
    records = s.stop()
    del blob

    assert all(isinstance(r, AllocRecord) for r in records)
    funcs = {r.func for r in records}
    # The blob was allocated inside _make_blob in this test module.
    assert any(f.endswith(":_make_blob") for f in funcs)
    assert any(r.size_mb > 10 for r in records)
