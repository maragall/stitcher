import time
from profiling.sampler import RSSSampler, Sample


def test_sampler_collects_monotonic_positive_samples():
    t0 = time.perf_counter()
    s = RSSSampler(t0, interval_s=0.01)
    s.start()
    # Hold a chunk of memory while sampling.
    blob = bytearray(20 * 1024 * 1024)  # 20 MB
    time.sleep(0.1)
    del blob
    samples = s.stop()

    assert len(samples) >= 2
    assert all(isinstance(x, Sample) for x in samples)
    times = [x.t_ms for x in samples]
    assert times == sorted(times)  # monotonic non-decreasing
    assert all(x.rss_mb > 0 for x in samples)  # RSS always positive
