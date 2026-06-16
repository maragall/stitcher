"""Record stage time-spans and label samples by stage."""

import time
from contextlib import contextmanager
from typing import List, Tuple

from profiling.sampler import Sample

Span = Tuple[str, float, float]  # (name, start_ms, end_ms)


class StageTimer:
    def __init__(self, t0: float):
        self._t0 = t0
        self.spans: List[Span] = []

    @contextmanager
    def stage(self, name: str):
        start = (time.perf_counter() - self._t0) * 1000.0
        try:
            yield
        finally:
            end = (time.perf_counter() - self._t0) * 1000.0
            self.spans.append((name, start, end))


def assign_stages(samples: List[Sample], spans: List[Span]) -> List[Tuple[float, float, str]]:
    """Return (t_ms, rss_mb, stage) for each sample; "(other)" if outside all spans."""
    rows = []
    for s in samples:
        label = "(other)"
        for name, start, end in spans:
            if start <= s.t_ms <= end:
                label = name
                break
        rows.append((s.t_ms, s.rss_mb, label))
    return rows
