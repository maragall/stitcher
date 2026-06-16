"""Background thread sampling process RSS via psutil."""

import threading
import time
from typing import List, NamedTuple

import psutil


class Sample(NamedTuple):
    t_ms: float
    rss_mb: float


class RSSSampler(threading.Thread):
    """Samples resident set size at a fixed interval until stopped."""

    def __init__(self, t0: float, interval_s: float = 0.05):
        super().__init__(daemon=True)
        self._t0 = t0
        self._interval = interval_s
        self._stop = threading.Event()
        self._proc = psutil.Process()
        self.samples: List[Sample] = []

    def run(self) -> None:
        while True:
            t_ms = (time.perf_counter() - self._t0) * 1000.0
            rss_mb = self._proc.memory_info().rss / 1e6
            self.samples.append(Sample(t_ms, rss_mb))
            if self._stop.wait(self._interval):
                break

    def stop(self) -> List[Sample]:
        self._stop.set()
        self.join()
        return self.samples
