"""Attribute tracemalloc allocations to the enclosing Python function."""

import ast
import functools
import threading
import time
import tracemalloc
from pathlib import Path
from typing import List, NamedTuple, Tuple


@functools.lru_cache(maxsize=None)
def _func_spans(filename: str) -> Tuple[Tuple[int, int, str], ...]:
    """Return (start_line, end_line, name) for every function in a file."""
    try:
        src = Path(filename).read_text()
        tree = ast.parse(src)
    except (OSError, SyntaxError, ValueError):
        return ()
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            spans.append((node.lineno, end, node.name))
    return tuple(spans)


def function_for(filename: str, lineno: int) -> str:
    """Label "<module-stem>:<func>" for the innermost function covering lineno."""
    stem = Path(filename).stem
    best = None  # (start_line, name) of innermost enclosing function
    for start, end, name in _func_spans(filename):
        if start <= lineno <= end and (best is None or start > best[0]):
            best = (start, name)
    return f"{stem}:{best[1]}" if best else f"{stem}:<module>"


class AllocRecord(NamedTuple):
    t_ms: float
    func: str
    size_mb: float


class AllocationSampler(threading.Thread):
    """Periodically snapshots tracemalloc and attributes live bytes per function."""

    def __init__(self, t0: float, interval_s: float = 0.25):
        super().__init__(daemon=True)
        self._t0 = t0
        self._interval = interval_s
        self._stop = threading.Event()
        self.records: List[AllocRecord] = []

    def start_tracing(self) -> None:
        if not tracemalloc.is_tracing():
            tracemalloc.start()

    def run(self) -> None:
        while not self._stop.wait(self._interval):
            t_ms = (time.perf_counter() - self._t0) * 1000.0
            snapshot = tracemalloc.take_snapshot()
            agg = {}
            for stat in snapshot.statistics("lineno"):
                frame = stat.traceback[0]
                func = function_for(frame.filename, frame.lineno)
                agg[func] = agg.get(func, 0) + stat.size
            for func, size in agg.items():
                self.records.append(AllocRecord(t_ms, func, size / 1e6))

    def stop(self) -> List[AllocRecord]:
        self._stop.set()
        self.join()
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        return self.records
