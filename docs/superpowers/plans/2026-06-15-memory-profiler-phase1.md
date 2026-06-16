# Memory Profiler — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a headless profiler that runs the TileFusion pipeline on a real dataset and produces a total-RSS-over-time timeline, per-function memory attribution, a Pareto ranking, and `timeline.csv` / `functions.csv`.

**Architecture:** A self-contained, root-level `profiling/` package. A background thread samples process RSS (`psutil`); a second thread periodically takes `tracemalloc` snapshots and attributes Python allocations to the enclosing function. A harness wraps the existing `TileFusion` stage methods (non-invasively, via per-instance method override) to record stage time-spans, runs the pipeline, and hands the collected series to pure CSV-writers and matplotlib plot functions.

**Tech Stack:** Python 3.9+, `psutil` (RSS), `tracemalloc` (stdlib attribution), `matplotlib` (Agg backend) + `seaborn` (figures), `pytest` (TDD).

---

## Background the engineer must know

- **Target:** `tilefusion.TileFusion` (`src/tilefusion/core.py`). `TileFusion(dataset_path, region="manual0").run()` executes the pipeline end-to-end for one region. The constructor auto-detects the OME-TIFF-tiles format.
- **Stage methods inside `run()`** (these are what we wrap):
  - `refine_tile_positions_with_cross_correlation` → **Register** (reads overlap patches + cross-correlates; only runs if no cached metrics file exists)
  - `optimize_shifts` → **Optimize**
  - `_fuse_tiles` → **Fuse**
  - `_create_multiscales` → **Write**
  - (There is no separate top-level "Read" stage; tile reads happen inside Register and Fuse. Phase 1 records these four spans.)
- **Metrics cache gotcha:** `run()` loads `<dataset_parent>/<metrics_filename>` if present and **skips registration**. The harness sets a dedicated `metrics_filename` and deletes that file before running, so the Register stage actually executes and is profiled.
- **Dataset:** `~/Cephla/Data/test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy` (OME-TIFF tiles, regions `manual0`/`manual1`).
- **`tracemalloc` only sees Python-level allocations** (not numpy/native buffers). That is expected; the "unattributed gap" between summed function lines and the `psutil` total is the native memory and is shown deliberately.
- **Shared clock:** all three collectors (RSS sampler, allocation sampler, stage timer) take a single `t0 = time.perf_counter()` so their timestamps align. Times are stored in **milliseconds** relative to `t0`.

All work happens on branch `feat/memory-profiler`. Profiling test files live in `tests/` as `test_prof_*.py`.

---

## Task 1: Scaffolding, dependencies, pytest path

**Files:**
- Create: `profiling/__init__.py`
- Modify: `pyproject.toml` (add `profiling` optional-deps group; add pytest `pythonpath`)

- [ ] **Step 1: Create the package init**

Create `profiling/__init__.py`:

```python
"""Headless memory-footprint profiler for the TileFusion pipeline.

Not part of the shipped runtime; a developer tool run against a dataset.
"""
```

- [ ] **Step 2: Add the optional-dependency group**

In `pyproject.toml`, inside `[project.optional-dependencies]`, add this group after the `dev = [...]` block:

```toml
profiling = [
    "matplotlib>=3.5",
    "seaborn>=0.12",
]
```

- [ ] **Step 3: Make the root `profiling` package importable in tests**

In `pyproject.toml`, replace the `[tool.pytest.ini_options]` block with:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=tilefusion --cov-report=term-missing"
pythonpath = ["."]
```

- [ ] **Step 4: Install the new deps**

Run: `pip install -e ".[dev,profiling]"`
Expected: installs matplotlib + seaborn, no errors.

- [ ] **Step 5: Verify the package imports**

Run: `python -c "import profiling; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 6: Commit**

```bash
git add profiling/__init__.py pyproject.toml
git commit -m "feat(profiling): scaffold package, deps, and pytest path"
```

---

## Task 2: RSS sampler

**Files:**
- Create: `profiling/sampler.py`
- Test: `tests/test_prof_sampler.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prof_sampler.py`:

```python
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
    assert times == sorted(times)            # monotonic non-decreasing
    assert all(x.rss_mb > 0 for x in samples)  # RSS always positive
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prof_sampler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'profiling.sampler'`.

- [ ] **Step 3: Write minimal implementation**

Create `profiling/sampler.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prof_sampler.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add profiling/sampler.py tests/test_prof_sampler.py
git commit -m "feat(profiling): RSS sampler thread"
```

---

## Task 3: Function attribution helper (`function_for`)

**Files:**
- Create: `profiling/attribution.py`
- Test: `tests/test_prof_attribution.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prof_attribution.py`:

```python
import textwrap
from profiling.attribution import function_for


def test_function_for_maps_line_to_enclosing_function(tmp_path):
    src = textwrap.dedent('''\
        x = 1

        def outer():
            a = 1
            b = 2
            return a + b

        def other():
            return 0
    ''')
    f = tmp_path / "mod.py"
    f.write_text(src)

    assert function_for(str(f), 5) == "mod:outer"   # line "b = 2"
    assert function_for(str(f), 9) == "mod:other"   # line "return 0"
    assert function_for(str(f), 1) == "mod:<module>"  # top-level line
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prof_attribution.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'profiling.attribution'`.

- [ ] **Step 3: Write minimal implementation**

Create `profiling/attribution.py`:

```python
"""Attribute tracemalloc allocations to the enclosing Python function."""
import ast
import functools
from pathlib import Path
from typing import Tuple


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prof_attribution.py -v`
Expected: PASS (all three asserts).

- [ ] **Step 5: Commit**

```bash
git add profiling/attribution.py tests/test_prof_attribution.py
git commit -m "feat(profiling): map allocation lines to enclosing function"
```

---

## Task 4: Allocation sampler (tracemalloc)

**Files:**
- Modify: `profiling/attribution.py` (append `AllocRecord` + `AllocationSampler`)
- Test: `tests/test_prof_alloc_sampler.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prof_alloc_sampler.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prof_alloc_sampler.py -v`
Expected: FAIL with `ImportError: cannot import name 'AllocationSampler'`.

- [ ] **Step 3: Write minimal implementation**

Append to `profiling/attribution.py`:

```python
import threading
import time
import tracemalloc
from typing import List, NamedTuple


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prof_alloc_sampler.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add profiling/attribution.py tests/test_prof_alloc_sampler.py
git commit -m "feat(profiling): tracemalloc allocation sampler"
```

---

## Task 5: Ranking computation

**Files:**
- Create: `profiling/ranking.py`
- Test: `tests/test_prof_ranking.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prof_ranking.py`:

```python
from profiling.attribution import AllocRecord
from profiling.ranking import compute_ranking


def test_compute_ranking_integrates_and_ranks():
    # funcA: flat 100 MB from t=0..1000ms -> 100 MB * 1 s = 100 MB·s
    # funcB: flat 50 MB from t=0..1000ms -> 50 MB·s
    records = [
        AllocRecord(0.0, "m:funcA", 100.0),
        AllocRecord(1000.0, "m:funcA", 100.0),
        AllocRecord(0.0, "m:funcB", 50.0),
        AllocRecord(1000.0, "m:funcB", 50.0),
    ]
    ranking = compute_ranking(records)

    assert [r["function"] for r in ranking] == ["m:funcA", "m:funcB"]
    assert abs(ranking[0]["integrated_mb_s"] - 100.0) < 1e-6
    assert abs(ranking[1]["integrated_mb_s"] - 50.0) < 1e-6
    assert abs(ranking[0]["peak_mb"] - 100.0) < 1e-6
    assert abs(ranking[0]["pct_of_total"] - (100.0 / 150.0 * 100.0)) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prof_ranking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'profiling.ranking'`.

- [ ] **Step 3: Write minimal implementation**

Create `profiling/ranking.py`:

```python
"""Rank functions by integrated memory cost (MB·seconds)."""
from typing import Dict, List

from profiling.attribution import AllocRecord


def _trapz_mb_s(series: List) -> float:
    """Trapezoidal integral of (t_ms, mb) points -> MB·seconds."""
    total = 0.0
    for (t0, v0), (t1, v1) in zip(series, series[1:]):
        dt_s = (t1 - t0) / 1000.0
        total += 0.5 * (v0 + v1) * dt_s
    return total


def compute_ranking(records: List[AllocRecord]) -> List[Dict]:
    """Aggregate per-function: peak_mb, integrated_mb_s, pct_of_total; sorted desc."""
    by_func: Dict[str, List] = {}
    for r in records:
        by_func.setdefault(r.func, []).append((r.t_ms, r.size_mb))

    out = []
    for func, series in by_func.items():
        series.sort()
        out.append(
            {
                "function": func,
                "peak_mb": max(v for _, v in series),
                "integrated_mb_s": _trapz_mb_s(series),
            }
        )

    total = sum(o["integrated_mb_s"] for o in out) or 1.0
    for o in out:
        o["pct_of_total"] = 100.0 * o["integrated_mb_s"] / total

    out.sort(key=lambda o: o["integrated_mb_s"], reverse=True)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prof_ranking.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add profiling/ranking.py tests/test_prof_ranking.py
git commit -m "feat(profiling): per-function MB·s ranking"
```

---

## Task 6: Stage timer + stage assignment

**Files:**
- Create: `profiling/stages.py`
- Test: `tests/test_prof_stages.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prof_stages.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prof_stages.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'profiling.stages'`.

- [ ] **Step 3: Write minimal implementation**

Create `profiling/stages.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prof_stages.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add profiling/stages.py tests/test_prof_stages.py
git commit -m "feat(profiling): stage timer and sample-to-stage assignment"
```

---

## Task 7: CSV writers

**Files:**
- Create: `profiling/record.py`
- Test: `tests/test_prof_record.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prof_record.py`:

```python
import csv
from profiling.sampler import Sample
from profiling.record import write_timeline_csv, write_functions_csv


def test_write_timeline_csv(tmp_path):
    samples = [Sample(0.0, 100.0), Sample(50.0, 120.0)]
    spans = [("Register", 0.0, 100.0)]
    path = tmp_path / "timeline.csv"
    write_timeline_csv(str(path), samples, spans)

    rows = list(csv.DictReader(path.open()))
    assert rows[0]["stage"] == "Register"
    assert float(rows[1]["rss_mb"]) == 120.0


def test_write_functions_csv(tmp_path):
    ranking = [
        {"function": "m:funcA", "peak_mb": 100.0, "integrated_mb_s": 100.0, "pct_of_total": 66.67}
    ]
    path = tmp_path / "functions.csv"
    write_functions_csv(str(path), ranking)

    rows = list(csv.DictReader(path.open()))
    assert rows[0]["function"] == "m:funcA"
    assert float(rows[0]["pct_of_total"]) == 66.67
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prof_record.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'profiling.record'`.

- [ ] **Step 3: Write minimal implementation**

Create `profiling/record.py`:

```python
"""Write profiler results to CSV."""
import csv
from typing import Dict, List

from profiling.sampler import Sample
from profiling.stages import Span, assign_stages


def write_timeline_csv(path: str, samples: List[Sample], spans: List[Span]) -> None:
    rows = assign_stages(samples, spans)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ms", "rss_mb", "stage"])
        for t_ms, rss_mb, stage in rows:
            w.writerow([f"{t_ms:.1f}", f"{rss_mb:.3f}", stage])


def write_functions_csv(path: str, ranking: List[Dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["function", "peak_mb", "integrated_mb_s", "pct_of_total"])
        for o in ranking:
            w.writerow(
                [
                    o["function"],
                    f'{o["peak_mb"]:.3f}',
                    f'{o["integrated_mb_s"]:.3f}',
                    f'{o["pct_of_total"]:.2f}',
                ]
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prof_record.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add profiling/record.py tests/test_prof_record.py
git commit -m "feat(profiling): timeline and functions CSV writers"
```

---

## Task 8: Plots (timeline, per-function lines, Pareto)

**Files:**
- Create: `profiling/plots.py`
- Test: `tests/test_prof_plots.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prof_plots.py`:

```python
from profiling.sampler import Sample
from profiling.attribution import AllocRecord
from profiling.ranking import compute_ranking
from profiling.plots import plot_timeline, plot_function_lines, plot_pareto


def _records():
    return [
        AllocRecord(0.0, "m:funcA", 100.0),
        AllocRecord(1000.0, "m:funcA", 100.0),
        AllocRecord(0.0, "m:funcB", 50.0),
        AllocRecord(1000.0, "m:funcB", 50.0),
    ]


def test_plots_write_nonempty_files(tmp_path):
    samples = [Sample(0.0, 100.0), Sample(500.0, 180.0), Sample(1000.0, 120.0)]
    spans = [("Register", 0.0, 500.0), ("Fuse", 500.0, 1000.0)]
    records = _records()
    ranking = compute_ranking(records)

    p1 = tmp_path / "timeline.png"
    p2 = tmp_path / "functions.png"
    p3 = tmp_path / "pareto.png"
    plot_timeline(samples, spans, str(p1))
    plot_function_lines(samples, records, ranking, str(p2), top_k=2)
    plot_pareto(ranking, str(p3))

    for p in (p1, p2, p3):
        assert p.exists() and p.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prof_plots.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'profiling.plots'`.

- [ ] **Step 3: Write minimal implementation**

Create `profiling/plots.py`:

```python
"""Phase-1 figures. Agg backend so it runs headless."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from profiling.sampler import Sample  # noqa: E402
from profiling.stages import Span  # noqa: E402

_PALETTE = ["#1565c0", "#26a69a", "#5c6bc0", "#c9a227", "#a1707f", "#78909c"]


def _draw_stage_boundaries(ax, spans):
    """Dashed vertical at each span end; stage name rotated -90 at the top."""
    ymax = ax.get_ylim()[1]
    for name, _start, end in spans:
        ax.axvline(end / 1000.0, color="#90a4ae", linestyle="--", linewidth=1)
        ax.text(end / 1000.0, ymax, name, rotation=-90, va="top", ha="right",
                fontsize=8, color="#37474f")


def plot_timeline(samples, spans, out_path):
    t_s = [s.t_ms / 1000.0 for s in samples]
    rss = [s.rss_mb for s in samples]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t_s, rss, color="#1565c0", linewidth=2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Total RSS over time")
    _draw_stage_boundaries(ax, spans)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_function_lines(samples, records, ranking, out_path, top_k=5):
    top = [r["function"] for r in ranking[:top_k]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, func in enumerate(top):
        pts = sorted((r.t_ms / 1000.0, r.size_mb) for r in records if r.func == func)
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, label=func, color=_PALETTE[i % len(_PALETTE)], linewidth=1.5)
    # Bold total RSS line.
    t_s = [s.t_ms / 1000.0 for s in samples]
    rss = [s.rss_mb for s in samples]
    ax.plot(t_s, rss, label="TOTAL RSS", color="#263238", linewidth=2.4)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Per-function memory (top functions) + total")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pareto(ranking, out_path, top_k=10):
    top = ranking[:top_k]
    names = [r["function"] for r in top]
    vals = [r["integrated_mb_s"] for r in top]
    total = sum(r["integrated_mb_s"] for r in ranking) or 1.0
    cum = []
    running = 0.0
    for v in vals:
        running += v
        cum.append(100.0 * running / total)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(names)), vals, color="#26a69a")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=-90, fontsize=7)
    ax.set_ylabel("Integrated cost (MB·s)")
    ax2 = ax.twinx()
    ax2.plot(range(len(names)), cum, color="#263238", marker="o", linewidth=1.5)
    ax2.set_ylabel("Cumulative %")
    ax2.set_ylim(0, 105)
    ax.set_title("Function ranking (Pareto)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prof_plots.py -v`
Expected: PASS (all three files non-empty).

- [ ] **Step 5: Commit**

```bash
git add profiling/plots.py tests/test_prof_plots.py
git commit -m "feat(profiling): timeline, per-function, and Pareto figures"
```

---

## Task 9: Harness (wire pipeline + collectors)

**Files:**
- Create: `profiling/harness.py`
- Test: `tests/test_prof_harness.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prof_harness.py`. The full run is gated on the real dataset (skipped if absent); the method-wrapping logic is unit-tested without it.

```python
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

    assert obj.step(21) == 42                 # behavior preserved
    assert [s[0] for s in timer.spans] == ["Register"]


@pytest.mark.skipif(not os.path.isdir(DATASET), reason="profiling dataset not present")
def test_profile_dataset_smoke(tmp_path):
    result = profile_dataset(DATASET, region="manual0")
    assert len(result.samples) > 0
    assert len(result.stage_spans) >= 1
    assert any(s[0] == "Register" for s in result.stage_spans)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prof_harness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'profiling.harness'`.

- [ ] **Step 3: Write minimal implementation**

Create `profiling/harness.py`:

```python
"""Headless harness: wrap TileFusion stages, run pipeline, collect series."""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from profiling.sampler import RSSSampler, Sample
from profiling.attribution import AllocationSampler, AllocRecord
from profiling.stages import Span, StageTimer

# (method_name, stage_label) for the four demarcatable stages inside run().
_STAGE_METHODS = [
    ("refine_tile_positions_with_cross_correlation", "Register"),
    ("optimize_shifts", "Optimize"),
    ("_fuse_tiles", "Fuse"),
    ("_create_multiscales", "Write"),
]


@dataclass
class ProfileResult:
    samples: List[Sample]
    alloc_records: List[AllocRecord]
    stage_spans: List[Span]


def _wrap_stage(obj, method_name: str, stage: str, timer: StageTimer) -> None:
    """Override a bound method on `obj` to time it under `stage`."""
    original = getattr(obj, method_name)

    def wrapped(*args, **kwargs):
        with timer.stage(stage):
            return original(*args, **kwargs)

    setattr(obj, method_name, wrapped)


def profile_dataset(
    dataset: str,
    region: str = "manual0",
    rss_interval: float = 0.05,
    alloc_interval: float = 0.25,
) -> ProfileResult:
    from tilefusion import TileFusion

    t0 = time.perf_counter()
    metrics_name = f"profile_metrics_{region}.json"

    tf = TileFusion(dataset, region=region, metrics_filename=metrics_name)

    # Force the Register stage to actually run (don't load cached metrics).
    metrics_path = Path(dataset).parent / metrics_name
    if metrics_path.exists():
        metrics_path.unlink()

    timer = StageTimer(t0)
    for method_name, stage in _STAGE_METHODS:
        _wrap_stage(tf, method_name, stage, timer)

    rss = RSSSampler(t0, rss_interval)
    alloc = AllocationSampler(t0, alloc_interval)
    alloc.start_tracing()
    rss.start()
    alloc.start()
    try:
        tf.run()
    finally:
        samples = rss.stop()
        records = alloc.stop()

    return ProfileResult(samples, records, timer.spans)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prof_harness.py -v`
Expected: `test_wrap_stage_records_span_and_preserves_return` PASS; `test_profile_dataset_smoke` PASS if the dataset is present, else SKIPPED.

- [ ] **Step 5: Commit**

```bash
git add profiling/harness.py tests/test_prof_harness.py
git commit -m "feat(profiling): pipeline harness with stage wrapping"
```

---

## Task 10: CLI entrypoint

**Files:**
- Create: `profiling/cli.py`
- Test: `tests/test_prof_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prof_cli.py`:

```python
import pytest
from profiling.cli import build_parser


def test_parser_requires_dataset_and_defaults():
    parser = build_parser()
    args = parser.parse_args(["/some/dataset", "--out", "/tmp/out"])
    assert args.dataset == "/some/dataset"
    assert args.out == "/tmp/out"
    assert args.region == "manual0"  # default


def test_parser_errors_without_dataset():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prof_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'profiling.cli'`.

- [ ] **Step 3: Write minimal implementation**

Create `profiling/cli.py`:

```python
"""CLI: python -m profiling.cli <dataset> --out <dir> [--region manual0]"""
import argparse
import os

from profiling.harness import profile_dataset
from profiling.ranking import compute_ranking
from profiling.record import write_timeline_csv, write_functions_csv
from profiling.plots import plot_timeline, plot_function_lines, plot_pareto


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Profile TileFusion memory footprint (Phase 1).")
    p.add_argument("dataset", help="Path to the dataset folder")
    p.add_argument("--out", default="profile_out", help="Output directory")
    p.add_argument("--region", default="manual0", help="Region to profile")
    p.add_argument("--top-k", type=int, default=5, help="Functions to plot as lines")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    result = profile_dataset(args.dataset, region=args.region)
    ranking = compute_ranking(result.alloc_records)

    write_timeline_csv(os.path.join(args.out, "timeline.csv"), result.samples, result.stage_spans)
    write_functions_csv(os.path.join(args.out, "functions.csv"), ranking)
    plot_timeline(result.samples, result.stage_spans, os.path.join(args.out, "timeline.png"))
    plot_function_lines(
        result.samples, result.alloc_records, ranking,
        os.path.join(args.out, "functions.png"), top_k=args.top_k,
    )
    plot_pareto(ranking, os.path.join(args.out, "pareto.png"))

    print(f"Wrote profile to {args.out}")
    if ranking:
        top2 = sum(r["pct_of_total"] for r in ranking[:2])
        print(f"Top 2 functions explain {top2:.1f}% of integrated memory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prof_cli.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Run the full suite + a real profile (manual QC)**

Run: `pytest tests/test_prof_*.py -v`
Expected: all pass (harness smoke test SKIPPED only if dataset absent).

Then the real end-to-end run (this is the Phase-1 QC artifact):

Run: `python -m profiling.cli "$HOME/Cephla/Data/test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy" --out profile_out --region manual0`
Expected: prints `Wrote profile to profile_out` and the "Top 2 functions explain X%" line; `profile_out/` contains `timeline.csv`, `functions.csv`, `timeline.png`, `functions.png`, `pareto.png`.

- [ ] **Step 6: Commit**

```bash
git add profiling/cli.py tests/test_prof_cli.py
git commit -m "feat(profiling): CLI entrypoint for Phase 1 profiling"
```

---

## Phase 1 QC checklist (review before merge)

- [ ] All `tests/test_prof_*.py` pass.
- [ ] `profile_out/timeline.png` shows the RSS curve with dashed stage boundaries and rotated stage labels.
- [ ] `profile_out/functions.png` shows top-k function lines under a bold total-RSS line; the gap (native memory) is visible.
- [ ] `profile_out/pareto.png` shows sorted bars + cumulative-% line; CLI prints the "top 2 = X%" insight.
- [ ] `functions.csv` ranking is plausible (registration / fusion / tensorstore functions near the top).
- [ ] No changes were made to `src/tilefusion/` (pipeline behavior unchanged).

---

## Self-Review notes (completed by plan author)

- **Spec coverage:** Phase-1 spec items map to tasks — total RSS timeline (T2 sampler, T6 stages, T8 plot), per-function attribution (T3/T4), ranking (T5), `timeline.csv`/`functions.csv` (T7), three core figures (T8), harness + Run A (T9), CLI (T10). Per-pair/swimlanes/variability/scan-pattern are Phase 2 (out of scope here). `substep_stats.csv` is Phase 2.
- **Stage count:** spec lists 5 conceptual stages; `run()` exposes 4 wrappable methods (Read is subsumed into Register/Fuse). Documented in Background and Task 9.
- **Type consistency:** `Sample(t_ms, rss_mb)`, `AllocRecord(t_ms, func, size_mb)`, `Span=(name,start_ms,end_ms)`, ranking dict keys `function/peak_mb/integrated_mb_s/pct_of_total`, and `ProfileResult(samples, alloc_records, stage_spans)` are used consistently across tasks 2–10.
- **No placeholders:** every code/test step contains complete code and exact run commands.
