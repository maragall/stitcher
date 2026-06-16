# Memory Profiler — Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-pair registration analysis — `pairs.csv`, a reconstructed concurrency swimlane figure, a variability/CV summary, and a scan-pattern figure — all NON-INVASIVELY (no edits to `src/tilefusion/`).

**Architecture:** A "Run B" serialized registration pass (`max_workers=1`) wraps the module-level `register_pair_worker` at runtime (monkeypatch on `tilefusion.core`) to record per-pair tile indices, overlap-patch byte sizes, and wall-clock duration. Tile indices are mapped to `(region, fov)` + grid `(row, col)` from the `TileFusion` metadata. Pure functions compute variability (CV) and infer the scan pattern; matplotlib renders the figures; everything also lands in CSV.

**Tech Stack:** Python 3.9+, matplotlib (Agg), pytest. Reuses Phase 1's `profiling` package.

---

## Background the engineer must know

- Phase 1 shipped the `profiling/` package (sampler, attribution, ranking, stages, record, plots, harness, cli) — all on `main`. This phase ADDS to it on branch `feat/memory-profiler-phase2`.
- **Why per-pair memory = patch bytes, not tracemalloc:** the registration compute (`register_pair_worker`) does FFT/phase-correlation whose big buffers are native (numpy/pocketfft), which `tracemalloc` cannot see. The deterministic, cross-platform per-pair memory metric is the **overlap-patch size** (`patch_i.nbytes + patch_j.nbytes`) — the actual input each pair loads. We record that plus wall-clock duration. (This is why Phase 2 is "per-pair, non-invasive": no sub-step read/FFT/corr breakdown.)
- **How registration is called** (`src/tilefusion/core.py`): `_register_parallel` builds `work_items = [(i, j, pi, pj, df, sw, th, max_shift), ...]` and runs `executor.map(register_pair_worker, work_items)`. `register_pair_worker` is imported into `core` via `from .registration import register_pair_worker`, so the bound name to wrap is `tilefusion.core.register_pair_worker`. Each arg tuple's first four entries are `(i_pos, j_pos, patch_i, patch_j)` — `i_pos`/`j_pos` are tile indices; `patch_i`/`patch_j` are numpy arrays (have `.nbytes`).
- **Serialized run:** with `max_workers=1`, `_register_parallel` still uses a `ThreadPoolExecutor` but with one worker, so calls happen one at a time — per-pair durations are clean (no concurrency overlap inflating them).
- **Registration entry point:** `tf.refine_tile_positions_with_cross_correlation(downsample_factors=tf.downsample_factors, ch_idx=tf.channel_to_use, threshold=tf.threshold)` runs registration only (no fuse/write). It only runs if no cached metrics file exists — so delete it first (same as Phase 1).
- **Tile identity:** `tf._tile_positions` = list of `(y_um, x_um)`; `tf._tile_identifiers` = list of `(region, fov)`; `tf.Y, tf.X` = tile shape. Indices in pair records map into these lists.
- **Swimlanes are a RECONSTRUCTION:** we measure pairs serially, then simulate the real 8-worker schedule by greedily packing pair durations into 8 lanes. The figure is labeled as a reconstruction.
- CI enforces only `black --line-length 100`. Keep `from typing import ...` style. Tests are `tests/test_prof2_*.py`. The dataset for smoke tests: `~/Cephla/Data/test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy`.

---

## Task 1: Per-pair recorder

**Files:**
- Create: `profiling/perpair.py`
- Test: `tests/test_prof2_perpair.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prof2_perpair.py`:

```python
import types
from profiling.perpair import PairRecorder, PairRecord


class _FakePatch:
    def __init__(self, nbytes):
        self.nbytes = nbytes


def test_pair_recorder_wraps_records_and_restores():
    calls = []

    def fake_worker(args):
        calls.append(args)
        return (args[0], args[1], 1, 2, 0.9)  # mimic (i, j, dy, dx, score)

    target = types.SimpleNamespace(register_pair_worker=fake_worker)
    original = target.register_pair_worker

    with PairRecorder(target=target) as rec:
        args = (3, 4, _FakePatch(100), _FakePatch(150), None, None, None, None)
        out = target.register_pair_worker(args)

    # original behavior preserved (delegates + returns)
    assert out == (3, 4, 1, 2, 0.9)
    assert calls == [args]
    # restored after context
    assert target.register_pair_worker is original

    assert len(rec.records) == 1
    r = rec.records[0]
    assert isinstance(r, PairRecord)
    assert (r.i, r.j) == (3, 4)
    assert r.patch_i_bytes == 100
    assert r.patch_j_bytes == 150
    assert r.patch_bytes_total == 250
    assert r.duration_ms >= 0.0
    assert r.pair_id == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prof2_perpair.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'profiling.perpair'`.

- [ ] **Step 3: Write minimal implementation**

Create `profiling/perpair.py`:

```python
"""Non-invasive per-pair recorder for the registration stage.

Wraps the module-level ``register_pair_worker`` that ``tilefusion.core`` calls,
recording each pair's tile indices, overlap-patch byte sizes, and wall-clock
duration. Restores the original on exit. Use with ``max_workers=1`` so per-pair
durations are not inflated by concurrency.
"""
import time
from typing import List, NamedTuple, Optional


class PairRecord(NamedTuple):
    pair_id: int
    i: int
    j: int
    patch_i_bytes: int
    patch_j_bytes: int
    patch_bytes_total: int
    duration_ms: float


class PairRecorder:
    def __init__(self, target=None, attr: str = "register_pair_worker"):
        self._target = target  # resolved to tilefusion.core on enter if None
        self._attr = attr
        self._original = None
        self.records: List[PairRecord] = []

    def __enter__(self) -> "PairRecorder":
        if self._target is None:
            import tilefusion.core as core

            self._target = core
        self._original = getattr(self._target, self._attr)
        original = self._original
        records = self.records

        def wrapped(args):
            i_pos, j_pos, patch_i, patch_j = args[0], args[1], args[2], args[3]
            pb_i = int(getattr(patch_i, "nbytes", 0) or 0)
            pb_j = int(getattr(patch_j, "nbytes", 0) or 0)
            t0 = time.perf_counter()
            result = original(args)
            dt = (time.perf_counter() - t0) * 1000.0
            records.append(
                PairRecord(len(records), i_pos, j_pos, pb_i, pb_j, pb_i + pb_j, dt)
            )
            return result

        setattr(self._target, self._attr, wrapped)
        return self

    def __exit__(self, *exc) -> Optional[bool]:
        setattr(self._target, self._attr, self._original)
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prof2_perpair.py -v`
Expected: PASS.

- [ ] **Step 5: Black + commit**

Run: `black --line-length 100 profiling/perpair.py tests/test_prof2_perpair.py`
```bash
git add profiling/perpair.py tests/test_prof2_perpair.py
git commit -m "feat(profiling): non-invasive per-pair registration recorder"
```

---

## Task 2: Tile grid + scan-pattern inference

**Files:**
- Create: `profiling/tiles.py`
- Test: `tests/test_prof2_tiles.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prof2_tiles.py`:

```python
from profiling.tiles import build_grid, tile_label, infer_scan_pattern


def test_build_grid_assigns_row_col():
    # 2 rows x 3 cols raster; positions are (y_um, x_um)
    positions = [
        (10.0, 0.0), (10.0, 5.0), (10.0, 10.0),
        (20.0, 0.0), (20.0, 5.0), (20.0, 10.0),
    ]
    grid = build_grid(positions)
    assert grid[0] == (0, 0)
    assert grid[2] == (0, 2)
    assert grid[3] == (1, 0)
    assert grid[5] == (1, 2)


def test_tile_label_uses_identifier_and_grid():
    positions = [(10.0, 0.0), (10.0, 5.0)]
    identifiers = [("manual0", 0), ("manual0", 1)]
    grid = build_grid(positions)
    assert tile_label(identifiers, grid, 1) == "manual0/fov1@(r0,c1)"


def test_infer_scan_pattern_raster_vs_serpentine():
    # raster: every row left->right
    raster = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert infer_scan_pattern(raster) == "raster"
    # serpentine: row 0 left->right, row 1 right->left
    serp = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]
    assert infer_scan_pattern(serp) == "serpentine"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prof2_tiles.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'profiling.tiles'`.

- [ ] **Step 3: Write minimal implementation**

Create `profiling/tiles.py`:

```python
"""Map tile indices to grid (row, col), labels, and infer the scan pattern."""
from typing import Dict, List, Tuple


def build_grid(tile_positions: List[Tuple[float, float]], decimals: int = 0) -> Dict[int, Tuple[int, int]]:
    """Map each tile index to (row, col) by ranking unique y (rows) and x (cols).

    tile_positions are (y_um, x_um). Rows increase with y, cols with x.
    """
    ys = sorted({round(y, decimals) for y, _ in tile_positions})
    xs = sorted({round(x, decimals) for _, x in tile_positions})
    row_of = {y: r for r, y in enumerate(ys)}
    col_of = {x: c for c, x in enumerate(xs)}
    return {
        idx: (row_of[round(y, decimals)], col_of[round(x, decimals)])
        for idx, (y, x) in enumerate(tile_positions)
    }


def tile_label(identifiers: List[Tuple], grid: Dict[int, Tuple[int, int]], idx: int) -> str:
    """Human label like "manual0/fov1@(r0,c1)" (falls back to index if no id)."""
    r, c = grid[idx]
    if identifiers and idx < len(identifiers):
        region, fov = identifiers[idx]
        return f"{region}/fov{fov}@(r{r},c{c})"
    return f"tile{idx}@(r{r},c{c})"


def infer_scan_pattern(grid_sequence: List[Tuple[int, int]]) -> str:
    """Classify acquisition order (a list of (row, col) in tile-index order).

    Returns "raster" (every row same column direction), "serpentine"
    (direction alternates per row), or "unknown".
    """
    by_row: Dict[int, List[int]] = {}
    for r, c in grid_sequence:
        by_row.setdefault(r, []).append(c)

    directions = []
    for r in sorted(by_row):
        cols = by_row[r]
        if len(cols) < 2:
            continue
        directions.append(1 if cols[-1] > cols[0] else -1)

    if not directions:
        return "unknown"
    if all(d == 1 for d in directions):
        return "raster"
    if all(directions[k] == -directions[k - 1] for k in range(1, len(directions))):
        return "serpentine"
    return "unknown"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prof2_tiles.py -v`
Expected: PASS (all three tests).

- [ ] **Step 5: Black + commit**

Run: `black --line-length 100 profiling/tiles.py tests/test_prof2_tiles.py`
```bash
git add profiling/tiles.py tests/test_prof2_tiles.py
git commit -m "feat(profiling): tile grid mapping, labels, and scan-pattern inference"
```

---

## Task 3: Variability / CV stats

**Files:**
- Create: `profiling/variability.py`
- Test: `tests/test_prof2_variability.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prof2_variability.py`:

```python
from profiling.perpair import PairRecord
from profiling.variability import compute_pair_stats


def _rec(pid, dur, total):
    return PairRecord(pid, pid, pid + 1, total // 2, total - total // 2, total, dur)


def test_compute_pair_stats_mean_std_cv():
    records = [_rec(0, 10.0, 100), _rec(1, 20.0, 100), _rec(2, 30.0, 100)]
    stats = compute_pair_stats(records)

    assert stats["n_pairs"] == 3
    assert abs(stats["duration_ms"]["mean"] - 20.0) < 1e-9
    # population std of [10,20,30] = sqrt(200/3) ~= 8.16497
    assert abs(stats["duration_ms"]["std"] - 8.16496580927726) < 1e-6
    assert abs(stats["duration_ms"]["cv"] - (8.16496580927726 / 20.0)) < 1e-9
    # patch_bytes_total all equal -> cv 0
    assert abs(stats["patch_bytes_total"]["cv"] - 0.0) < 1e-12


def test_compute_pair_stats_empty():
    stats = compute_pair_stats([])
    assert stats["n_pairs"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prof2_variability.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'profiling.variability'`.

- [ ] **Step 3: Write minimal implementation**

Create `profiling/variability.py`:

```python
"""Variability (coefficient of variation) of per-pair metrics.

A high CV across pairs flags inconsistent per-pair work — a signal of
algorithmic inefficiency worth investigating.
"""
import math
from typing import Dict, List

from profiling.perpair import PairRecord

_METRICS = ("duration_ms", "patch_bytes_total")


def _stats(values: List[float]) -> Dict[str, float]:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n  # population variance
    std = math.sqrt(var)
    cv = (std / mean) if mean else 0.0
    return {"mean": mean, "std": std, "cv": cv, "min": min(values), "max": max(values)}


def compute_pair_stats(records: List[PairRecord]) -> Dict:
    """Per-metric mean/std/cv/min/max across all pairs."""
    out: Dict = {"n_pairs": len(records)}
    if not records:
        return out
    for metric in _METRICS:
        out[metric] = _stats([float(getattr(r, metric)) for r in records])
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prof2_variability.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Black + commit**

Run: `black --line-length 100 profiling/variability.py tests/test_prof2_variability.py`
```bash
git add profiling/variability.py tests/test_prof2_variability.py
git commit -m "feat(profiling): per-pair variability (CV) stats"
```

---

## Task 4: pairs.csv writer

**Files:**
- Modify: `profiling/record.py` (append `write_pairs_csv`)
- Test: `tests/test_prof2_record.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prof2_record.py`:

```python
import csv
from profiling.perpair import PairRecord
from profiling.record import write_pairs_csv


def test_write_pairs_csv(tmp_path):
    records = [PairRecord(0, 3, 4, 100, 150, 250, 12.5)]
    grid = {3: (0, 3), 4: (0, 4)}
    identifiers = [("m", 0), ("m", 1), ("m", 2), ("m", 3), ("m", 4)]
    path = tmp_path / "pairs.csv"
    write_pairs_csv(str(path), records, grid, identifiers)

    rows = list(csv.DictReader(path.open()))
    assert rows[0]["pair_id"] == "0"
    assert rows[0]["i"] == "3"
    assert rows[0]["tile_i"] == "m/fov3@(r0,c3)"
    assert rows[0]["tile_j"] == "m/fov4@(r0,c4)"
    assert rows[0]["patch_bytes_total"] == "250"
    assert float(rows[0]["duration_ms"]) == 12.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prof2_record.py -v`
Expected: FAIL with `ImportError: cannot import name 'write_pairs_csv'`.

- [ ] **Step 3: Write minimal implementation**

Append to `profiling/record.py` (keep existing functions and imports; add the `tile_label` import to the existing import block at the top, and `PairRecord` is only needed for typing — you may import it for the annotation):

```python
from profiling.perpair import PairRecord  # noqa: E402  (append near other imports)
from profiling.tiles import tile_label  # noqa: E402


def write_pairs_csv(path, records, grid, identifiers) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "pair_id",
                "i",
                "j",
                "tile_i",
                "tile_j",
                "row_i",
                "col_i",
                "row_j",
                "col_j",
                "patch_i_bytes",
                "patch_j_bytes",
                "patch_bytes_total",
                "duration_ms",
            ]
        )
        for r in records:
            ri, ci = grid[r.i]
            rj, cj = grid[r.j]
            w.writerow(
                [
                    r.pair_id,
                    r.i,
                    r.j,
                    tile_label(identifiers, grid, r.i),
                    tile_label(identifiers, grid, r.j),
                    ri,
                    ci,
                    rj,
                    cj,
                    r.patch_i_bytes,
                    r.patch_j_bytes,
                    r.patch_bytes_total,
                    f"{r.duration_ms:.3f}",
                ]
            )
```

NOTE: put the two new imports with the other `from profiling...` imports at the top of `record.py` (not mid-file); the `# noqa` hints above are only to indicate placement, drop them if the imports are at the top. Ensure `black` passes.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prof2_record.py tests/test_prof_record.py -v`
Expected: PASS (new test AND the Phase 1 record test — no regression).

- [ ] **Step 5: Black + commit**

Run: `black --line-length 100 profiling/record.py tests/test_prof2_record.py`
```bash
git add profiling/record.py tests/test_prof2_record.py
git commit -m "feat(profiling): pairs.csv writer"
```

---

## Task 5: Swimlane scheduler + figure

**Files:**
- Create: `profiling/swimlanes.py` (pure scheduler)
- Modify: `profiling/plots.py` (append `plot_swimlanes`)
- Test: `tests/test_prof2_swimlanes.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prof2_swimlanes.py`:

```python
from profiling.perpair import PairRecord
from profiling.swimlanes import schedule_lanes
from profiling.plots import plot_swimlanes


def _rec(pid, dur):
    return PairRecord(pid, pid, pid + 1, 50, 50, 100, dur)


def test_schedule_lanes_packs_into_n_lanes_greedily():
    # 4 pairs, 2 lanes: durations 10, 20, 5, 5
    records = [_rec(0, 10.0), _rec(1, 20.0), _rec(2, 5.0), _rec(3, 5.0)]
    placed = schedule_lanes(records, n_lanes=2)

    # one entry per pair
    assert len(placed) == 4
    # first two pairs seed the two lanes at t=0
    starts = {p["pair_id"]: p["start_ms"] for p in placed}
    assert starts[0] == 0.0
    assert starts[1] == 0.0
    # pair 2 goes to the lane that frees first (lane of pair 0, free at 10)
    assert starts[2] == 10.0
    # pair 3 goes to next-free lane (pair 2 ends at 15 vs lane1 free at 20) -> 15
    assert starts[3] == 15.0
    # lanes are within range
    assert all(0 <= p["lane"] < 2 for p in placed)


def test_plot_swimlanes_writes_file(tmp_path):
    records = [_rec(i, 5.0 + i) for i in range(6)]
    out = tmp_path / "swimlanes.png"
    plot_swimlanes(records, str(out), n_lanes=3, labels={i: f"p{i}" for i in range(6)})
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prof2_swimlanes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'profiling.swimlanes'`.

- [ ] **Step 3: Write minimal implementation**

Create `profiling/swimlanes.py`:

```python
"""Reconstruct a concurrent schedule from serially-measured pair durations.

Pairs were measured one at a time (max_workers=1). To visualize how they would
overlap across N real workers, greedily pack each pair (in visit order) onto the
lane that frees earliest. This is a RECONSTRUCTION, not a measured concurrency.
"""
from typing import Dict, List

from profiling.perpair import PairRecord


def schedule_lanes(records: List[PairRecord], n_lanes: int = 8) -> List[Dict]:
    """Greedy earliest-free-lane packing. Returns one dict per pair:
    {pair_id, i, j, lane, start_ms, end_ms}.
    """
    lane_free = [0.0] * max(1, n_lanes)
    placed = []
    for r in records:
        lane = min(range(len(lane_free)), key=lambda l: lane_free[l])
        start = lane_free[lane]
        end = start + r.duration_ms
        lane_free[lane] = end
        placed.append(
            {"pair_id": r.pair_id, "i": r.i, "j": r.j, "lane": lane, "start_ms": start, "end_ms": end}
        )
    return placed
```

Append to `profiling/plots.py` (reuse the existing `_PALETTE`):

```python
from profiling.swimlanes import schedule_lanes  # noqa: E402  (place with top imports)


def plot_swimlanes(records, out_path, n_lanes=8, labels=None):
    """Reconstructed registration concurrency: each pair a bar on a worker lane."""
    placed = schedule_lanes(records, n_lanes=n_lanes)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for p in placed:
        x = p["start_ms"] / 1000.0
        w = (p["end_ms"] - p["start_ms"]) / 1000.0
        ax.barh(
            p["lane"],
            w,
            left=x,
            height=0.7,
            color=_PALETTE[p["pair_id"] % len(_PALETTE)],
            edgecolor="white",
            linewidth=0.3,
        )
    ax.set_xlabel("reconstructed time (s)")
    ax.set_ylabel("worker lane")
    ax.set_yticks(range(n_lanes))
    ax.set_title(f"Registration pairs across {n_lanes} workers (reconstructed)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
```

(The `labels` parameter is accepted for future per-bar annotation; it is not required to be drawn for Phase 2 — keep the signature so callers can pass it.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prof2_swimlanes.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Black + commit**

Run: `black --line-length 100 profiling/swimlanes.py profiling/plots.py tests/test_prof2_swimlanes.py`
```bash
git add profiling/swimlanes.py profiling/plots.py tests/test_prof2_swimlanes.py
git commit -m "feat(profiling): swimlane scheduler and figure"
```

---

## Task 6: Variability + scan-pattern figures

**Files:**
- Modify: `profiling/plots.py` (append `plot_pair_variability`, `plot_scan_pattern`)
- Test: `tests/test_prof2_plots.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prof2_plots.py`:

```python
from profiling.perpair import PairRecord
from profiling.plots import plot_pair_variability, plot_scan_pattern


def test_plot_pair_variability_writes_file(tmp_path):
    records = [PairRecord(i, i, i + 1, 50, 50, 100 + i, 5.0 + i) for i in range(8)]
    out = tmp_path / "variability.png"
    plot_pair_variability(records, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_scan_pattern_writes_file(tmp_path):
    # 2x3 grid in raster index order
    grid = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1), 5: (1, 2)}
    out = tmp_path / "scan.png"
    plot_scan_pattern(grid, str(out), pattern="raster")
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prof2_plots.py -v`
Expected: FAIL with `ImportError: cannot import name 'plot_pair_variability'`.

- [ ] **Step 3: Write minimal implementation**

Append to `profiling/plots.py`:

```python
def plot_pair_variability(records, out_path):
    """Per-pair duration distribution with mean and CV annotated."""
    durations = [r.duration_ms for r in records]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if durations:
        n = len(durations)
        mean = sum(durations) / n
        var = sum((d - mean) ** 2 for d in durations) / n
        std = var**0.5
        cv = (std / mean) if mean else 0.0
        ax.hist(durations, bins=min(20, max(5, n // 3)), color="#26a69a", edgecolor="white")
        ax.axvline(mean, color="#263238", linestyle="--", linewidth=1.5, label=f"mean {mean:.1f} ms")
        ax.set_title(f"Per-pair registration duration (CV = {cv:.2f})")
        ax.legend(fontsize=8)
    ax.set_xlabel("duration (ms)")
    ax.set_ylabel("pairs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scan_pattern(grid, out_path, pattern="unknown"):
    """Scatter tiles at (col, row) and connect them in acquisition (index) order."""
    items = sorted(grid.items())  # by tile index
    cols = [c for _, (r, c) in items]
    rows = [r for _, (r, c) in items]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cols, rows, color="#90a4ae", linewidth=1, zorder=1)
    ax.scatter(cols, rows, c=range(len(items)), cmap="viridis", s=60, zorder=2)
    for order, (idx, (r, c)) in enumerate(items):
        ax.annotate(str(order), (c, r), fontsize=6, ha="center", va="center", color="white")
    ax.set_xlabel("grid column")
    ax.set_ylabel("grid row")
    ax.invert_yaxis()  # row 0 at top
    ax.set_title(f"Tile acquisition order — scan pattern: {pattern}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prof2_plots.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Black + commit**

Run: `black --line-length 100 profiling/plots.py tests/test_prof2_plots.py`
```bash
git add profiling/plots.py tests/test_prof2_plots.py
git commit -m "feat(profiling): variability and scan-pattern figures"
```

---

## Task 7: Run B harness

**Files:**
- Modify: `profiling/harness.py` (append `PairProfileResult` + `profile_registration_perpair`)
- Test: `tests/test_prof2_harness.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prof2_harness.py`:

```python
import os
import pytest
from profiling.harness import profile_registration_perpair, PairProfileResult

DATASET = os.path.expanduser(
    "~/Cephla/Data/test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy"
)


def test_pair_profile_result_fields_exist():
    # construct directly to lock the field names downstream code depends on
    r = PairProfileResult(records=[], tile_positions=[], tile_identifiers=[], tile_shape=(0, 0))
    assert r.records == []
    assert r.tile_positions == []
    assert r.tile_identifiers == []
    assert r.tile_shape == (0, 0)


@pytest.mark.skipif(not os.path.isdir(DATASET), reason="profiling dataset not present")
def test_profile_registration_perpair_smoke():
    result = profile_registration_perpair(DATASET, region="manual0")
    assert len(result.records) > 0
    assert len(result.tile_positions) == len(result.tile_identifiers)
    # every recorded pair indexes valid tiles
    n = len(result.tile_positions)
    assert all(0 <= rec.i < n and 0 <= rec.j < n for rec in result.records)
    assert all(rec.patch_bytes_total > 0 for rec in result.records)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prof2_harness.py -v`
Expected: FAIL with `ImportError: cannot import name 'profile_registration_perpair'`.

- [ ] **Step 3: Write minimal implementation**

Append to `profiling/harness.py` (add `from profiling.perpair import PairRecord, PairRecorder` to the top imports):

```python
@dataclass
class PairProfileResult:
    records: List["PairRecord"]
    tile_positions: List
    tile_identifiers: List
    tile_shape: tuple


def profile_registration_perpair(dataset: str, region: str = "manual0") -> PairProfileResult:
    """Run B: serialized registration with per-pair recording (non-invasive)."""
    from tilefusion import TileFusion

    metrics_name = f"profile_perpair_metrics_{region}.json"
    tf = TileFusion(dataset, region=region, metrics_filename=metrics_name, max_workers=1)

    metrics_path = Path(dataset).parent / metrics_name
    metrics_path.unlink(missing_ok=True)

    with PairRecorder() as rec:
        tf.refine_tile_positions_with_cross_correlation(
            downsample_factors=tf.downsample_factors,
            ch_idx=tf.channel_to_use,
            threshold=tf.threshold,
        )

    return PairProfileResult(
        records=list(rec.records),
        tile_positions=list(tf._tile_positions),
        tile_identifiers=list(tf._tile_identifiers),
        tile_shape=(tf.Y, tf.X),
    )
```

Also add `PairRecord` to the typing import usage. Put `from profiling.perpair import PairRecord, PairRecorder` near the other `from profiling...` imports at the top of `harness.py`. (The `List["PairRecord"]` forward-ref in the dataclass is fine; you may use a plain `List` if simpler — keep consistent with repo style.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prof2_harness.py -v`
Expected: `test_pair_profile_result_fields_exist` PASS; smoke test PASS if dataset present (it runs registration only, ~seconds–tens of seconds), else SKIPPED. If the smoke test ERRORS (e.g. a signature mismatch on `refine_tile_positions_with_cross_correlation`), DO NOT fake it — report DONE_WITH_CONCERNS with the real traceback; the field-name test is enough to commit.

- [ ] **Step 5: Black + commit**

Run: `black --line-length 100 profiling/harness.py tests/test_prof2_harness.py`
```bash
git add profiling/harness.py tests/test_prof2_harness.py
git commit -m "feat(profiling): Run B per-pair registration harness"
```

---

## Task 8: CLI `--perpair` wiring + real QC

**Files:**
- Modify: `profiling/cli.py` (add `--perpair`; when set, run Run B and write per-pair artifacts)
- Test: `tests/test_prof2_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prof2_cli.py`:

```python
from profiling.cli import build_parser


def test_parser_has_perpair_flag_default_false():
    args = build_parser().parse_args(["/some/dataset"])
    assert args.perpair is False


def test_parser_perpair_flag_sets_true():
    args = build_parser().parse_args(["/some/dataset", "--perpair"])
    assert args.perpair is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prof2_cli.py -v`
Expected: FAIL (`AttributeError: 'Namespace' object has no attribute 'perpair'`).

- [ ] **Step 3: Write minimal implementation**

In `profiling/cli.py`:

(a) add the flag in `build_parser()` after the `--top-k` line:

```python
    p.add_argument(
        "--perpair",
        action="store_true",
        help="Also run serialized per-pair registration analysis (Phase 2)",
    )
```

(b) add the imports at the top:

```python
from profiling.harness import profile_dataset, profile_registration_perpair
from profiling.tiles import build_grid, infer_scan_pattern
from profiling.variability import compute_pair_stats
from profiling.record import write_timeline_csv, write_functions_csv, write_pairs_csv
from profiling.plots import (
    plot_timeline,
    plot_function_lines,
    plot_pareto,
    plot_swimlanes,
    plot_pair_variability,
    plot_scan_pattern,
)
```

(c) in `main()`, after the existing Phase 1 outputs and the `Top 2 ...` print, append:

```python
    if args.perpair:
        pair = profile_registration_perpair(args.dataset, region=args.region)
        grid = build_grid(pair.tile_positions)
        sequence = [grid[idx] for idx in range(len(pair.tile_positions))]
        pattern = infer_scan_pattern(sequence)
        stats = compute_pair_stats(pair.records)

        write_pairs_csv(
            os.path.join(args.out, "pairs.csv"),
            pair.records,
            grid,
            pair.tile_identifiers,
        )
        plot_swimlanes(pair.records, os.path.join(args.out, "swimlanes.png"))
        plot_pair_variability(pair.records, os.path.join(args.out, "variability.png"))
        plot_scan_pattern(grid, os.path.join(args.out, "scan_pattern.png"), pattern=pattern)

        print(f"Per-pair: {stats['n_pairs']} pairs, scan pattern = {pattern}")
        if stats.get("duration_ms"):
            print(f"Per-pair duration CV = {stats['duration_ms']['cv']:.2f}")
```

- [ ] **Step 4: Run test + full suites**

Run: `pytest tests/test_prof2_cli.py tests/test_prof_cli.py -v`
Expected: PASS (new flag tests + Phase 1 CLI test unchanged).

Then run all Phase 2 fast tests:
`pytest tests/test_prof2_perpair.py tests/test_prof2_tiles.py tests/test_prof2_variability.py tests/test_prof2_record.py tests/test_prof2_swimlanes.py tests/test_prof2_plots.py tests/test_prof2_cli.py "tests/test_prof2_harness.py::test_pair_profile_result_fields_exist" -v`
Expected: all pass.

- [ ] **Step 5: REAL END-TO-END QC RUN**

The dataset is present. Run:
```bash
python -m profiling.cli "$HOME/Cephla/Data/test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy" --out profile_out --region manual0 --perpair
```
Expected: the Phase 1 lines, plus `Per-pair: N pairs, scan pattern = raster` and a `Per-pair duration CV = ...` line; and `profile_out/` now also contains `pairs.csv`, `swimlanes.png`, `variability.png`, `scan_pattern.png`. Verify they exist & are non-empty (`ls -la profile_out/`) and show the first ~12 lines of `pairs.csv` + the printed per-pair summary in your report. If the run errors, report honestly (DONE_WITH_CONCERNS / BLOCKED with the traceback) — do not fake it.

Do NOT commit `profile_out/` (already gitignored).

- [ ] **Step 6: Black + commit**

Run: `black --line-length 100 profiling/cli.py tests/test_prof2_cli.py`
```bash
git add profiling/cli.py tests/test_prof2_cli.py
git commit -m "feat(profiling): CLI --perpair wiring for Phase 2 analysis"
```

---

## Phase 2 QC checklist (review before merge)

- [ ] All `tests/test_prof2_*.py` pass; Phase 1 tests still pass.
- [ ] `swimlanes.png` shows pairs packed across 8 lanes (reconstructed), staggered finishes visible.
- [ ] `scan_pattern.png` shows the raster traversal over the FOV grid (numbered order).
- [ ] `variability.png` shows the per-pair duration distribution with CV annotation.
- [ ] `pairs.csv` is plausible (per-pair tile labels, grid row/col, patch bytes, durations).
- [ ] No changes under `src/tilefusion/` (`git diff --stat main..HEAD -- src/` is empty).

---

## Self-Review notes (completed by plan author)

- **Spec coverage:** Phase 2 spec items → tasks: per-pair recording (T1), grid/scan (T2), CV (T3), `pairs.csv` (T4), swimlanes (T5), variability + scan-pattern figures (T6), Run B harness (T7), CLI wiring + real QC (T8). Non-invasive: only runtime monkeypatch of `tilefusion.core.register_pair_worker`; `git diff src/` stays empty (checklist item).
- **Type consistency:** `PairRecord(pair_id, i, j, patch_i_bytes, patch_j_bytes, patch_bytes_total, duration_ms)` used identically across perpair/variability/record/swimlanes/plots/harness. `build_grid` returns `{idx: (row, col)}`; `tile_label(identifiers, grid, idx)`; `compute_pair_stats` keys `n_pairs`/`duration_ms`/`patch_bytes_total` each with mean/std/cv/min/max; `PairProfileResult(records, tile_positions, tile_identifiers, tile_shape)`.
- **Non-placeholder:** every step has complete code + exact commands.
- **Reuse:** swimlane/variability/scan figures live in `plots.py` (Agg already configured) alongside Phase 1 figures, so a single `--perpair` run regenerates the full set for the eventual before/after (Phase 5).
