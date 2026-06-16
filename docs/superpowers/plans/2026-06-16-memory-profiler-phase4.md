# Memory Profiler — Phase 4 (Fuse Memory Optimization) + Conclusion

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Cut the peak memory of the chunked fuse path (96% of the footprint) with output-identical changes, prove the win with before/after profiling, and produce the Phase 5 conclusion (timeline overlay + metric bullets).

**Architecture:** Optimize `_fuse_tiles_chunked_plane` in `src/tilefusion/core.py`: (1) in-place masked divide, (2) reuse fused/weight buffers across blocks, (3) honest block sizing so `ram_fraction` is a real, safe cap. Guard correctness with a new chunked-vs-full output-equivalence test plus the existing fusion suite. Add a small `profiling/conclude.py` that overlays before/after timelines and prints improvement metrics.

**Tech Stack:** Python 3.9+, numpy, tensorstore, pytest, matplotlib (Agg). Branch: `feat/fuse-memory-opt`.

---

## Background the engineer must know

- **This phase EDITS production code** (`src/tilefusion/core.py`) — unlike Phases 1–2. The fused output MUST stay numerically identical. The guard is Task 1's equivalence test + the existing `tests/test_fusion.py` and `tests/test_integration.py`.
- **The target** is `_fuse_tiles_chunked_plane` (`core.py:1062`). Its per-block loop currently does:
  ```python
  fused_block = np.zeros((C, bh, bw), dtype=np.float32)   # line 1115
  weight_sum = np.zeros_like(fused_block)                 # line 1116
  for t_idx in overlapping:
      ... fused_block[c, ...] += tile * w2d ; weight_sum[c, ...] += w2d
  mask = weight_sum > 0                                   # line 1148
  fused_block[mask] /= weight_sum[mask]                   # line 1149  (2 big temporaries)
  self.fused_ts[...].write(fused_block.astype(np.uint16)).result()
  del fused_block, weight_sum
  ```
- **Why it's the hotspot:** per block it allocates two `float32` block buffers + a bool mask + two 1-D fancy-index temporaries (`fused_block[mask]`, `weight_sum[mask]`, each up to `C*bh*bw` elements) + a uint16 cast copy. `functions.csv`: `_fuse_tiles_chunked_plane` ≈61% + `zeros_like` ≈35%.
- **Block sizing** (`core.py:1070-1075`): `bytes_per_pixel = 4 * 2 * self.channels` only counts the two float32 buffers — it ignores the mask, the fancy-index temporaries, and the uint16 cast. So the real peak overshoots `ram_fraction` (~4 GB peak vs an intended 0.4·available). Fixing this divisor makes the cap honest and safe.
- **Baseline (fixed "before"):** `docs/superpowers/baselines/pre-opt/` — peak ≈ 3971 MB, mean ≈ 2225 MB (3 runs).
- **Dataset** (smoke/QC): `~/Cephla/Data/test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy` (uses the chunked path — confirmed by the profile).
- CI enforces only `black --line-length 100`. Keep `from typing import ...` style.

---

## Task 1: Chunked-vs-full output-equivalence regression test (the guard)

**Files:**
- Test: `tests/test_fuse_equivalence.py`

This test pins current behavior: the chunked path must produce the same fused output as the full path. It MUST pass on the current (unoptimized) code before any optimization.

- [ ] **Step 1: Write the test**

Create `tests/test_fuse_equivalence.py`:

```python
"""Chunked fuse must equal full-plane fuse (the optimization's correctness guard)."""
import numpy as np
import pytest

tilefusion = pytest.importorskip("tilefusion")
from tilefusion import TileFusion  # noqa: E402


def _read_scale0(path):
    import tensorstore as ts

    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(path / "scale0" / "image")},
    }
    return ts.open(spec).result().read().result()


def _build_tf(tmp_path, out_name):
    """Synthetic 2x2 grid of overlapping single-channel tiles -> a TileFusion."""
    from tilefusion.io import individual_tiffs  # noqa: F401

    # Use the public TileFusion on a tiny synthetic individual-tiffs dataset.
    # Build 4 tiles (64x64) on a 2x2 grid with 16px overlap, constant values.
    import tifffile

    folder = tmp_path / "data"
    folder.mkdir()
    vals = [100, 150, 200, 250]
    positions = [(0, 0), (0, 48), (48, 0), (48, 48)]  # microns == pixels (pixel_size=1)
    rows = ["filename,x (um),y (um)"]
    for i, ((py, px), v) in enumerate(zip(positions, vals)):
        tifffile.imwrite(folder / f"tile_{i}.tif", np.full((64, 64), v, dtype=np.uint16))
        rows.append(f"tile_{i}.tif,{px},{py}")
    (folder / "coordinates.csv").write_text("\n".join(rows))

    return TileFusion(folder, output_path=tmp_path / out_name)


def test_chunked_equals_full_plane(tmp_path, monkeypatch):
    # Two independent runs of the same dataset: one forced to chunked, one full.
    tf_full = _build_tf(tmp_path, "full.ome.zarr")
    tf_full.refine_tile_positions_with_cross_correlation()
    tf_full.optimize_shifts(method="TWO_ROUND_ITERATIVE")
    tf_full._tile_positions = [
        tuple(np.array(p) + o * np.array(tf_full.pixel_size))
        for p, o in zip(tf_full._tile_positions, tf_full.global_offsets)
    ]
    tf_full._compute_fused_image_space()
    tf_full._pad_to_chunk_multiple()
    scale0 = tf_full.output_path / "scale0" / "image"
    scale0.parent.mkdir(parents=True, exist_ok=True)
    tf_full._create_fused_tensorstore(output_path=scale0)
    tf_full._fuse_tiles_full_plane()
    full_out = np.asarray(_read_scale0(tf_full.output_path))

    tf_chunk = _build_tf(tmp_path, "chunk.ome.zarr")
    tf_chunk.refine_tile_positions_with_cross_correlation()
    tf_chunk.optimize_shifts(method="TWO_ROUND_ITERATIVE")
    tf_chunk._tile_positions = [
        tuple(np.array(p) + o * np.array(tf_chunk.pixel_size))
        for p, o in zip(tf_chunk._tile_positions, tf_chunk.global_offsets)
    ]
    tf_chunk._compute_fused_image_space()
    tf_chunk._pad_to_chunk_multiple()
    scale0c = tf_chunk.output_path / "scale0" / "image"
    scale0c.parent.mkdir(parents=True, exist_ok=True)
    tf_chunk._create_fused_tensorstore(output_path=scale0c)
    # Force several small blocks by monkeypatching the chunked sizing knobs:
    # call the chunked plane directly with a tiny ram_fraction so block_size < image.
    tf_chunk._fuse_tiles_chunked_plane(ram_fraction=1e-6)
    chunk_out = np.asarray(_read_scale0(tf_chunk.output_path))

    np.testing.assert_array_equal(chunk_out, full_out)
```

- [ ] **Step 2: Run it on current code to confirm it PASSES (this is the guard, not a red test)**

Run: `pytest tests/test_fuse_equivalence.py -v`
Expected: PASS on the unoptimized code. (If it ERRORS due to a dataset/loader detail — e.g. `coordinates.csv` columns or `individual_tiffs` format expectations — STOP and report DONE_WITH_CONCERNS with the real error; the controller will adjust the synthetic-dataset construction. Do NOT weaken the equality assertion.)

- [ ] **Step 3: Black + commit**

Run: `black --line-length 100 tests/test_fuse_equivalence.py`
```bash
git add tests/test_fuse_equivalence.py
git commit -m "test(fuse): pin chunked == full-plane output equivalence (opt guard)"
```

---

## Task 2: Optimize the chunked fuse plane (output-identical)

**Files:**
- Modify: `src/tilefusion/core.py` (`_fuse_tiles_chunked_plane`, ~1062-1161)

Apply three coupled changes. The Task 1 test + existing fusion suite are the guard.

- [ ] **Step 1: Implement the optimization**

In `_fuse_tiles_chunked_plane`:

(a) **Honest block sizing.** Replace:
```python
        bytes_per_pixel = 4 * 2 * self.channels
```
with (account for fused f32 + weight f32 + bool mask + uint16 cast, with margin):
```python
        # Per output pixel-channel, the block loop holds: fused (f32=4) + weight
        # (f32=4) + mask (bool=1) + uint16 cast for the write (2) ~= 11 bytes.
        # Round up to 12 for margin so ram_fraction is a real, safe ceiling.
        bytes_per_pixel = 12 * self.channels
```

(b) **Allocate buffers once, reuse across blocks.** Immediately before the
`for block_y in range(0, pad_Y, block_size):` loop, add:
```python
        # Reusable per-block accumulators (sized to the largest block); we zero
        # and view sub-regions instead of re-allocating each block.
        max_bh = min(block_size, pad_Y)
        max_bw = min(block_size, pad_X)
        fused_buf = np.zeros((C, max_bh, max_bw), dtype=np.float32)
        weight_buf = np.zeros((C, max_bh, max_bw), dtype=np.float32)
```

(c) **Per-block: use views + zero them; in-place masked divide.** Replace the
block body from `fused_block = np.zeros(...)` through `del fused_block, weight_sum`:
```python
                fused_block = fused_buf[:, :bh, :bw]
                weight_sum = weight_buf[:, :bh, :bw]
                fused_block[...] = 0.0
                weight_sum[...] = 0.0

                desc = f"block {block_idx}/{total_blocks}"
                iterator = (
                    tqdm(overlapping, desc=desc, leave=False) if show_progress else overlapping
                )
                for t_idx in iterator:
                    tile_all = self._read_tile(t_idx, z_level=z_level, time_idx=time_idx)

                    if tile_all.shape[0] == 1 and C > 1:
                        tile_all = np.broadcast_to(
                            tile_all, (C, tile_all.shape[1], tile_all.shape[2])
                        )

                    ty0, ty1, tx0, tx1 = tile_bounds[t_idx]

                    oy0 = max(ty0, block_y) - block_y
                    oy1 = min(ty1, by_end) - block_y
                    ox0 = max(tx0, block_x) - block_x
                    ox1 = min(tx1, bx_end) - block_x

                    sy0 = max(block_y - ty0, 0)
                    sy1 = sy0 + (oy1 - oy0)
                    sx0 = max(block_x - tx0, 0)
                    sx1 = sx0 + (ox1 - ox0)

                    w2d = self.y_profile[sy0:sy1, None] * self.x_profile[None, sx0:sx1]

                    for c in range(C):
                        fused_block[c, oy0:oy1, ox0:ox1] += tile_all[c, sy0:sy1, sx0:sx1] * w2d
                        weight_sum[c, oy0:oy1, ox0:ox1] += w2d

                # In-place masked divide: avoids the large fancy-index temporaries
                # that fused_block[mask] /= weight_sum[mask] would allocate.
                mask = weight_sum > 0
                np.divide(fused_block, weight_sum, out=fused_block, where=mask)

                # Write to 5D output: (T, C, Z, Y, X)
                self.fused_ts[time_idx, :, z_level, block_y:by_end, block_x:bx_end].write(
                    fused_block.astype(np.uint16)
                ).result()
```
(Note: remove the old `del fused_block, weight_sum` — the buffers are now reused.
After the `for block_y` loops complete, you may add `del fused_buf, weight_buf`
before the final `gc.collect()`.)

- [ ] **Step 2: Run the equivalence guard + full fusion suite**

Run: `pytest tests/test_fuse_equivalence.py tests/test_fusion.py tests/test_integration.py -v`
Expected: ALL pass. If any fail, the optimization changed behavior — fix until green. Do NOT modify the tests to pass.

- [ ] **Step 3: Black + commit**

Run: `black --line-length 100 src/tilefusion/core.py`
```bash
git add src/tilefusion/core.py
git commit -m "perf(fuse): reuse block buffers + in-place divide + honest sizing"
```

- [ ] **Step 4: Report** the diff and test results. If the equivalence test or fusion suite needed any judgment calls, surface them.

---

## Task 3: Timeline-overlay figure (before vs after)

**Files:**
- Modify: `profiling/plots.py` (append `plot_timeline_overlay`)
- Test: `tests/test_prof_overlay.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prof_overlay.py`:

```python
from profiling.plots import plot_timeline_overlay


def test_plot_timeline_overlay_writes_file(tmp_path):
    before = [(0.0, 1000.0), (1.0, 3971.0), (2.0, 1200.0)]
    after = [(0.0, 800.0), (1.0, 2200.0), (2.0, 900.0)]
    out = tmp_path / "overlay.png"
    plot_timeline_overlay(before, after, str(out), before_peak=3971.0, after_peak=2200.0)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run it — expect FAIL** (`ImportError: cannot import name 'plot_timeline_overlay'`).

Run: `pytest tests/test_prof_overlay.py -v`

- [ ] **Step 3: Implement** — append to `profiling/plots.py`:

```python
def plot_timeline_overlay(before, after, out_path, before_peak=None, after_peak=None):
    """Overlay two RSS-vs-time series (before vs after optimization).

    `before`/`after` are lists of (t_s, rss_mb).
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if before:
        bx, by = zip(*before)
        ax.plot(bx, by, color="#b0676f", linewidth=1.8, label="before")
    if after:
        ax_, ay = zip(*after)
        ax.plot(ax_, ay, color="#26a69a", linewidth=1.8, label="after")
    if before_peak is not None:
        ax.axhline(before_peak, color="#b0676f", linestyle=":", linewidth=1)
    if after_peak is not None:
        ax.axhline(after_peak, color="#26a69a", linestyle=":", linewidth=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory footprint: before vs after optimization")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Run — expect PASS.** `pytest tests/test_prof_overlay.py -v`

- [ ] **Step 5: Black + commit**

Run: `black --line-length 100 profiling/plots.py tests/test_prof_overlay.py`
```bash
git add profiling/plots.py tests/test_prof_overlay.py
git commit -m "feat(profiling): before/after timeline-overlay figure"
```

---

## Task 4: Conclusion CLI (overlay + metric bullets)

**Files:**
- Create: `profiling/conclude.py`
- Test: `tests/test_prof_conclude.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prof_conclude.py`:

```python
import csv
from profiling.conclude import _read_timeline, _metrics, build_parser


def _write_timeline(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ms", "rss_mb", "stage"])
        for t_ms, rss in rows:
            w.writerow([t_ms, rss, "Fuse"])


def test_read_timeline_and_metrics(tmp_path):
    p = tmp_path / "timeline.csv"
    _write_timeline(p, [(0, 1000.0), (1000, 4000.0), (2000, 1000.0)])
    series = _read_timeline(str(p))
    assert series[1] == (1.0, 4000.0)  # (t_s, rss_mb)

    m = _metrics([(0.0, 1000.0), (1.0, 4000.0)], [(0.0, 800.0), (1.0, 2000.0)])
    assert m["before_peak"] == 4000.0
    assert m["after_peak"] == 2000.0
    assert abs(m["peak_pct_improvement"] - 50.0) < 1e-9


def test_parser_requires_before_and_after():
    args = build_parser().parse_args(["--before", "b/timeline.csv", "--after", "a/timeline.csv"])
    assert args.before.endswith("timeline.csv")
    assert args.after.endswith("timeline.csv")
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError: No module named 'profiling.conclude'`).

- [ ] **Step 3: Implement** — create `profiling/conclude.py`:

```python
"""Phase 5 conclusion: overlay before/after timelines and print metric bullets.

Usage:
  python -m profiling.conclude --before <dir-or-csv> --after <dir-or-csv> --out <dir>
"""
import argparse
import csv
import os

from profiling.plots import plot_timeline_overlay


def _read_timeline(path):
    """Read a timeline.csv into a list of (t_s, rss_mb)."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [(float(r["t_ms"]) / 1000.0, float(r["rss_mb"])) for r in rows]


def _peak(series):
    return max((rss for _, rss in series), default=0.0)


def _mean(series):
    return (sum(rss for _, rss in series) / len(series)) if series else 0.0


def _metrics(before, after):
    bp, ap = _peak(before), _peak(after)
    bm, am = _mean(before), _mean(after)
    return {
        "before_peak": bp,
        "after_peak": ap,
        "before_mean": bm,
        "after_mean": am,
        "peak_pct_improvement": (100.0 * (bp - ap) / bp) if bp else 0.0,
        "mean_pct_improvement": (100.0 * (bm - am) / bm) if bm else 0.0,
    }


def _resolve(path):
    """Accept a directory (use timeline.csv inside) or a direct CSV path."""
    return os.path.join(path, "timeline.csv") if os.path.isdir(path) else path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Before/after memory conclusion.")
    p.add_argument("--before", required=True, help="Baseline dir or timeline.csv")
    p.add_argument("--after", required=True, help="Post-opt dir or timeline.csv")
    p.add_argument("--out", default="profile_out", help="Output directory")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    before = _read_timeline(_resolve(args.before))
    after = _read_timeline(_resolve(args.after))
    m = _metrics(before, after)

    plot_timeline_overlay(
        before,
        after,
        os.path.join(args.out, "overlay.png"),
        before_peak=m["before_peak"],
        after_peak=m["after_peak"],
    )

    print("## Memory footprint — before vs after")
    print(f"- Peak: {m['before_peak']:.0f} MB -> {m['after_peak']:.0f} MB "
          f"({m['peak_pct_improvement']:.1f}% lower)")
    print(f"- Mean: {m['before_mean']:.0f} MB -> {m['after_mean']:.0f} MB "
          f"({m['mean_pct_improvement']:.1f}% lower)")
    print(f"- Overlay figure: {os.path.join(args.out, 'overlay.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run — expect PASS.** `pytest tests/test_prof_conclude.py -v`

- [ ] **Step 5: Black + commit**

Run: `black --line-length 100 profiling/conclude.py tests/test_prof_conclude.py`
```bash
git add profiling/conclude.py tests/test_prof_conclude.py
git commit -m "feat(profiling): conclusion CLI (overlay + improvement metrics)"
```

---

## Task 5: Real before/after QC + conclusion artifacts

**Files:** none (runs the tools; produces artifacts under `profile_out/` which is gitignored)

- [ ] **Step 1: Run the optimized profiler ("after"), 3 runs, capture peak/mean**

```bash
DS="$HOME/Cephla/Data/test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy"
for run in 1 2 3; do
  python -m profiling.cli "$DS" --out profile_out_after --region manual0 --perpair >/dev/null 2>&1
  python - <<'PY'
import csv
r=list(csv.DictReader(open("profile_out_after/timeline.csv")))
v=[float(x["rss_mb"]) for x in r]
print(f"after peak/mean = {max(v):.0f} / {sum(v)/len(v):.0f} MB")
PY
done
```
Record the three peak/mean numbers in your report.

- [ ] **Step 2: Generate the conclusion** (overlay + metrics) against the committed baseline:

```bash
python -m profiling.conclude \
  --before docs/superpowers/baselines/pre-opt \
  --after profile_out_after \
  --out profile_out_after
```
Paste the printed metric bullets and confirm `profile_out_after/overlay.png` exists & is non-empty.

- [ ] **Step 3: Verify the fused OUTPUT is unchanged by the optimization** (correctness in production, not just unit tests). Confirm `tests/test_fuse_equivalence.py tests/test_fusion.py tests/test_integration.py` all pass:

```bash
pytest tests/test_fuse_equivalence.py tests/test_fusion.py tests/test_integration.py -v
```

- [ ] **Step 4: Report** the before→after peak/mean, the % improvement, and whether peak dropped meaningfully. If peak did NOT improve (e.g. the honest-sizing change shrank blocks but temporaries still dominate), report DONE_WITH_CONCERNS with the numbers — the controller may iterate on the optimization. (Do not commit `profile_out_after/`.)

---

## Phase 4 QC checklist (before merge)

- [ ] `tests/test_fuse_equivalence.py` passes on BOTH pre- and post-optimization code.
- [ ] `tests/test_fusion.py` + `tests/test_integration.py` pass (output unchanged).
- [ ] All `profiling` unit tests pass; `black --check` clean.
- [ ] Measured peak RSS dropped vs baseline (3971 MB) by a meaningful margin; numbers recorded.
- [ ] `overlay.png` + printed metric bullets produced for the Notion conclusion.

---

## Self-Review notes (plan author)

- **Correctness-first:** Task 1 establishes the chunked==full guard BEFORE any production edit; Task 2's optimization is purely a memory-layout change (same arithmetic: accumulate then divide where weight>0), so the guard + fusion suite fully cover it.
- **Safety/cap:** honest `bytes_per_pixel` makes `ram_fraction` a real ceiling (blocks sized so fused+weight+mask+cast fit) — addresses "we cannot afford crashing the computer."
- **Scope:** fuse-only (the 96%); registration RAM-cap throughput tweak deferred (would muddy the memory before/after).
- **Phase 5 folded in:** Tasks 3–5 are the conclusion (overlay + metrics), per the decision that Phase 5 is the conclusion of Phase 4.
- **No placeholders:** every step has complete code + exact commands.
