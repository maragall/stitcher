# Stitcher Memory Footprint Profiler — Design

**Date:** 2026-06-15
**Branch:** `feat/memory-profiler`
**Status:** Approved design, pre-implementation

## Goal

Quantify and visualize the memory footprint of the TileFusion stitching +
registration pipeline so we can (a) show the CTO where memory goes, and
(b) pinpoint the functions / lines / tile-pairs that explain most of the
footprint, in order to optimize space complexity and tune throughput.

This is a **measurement and visualization tool**, run headless against a real
dataset. It does not change pipeline behavior.

## Non-goals

- Changing the acquisition scan pattern or adding metadata-aware traversal
  (deferred — future work).
- A time-lapse video of memory. Static, Notion-embeddable figures only.
- CPU-register / assembly-level profiling (not meaningful for interpreted
  Python; see "Measurement reality").
- Cross-platform native memory profiling via `memray` (Mac/Linux only).

## Target pipeline (what we profile)

Orchestrator: `TileFusion` in `src/tilefusion/core.py`. Stages, in order
(a strict sequential state machine):

1. **Read** — load tile metadata + tile pixel data
2. **Register** — pairwise cross-correlation of adjacent FOVs
   (`_register_parallel` → `register_pair_worker`), 8 worker threads by default
3. **Optimize** — global shift solve (`optimize_shifts`, single-threaded)
4. **Fuse** — write fused image (`_fuse_tiles_*`), thread pool + tensorstore
5. **Write** — multiscale / NGFF output

Concurrency model (verified in code): **multi-threaded** (`ThreadPoolExecutor`,
default 8 workers), **one process, one shared RSS**. Not multiprocessing.
Within a parallel stage the same function runs on independent work items
concurrently (e.g. 8 threads all in `register_pair_worker` on different
tile-pairs). Stages themselves are sequential and dependent.

## Test dataset

`~/Cephla/Data/test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy`

- Format: **OME-TIFF tiles** (`ome_tiff/manual{region}_{fov}.ome.tiff`).
- 2 regions: `manual0` (27 FOVs), `manual1` (28 FOVs).
- Grid: ~5-column **raster, left-to-right, row-by-row** (confirmed from
  `coordinates.csv`; not serpentine).
- Per tile: 10 z-levels (dz 1.5 µm), 1 timepoint, multiple channels (axes
  `ZCYX`). Registration aligns adjacent FOVs on a chosen channel/z (~90 pairs).

## Measurement reality (the constraints that shape the design)

- **`psutil` RSS** = total process memory at an instant. Cross-platform
  identical (Linux/Mac/Windows). Captures everything incl. native/numpy.
  → Source of truth for the **total** timeline.
- **`tracemalloc`** = Python-level allocations attributed by code location
  (function/line), thread-correct. Cross-platform (stdlib). **Blind to native
  C allocations** (numpy data buffers, FFT/pocketfft). → Source for
  **attribution** (function / line / object), with a known undercount.
- **Unattributed gap** = `RSS total − sum(tracemalloc-attributed)`. We show it
  explicitly rather than hide it; it is the native/numpy memory.
- **Per-pair attribution under concurrency is not clean** (process-global
  snapshots smear concurrent calls). Solved with a two-run approach below.
- **`memray`** (native-aware, per-line, Mac/Linux only) is an **optional
  Mac-only deep-dive**, not part of the cross-platform tool.

**Cross-OS verdict:** `psutil` + `tracemalloc` are sufficient and behave
identically on all three OSes. No per-OS code. Absolute MB differ between OSes
(allocators/libs); methodology and figures are identical. Profile on Mac.

## Two-run strategy

- **Run A — concurrent (production, 8 workers):** real total RSS timeline and
  the true overlapping-worker envelope. The honest "what actually happens."
- **Run B — serialized registration (`max_workers=1`):** each tile-pair
  isolated → deterministic per-pair allocated bytes + per-substep timing.
  Source for per-pair ranking, variability analysis, and reconstructed
  swimlanes.

## Architecture

A new, self-contained profiling package, kept out of the shipped GUI/runtime
path. Proposed layout (final names settled in the implementation plan):

```
profiling/
  harness.py      # headless TileFusion runner; stage markers; Run A / Run B
  sampler.py      # background psutil RSS sampler (thread, ~50-100ms)
  attribution.py  # tracemalloc snapshots → per-function / per-line / per-object
  perpair.py      # per-pair bytes + substep timing capture (Run B)
  record.py       # write CSV/parquet records
  plots.py        # figure generation (seaborn/matplotlib)
  report.py       # assemble PNG/SVG + tables into a Notion-embeddable PDF
  cli.py          # entrypoint: profile <dataset> [--mode A|B|both]
```

Instrumentation is **non-invasive**: stage boundaries and per-pair hooks are
attached via lightweight wrappers / context managers around the existing
`TileFusion` calls (and a profiling-only `max_workers=1` for Run B). No
edits to pipeline logic.

### Data flow

1. `harness` constructs `TileFusion` for the dataset (mirroring how
   `gui/app.py` builds it), wraps each stage in a timing/marker context.
2. `sampler` thread records `(t, rss_mb)` throughout (Run A).
3. `attribution` takes `tracemalloc` snapshots at stage boundaries and on a
   timer to attribute Python allocations to functions/lines.
4. `perpair` (Run B) wraps `read_pair_patches` and `register_pair_worker` to
   record per-pair bytes + read/fft/corr timings.
5. `record` flushes everything to CSV (one row per sample, one per pair).
6. `plots` + `report` render figures and the bundle.

## Outputs

### Quantitative (CSV — primary, for further analysis)

- `timeline.csv`: `t_ms, rss_mb, stage` (Run A)
- `functions.csv`: `function, peak_mb, integrated_mb_s, pct_of_total` (ranking)
- `pairs.csv`: `pair_id, tile_i, tile_j, row, col, z, channel,
  alloc_bytes, read_ms, fft_ms, corr_ms, peak_kb` (Run B)
- `substep_stats.csv`: per-substep mean / std / **coefficient of variation**
  across pairs (variability → inefficiency flag)

### Figures ("less is more" — earn their place)

**Core (Phase 1) — exactly three:**

1. **Total RSS over time**, stage boundaries as labeled dashed verticals
   (stage name rotated -90° on top, plain seconds on x-axis). The "piano plot."
2. **Per-function overlaid lines** (top-ranked only) + **bold black total RSS**
   line. Gap between summed lines and total = unattributed native memory.
   No stacked areas (avoids false-overlap reading).
3. **Pareto ranking** — sorted bars by integrated cost (MB·s) + cumulative-%
   line ("top N functions = X% of footprint").

**Added later only as they prove useful at QC (not built up front):**

- **Per-pair registration swimlanes** (Phase 2, Run B) — each pair a segment,
  labeled by tile id; shows overlap + staggered finishes.
- **Variability** — per-substep distribution (read/fft/corr) with CV annotation.
- **Scan-pattern / grid traversal** — pair-visit order over the FOV grid.
- **Per-line drill-down** of the #1 function (annotated source, single accent
  color, truncated, frees shown) and **per-object ranking** (live arrays by
  bytes + alloc site) — deepest levels, on demand.

### Report

Notion-embeddable: high-res PNG/SVG figures + the ranking tables, assembled
into a single PDF (overview → per-function → per-pair). CTO opens one file.

## Tile identity / naming

Canonical id derived from existing `(region, fov)` + position + axis indices,
e.g. `manual0/fov12@(r2,c3) z=9 c=DAPI t=0`. Pair label: `A ↔ B`. Pair-visit
order recorded to infer scan pattern.

## Follow-on analyses (enabled, not built in Phase 1)

- **Optimal memory cap:** the registration batching threshold is
  `ram_budget = available_RAM * 0.30` (`_register_parallel`). We capture peak
  RSS, available RAM, and batching behavior so we can later sweep the cap vs
  throughput and find a safe higher setting (more concurrent pairs → speed).
- **Algorithmic waste:** high substep-timing CV across pairs flags redundant
  / inconsistent computation to investigate.
- **memray Mac-only deep dive** for true native intra-FFT memory.

## Dependencies

`psutil` (already a dep), `tracemalloc` (stdlib), `matplotlib` + `seaborn`
(figures), `pandas` (already a dep). Optional/dev: `memray` (Mac/Linux),
`pyarrow` (parquet). New deps go under a `profiling` optional-dependency group.

## Phasing & QC

QC review after each phase (per user). Each phase is its own branch → plan →
build → QC cycle (not "a bit more on the previous branch").

- **Phase 1 — DONE (merged to main 2026-06-15):** harness + Run A → timeline
  figure, per-function overlaid lines, Pareto ranking, `functions.csv`.
  Answers "monitor memory, find expensive functions."
  Baseline result on `test_10x` (manual0): peak RSS ≈ 3.4 GB, mean ≈ 2.0 GB;
  `_fuse_tiles_chunked_plane` (~61%) + `numpy zeros_like` (~35%) ≈ 96% of
  integrated memory.
- **Phase 2:** Run B → `pairs.csv`, swimlanes, variability/CV, scan-pattern. → QC.
- **Phase 3:** report packaging (PDF / Notion assets). → QC.
- **Phase 4 — algorithm optimization:** reduce the memory footprint of the
  hotspots identified above (start with the fuse path: reuse the per-plane
  buffer instead of re-allocating via `zeros_like`; tune the registration
  batching cap). BEFORE the first change, archive a committed **baseline**
  (timeline CSV + peak/mean over several runs, since peak RSS is noisy
  run-to-run) so "before" is a fixed reference. Each optimization is profiled
  before/after to prove the win. → QC.
- **Phase 5 — improvement conclusion (CTO deliverable):** one figure — the
  timeline with TWO RSS lines (before vs after) — plus metric bullets only
  (no narrative): peak before vs after, mean before vs after, **% improvement**.
  Built from the archived baseline + post-optimization profile. → QC.

## Success criteria

- Headless run on the test dataset produces all Phase-1 CSVs + figures with no
  changes to pipeline behavior.
- Total RSS timeline matches OS-reported peak within sampling resolution.
- Function ranking identifies the top contributors with a stated
  attributed-vs-total coverage %.
- Figures are legible, few, and embed in Notion without further editing.
