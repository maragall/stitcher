# Memory optimization — conclusion (`test_10x`, manual0)

Result of the Phase 4 fuse-memory optimization (branch `feat/fuse-memory-opt`),
measured on the same dataset/machine as the pre-optimization baseline.

## Headline metrics (copy-paste for Notion)

- **Peak RSS: ~3971 MB → ~2527 MB — ≈ 36% lower**
- **Mean RSS: ~2225 MB → ~1842 MB — ≈ 17% lower**
- Fused output is **byte-identical** (guarded by `tests/test_fuse_equivalence.py`
  + the full fusion/integration suites — 20/20 pass).
- Side effect: the run also finished **faster** (~43 s vs ~55 s wall-clock) —
  less allocation churn.

### Per-run numbers

| | peak RSS (MB) | mean RSS (MB) |
|---|---|---|
| before (3-run avg) | 3971 (3993/3961/3960) | 2225 |
| after (3-run avg)  | 2527 (2491/2578/2512) | 1842 |

## What changed (output-identical)

`_fuse_tiles_chunked_plane` in `src/tilefusion/core.py`:
1. **Reuse** the per-block `fused`/`weight` `float32` buffers across blocks
   (allocate once, zero per block) — removes the `np.zeros`/`zeros_like` churn
   (the ~35% contributor) and the volatile sawtooth.
2. **In-place masked divide** (`np.divide(..., out=, where=mask)`) — removes the
   large fancy-index temporaries from `fused_block[mask] /= weight_sum[mask]`.
3. **Honest block sizing** (`bytes_per_pixel = 12*C`, was `8*C`) — the divisor
   now reflects true transient cost (fused+weight+mask+uint16 cast), so the
   `ram_fraction` budget is a real, **safe** ceiling (no more silent overshoot
   to ~4 GB).

## Figures (this folder)

- `overlay.png` — before vs after RSS over time (the conclusion figure).
- `timeline.png`, `functions.png`, `pareto.png` — the "after" profile set.
  (Pair with `../pre-opt/` for the "before" set in the Notion page.)
