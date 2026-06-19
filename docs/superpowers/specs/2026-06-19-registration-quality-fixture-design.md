# Registration Quality Fixture: Synthetic Ground Truth by Tiling One FOV

**Date:** 2026-06-19
**Status:** design, awaiting review
**Author:** Julio (with Claude)

## 1. Background and context

We are about to change how the registration and optimization phases use memory, and then how
accurately they register. Every one of those changes will be its own spec, its own plan, and
its own commit. None of them is in this spec. This spec delivers only the safety net they all
depend on: a deterministic, end-to-end registration test with ground truth by construction.

Why it must come first. Today the only registration coverage is unit tests of the kernels
(`tests/test_registration.py`, `tests/test_optimization.py`). There is no end-to-end pin of
`pairwise_metrics` or `global_offsets`, and no input whose correct answer is known. Without
that, a memory refactor that silently changes a result, or a quality change that regresses an
existing case, would pass unnoticed. The fixture built here is the thing each later spec
verifies against.

Sequence, each step a separate spec and commit: **this fixture and test first; then the
memory fixes, driven by `/profiling` and gated on this test; then the accuracy fixes, also
gated on this test.** No fix is designed or named here.

## 2. Requirements (EARS)

- R1: The fixture generator **shall** produce four overlapping tiles from the single real
  source FOV at deterministic offsets, given a fixed seed.
- R2: WHEN the registration pipeline runs on the fixture, the recovered relative offsets
  **shall** match the constructed ground-truth offsets within 0.5 px per pair.
- R3: WHEN a change meant to preserve the numerics is applied (for example a memory
  refactor), the pipeline's `pairwise_metrics` and `global_offsets` on the fixture **shall**
  match the committed golden within `atol = 1e-6` px.
- R4: WHERE a change deliberately alters the numerics (for example a different solver), the
  recovered `global_offsets` **shall** match the committed golden within a looser tolerance
  justified and recorded in that change's own spec (default `atol = 1e-3` px).
- R5: IF the real source FOV under `/Users/.../Data` is unavailable, THEN the committed
  fixture **shall** still let the test run (the test is hermetic; the source is needed only
  to regenerate the fixture).
- R6: The fixture **shall** be laid out exactly as the `ome_tiff_tiles` reader expects, so
  the test exercises the real read, register, and optimize path end to end.

## 3. Goals and assumptions

Goals:

1. A deterministic, hermetic, end-to-end registration test with ground truth by
   construction.
2. Two assertions from one fixture: accuracy (recovered vs constructed truth) and
   regression (recovered vs committed golden).
3. Exercise the real reader and the real register plus optimize path, not a mock.

Assumptions:

1. The pipeline is translation only. Phase cross-correlation returns `(dy, dx)`, never an
   angle. **Decision: no rotation in the error model** (a constant camera rotation cancels
   in the relative transform between adjacent tiles and is therefore a no-op for
   registration; a per-tile rotation is a separate quality stress test, out of scope here).
2. The source FOV is representative texture. **Decision: channel 561 nm** (highest
   broadband, non-repetitive texture of the three; 488 is too diffuse, 638 is sparse and
   repetitive).
3. Physical parameters are inherited from the source metadata, not hard-coded: pixel size
   `6.5 / 20 = 0.325` um/px, dtype uint16, FOV 2304 x 2304.

## 4. Design discussion

### 4.1 Construction: one FOV into four

Take the 561 nm channel at mid-z (z = 21) of the real FOV (2304 x 2304, uint16). Cut a 2x2
grid of 1280 x 1280 tiles with step 1024, giving 256 px (20 percent) overlap. The base
top-left offsets (y, x) in pixels:

```
fov0 (0,    0)      fov1 (0,    1024)
fov2 (1024, 0)      fov3 (1024, 1024)
```

`1024 + 1280 = 2304`, so the four tiles exactly cover the FOV and the pairwise overlap is
`1280 - 1024 = 256` px. These base offsets are the regular grid we report to the stitcher.

### 4.2 Error model: where content actually sits vs what we report

The stitcher is handed the clean regular grid as the reported stage positions. The actual
content of each tile is sampled at `base + jitter_k + backlash_k` (sub-pixel), so
registration has to recover the offset we baked in.

**Jitter** (stage positioning noise): zero-mean, isotropic, independent per axis and tile.

```
rng = np.random.default_rng(seed=42)           # 42 to match core.py's seed convention
jitter_k = rng.normal(0.0, sigma_px, size=2)   # (dy, dx) px, sigma_px = 1.5 (about 0.49 um)
```

**Backlash** (mechanical lost motion): on an axis-direction reversal the stage undershoots
the target by a fixed `b`, in the prior direction of travel. Deterministic given the scan
order. **Decision: raster row-major scan, b = 3 px.** Walking the path 0 to 1 to 2 to 3 and
applying `-b * sign(current_move)` on a reversal gives, in (dy, dx) px:

| fov | grid (row,col) | move in | reversal | backlash (dy, dx) |
|-----|----------------|---------|----------|-------------------|
| 0   | (0,0)          | start   | none     | (0, 0)            |
| 1   | (0,1)          | +x      | no       | (0, 0)            |
| 2   | (1,0)          | -x, +y  | x        | (0, +3)           |
| 3   | (1,1)          | +x      | x        | (0, -3)           |

So each tile's actual content offset is `o_k = base_k + jitter_k + backlash_k` (fractional).
Tiles are sampled at that fractional window with `scipy.ndimage.map_coordinates(order=3)`,
which injects the sub-pixel part exactly with no edge wrap (windows stay inside 2304 x 2304)
and exercises the `upsample_factor = 10` (0.1 px) path in `phase_cross_correlation`.

### 4.3 What ground truth is, and why recovery works

Reported displacement for pair `(i, j)` is `base_j - base_i`. Actual content displacement is
`o_j - o_i = (base_j - base_i) + (e_j - e_i)` where `e = jitter + backlash`. Registration
measures the residual `e_j - e_i` from image content and adds it to the reported
displacement, so the recovered relative offset equals `o_j - o_i`. Therefore:

- **Ground truth, per pair:** `o_j - o_i`.
- **Ground truth, global:** `{o_k}` up to the anchor translation (fov0 is pinned).

The accuracy assertion (R2) compares recovered to this. The errors are a few px inside a
256 px overlap, so the overlap window always holds valid matching texture and the search
never escapes.

### 4.4 Fixture layout (matches the reader contract)

`load_ome_tiff_tiles_metadata` expects: `ome_tiff/{region}_{fov}.ome.tiff`, a top-level
`coordinates.csv` with columns `region`, `x (mm)`, `y (mm)` where **fov index is the row
order**, positions read as `(y_um, x_um)`, and pixel size from `acquisition
parameters.json` (`sensor_pixel_size_um / magnification`, fallback 0.752 if the file is
missing). So the committed fixture is:

```
tests/fixtures/synth_4fov/
  ome_tiff/synth_0.ome.tiff ... synth_3.ome.tiff   # 1280x1280 uint16, single channel, single z
  coordinates.csv                                  # 4 rows, region "synth", x/y (mm) = clean base grid
  acquisition parameters.json                      # sensor_pixel_size_um 6.5, magnification 20
  ground_truth.json                                # o_k (true offsets), seed, sigma, b, scan order, channel, z
  golden_metrics.json                              # committed pairwise_metrics + global_offsets snapshot
```

Reported positions in `coordinates.csv` are the clean grid: `x_mm = origin_x + col * 1024 *
0.000325`, `y_mm = origin_y + row * 1024 * 0.000325`, origin from the source
`coordinates.csv`. `acquisition parameters.json` must be present or the reader falls back to
0.752 um/px and the overlap math is wrong.

Size: four 1280 x 1280 uint16 tiles are about 13 MB raw. **Decision: write them as
compressed OME-TIFF (zlib) to keep the committed fixture near 6 to 8 MB.** Compression does
not change pixel values, so the golden is unaffected.

### 4.5 The two assertions

1. **Accuracy (portable correctness gate):** recovered relative offset per pair within
   0.5 px of `o_j - o_i`; recovered global positions within 0.5 px of `{o_k}` after aligning
   the anchor. Tolerance absorbs float and platform noise; this is the machine-portable
   check.
2. **Regression (refactor-equivalence gate):** load `golden_metrics.json`, compare current
   `pairwise_metrics` and `global_offsets`. `atol = 1e-6` px for changes meant to preserve
   numerics (tight enough to catch any algorithmic change, loose enough for float noise); a
   looser, explicitly justified tolerance for changes that deliberately alter numerics, set
   in that change's own spec (default `atol = 1e-3` px). The golden is generated once on the
   dev machine and committed.

### 4.6 Fit into the stitcher workflow

The fixture is consumed by the real entry path, no test-only shim:

- `TileFusion(tiff_path=tests/fixtures/synth_4fov, region="synth", channel_to_use=0)`. The
  constructor auto-detects the ome_tiff_tiles format from the `ome_tiff/` folder
  (`core.py:141`), loads metadata, and filters to region "synth" (4 tiles, `core.py:179`).
- Single channel, so `channel_to_use = 0`. Single z, so `n_z = 1` and `registration_z`
  defaults to its only plane, 0 (`core.py:203-207`); nothing to pass.
- The test then calls `refine_tile_positions_with_cross_correlation()` (fills
  `self.pairwise_metrics`) and `optimize_shifts()` (fills `self.global_offsets`). Those two
  attributes are the golden. This is the same register-then-optimize sequence a full run
  uses; fusion is not exercised here.

## 5. How to implement (ordered, TDD; detailed steps go to writing-plans)

1. **Generator** `tests/fixtures/generate_synth_4fov.py`: read source FOV metadata and the
   561/mid-z plane, build `o_k` from seed/sigma/b/scan, sample tiles via `map_coordinates`,
   write the fixture layout in 4.4 plus `ground_truth.json`. Deterministic. Check-in point:
   eyeball the four tiles and confirm overlaps line up.
2. **Commit the fixture** artifacts (compressed tiles, coordinates, params, ground_truth).
3. **Accuracy test** `tests/test_registration_quality.py`: load via the reader, run register
   plus optimize, assert R2 against `ground_truth.json`. This is the failing-first test that
   defines correct.
4. **Capture the golden:** run the pipeline once, write `golden_metrics.json`, commit it, add
   the R3/R4 regression assertion.

This spec ends at a committed, passing golden. From here, each memory fix and each quality
fix is its own spec, plan, and commit, and every one re-runs this test.

## 6. Other options considered

- **Real multi-tile dataset (e.g. test_10x) as the golden.** No ground truth (true offsets
  unknown), so accuracy is impossible; only regression. Not in the repo and large. Rejected
  as the primary; may complement later for scale.
- **Fully synthetic random-texture image.** Less representative texture statistics (SSIM,
  contrast) than a real FOV. Rejected; the real FOV is free and representative.
- **Integer-only offsets.** Simpler, exact crops, but skips the sub-pixel machinery the
  pipeline advertises. Rejected; sub-pixel chosen.
- **Rotation in the error model.** Constant rotation is a no-op for relative registration;
  per-tile rotation is unrecoverable by a translation-only pipeline and belongs to the
  separate quality track. Rejected here.
- **Calibrate sigma and b empirically from a real mosaic's coordinates.csv.** Viable bridge
  to realistic magnitudes; deferred. The injected value must still be a fixed known constant
  for ground truth, so picked defaults are sufficient for S0.

## 7. Open questions and decision log

- 2026-06-19: channel **561**, **no rotation**, **jitter + backlash**, geometry **1280/1024
  (256 px overlap)**, **sigma 1.5 px**, **b 3 px**, **raster** scan, **sub-pixel** via
  interpolation, **seed 42** (matches core.py's seed convention). (Julio, "use defaults".)
- Open: regression tolerance across machines. Leaning `atol 1e-6` px rather than strict
  equality so trivial FFT/BLAS float noise does not flake the test, while still catching any
  real algorithmic change. Revisit if the test ever runs in CI on a different platform.
- Open: single mid-z plane (chosen) vs z-MIP. Mid-z is simpler and deterministic; MIP adds
  texture but is unnecessary given 561 already fills the frame.
