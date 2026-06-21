# Stitcher architecture

This is the map of the codebase after the 2026-06 maintainability refactor. It is meant to be
read top to bottom once, so that the next person (or the next you) can find where a thing lives
without reading `core.py` end to end.

## The one-paragraph version

`TileFusion` (in `core.py`) is the orchestrator. It reads a microscope acquisition through a
format-agnostic `Reader`, registers neighbouring tiles against each other to correct stage
error, solves one global set of tile offsets from those pairwise measurements, fuses the tiles
into a single feathered image, and writes it as a multiscale OME-Zarr. Each of those verbs is a
module; `core.py` holds the state and the sequencing, and delegates the real work outward.

## The pipeline (what `TileFusion.run()` does, in order)

```
 input acquisition (OME-TIFF / OME-TIFF tiles / individual TIFFs / Zarr)
        │
        ▼
 1. READ      io.open_reader(path) -> Reader      core._read_tile / _read_tile_region
        │     (flatfield/darkfield applied here, in the read path)
        ▼
 2. REGISTER  registration.find_adjacent_pairs -> compute_pair_bounds
        │     -> register_pairs_batched | register_pairs_readahead
        │     => core.pairwise_metrics  { (i,j): (dy, dx, score) }
        ▼
 3. OPTIMIZE  optimization._edges_from_pairwise_metrics
        │     -> solve_least_squares | two_round_optimization
        │     => core.global_offsets  (n_tiles, 2)
        ▼
 4. PLACE     offsets applied to tile stage positions
        │     core._compute_fused_image_space -> _pad_to_chunk_multiple
        ▼
 5. FUSE      fusion.fuse_plane (block-iterating, one feathered plane per z/t)
        │     => writes into the output tensorstore
        ▼
 6. WRITE     core._create_multiscales -> io.zarr (NGFF metadata, pyramid)
        ▼
 output: <name>_fused.ome.zarr   (T, C, Z, Y, X)
```

## Modules

### `io/` — reading acquisitions, writing Zarr

The input side is unified behind one abstraction so `core.py` never branches on format:

- **`io/base.py`** — the `Reader` protocol (`load_metadata`, `read_tile`, `read_region`,
  `is_multi_file`) and the `open_reader(path)` factory. `open_reader` detects the format and
  returns the matching concrete reader; everything downstream talks to the protocol.
- **`io/_squid.py`** — shared Squid-acquisition helpers (`load_acquisition_params`,
  `channel_names_or_default`) used by more than one loader, so the parsing of
  `acquisition parameters.json` and channel-name fallback lives in one place.
- **`io/ome_tiff.py`, `io/ome_tiff_tiles.py`, `io/individual_tiffs.py`, `io/zarr.py`** — the
  per-format metadata loaders and tile/region readers. The two metadata parsers
  (`load_ome_tiff_tiles_metadata`, `load_individual_tiffs_metadata`) are the longest functions
  left in the tree; they are inherently long because they parse real-world acquisition layouts,
  and their coordinate parsers are deliberately NOT merged (the two CSV layouts genuinely diverge).
- **`io/zarr.py`** — the output side too: the OME-Zarr v3 store, NGFF metadata, scale-group
  metadata.

`is_multi_file` is the one piece of format knowledge that escapes the reader: it drives both the
registration parallel-mode auto-detect and the single-OME-TIFF thread-local-handle path in core.

### `registration.py` — measuring tile-to-tile displacement

Pure functions over explicit inputs (no `TileFusion` dependency):

- **`find_adjacent_pairs`** — which tiles physically overlap (from stage positions).
- **`compute_pair_bounds`** — the pixel crop geometry for each overlapping pair.
- **`register_pair_worker`** — the numpy compute kernel for one pair (used by the batched path).
- **`register_and_score`** — the GPU-capable compute kernel (used by the read-ahead path).
- **`register_pairs_batched`** — bounded-memory batches over a CPU compute pool; the default path
  for multi-file inputs with many pairs.
- **`register_pairs_readahead`** — one pair at a time, reading each pair's two patches via a
  2-worker read-ahead pool; the GPU path and the small-dataset fallback. (It was once misnamed
  "sequential" — it is not sequential.)

Both `register_pairs_*` take an injected `read_region` callable (core passes the bound
`self._read_tile_region`) and return a `pairwise_metrics` dict. The thread-local-handle lifecycle
stays in core by construction.

> Known sub-pixel limitation: both kernels store `int(np.round(shift * df))`, discarding the
> `upsample_factor=10` sub-pixel precision. This is pinned by the strict-xfail
> `test_accuracy_pairwise` and is the first scheduled quality fix (separate from this refactor).

### `optimization.py` — one global solution from many pairwise measurements

- **`solve_least_squares`** — the weighted least-squares solve (per axis) that turns the edge set
  into per-tile offsets, with chosen tiles anchored at the origin.
- **`two_round_optimization`** — robust variant: select a maximum-weight spanning tree
  (`_build_mst`), solve, drop outlier edges by residual, re-solve (optionally to convergence).
- **`_edges_from_pairwise_metrics`** — the single, private bridge from the canonical
  `pairwise_metrics` (the serializable source of truth) to the solver-native edge list.
  `pairwise_metrics` is canonical; the edge list is a transient derived value.

### `fusion.py` — blending tiles into one image

- **`fuse_plane`** — the one block-iterating fuser. It walks the output plane block by block,
  accumulates each overlapping tile's feathered contribution, normalizes, and writes the block.
  `block_size` is the memory-budget knob: small blocks bound peak memory; a block as large as the
  plane is the whole-plane case. Output is identical regardless of `block_size` (the full-vs-chunked
  equivalence, guarded byte-for-byte by `test_fuse_equivalence`, including a feathered case).
- **`accumulate_tile_shard`, `normalize_shard`, `blend_numba_2d`** — the numba blend kernels.

### `flatfield.py` — illumination correction

`apply_flatfield` / `apply_flatfield_region`, applied inside core's read path (`_read_tile` /
`_read_tile_region`), so every consumer of a tile sees corrected pixels and fusion never has to
know about it.

### `core.py` — `TileFusion`, the orchestrator

What remains in core after the refactor is genuinely core: construction/state, the read path
(including the thread-local TiffFile handle system and `close()`/context-manager lifecycle), the
pipeline sequencing (`run`), the fused-image-space geometry, the output-store creation, and the
multiscale pyramid. `__init__` is now a short call sequence over nine named initializers
(`_resolve_paths`, `_load_metadata`, `_apply_region_filter`, `_configure_z_t_planes`,
`_configure_registration_params`, `_init_chunking`, `_init_pipeline_state`, `_init_corrections`,
`_init_handle_storage`).

Key state set up front and consumed across the pipeline:
`pairwise_metrics` (stage 2 output), `global_offsets` (stage 3 output), `padded_shape` /
`center` (stage 4 geometry), `fused_ts` (the output tensorstore).

## Tests that gate changes

- **`tests/test_registration_quality.py`** — the registration golden. A synthetic 4-FOV fixture
  built from one real FOV with known offsets + realistic per-tile degradations; pins recovered
  offsets to ground truth (accuracy) and to a committed `golden_metrics.json` (regression). See
  `docs/registration-quality-fixture-deep-dive.md`.
- **`tests/test_fuse_equivalence.py`** — byte-identical fusion under different block sizes
  (uniform and feathered).
- **`tests/test_io_readers.py` / `test_io_factory.py` / `test_io_squid.py`** — characterization of
  every reader's metadata and region pixels.
- **`tests/test_registration.py` / `test_optimization.py` / `test_fusion.py`** — unit tests of the
  free functions in each stage module.

## Measuring the codebase

`scripts/loc_by_function.py` emits per-stage function-size histograms and a name-agnostic CSV
inventory (`docs/loc_by_function.csv`). It is the instrument used to measure this refactor; see
`docs/refactor-overview.md` for the before/after.
