# Registration Quality Fixture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, hermetic, end-to-end registration test by cutting one real 20x FOV into four overlapping tiles at known offsets with an injected jitter+backlash stage error, so registration must recover the offsets we baked in.

**Architecture:** A fixture generator (reads the real FOV, injects sub-pixel error via interpolation, writes tiles in the `ome_tiff_tiles` layout) plus two pytest tests against the committed fixture: an accuracy test (recovered vs constructed ground truth) and a regression test (recovered vs committed golden snapshot).

**Tech Stack:** Python, numpy, scipy.ndimage, tifffile, pytest. Consumes `tilefusion.core.TileFusion` and `tilefusion.io.ome_tiff_tiles`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-19-registration-quality-fixture-design.md`.
- Seed `42`, `sigma_px = 1.5`, `b_px = 3.0`, raster scan, channel index `1` (561 nm), `z_level = 21`.
- Geometry: tile `1280`, step `1024`, 256 px overlap, source FOV `2304 x 2304` uint16.
- Pixel size is read from source metadata (`sensor_pixel_size_um / magnification = 0.325` um/px), never hard-coded into the recovered values.
- Fixture must match the reader contract: `ome_tiff/{region}_{fov}.ome.tiff`, top-level `coordinates.csv` (cols `region`, `x (mm)`, `y (mm)`; fov = row order), and `acquisition parameters.json` for pixel size.
- Tests against the committed fixture are hermetic (no `/Data` dependency). Only the generator needs the real source; its test is `skipif` the source is absent.
- `global_offsets`, `pairwise_metrics` are in pixels (verified: `links_from_pairwise_metrics` passes pixel shifts straight through).
- Tolerances: accuracy `atol = 0.5` px; regression `atol = 1e-6` px.
- Source FOV: `/Users/julioamaragall/CEPHLA/Data/20x_FoxChase_488_555_640`.

---

### Task 1: Error model (pure functions)

**Files:**
- Create: `tests/fixtures/__init__.py` (empty, makes the folder importable)
- Create: `tests/fixtures/generate_synth_4fov.py` (error-model functions only in this task)
- Test: `tests/test_synth_fixture.py`

**Interfaces:**
- Produces: `BASE_OFFSETS` (np.ndarray `(4,2)` float64, (y,x) px), `backlash_offsets(b_px: float) -> np.ndarray (4,2)`, `compute_content_offsets(seed: int = 42, sigma_px: float = 1.5, b_px: float = 3.0) -> np.ndarray (4,2)` returning the actual content offset `o_k` (y,x) px per fov.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synth_fixture.py
import numpy as np
from tests.fixtures.generate_synth_4fov import (
    BASE_OFFSETS,
    backlash_offsets,
    compute_content_offsets,
)


def test_base_offsets_are_the_raster_grid():
    expected = np.array([(0, 0), (0, 1024), (1024, 0), (1024, 1024)], dtype=np.float64)
    np.testing.assert_array_equal(BASE_OFFSETS, expected)


def test_backlash_matches_spec_table():
    # raster row-major: x reverses into fov2 (+b) and fov3 (-b); y untouched
    expected = np.array([(0, 0), (0, 0), (0, 3), (0, -3)], dtype=np.float64)
    np.testing.assert_array_equal(backlash_offsets(3.0), expected)


def test_content_offsets_deterministic():
    a = compute_content_offsets(seed=42, sigma_px=1.5, b_px=3.0)
    b = compute_content_offsets(seed=42, sigma_px=1.5, b_px=3.0)
    np.testing.assert_array_equal(a, b)


def test_zero_jitter_reduces_to_base_plus_backlash():
    o = compute_content_offsets(seed=42, sigma_px=0.0, b_px=3.0)
    np.testing.assert_array_equal(o, BASE_OFFSETS + backlash_offsets(3.0))


def test_jitter_is_small_and_present():
    o = compute_content_offsets(seed=42, sigma_px=1.5, b_px=3.0)
    resid = o - (BASE_OFFSETS + backlash_offsets(3.0))  # the jitter component
    assert np.any(resid != 0.0)
    assert np.all(np.abs(resid) < 10.0)  # 1.5 px sigma stays well under 10 px
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_synth_fixture.py -v`
Expected: FAIL at import (`ModuleNotFoundError: tests.fixtures.generate_synth_4fov`).

- [ ] **Step 3: Write minimal implementation**

```python
# tests/fixtures/generate_synth_4fov.py
"""Generate a synthetic 4-FOV registration fixture by tiling one real FOV.

See docs/superpowers/specs/2026-06-19-registration-quality-fixture-design.md.
"""
import numpy as np

TILE = 1280
STEP = 1024
# raster row-major fov order; (row, col)
GRID = [(0, 0), (0, 1), (1, 0), (1, 1)]
BASE_OFFSETS = np.array([(r * STEP, c * STEP) for (r, c) in GRID], dtype=np.float64)  # (y,x) px


def backlash_offsets(b_px: float) -> np.ndarray:
    """Raster row-major backlash: x reverses into fov2 (+b) and fov3 (-b). (y,x) px."""
    return np.array([(0.0, 0.0), (0.0, 0.0), (0.0, b_px), (0.0, -b_px)], dtype=np.float64)


def compute_content_offsets(seed: int = 42, sigma_px: float = 1.5, b_px: float = 3.0) -> np.ndarray:
    """Actual content offset o_k = base + jitter + backlash, (4,2) (y,x) px."""
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0.0, sigma_px, size=(4, 2))
    return BASE_OFFSETS + jitter + backlash_offsets(b_px)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_synth_fixture.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/__init__.py tests/fixtures/generate_synth_4fov.py tests/test_synth_fixture.py
git commit -m "test(reg-quality): error model for synthetic 4-FOV fixture (jitter + raster backlash)"
```

---

### Task 2: Fixture generator and committed fixture

**Files:**
- Modify: `tests/fixtures/generate_synth_4fov.py` (add `sample_tile`, `generate_fixture`, `__main__`)
- Modify: `tests/test_synth_fixture.py` (add the round-trip test)
- Create (generated, committed): `tests/fixtures/synth_4fov/ome_tiff/synth_{0..3}.ome.tiff`, `coordinates.csv`, `acquisition parameters.json`, `ground_truth.json`

**Interfaces:**
- Consumes: `BASE_OFFSETS`, `compute_content_offsets` from Task 1.
- Produces: `sample_tile(plane: np.ndarray, oy: float, ox: float, size: int = TILE) -> np.ndarray (uint16)`; `generate_fixture(out_dir, source=SOURCE, channel_idx=1, z_level=21, seed=42, sigma_px=1.5, b_px=3.0) -> None` writing the fixture layout plus `ground_truth.json` with key `tiles` (list of `{fov, base, offset, error}`) and `params`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_synth_fixture.py
import json
from pathlib import Path

import pytest

from tests.fixtures.generate_synth_4fov import SOURCE, generate_fixture

pytestmark_source = pytest.mark.skipif(
    not (SOURCE / "ome_tiff" / "current_0.ome.tiff").exists(),
    reason="real source FOV not present; fixture regeneration test skipped",
)


@pytestmark_source
def test_generator_roundtrip_and_determinism(tmp_path):
    from tilefusion.core import TileFusion

    a = tmp_path / "a"
    generate_fixture(a)

    # layout exists
    assert (a / "coordinates.csv").exists()
    assert (a / "acquisition parameters.json").exists()
    assert sorted(p.name for p in (a / "ome_tiff").glob("*.ome.tiff")) == [
        f"synth_{k}.ome.tiff" for k in range(4)
    ]

    # loads through the real reader as 4 single-channel single-z 1280^2 tiles
    tf = TileFusion(a, region="synth", channel_to_use=0)
    assert tf.n_tiles == 4
    assert tf.channels == 1
    assert tf.n_z == 1
    assert (tf.Y, tf.X) == (1280, 1280)

    # ground truth has 4 tiles with base/offset/error
    gt = json.loads((a / "ground_truth.json").read_text())
    assert len(gt["tiles"]) == 4
    assert gt["params"]["seed"] == 42

    # determinism: regenerate and compare bytes + ground truth
    b = tmp_path / "b"
    generate_fixture(b)
    assert (a / "ground_truth.json").read_bytes() == (b / "ground_truth.json").read_bytes()
    for k in range(4):
        assert (
            (a / "ome_tiff" / f"synth_{k}.ome.tiff").read_bytes()
            == (b / "ome_tiff" / f"synth_{k}.ome.tiff").read_bytes()
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_synth_fixture.py::test_generator_roundtrip_and_determinism -v`
Expected: FAIL with `ImportError: cannot import name 'SOURCE'` / `generate_fixture` (not yet defined).

- [ ] **Step 3: Write minimal implementation**

```python
# add to tests/fixtures/generate_synth_4fov.py
import csv
import json
from pathlib import Path

import tifffile
from scipy.ndimage import map_coordinates

SOURCE = Path("/Users/julioamaragall/CEPHLA/Data/20x_FoxChase_488_555_640")
FIXTURE_DIR = Path(__file__).parent / "synth_4fov"


def sample_tile(plane: np.ndarray, oy: float, ox: float, size: int = TILE) -> np.ndarray:
    """Sample a size x size window at fractional top-left (oy, ox), bicubic. uint16."""
    ys = oy + np.arange(size)
    xs = ox + np.arange(size)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    coords = np.stack([gy.ravel(), gx.ravel()])
    vals = map_coordinates(plane.astype(np.float64), coords, order=3, mode="reflect")
    return np.clip(vals, 0, 65535).round().astype(np.uint16).reshape(size, size)


def generate_fixture(
    out_dir,
    source: Path = SOURCE,
    channel_idx: int = 1,
    z_level: int = 21,
    seed: int = 42,
    sigma_px: float = 1.5,
    b_px: float = 3.0,
) -> None:
    out_dir = Path(out_dir)
    (out_dir / "ome_tiff").mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(source / "ome_tiff" / "current_0.ome.tiff") as tf:
        plane = tf.series[0].asarray()[z_level, channel_idx]  # (2304, 2304)

    params = json.loads((source / "acquisition parameters.json").read_text())
    px_um = params["sensor_pixel_size_um"] / params["objective"]["magnification"]
    px_mm = px_um / 1000.0

    o = compute_content_offsets(seed, sigma_px, b_px)  # (4,2) (y,x) px

    for k in range(4):
        tile = sample_tile(plane, o[k, 0], o[k, 1])
        tifffile.imwrite(out_dir / "ome_tiff" / f"synth_{k}.ome.tiff", tile, compression="zlib")

    # reported positions = clean base grid, in mm; fov index = row order
    with open(out_dir / "coordinates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "x (mm)", "y (mm)", "z (mm)"])
        for k in range(4):
            by, bx = BASE_OFFSETS[k]
            w.writerow(["synth", bx * px_mm, by * px_mm, 0.0])

    json.dump(
        {"sensor_pixel_size_um": 6.5, "objective": {"magnification": 20.0}, "Nz": 1},
        open(out_dir / "acquisition parameters.json", "w"),
    )

    gt = {
        "params": {
            "seed": seed, "sigma_px": sigma_px, "b_px": b_px, "scan": "raster",
            "channel_idx": channel_idx, "z_level": z_level, "pixel_size_um": px_um,
        },
        "tiles": [
            {
                "fov": k,
                "base": BASE_OFFSETS[k].tolist(),
                "offset": o[k].tolist(),
                "error": (o[k] - BASE_OFFSETS[k]).tolist(),
            }
            for k in range(4)
        ],
    }
    with open(out_dir / "ground_truth.json", "w") as f:
        json.dump(gt, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    generate_fixture(FIXTURE_DIR)
    print(f"wrote fixture to {FIXTURE_DIR}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_synth_fixture.py -v`
Expected: PASS (the round-trip test runs because the source FOV is present on this machine).
If `TileFusion` reports `channels != 1` or `n_z != 1`, the OME-TIFF axis layout is wrong; write the tile with an explicit single-page 2D array (as above, `imwrite` of a `(1280,1280)` array yields one page, which the reader treats as 1 channel, 1 z). Fix and rerun before continuing.

- [ ] **Step 5: Generate and commit the fixture**

```bash
python -m tests.fixtures.generate_synth_4fov
git add tests/fixtures/generate_synth_4fov.py tests/test_synth_fixture.py
git add tests/fixtures/synth_4fov/ome_tiff/*.ome.tiff
git add "tests/fixtures/synth_4fov/coordinates.csv" "tests/fixtures/synth_4fov/acquisition parameters.json" tests/fixtures/synth_4fov/ground_truth.json
git commit -m "test(reg-quality): synthetic 4-FOV fixture generator + committed fixture"
```

---

### Task 3: Accuracy test (recovered vs constructed ground truth, R2)

**Files:**
- Create: `tests/test_registration_quality.py`

**Interfaces:**
- Consumes: the committed fixture at `tests/fixtures/synth_4fov/`, `ground_truth.json` schema from Task 2 (`tiles[k].error` = `e_k` (y,x) px), `TileFusion.global_offsets` (px, tile 0 fixed at origin), `TileFusion.pairwise_metrics` (`{(i,j): (dy,dx,score)}` px).
- Produces: `run_pipeline(fixture_dir) -> TileFusion` (loads, registers, optimizes) reused by Task 4.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registration_quality.py
import json
from pathlib import Path

import numpy as np

from tilefusion.core import TileFusion

FIXTURE = Path(__file__).parent / "fixtures" / "synth_4fov"


def run_pipeline(fixture_dir=FIXTURE) -> TileFusion:
    tf = TileFusion(fixture_dir, region="synth", channel_to_use=0)
    tf.refine_tile_positions_with_cross_correlation()
    tf.optimize_shifts()
    return tf


def _errors_px():
    gt = json.loads((FIXTURE / "ground_truth.json").read_text())
    e = np.array([t["error"] for t in sorted(gt["tiles"], key=lambda t: t["fov"])])
    return e  # (4,2) (y,x) px


def test_accuracy_global_offsets():
    tf = run_pipeline()
    e = _errors_px()
    expected = e - e[0]  # tile 0 is the fixed anchor
    recovered = np.asarray(tf.global_offsets, dtype=np.float64)
    np.testing.assert_allclose(recovered, expected, atol=0.5)


def test_accuracy_pairwise():
    tf = run_pipeline()
    e = _errors_px()
    assert len(tf.pairwise_metrics) >= 3  # connected graph over 4 tiles
    for (i, j), (dy, dx, score) in tf.pairwise_metrics.items():
        expected = e[j] - e[i]
        np.testing.assert_allclose([dy, dx], expected, atol=0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_registration_quality.py -v`
Expected initial state before the fixture/pipeline line up: FAIL. Two legitimate first-run outcomes to handle:
- A uniform sign flip on every tile/pair (recovered ~ -expected): the pipeline's correction convention is negated relative to our injection. Negate `expected` in both tests (`expected = e[0] - e` and `e[i] - e[j]`), add a one-line comment citing this step, and rerun. This is the documented convention resolution, not a placeholder.
- Any non-uniform mismatch above 0.5 px: a real signal (generator bug or genuine accuracy gap). Stop and investigate before proceeding; do not loosen the tolerance to force a pass.

- [ ] **Step 3: Resolve convention if needed**

Apply only the sign correction described in Step 2 if the run showed a uniform flip. No other code changes.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_registration_quality.py -v`
Expected: PASS (2 passed). Recovered offsets match the injected jitter+backlash within 0.5 px.

- [ ] **Step 5: Commit**

```bash
git add tests/test_registration_quality.py
git commit -m "test(reg-quality): accuracy test, registration recovers injected offsets within 0.5 px"
```

---

### Task 4: Golden capture and regression test (R3)

**Files:**
- Modify: `tests/fixtures/generate_synth_4fov.py` (add `capture_golden`)
- Create (generated, committed): `tests/fixtures/synth_4fov/golden_metrics.json`
- Modify: `tests/test_registration_quality.py` (add the regression test)

**Interfaces:**
- Consumes: `run_pipeline` from Task 3.
- Produces: `capture_golden(fixture_dir=FIXTURE_DIR) -> None` writing `golden_metrics.json` with `{"pairwise": {"i,j": [dy,dx,score]}, "global": [[y,x], ...]}`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_registration_quality.py
def test_regression_matches_committed_golden():
    golden = json.loads((FIXTURE / "golden_metrics.json").read_text())
    tf = run_pipeline()

    # global_offsets
    np.testing.assert_allclose(
        np.asarray(tf.global_offsets, dtype=np.float64),
        np.asarray(golden["global"], dtype=np.float64),
        atol=1e-6,
    )

    # pairwise_metrics
    cur = {f"{i},{j}": [float(dy), float(dx), float(s)] for (i, j), (dy, dx, s) in tf.pairwise_metrics.items()}
    assert set(cur) == set(golden["pairwise"])
    for key in cur:
        np.testing.assert_allclose(cur[key], golden["pairwise"][key], atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_registration_quality.py::test_regression_matches_committed_golden -v`
Expected: FAIL (`FileNotFoundError: golden_metrics.json`), because the golden has not been captured yet.

- [ ] **Step 3: Write minimal implementation (the capture function)**

```python
# add to tests/fixtures/generate_synth_4fov.py
def capture_golden(fixture_dir=FIXTURE_DIR) -> None:
    """Run the pipeline on the committed fixture and freeze its output as the golden."""
    from tilefusion.core import TileFusion

    tf = TileFusion(fixture_dir, region="synth", channel_to_use=0)
    tf.refine_tile_positions_with_cross_correlation()
    tf.optimize_shifts()
    golden = {
        "pairwise": {f"{i},{j}": [float(dy), float(dx), float(s)]
                     for (i, j), (dy, dx, s) in tf.pairwise_metrics.items()},
        "global": np.asarray(tf.global_offsets, dtype=np.float64).tolist(),
    }
    with open(Path(fixture_dir) / "golden_metrics.json", "w") as f:
        json.dump(golden, f, indent=2, sort_keys=True)
```

- [ ] **Step 4: Capture the golden, then run the test to verify it passes**

```bash
python -c "from tests.fixtures.generate_synth_4fov import capture_golden; capture_golden()"
pytest tests/test_registration_quality.py -v
```
Expected: PASS (3 passed: 2 accuracy + 1 regression).

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/generate_synth_4fov.py tests/fixtures/synth_4fov/golden_metrics.json tests/test_registration_quality.py
git commit -m "test(reg-quality): capture golden metrics + regression pin (atol 1e-6 px)"
```

---

### Task 5: Full-suite check

**Files:** none (verification only)

- [ ] **Step 1: Run the whole suite**

Run: `pytest -q`
Expected: all tests pass, including the existing `tests/test_registration.py`, `tests/test_optimization.py`, and the new `tests/test_synth_fixture.py`, `tests/test_registration_quality.py`. No regressions introduced.

- [ ] **Step 2: Confirm hermeticity**

The accuracy and regression tests must not read `/Data`. Confirm by inspection that `tests/test_registration_quality.py` references only `tests/fixtures/synth_4fov/`. (The generator round-trip test in `tests/test_synth_fixture.py` is the only `/Data` consumer and is `skipif`-guarded.)

---

## Self-Review

**Spec coverage:**
- R1 (deterministic generator): Task 1 (determinism test) + Task 2 (byte-identical regen test).
- R2 (accuracy within 0.5 px): Task 3.
- R3 (numerics-preserving regression within 1e-6 px): Task 4.
- R4 (looser tolerance for numerics-changing fixes): not implemented here by design; it is invoked by each future fix's own spec, which sets its tolerance. The regression test's structure supports passing a different atol then. No task needed in S0.
- R5 (hermetic): Task 2 `skipif` guard on the only `/Data` consumer; Task 5 Step 2 confirms.
- R6 (reader contract): Task 2 round-trip loads through `TileFusion`/the real reader.

**Placeholder scan:** No TBD/TODO. The two first-run contingencies in Task 3 Step 2 (sign convention; real mismatch) are concrete documented branches, not deferred work.

**Type consistency:** `compute_content_offsets`/`BASE_OFFSETS`/`backlash_offsets` return `(4,2)` float64 (y,x) px throughout. `ground_truth.json` `tiles[k].error` is `e_k` (y,x) px, consumed as such in Task 3. `global_offsets` and `pairwise_metrics` treated as px in Tasks 3 and 4. `run_pipeline` defined in Task 3, reused in Task 4. `capture_golden` writes the exact schema Task 4's regression test reads (`pairwise` dict keyed `"i,j"`, `global` list of `[y,x]`).
