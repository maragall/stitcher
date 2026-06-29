# TileFusion

GPU/CPU-accelerated tile registration and fusion for 2D microscopy images.

## Features

- **Squid-native input**: stitches Squid OME-TIFF / individual-TIFF acquisitions
- **Robust registration**: phase cross-correlation with NCC scoring and outlier rejection
- **Global optimization**: BigStitcher-style least-squares pose solve over the tile graph
- **Per-seam distortion correction**: elastic correction chosen per seam by cross-validation
- **Feathered fusion**: sub-pixel-accurate blending, streamed at fixed low memory
- **Flatfield correction**: built-in BaSiC illumination correction (pure numpy — no torch/jax)
- **Brightfield-capable**: handles brightfield (B/G/R/RGB) as well as fluorescence
- **OME-NGFF output**: Zarr v3 with multiscale pyramids, plus optional Squid-style OME-TIFF export
- **Optional GPU acceleration**: CUDA via CuPy/cuCIM when available

## Installation

### Desktop application (for end users)

Prebuilt desktop binaries are produced for **macOS, Linux, and Windows** (see the
repository's GitHub Actions build artifacts / Releases). Download the artifact for your
OS and launch it — no Python setup required.

### From source (developers)

```bash
git clone https://github.com/cephla/tilefusion.git
cd tilefusion

pip install -e .            # core (CPU)
pip install -e ".[gui]"     # + napari GUI
pip install -e ".[gpu]"     # + CUDA (CuPy/cuCIM)
pip install -e ".[dev]"     # + linting/test tools
```

## Quick Start

### GUI

```bash
stitcher-gui
```

Workflow: drag in an acquisition folder → **select the registration channel** (required
for multi-channel data — see note below) → optionally enable flatfield → **Run
Stitching**. After it finishes you can **Open in Napari**, compute a **Max Projection**,
or **Export OME-TIFF** (Squid-style, on demand).

### Python API

```python
from tilefusion import TileFusion

with TileFusion(
    "path/to/acquisition",   # Squid OME-TIFF or individual-TIFF folder
    channel_to_use=0,        # registration channel — REQUIRED for multi-channel data
    blend_pixels=(50, 50),
) as tf:
    tf.run()                 # writes <input>_fused.ome.zarr
```

### Registration channel (important)

The registration channel is an **explicit choice**, not auto-detected. There is no
automatic channel picker: contrast/entropy heuristics don't reliably pick the
best-registering channel across datasets, and an automatic guess can silently misalign
the whole mosaic (cf. ASHLAR's `--align-channel`). So:

- **Single-channel** data resolves trivially to channel 0.
- **Multi-channel** data **requires** a channel — pass `channel_to_use=N` (Python) or pick
  one in the GUI dropdown. If you don't, registration raises a clear error (Python) or the
  GUI prompts you to choose before running. Pick a channel with good, **unsaturated**
  structure (e.g. nuclei for fluorescence; a non-blown channel for brightfield).

## Supported formats

### Input
- **Squid OME-TIFF** — the primary, recommended input.
- **Individual TIFFs** — a folder with `manual_{fov}_{z}_{channel}.tiff` + `coordinates.csv`.

> **Note:** OME-**Zarr** is not a stitching input — Squid OME-Zarr carries only identity
> transforms (no per-tile stage positions), so there is nothing to register from. Use the
> OME-TIFF acquisition.

### Output
- **OME-NGFF Zarr v3** (always) — multiscale pyramids for efficient visualization.
- **Squid-style OME-TIFF** (on demand) — tiled BigTIFF with OME-XML (axis order `TZCYX`,
  `PhysicalSize` in µm, channel names), via the GUI's **Export OME-TIFF** button or
  `tilefusion.ome_tiff_export.export_zarr_to_ome_tiff(...)`.

## Flatfield correction

Illumination (flatfield) and optional darkfield are estimated from the tiles themselves
using a pure-numpy port of **BaSiC** (Peng et al., *Nat. Commun.* 2017) — a low-rank +
sparse decomposition, with **no torch/jax dependency**. In the GUI, click **Calculate
Flatfield**; it is then applied during fusion.

## Thread safety

TileFusion uses thread-local file handles for safe concurrent tile reads. BLAS is pinned
to one thread inside the parallel pools to avoid oversubscription. Prefer the context
manager so handles close cleanly:

```python
with TileFusion("tiles.ome.tiff", channel_to_use=0) as tf:
    tf.run()
```

**Limitations:** don't call `close()` while reads are in flight; thread-local handles
consume file descriptors (one per worker thread).

## Acknowledgments

Based on the [tilefusion module](https://github.com/QI2lab/opm-processing-v2/blob/tilefusion2D/src/opm_processing/imageprocessing/tilefusion.py)
from [opm-processing-v2](https://github.com/QI2lab/opm-processing-v2) by
[Doug Shepherd](https://github.com/dpshepherd) and the QI2lab team at Arizona State University.

## License

BSD-3-Clause
