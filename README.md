# TileFusion

GPU/CPU-accelerated tile registration and fusion for 2D microscopy images.

## Features

- **Multi-format support**: OME-TIFF, individual TIFFs with coordinates.csv, Zarr
- **GPU acceleration**: Optional CUDA support via CuPy/cuCIM for 10-50x speedup
- **Robust registration**: Phase cross-correlation with SSIM scoring and outlier rejection
- **Memory-efficient**: Chunked processing for large datasets
- **OME-NGFF output**: Zarr v3 with multiscale pyramids

## Installation

### From source (recommended)

```bash
# Clone the repository
git clone https://github.com/cephla/tilefusion.git
cd tilefusion

# Basic installation (CPU only)
pip install -e .

# For development (includes linting tools)
pip install -e ".[dev]"
pre-commit install  # Enable automatic formatting on commit

# With GUI support
pip install -e ".[gui]"

# With GPU support (requires CUDA)
pip install -e ".[gpu]"

# Full installation (GPU + GUI + dev tools)
pip install -e ".[all]"
```

### From PyPI (when published)

```bash
pip install tilefusion
pip install tilefusion[gui]  # With GUI
pip install tilefusion[gpu]  # With GPU support
```

## Quick Start

### Python API

```python
from tilefusion import TileFusion

# Create stitcher instance
tf = TileFusion(
    "path/to/tiles.ome.tiff",
    blend_pixels=(50, 50),
    downsample_factors=(2, 2),
)

# Run full pipeline
tf.run()
```

### GUI

```bash
stitcher-gui
```

### CLI

```bash
# Convert individual TIFFs to Zarr
convert-to-zarr path/to/folder -o output.zarr
```

## Supported Formats

### Input
- **OME-TIFF**: Multi-series TIFF with OME-XML metadata
- **Individual TIFFs**: Folder with `manual_{fov}_{z}_{channel}.tiff` and `coordinates.csv`
- **Zarr**: Zarr v3 with `per_index_metadata` stage positions

### Output
- **OME-NGFF Zarr v3**: With multiscale pyramids for efficient visualization

## Acknowledgments

This project is based on the [tilefusion module](https://github.com/QI2lab/opm-processing-v2/blob/tilefusion2D/src/opm_processing/imageprocessing/tilefusion.py) from [opm-processing-v2](https://github.com/QI2lab/opm-processing-v2) by [Doug Shepherd](https://github.com/dpshepherd) and the QI2lab team at Arizona State University.

## License

BSD-3-Clause
