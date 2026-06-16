# Test condition — acquisition profiled (before & after)

The same dataset/region was used for the pre- and post-optimization profiles.

| Field | Value |
|---|---|
| Dataset | `test_10x_laser_af_z_stack` (acquired 2025-10-28) |
| Format | OME-TIFF tiles (one OME-TIFF file per FOV) |
| Profiled region | `manual0` — 27 FOVs (dataset also contains `manual1`, 28 FOVs) |
| Tile size | 2084 × 2084 px (Y × X) |
| Per-tile data | axes `ZCYX` = 10 z × 4 channels × 2084 × 2084, `uint16` (~347 MB/tile) |
| Channels | 4 |
| Z-levels | 10 (Δz = 1.5 µm) |
| Timepoints | 1 |
| Pixel size | 0.752 µm/px (10× objective, NA 0.30, sensor 7.52 µm) |
| Grid / scan pattern | ~5-column **raster** |
| Registration | 43 adjacent FOV pairs; channel 0, middle z-level |
| Fused output (manual0) | 12,288 × 10,240 px per plane (~126 megapixels) × 4 ch × 10 z, `uint16` |
| Dataset size on disk | ~18 GB (`ome_tiff/`) |
| Hardware | profiled on the dev Mac (Apple Silicon) |

Notes
- Memory profiling used `psutil` RSS + `tracemalloc`; cross-platform, no per-OS code.
- The fuse optimization is output-identical (guarded by `tests/test_fuse_equivalence.py`).
- The registration **swimlanes** figure is unchanged by the fuse optimization
  (registration was not modified); the same figure appears in both `pre-opt/`
  and `post-opt/` for convenience.
