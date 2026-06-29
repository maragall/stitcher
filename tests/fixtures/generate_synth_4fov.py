"""Generate a synthetic 4-FOV registration fixture by tiling one real FOV.

Geometry is exact ground truth: four tiles are cropped from one source plane at known
fractional offsets, so the true pairwise displacement is `o_j - o_i` by construction. Each
tile is then an INDEPENDENT, realistically degraded observation of its crop — defocus,
illumination/vignette + background, scan-order bleaching, and independent Poisson+Gaussian
noise. A real overlap images the same physical sample, so a single source is the correct base
for the shared signal; the per-tile nuisances are what make registration generalize to real
data. Degradations change intensity/sharpness only — they never move a pixel, so the offsets
stay exact.

The grid is inset (windows stay strictly inside the source) so no pixel is ever fabricated by
edge handling. See docs/registration-quality-fixture-deep-dive.md.
"""

import csv
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, map_coordinates

TILE = 1280
# Inset the grid on ALL sides by MARGIN so o = base + jitter + backlash stays inside
# [0, source - TILE] at both the low edge (negative jitter on fov0) and the high edge
# (positive jitter on the far tiles) -> no window ever leaves the source -> no edge-reflect
# fabrication. MARGIN 24 comfortably covers max |jitter|+|backlash| (~8 px) + the bicubic halo;
# STEP 984 keeps the far tiles in-bounds for the 2304 source. Overlap = TILE - STEP = 296 px.
STEP = 984
MARGIN = 24
GRID = [(0, 0), (0, 1), (1, 0), (1, 1)]  # raster row-major fov order; (row, col)
BASE_OFFSETS = np.array(
    [(MARGIN + r * STEP, MARGIN + c * STEP) for (r, c) in GRID], dtype=np.float64
)  # (y,x) px

SOURCE = Path("/Users/julioamaragall/CEPHLA/Data/20x_FoxChase_488_555_640")
FIXTURE_DIR = Path(__file__).parent / "synth_4fov"

# Moderate, realistic default degradation severities — chosen so registration still recovers
# the injected offset within tolerance, while being representative. All are parameters so a
# later (non-gating) robustness sweep can push them harder.
DEFOCUS_MAX = 0.6  # per-tile Gaussian blur sigma drawn from [0, this] px (focus drift)
VIGNETTE_MAX = 0.25  # per-tile radial illumination falloff, fraction
BG_FRAC = 0.02  # additive background, fraction of the tile median
BLEACH = 0.03  # scan-order intensity drop: gain = 1 - BLEACH * fov_index
READ_SIGMA = 2.0  # Gaussian read-noise std (counts); Poisson supplies shot noise


def backlash_offsets(b_px: float) -> np.ndarray:
    """Raster row-major backlash: x reverses into fov2 (+b) and fov3 (-b). (y,x) px."""
    return np.array([(0.0, 0.0), (0.0, 0.0), (0.0, b_px), (0.0, -b_px)], dtype=np.float64)


def compute_content_offsets(seed: int = 42, sigma_px: float = 1.5, b_px: float = 3.0) -> np.ndarray:
    """Actual content offset o_k = base + jitter + backlash, (4,2) (y,x) px."""
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0.0, sigma_px, size=(4, 2))
    return BASE_OFFSETS + jitter + backlash_offsets(b_px)


def sample_tile(plane: np.ndarray, oy: float, ox: float, size: int = TILE) -> np.ndarray:
    """Bicubic-resample a size x size window at fractional top-left (oy, ox). Returns float64.

    Caller must keep the window in-bounds; generate_fixture asserts this, so map_coordinates
    never falls off the edge and never fabricates pixels.
    """
    ys = oy + np.arange(size)
    xs = ox + np.arange(size)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    coords = np.stack([gy.ravel(), gx.ravel()])
    return map_coordinates(plane.astype(np.float64), coords, order=3, mode="reflect").reshape(
        size, size
    )


def _illumination_field(shape, rng, strength) -> np.ndarray:
    """Smooth multiplicative radial vignette with a per-tile center. Values ~[1-strength, 1]."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    yy /= h - 1
    xx /= w - 1
    cy, cx = rng.uniform(0.35, 0.65, size=2)
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return 1.0 - strength * (r2 / r2.max())


def degrade_tile(clean: np.ndarray, fov: int, seed: int) -> np.ndarray:
    """Independent per-tile nuisances on a clean float crop -> uint16. Deterministic per (seed, fov).

    Intensity/sharpness only; no pixel is moved, so the geometric offset is unchanged.
    """
    rng = np.random.default_rng([seed, 1000 + fov])
    img = clean.astype(np.float64)
    sigma = float(rng.uniform(0.0, DEFOCUS_MAX))
    if sigma > 0:
        img = gaussian_filter(img, sigma)
    img = img * _illumination_field(
        img.shape, rng, float(rng.uniform(0.5 * VIGNETTE_MAX, VIGNETTE_MAX))
    )
    img = img * (1.0 - BLEACH * fov)
    img = img + BG_FRAC * float(np.median(clean))
    noisy = rng.poisson(np.clip(img, 0.0, None)) + rng.normal(0.0, READ_SIGMA, size=img.shape)
    return np.clip(noisy, 0, 65535).round().astype(np.uint16)


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
    h, w = plane.shape

    params = json.loads((source / "acquisition parameters.json").read_text())
    px_um = params["sensor_pixel_size_um"] / params["objective"]["magnification"]
    px_mm = px_um / 1000.0

    o = compute_content_offsets(seed, sigma_px, b_px)  # (4,2) (y,x) px
    # No fabrication: every window must stay strictly inside the source.
    for k in range(4):
        assert 0.0 <= o[k, 0] and o[k, 0] + TILE <= h, f"fov{k} y-window out of bounds: {o[k]}"
        assert 0.0 <= o[k, 1] and o[k, 1] + TILE <= w, f"fov{k} x-window out of bounds: {o[k]}"

    for k in range(4):
        clean = sample_tile(plane, o[k, 0], o[k, 1])
        tile = degrade_tile(clean, k, seed)
        tifffile.imwrite(
            out_dir / "ome_tiff" / f"synth_{k}.ome.tiff",
            tile,
            compression="zlib",
            datetime=False,
            software=False,
            metadata=None,
            ome=False,
        )

    with open(out_dir / "coordinates.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["region", "x (mm)", "y (mm)", "z (mm)"])
        for k in range(4):
            by, bx = BASE_OFFSETS[k]
            wr.writerow(["synth", bx * px_mm, by * px_mm, 0.0])

    json.dump(
        {"sensor_pixel_size_um": 6.5, "objective": {"magnification": 20.0}, "Nz": 1},
        open(out_dir / "acquisition parameters.json", "w"),
    )

    gt = {
        "params": {
            "seed": seed,
            "sigma_px": sigma_px,
            "b_px": b_px,
            "scan": "raster",
            "channel_idx": channel_idx,
            "z_level": z_level,
            "pixel_size_um": px_um,
            "step_px": STEP,
            "overlap_px": TILE - STEP,
            "degradation": {
                "defocus_max": DEFOCUS_MAX,
                "vignette_max": VIGNETTE_MAX,
                "bg_frac": BG_FRAC,
                "bleach": BLEACH,
                "read_sigma": READ_SIGMA,
            },
        },
        # geometry is ground truth; degradations do not move pixels, so offset/error are exact.
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


def capture_golden(fixture_dir=FIXTURE_DIR) -> None:
    """Run the pipeline on the committed fixture and freeze its output as the golden."""
    from tilefusion.core import TileFusion

    tf = TileFusion(fixture_dir, region="synth", channel_to_use=0)
    tf.refine_tile_positions_with_cross_correlation()
    tf.optimize_shifts()
    golden = {
        "pairwise": {
            f"{i},{j}": [float(dy), float(dx), float(s)]
            for (i, j), (dy, dx, s) in tf.pairwise_metrics.items()
        },
        "global": np.asarray(tf.global_offsets, dtype=np.float64).tolist(),
    }
    with open(Path(fixture_dir) / "golden_metrics.json", "w") as f:
        json.dump(golden, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    generate_fixture(FIXTURE_DIR)
    print(f"wrote fixture to {FIXTURE_DIR}")
