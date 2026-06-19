"""Generate a synthetic 4-FOV registration fixture by tiling one real FOV.

See docs/superpowers/specs/2026-06-19-registration-quality-fixture-design.md.
"""
import csv
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import map_coordinates

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
