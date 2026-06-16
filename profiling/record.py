"""Write profiler results to CSV."""

import csv
from typing import Dict, List

from profiling.sampler import Sample
from profiling.stages import Span, assign_stages
from profiling.tiles import tile_label


def write_timeline_csv(path: str, samples: List[Sample], spans: List[Span]) -> None:
    rows = assign_stages(samples, spans)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ms", "rss_mb", "stage"])
        for t_ms, rss_mb, stage in rows:
            w.writerow([f"{t_ms:.1f}", f"{rss_mb:.3f}", stage])


def write_functions_csv(path: str, ranking: List[Dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["function", "peak_mb", "integrated_mb_s", "pct_of_total"])
        for o in ranking:
            w.writerow(
                [
                    o["function"],
                    f'{o["peak_mb"]:.3f}',
                    f'{o["integrated_mb_s"]:.3f}',
                    f'{o["pct_of_total"]:.2f}',
                ]
            )


def write_pairs_csv(path, records, grid, identifiers) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "pair_id",
                "i",
                "j",
                "tile_i",
                "tile_j",
                "row_i",
                "col_i",
                "row_j",
                "col_j",
                "patch_i_bytes",
                "patch_j_bytes",
                "patch_bytes_total",
                "duration_ms",
            ]
        )
        for r in records:
            ri, ci = grid[r.i]
            rj, cj = grid[r.j]
            w.writerow(
                [
                    r.pair_id,
                    r.i,
                    r.j,
                    tile_label(identifiers, grid, r.i),
                    tile_label(identifiers, grid, r.j),
                    ri,
                    ci,
                    rj,
                    cj,
                    r.patch_i_bytes,
                    r.patch_j_bytes,
                    r.patch_bytes_total,
                    f"{r.duration_ms:.3f}",
                ]
            )
