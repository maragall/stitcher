"""Map tile indices to grid (row, col), labels, and infer the scan pattern."""

from typing import Dict, List, Tuple


def build_grid(
    tile_positions: List[Tuple[float, float]], decimals: int = 0
) -> Dict[int, Tuple[int, int]]:
    """Map each tile index to (row, col) by ranking unique y (rows) and x (cols).

    tile_positions are (y_um, x_um). Rows increase with y, cols with x.
    """
    ys = sorted({round(y, decimals) for y, _ in tile_positions})
    xs = sorted({round(x, decimals) for _, x in tile_positions})
    row_of = {y: r for r, y in enumerate(ys)}
    col_of = {x: c for c, x in enumerate(xs)}
    return {
        idx: (row_of[round(y, decimals)], col_of[round(x, decimals)])
        for idx, (y, x) in enumerate(tile_positions)
    }


def tile_label(identifiers: List[Tuple], grid: Dict[int, Tuple[int, int]], idx: int) -> str:
    """Human label like "manual0/fov1@(r0,c1)" (falls back to index if no id)."""
    r, c = grid[idx]
    if identifiers and idx < len(identifiers):
        region, fov = identifiers[idx]
        return f"{region}/fov{fov}@(r{r},c{c})"
    return f"tile{idx}@(r{r},c{c})"


def infer_scan_pattern(grid_sequence: List[Tuple[int, int]]) -> str:
    """Classify acquisition order (a list of (row, col) in tile-index order).

    Returns "raster" (every row same column direction), "serpentine"
    (direction alternates per row), or "unknown".
    """
    by_row: Dict[int, List[int]] = {}
    for r, c in grid_sequence:
        by_row.setdefault(r, []).append(c)

    directions = []
    for r in sorted(by_row):
        cols = by_row[r]
        if len(cols) < 2:
            continue
        directions.append(1 if cols[-1] > cols[0] else -1)

    if not directions:
        return "unknown"
    if all(d == 1 for d in directions):
        return "raster"
    if all(directions[k] == -directions[k - 1] for k in range(1, len(directions))):
        return "serpentine"
    return "unknown"
