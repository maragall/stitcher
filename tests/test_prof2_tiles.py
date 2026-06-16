from profiling.tiles import build_grid, tile_label, infer_scan_pattern


def test_tile_label_falls_back_for_non_pair_identifiers():
    grid = {0: (0, 0), 1: (0, 1)}
    # 1-element identifiers (simple directory layouts) must not crash
    assert tile_label([(7,), (8,)], grid, 1) == "tile1@(r0,c1)"
    # empty identifiers fall back too
    assert tile_label([], grid, 0) == "tile0@(r0,c0)"


def test_build_grid_assigns_row_col():
    # 2 rows x 3 cols raster; positions are (y_um, x_um)
    positions = [
        (10.0, 0.0),
        (10.0, 5.0),
        (10.0, 10.0),
        (20.0, 0.0),
        (20.0, 5.0),
        (20.0, 10.0),
    ]
    grid = build_grid(positions)
    assert grid[0] == (0, 0)
    assert grid[2] == (0, 2)
    assert grid[3] == (1, 0)
    assert grid[5] == (1, 2)


def test_tile_label_uses_identifier_and_grid():
    positions = [(10.0, 0.0), (10.0, 5.0)]
    identifiers = [("manual0", 0), ("manual0", 1)]
    grid = build_grid(positions)
    assert tile_label(identifiers, grid, 1) == "manual0/fov1@(r0,c1)"


def test_infer_scan_pattern_raster_vs_serpentine():
    # raster: every row left->right
    raster = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert infer_scan_pattern(raster) == "raster"
    # serpentine: row 0 left->right, row 1 right->left
    serp = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]
    assert infer_scan_pattern(serp) == "serpentine"
