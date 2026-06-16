import csv
from profiling.perpair import PairRecord
from profiling.record import write_pairs_csv


def test_write_pairs_csv(tmp_path):
    records = [PairRecord(0, 3, 4, 100, 150, 250, 12.5)]
    grid = {3: (0, 3), 4: (0, 4)}
    identifiers = [("m", 0), ("m", 1), ("m", 2), ("m", 3), ("m", 4)]
    path = tmp_path / "pairs.csv"
    write_pairs_csv(str(path), records, grid, identifiers)

    rows = list(csv.DictReader(path.open()))
    assert rows[0]["pair_id"] == "0"
    assert rows[0]["i"] == "3"
    assert rows[0]["tile_i"] == "m/fov3@(r0,c3)"
    assert rows[0]["tile_j"] == "m/fov4@(r0,c4)"
    assert rows[0]["patch_bytes_total"] == "250"
    assert float(rows[0]["duration_ms"]) == 12.5
