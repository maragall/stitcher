import os
import pytest
from profiling.harness import profile_registration_perpair, PairProfileResult

DATASET = os.path.expanduser(
    "~/Cephla/Data/test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy"
)


def test_pair_profile_result_fields_exist():
    r = PairProfileResult(records=[], tile_positions=[], tile_identifiers=[], tile_shape=(0, 0))
    assert r.records == []
    assert r.tile_positions == []
    assert r.tile_identifiers == []
    assert r.tile_shape == (0, 0)


@pytest.mark.skipif(not os.path.isdir(DATASET), reason="profiling dataset not present")
def test_profile_registration_perpair_smoke():
    result = profile_registration_perpair(DATASET, region="manual0")
    assert len(result.records) > 0
    assert len(result.tile_positions) == len(result.tile_identifiers)
    n = len(result.tile_positions)
    assert all(0 <= rec.i < n and 0 <= rec.j < n for rec in result.records)
    assert all(rec.patch_bytes_total > 0 for rec in result.records)
