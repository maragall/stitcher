import types
from profiling.perpair import PairRecorder, PairRecord


class _FakePatch:
    def __init__(self, nbytes):
        self.nbytes = nbytes


def test_pair_recorder_wraps_records_and_restores():
    calls = []

    def fake_worker(args):
        calls.append(args)
        return (args[0], args[1], 1, 2, 0.9)  # mimic (i, j, dy, dx, score)

    target = types.SimpleNamespace(register_pair_worker=fake_worker)
    original = target.register_pair_worker

    with PairRecorder(target=target) as rec:
        args = (3, 4, _FakePatch(100), _FakePatch(150), None, None, None, None)
        out = target.register_pair_worker(args)

    # original behavior preserved (delegates + returns)
    assert out == (3, 4, 1, 2, 0.9)
    assert calls == [args]
    # restored after context
    assert target.register_pair_worker is original

    assert len(rec.records) == 1
    r = rec.records[0]
    assert isinstance(r, PairRecord)
    assert (r.i, r.j) == (3, 4)
    assert r.patch_i_bytes == 100
    assert r.patch_j_bytes == 150
    assert r.patch_bytes_total == 250
    assert r.duration_ms >= 0.0
    assert r.pair_id == 0
