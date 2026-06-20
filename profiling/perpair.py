"""Non-invasive per-pair recorder for the registration stage.

Wraps the module-level ``register_pair_worker`` that ``tilefusion.core`` calls,
recording each pair's tile indices, overlap-patch byte sizes, and wall-clock
duration. Restores the original on exit. Use with ``max_workers=1`` so per-pair
durations are not inflated by concurrency.
"""

import time
from typing import List, NamedTuple, Optional


class PairRecord(NamedTuple):
    pair_id: int
    i: int
    j: int
    patch_i_bytes: int
    patch_j_bytes: int
    patch_bytes_total: int
    duration_ms: float


class PairRecorder:
    def __init__(self, target=None, attr: str = "register_pair_worker"):
        self._target = target  # resolved to tilefusion.core on enter if None
        self._attr = attr
        self._original = None
        self.records: List[PairRecord] = []

    def __enter__(self) -> "PairRecorder":
        if self._target is None:
            import tilefusion.core as core

            self._target = core
        self._original = getattr(self._target, self._attr)
        original = self._original
        records = self.records

        def wrapped(args):
            i_pos, j_pos, patch_i, patch_j = args[0], args[1], args[2], args[3]
            pb_i = int(getattr(patch_i, "nbytes", 0) or 0)
            pb_j = int(getattr(patch_j, "nbytes", 0) or 0)
            t0 = time.perf_counter()
            result = original(args)
            dt = (time.perf_counter() - t0) * 1000.0
            records.append(PairRecord(len(records), i_pos, j_pos, pb_i, pb_j, pb_i + pb_j, dt))
            return result

        setattr(self._target, self._attr, wrapped)
        return self

    def __exit__(self, *exc) -> Optional[bool]:
        setattr(self._target, self._attr, self._original)
        return False
