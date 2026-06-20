"""Reconstruct a concurrent schedule from serially-measured pair durations.

Pairs were measured one at a time (max_workers=1). To visualize how they would
overlap across N real workers, greedily pack each pair (in visit order) onto the
lane that frees earliest. This is a RECONSTRUCTION, not a measured concurrency.
"""

from typing import Dict, List

from profiling.perpair import PairRecord


def schedule_lanes(records: List[PairRecord], n_lanes: int = 8) -> List[Dict]:
    """Greedy earliest-free-lane packing. Returns one dict per pair:
    {pair_id, i, j, lane, start_ms, end_ms}.
    """
    lane_free = [0.0] * max(1, n_lanes)
    placed = []
    for r in records:
        lane = min(range(len(lane_free)), key=lambda l: lane_free[l])
        start = lane_free[lane]
        end = start + r.duration_ms
        lane_free[lane] = end
        placed.append(
            {
                "pair_id": r.pair_id,
                "i": r.i,
                "j": r.j,
                "lane": lane,
                "start_ms": start,
                "end_ms": end,
            }
        )
    return placed
