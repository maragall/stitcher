"""Headless harness: wrap TileFusion stages, run pipeline, collect series."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from profiling.sampler import RSSSampler, Sample
from profiling.attribution import AllocationSampler, AllocRecord
from profiling.stages import Span, StageTimer

# (method_name, stage_label) for the four demarcatable stages inside run().
_STAGE_METHODS = [
    ("refine_tile_positions_with_cross_correlation", "Register"),
    ("optimize_shifts", "Optimize"),
    ("_fuse_tiles", "Fuse"),
    ("_create_multiscales", "Write"),
]


@dataclass
class ProfileResult:
    samples: List[Sample]
    alloc_records: List[AllocRecord]
    stage_spans: List[Span]


def _wrap_stage(obj, method_name: str, stage: str, timer: StageTimer) -> None:
    """Override a bound method on `obj` to time it under `stage`."""
    original = getattr(obj, method_name)

    def wrapped(*args, **kwargs):
        with timer.stage(stage):
            return original(*args, **kwargs)

    setattr(obj, method_name, wrapped)


def profile_dataset(
    dataset: str,
    region: str = "manual0",
    rss_interval: float = 0.05,
    alloc_interval: float = 0.25,
) -> ProfileResult:
    from tilefusion import TileFusion

    t0 = time.perf_counter()
    metrics_name = f"profile_metrics_{region}.json"

    tf = TileFusion(dataset, region=region, metrics_filename=metrics_name)

    # Force the Register stage to actually run (don't load cached metrics).
    metrics_path = Path(dataset).parent / metrics_name
    if metrics_path.exists():
        metrics_path.unlink()

    timer = StageTimer(t0)
    for method_name, stage in _STAGE_METHODS:
        _wrap_stage(tf, method_name, stage, timer)

    rss = RSSSampler(t0, rss_interval)
    alloc = AllocationSampler(t0, alloc_interval)
    alloc.start_tracing()
    rss.start()
    alloc.start()
    try:
        tf.run()
    finally:
        samples = rss.stop()
        records = alloc.stop()

    return ProfileResult(samples, records, timer.spans)
