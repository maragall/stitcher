"""Headless harness: wrap TileFusion stages, run pipeline, collect series."""

import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from profiling.sampler import RSSSampler, Sample
from profiling.attribution import AllocationSampler, AllocRecord
from profiling.stages import Span, StageTimer
from profiling.perpair import PairRecord, PairRecorder

logger = logging.getLogger(__name__)

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
    error: Optional[str] = None


def _wrap_stage(obj, method_name: str, stage: str, timer: StageTimer) -> None:
    """Override a bound method on `obj` to time it under `stage`."""
    original = getattr(obj, method_name)

    def wrapped(*args, **kwargs):
        with timer.stage(stage):
            return original(*args, **kwargs)

    setattr(obj, method_name, wrapped)


def _safe_stop(sampler):
    """Best-effort stop; never raises (so cleanup of other resources continues)."""
    try:
        return sampler.stop()
    except Exception:  # pragma: no cover - cleanup must not mask the real error
        logger.warning("sampler stop failed", exc_info=True)
        return []


def _collect(
    run_callable: Callable[[], None],
    t0: float,
    timer: StageTimer,
    rss_interval: float = 0.05,
    alloc_interval: float = 0.25,
) -> ProfileResult:
    """Run `run_callable` under RSS + allocation sampling.

    Guarantees tracemalloc is stopped and both sampler threads are joined,
    and returns whatever was collected even if `run_callable` raises (the
    traceback is captured in ProfileResult.error).
    """
    rss = RSSSampler(t0, rss_interval)
    alloc = AllocationSampler(t0, alloc_interval)
    samples: List[Sample] = []
    records: List[AllocRecord] = []
    error: Optional[str] = None

    alloc.start_tracing()
    try:
        rss.start()
        alloc.start()
        try:
            run_callable()
        except Exception:
            error = traceback.format_exc()
            logger.warning("profiled run raised:\n%s", error)
        finally:
            samples = _safe_stop(rss)
            records = _safe_stop(alloc)
    finally:
        import tracemalloc

        if tracemalloc.is_tracing():
            tracemalloc.stop()

    return ProfileResult(samples, records, timer.spans, error)


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
    metrics_path.unlink(missing_ok=True)

    timer = StageTimer(t0)
    for method_name, stage in _STAGE_METHODS:
        _wrap_stage(tf, method_name, stage, timer)

    return _collect(tf.run, t0, timer, rss_interval, alloc_interval)


@dataclass
class PairProfileResult:
    records: List[PairRecord]
    tile_positions: List
    tile_identifiers: List
    tile_shape: tuple


def profile_registration_perpair(dataset: str, region: str = "manual0") -> PairProfileResult:
    """Run B: serialized registration with per-pair recording (non-invasive)."""
    from tilefusion import TileFusion

    metrics_name = f"profile_perpair_metrics_{region}.json"
    tf = TileFusion(dataset, region=region, metrics_filename=metrics_name, max_workers=1)

    metrics_path = Path(dataset).parent / metrics_name
    metrics_path.unlink(missing_ok=True)

    with PairRecorder() as rec:
        tf.refine_tile_positions_with_cross_correlation(
            downsample_factors=tf.downsample_factors,
            ch_idx=tf.channel_to_use,
        )

    if not rec.records:
        logger.warning(
            "No pairs recorded — registration likely took the sequential path "
            "(e.g. a single-file OME-TIFF). Per-pair analysis requires a "
            "multi-file format that uses the parallel registration path."
        )

    return PairProfileResult(
        records=list(rec.records),
        tile_positions=list(tf._tile_positions),
        tile_identifiers=list(tf._tile_identifiers),
        tile_shape=(tf.Y, tf.X),
    )
