"""Variability (coefficient of variation) of per-pair metrics.

A high CV across pairs flags inconsistent per-pair work — a signal of
algorithmic inefficiency worth investigating.
"""

import math
from typing import Dict, List

from profiling.perpair import PairRecord

_METRICS = ("duration_ms", "patch_bytes_total")


def _stats(values: List[float]) -> Dict[str, float]:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n  # population variance
    std = math.sqrt(var)
    cv = (std / mean) if mean else 0.0
    return {"mean": mean, "std": std, "cv": cv, "min": min(values), "max": max(values)}


def compute_pair_stats(records: List[PairRecord]) -> Dict:
    """Per-metric mean/std/cv/min/max across all pairs."""
    out: Dict = {"n_pairs": len(records)}
    if not records:
        return out
    for metric in _METRICS:
        out[metric] = _stats([float(getattr(r, metric)) for r in records])
    return out
