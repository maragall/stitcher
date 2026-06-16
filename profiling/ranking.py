"""Rank functions by integrated memory cost (MB·seconds)."""

from typing import Dict, List

from profiling.attribution import AllocRecord


def _trapz_mb_s(series: List) -> float:
    """Trapezoidal integral of (t_ms, mb) points -> MB·seconds."""
    total = 0.0
    for (t0, v0), (t1, v1) in zip(series, series[1:]):
        dt_s = (t1 - t0) / 1000.0
        total += 0.5 * (v0 + v1) * dt_s
    return total


def compute_ranking(records: List[AllocRecord]) -> List[Dict]:
    """Aggregate per-function: peak_mb, integrated_mb_s, pct_of_total; sorted desc."""
    by_func: Dict[str, List] = {}
    for r in records:
        by_func.setdefault(r.func, []).append((r.t_ms, r.size_mb))

    out = []
    for func, series in by_func.items():
        series.sort()
        out.append(
            {
                "function": func,
                "peak_mb": max(v for _, v in series),
                "integrated_mb_s": _trapz_mb_s(series),
            }
        )

    total = sum(o["integrated_mb_s"] for o in out) or 1.0
    for o in out:
        o["pct_of_total"] = 100.0 * o["integrated_mb_s"] / total

    out.sort(key=lambda o: o["integrated_mb_s"], reverse=True)
    return out
