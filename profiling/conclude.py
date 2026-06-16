"""Phase 5 conclusion: overlay before/after timelines and print metric bullets.

Usage:
  python -m profiling.conclude --before <dir-or-csv> --after <dir-or-csv> --out <dir>
"""

import argparse
import csv
import os

from profiling.plots import plot_timeline_overlay


def _read_timeline(path):
    """Read a timeline.csv into a list of (t_s, rss_mb)."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [(float(r["t_ms"]) / 1000.0, float(r["rss_mb"])) for r in rows]


def _peak(series):
    return max((rss for _, rss in series), default=0.0)


def _mean(series):
    return (sum(rss for _, rss in series) / len(series)) if series else 0.0


def _metrics(before, after):
    bp, ap = _peak(before), _peak(after)
    bm, am = _mean(before), _mean(after)
    return {
        "before_peak": bp,
        "after_peak": ap,
        "before_mean": bm,
        "after_mean": am,
        "peak_pct_improvement": (100.0 * (bp - ap) / bp) if bp else 0.0,
        "mean_pct_improvement": (100.0 * (bm - am) / bm) if bm else 0.0,
    }


def _resolve(path):
    """Accept a directory (use timeline.csv inside) or a direct CSV path."""
    return os.path.join(path, "timeline.csv") if os.path.isdir(path) else path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Before/after memory conclusion.")
    p.add_argument("--before", required=True, help="Baseline dir or timeline.csv")
    p.add_argument("--after", required=True, help="Post-opt dir or timeline.csv")
    p.add_argument("--out", default="profile_out", help="Output directory")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    before = _read_timeline(_resolve(args.before))
    after = _read_timeline(_resolve(args.after))
    m = _metrics(before, after)

    plot_timeline_overlay(
        before,
        after,
        os.path.join(args.out, "overlay.png"),
        before_peak=m["before_peak"],
        after_peak=m["after_peak"],
    )

    print("## Memory footprint — before vs after")
    print(
        f"- Peak: {m['before_peak']:.0f} MB -> {m['after_peak']:.0f} MB "
        f"({m['peak_pct_improvement']:.1f}% lower)"
    )
    print(
        f"- Mean: {m['before_mean']:.0f} MB -> {m['after_mean']:.0f} MB "
        f"({m['mean_pct_improvement']:.1f}% lower)"
    )
    print(f"- Overlay figure: {os.path.join(args.out, 'overlay.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
