"""Overlay memory-footprint timelines from multiple commits onto one figure.

One labeled curve per optimization, so the per-commit impact is visible at a
glance. Peak RSS is annotated and shown in the legend.

Usage:
  python -m profiling.overlay_commits --out overlay_commits.png \
      "Before (original):/path/prof_baseline" \
      "Round 1 — buffer reuse:/path/prof_round1" \
      "Round 2 — fixed block:/path/prof_round2"

Each positional arg is "<legend label>:<dir-or-timeline.csv>". A directory uses
its timeline.csv. Labels must not contain a colon.
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Muted red -> amber -> green: reads as "worse -> better" for a non-technical audience.
PALETTE = ["#C44E52", "#DD8452", "#55A868", "#4C72B0", "#8172B3", "#937860"]


def _read_timeline(path):
    if os.path.isdir(path):
        path = os.path.join(path, "timeline.csv")
    t_s, rss = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            t_s.append(float(row["t_ms"]) / 1000.0)
            rss.append(float(row["rss_mb"]))
    return t_s, rss


def build_parser():
    p = argparse.ArgumentParser(description="Overlay memory timelines from multiple commits.")
    p.add_argument("series", nargs="+", help='"<label>:<dir-or-csv>" per commit')
    p.add_argument("--out", default="overlay_commits.png")
    p.add_argument("--title", default="Fusion memory footprint per optimization")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    fig, ax = plt.subplots(figsize=(10, 6))
    peaks = []
    for i, spec in enumerate(args.series):
        label, _, path = spec.partition(":")
        t_s, rss = _read_timeline(path)
        if not rss:
            continue
        peak = max(rss)
        peak_t = t_s[rss.index(peak)]
        peaks.append((label, peak))
        color = PALETTE[i % len(PALETTE)]
        ax.plot(t_s, rss, color=color, linewidth=1.8, label=f"{label}  —  peak {peak:.0f} MB")
        ax.scatter([peak_t], [peak], s=28, color=color, zorder=5, edgecolor="white", linewidth=0.8)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("process RSS (MB)")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    for label, peak in peaks:
        print(f"  {label}: peak {peak:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
