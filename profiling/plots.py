"""Phase-1 figures. Agg backend so it runs headless."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from profiling.sampler import Sample  # noqa: E402
from profiling.stages import Span  # noqa: E402
from profiling.swimlanes import schedule_lanes  # noqa: E402

_PALETTE = ["#1565c0", "#26a69a", "#5c6bc0", "#c9a227", "#a1707f", "#78909c"]


def _draw_stage_boundaries(ax, spans, min_duration_frac=0.02):
    """Dashed verticals at stage boundaries; stage name centered over its region.

    Each stage's label is drawn at the MIDPOINT of its span (not its edge) so it
    sits over the region it names — a label at the edge is easily misread as
    belonging to the neighbouring region. Stages whose duration is below
    `min_duration_frac` of the total profiled time are omitted: a near-instant
    stage's boundary coincides with its neighbour's and would read as a single
    line. (Such stages are still recorded in the timeline CSV.)
    """
    if not spans:
        return
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    total_ms = max(e for _, _, e in spans) - min(s for _, s, _ in spans)
    min_dur = min_duration_frac * total_ms if total_ms > 0 else 0.0
    kept = [(n, s, e) for (n, s, e) in spans if (e - s) >= min_dur]

    for name, start, end in kept:
        ax.axvline(end / 1000.0, color="#90a4ae", linestyle="--", linewidth=1)
        midpoint = (start + end) / 2.0 / 1000.0
        ax.text(
            midpoint,
            ymax - 0.02 * yrange,
            name,
            rotation=-90,
            va="top",
            ha="center",
            fontsize=8,
            color="#607d8b",
        )


def plot_timeline(samples, spans, out_path):
    t_s = [s.t_ms / 1000.0 for s in samples]
    rss = [s.rss_mb for s in samples]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t_s, rss, color="#1565c0", linewidth=2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Total RSS over time")
    _draw_stage_boundaries(ax, spans)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_function_lines(samples, records, ranking, out_path, top_k=5):
    top = [r["function"] for r in ranking[:top_k]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, func in enumerate(top):
        pts = sorted((r.t_ms / 1000.0, r.size_mb) for r in records if r.func == func)
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, label=func, color=_PALETTE[i % len(_PALETTE)], linewidth=1.5)
    # Bold total RSS line.
    t_s = [s.t_ms / 1000.0 for s in samples]
    rss = [s.rss_mb for s in samples]
    ax.plot(t_s, rss, label="TOTAL RSS", color="#263238", linewidth=1.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Per-function memory (top functions) + total")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pareto(ranking, out_path, top_k=10):
    top = ranking[:top_k]
    names = [r["function"] for r in top]
    vals = [r["integrated_mb_s"] for r in top]
    total = sum(r["integrated_mb_s"] for r in ranking) or 1.0
    cum = []
    running = 0.0
    for v in vals:
        running += v
        cum.append(100.0 * running / total)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(names)), vals, color="#26a69a")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=-90, fontsize=7)
    ax.set_ylabel("Integrated cost (MB·s)")
    ax2 = ax.twinx()
    ax2.plot(range(len(names)), cum, color="#263238", marker="o", linewidth=1.5)
    ax2.set_ylabel("Cumulative %")
    ax2.set_ylim(0, 105)
    ax.set_title("Function ranking (Pareto)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_swimlanes(records, out_path, n_lanes=8, labels=None):
    """Reconstructed registration concurrency: each pair a bar on a worker lane."""
    placed = schedule_lanes(records, n_lanes=n_lanes)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for p in placed:
        x = p["start_ms"] / 1000.0
        w = (p["end_ms"] - p["start_ms"]) / 1000.0
        ax.barh(
            p["lane"],
            w,
            left=x,
            height=0.7,
            color=_PALETTE[p["pair_id"] % len(_PALETTE)],
            edgecolor="white",
            linewidth=0.3,
        )
    ax.set_xlabel("reconstructed time (s)")
    ax.set_ylabel("worker lane")
    ax.set_yticks(range(n_lanes))
    ax.set_title(f"Registration pairs across {n_lanes} workers (reconstructed)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pair_variability(records, out_path):
    """Per-pair duration distribution with mean and CV annotated."""
    durations = [r.duration_ms for r in records]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if durations:
        n = len(durations)
        mean = sum(durations) / n
        var = sum((d - mean) ** 2 for d in durations) / n
        std = var**0.5
        cv = (std / mean) if mean else 0.0
        ax.hist(durations, bins=min(20, max(5, n // 3)), color="#26a69a", edgecolor="white")
        ax.axvline(
            mean, color="#263238", linestyle="--", linewidth=1.5, label=f"mean {mean:.1f} ms"
        )
        ax.set_title(f"Per-pair registration duration (CV = {cv:.2f})")
        ax.legend(fontsize=8)
    ax.set_xlabel("duration (ms)")
    ax.set_ylabel("pairs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scan_pattern(grid, out_path, pattern="unknown"):
    """Scatter tiles at (col, row) and connect them in acquisition (index) order."""
    items = sorted(grid.items())  # by tile index
    cols = [c for _, (r, c) in items]
    rows = [r for _, (r, c) in items]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cols, rows, color="#90a4ae", linewidth=1, zorder=1)
    ax.scatter(cols, rows, c=range(len(items)), cmap="viridis", s=60, zorder=2)
    for order, (idx, (r, c)) in enumerate(items):
        ax.annotate(str(order), (c, r), fontsize=6, ha="center", va="center", color="white")
    ax.set_xlabel("grid column")
    ax.set_ylabel("grid row")
    ax.invert_yaxis()  # row 0 at top
    ax.set_title(f"Tile acquisition order — scan pattern: {pattern}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
