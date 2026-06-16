"""Phase-1 figures. Agg backend so it runs headless."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from profiling.sampler import Sample  # noqa: E402
from profiling.stages import Span  # noqa: E402

_PALETTE = ["#1565c0", "#26a69a", "#5c6bc0", "#c9a227", "#a1707f", "#78909c"]


def _draw_stage_boundaries(ax, spans):
    """Dashed vertical at each span end; stage name rotated -90 at the top."""
    ymax = ax.get_ylim()[1]
    for name, _start, end in spans:
        ax.axvline(end / 1000.0, color="#90a4ae", linestyle="--", linewidth=1)
        ax.text(
            end / 1000.0,
            ymax,
            name,
            rotation=-90,
            va="top",
            ha="right",
            fontsize=8,
            color="#37474f",
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
    ax.plot(t_s, rss, label="TOTAL RSS", color="#263238", linewidth=2.4)
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
