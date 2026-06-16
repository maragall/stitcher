"""CLI: python -m profiling.cli <dataset> --out <dir> [--region manual0]"""

import argparse
import os
import sys

from profiling.harness import profile_dataset
from profiling.ranking import compute_ranking
from profiling.record import write_timeline_csv, write_functions_csv
from profiling.plots import plot_timeline, plot_function_lines, plot_pareto


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Profile TileFusion memory footprint (Phase 1).")
    p.add_argument("dataset", help="Path to the dataset folder")
    p.add_argument("--out", default="profile_out", help="Output directory")
    p.add_argument("--region", default="manual0", help="Region to profile")
    p.add_argument("--top-k", type=int, default=5, help="Functions to plot as lines")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    result = profile_dataset(args.dataset, region=args.region)
    ranking = compute_ranking(result.alloc_records)

    write_timeline_csv(os.path.join(args.out, "timeline.csv"), result.samples, result.stage_spans)
    write_functions_csv(os.path.join(args.out, "functions.csv"), ranking)
    plot_timeline(result.samples, result.stage_spans, os.path.join(args.out, "timeline.png"))
    plot_function_lines(
        result.samples,
        result.alloc_records,
        ranking,
        os.path.join(args.out, "functions.png"),
        top_k=args.top_k,
    )
    plot_pareto(ranking, os.path.join(args.out, "pareto.png"))

    print(f"Wrote profile to {args.out}")
    if result.error:
        print(
            f"WARNING: profiled run failed mid-pipeline; results are partial:\n{result.error}",
            file=sys.stderr,
        )
    if ranking:
        top2 = sum(r["pct_of_total"] for r in ranking[:2])
        print(f"Top 2 functions explain {top2:.1f}% of integrated memory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
