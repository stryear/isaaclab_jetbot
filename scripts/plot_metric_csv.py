#!/usr/bin/env python3
"""Plot paper-style curves from exported metric CSV files.

Examples:
  python scripts/plot_metric_csv.py logs/exports/metrics_csv_test2/*.csv --output plots/runs.png
  python scripts/plot_metric_csv.py --summary-csv logs/exports/metrics_csv_test2/summary.csv --output plots/summary.png
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METRICS = ["success", "collision", "timeout", "avg_goal_end_dist"]


def read_detail_csv(path: Path) -> Dict[str, List[float]]:
    columns: Dict[str, List[float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key, value in row.items():
                columns.setdefault(key, [])
                try:
                    columns[key].append(float(value))
                except (TypeError, ValueError):
                    columns[key].append(np.nan)
    return columns


def moving_average(values: List[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    valid = np.convolve(arr, kernel, mode="valid")
    prefix = np.full(window - 1, np.nan)
    return np.concatenate([prefix, valid])


def resolve_detail_csvs(inputs: List[str], summary_csv: str | None) -> List[Tuple[str, Path]]:
    results: List[Tuple[str, Path]] = []
    seen = set()

    if summary_csv:
        with Path(summary_csv).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                path = Path(row["detail_csv"])
                key = (row["run_name"], path)
                if path.is_file() and key not in seen:
                    results.append((row["run_name"], path))
                    seen.add(key)

    for pattern in inputs:
        matched = [Path(p) for p in sorted(glob.glob(pattern))] if any(ch in pattern for ch in "*?[]") else [Path(pattern)]
        for path in matched:
            key = (path.stem, path)
            if path.is_file() and key not in seen:
                results.append((path.stem, path))
                seen.add(key)

    return results


def infer_ylabel(metric: str) -> str:
    if metric in {"success", "collision", "timeout", "coll_lidar", "coll_contact"}:
        return f"{metric.replace('_', ' ').title()} (%)"
    if metric == "avg_goal_end_dist":
        return "Goal Distance (m)"
    if metric == "avg_len":
        return "Episode Length"
    if metric == "avg_return":
        return "Return"
    return metric.replace("_", " ").title()


def plot_runs(
    runs: List[Tuple[str, Path]],
    metrics: List[str],
    x_key: str,
    smooth_window: int,
    output: Path,
    title: str | None,
    dpi: int,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 2.8 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs), 1)))

    for color, (label, path) in zip(colors, runs):
        data = read_detail_csv(path)
        x = np.asarray(data.get(x_key, []), dtype=float)
        if x.size == 0:
            continue

        for ax, metric in zip(axes, metrics):
            y = data.get(metric)
            if not y:
                continue
            y_smoothed = moving_average(y, smooth_window)
            ax.plot(x, y_smoothed, label=label, color=color, linewidth=2.0)
            ax.set_ylabel(infer_ylabel(metric))
            ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel(x_key.replace("_", " ").title())
    axes[0].legend(loc="best", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
    else:
        fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot paper-style curves from metric CSV files")
    parser.add_argument("inputs", nargs="*", help="Detail CSV files or glob patterns")
    parser.add_argument("--summary-csv", help="Optional summary.csv to auto-load detail_csv entries")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, help="Metrics to plot")
    parser.add_argument("--x-key", default="step", help="Column to use as x-axis")
    parser.add_argument("--smooth-window", type=int, default=15, help="Moving-average window")
    parser.add_argument("--output", default="logs/exports/paper_plots/training_curves.png", help="Output image path")
    parser.add_argument("--title", help="Optional figure title")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    args = parser.parse_args()

    runs = resolve_detail_csvs(args.inputs, args.summary_csv)
    if not runs:
        raise SystemExit("No detail CSV files found")

    plot_runs(
        runs=runs,
        metrics=args.metrics,
        x_key=args.x_key,
        smooth_window=args.smooth_window,
        output=Path(args.output),
        title=args.title,
        dpi=args.dpi,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
