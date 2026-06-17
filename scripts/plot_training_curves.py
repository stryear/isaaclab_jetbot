#!/usr/bin/env python3
"""
Training visualization tool - Plot training curves from logs
Usage: python scripts/plot_training_curves.py v17 --metrics success collision return
"""

import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List
import numpy as np


def parse_metrics_from_log(log_path: Path) -> Dict[str, List]:
    """Extract time-series metrics from log file."""
    data = {
        "iteration": [],
        "success": [],
        "collision": [],
        "timeout": [],
        "avg_return": [],
        "avg_len": [],
        "avg_goal_end_dist": [],
        "avg_min_lidar": [],
    }

    with open(log_path, 'r') as f:
        for line in f:
            if '[METRIC][LidarNav]' in line:
                # Parse metrics line
                match = re.search(
                    r'iter=(\d+).*success=([\d.]+)%.*collision=([\d.]+)%.*timeout=([\d.]+)%'
                    r'.*avg_len=([\d.]+).*avg_return=([-\d.]+)'
                    r'.*avg_goal_end_dist=([\d.]+).*avg_min_lidar=([\d.]+)',
                    line
                )
                if match:
                    data["iteration"].append(int(match.group(1)))
                    data["success"].append(float(match.group(2)))
                    data["collision"].append(float(match.group(3)))
                    data["timeout"].append(float(match.group(4)))
                    data["avg_len"].append(float(match.group(5)))
                    data["avg_return"].append(float(match.group(6)))
                    data["avg_goal_end_dist"].append(float(match.group(7)))
                    data["avg_min_lidar"].append(float(match.group(8)))

    return data


def plot_metrics(versions: List[str], metrics: List[str], log_dir: Path, output_path: Path = None):
    """Plot training curves for specified metrics."""
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(versions)))

    for version, color in zip(versions, colors):
        # Find log file
        pattern = f"short_nav_{version}_*.log"
        log_files = list(log_dir.glob(pattern))

        if not log_files:
            print(f"⚠️  No log found for {version}")
            continue

        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        data = parse_metrics_from_log(latest_log)

        if not data["iteration"]:
            print(f"⚠️  No metrics found in {latest_log.name}")
            continue

        # Plot each metric
        for ax, metric in zip(axes, metrics):
            if metric in data and data[metric]:
                ax.plot(data["iteration"], data[metric], label=version, color=color, linewidth=2)

    # Format plots
    for ax, metric in zip(axes, metrics):
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add target lines for key metrics
        if metric == "success":
            ax.axhline(y=55, color='green', linestyle='--', alpha=0.5, label='Target (55%)')
        elif metric == "collision":
            ax.axhline(y=40, color='red', linestyle='--', alpha=0.5, label='Target (<40%)')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument("versions", nargs="+", help="Versions to plot (e.g., v16 v17 v18)")
    parser.add_argument("--metrics", nargs="+",
                        default=["success", "collision", "avg_return"],
                        help="Metrics to plot")
    parser.add_argument("--log-dir", default="logs/skrl", help="Log directory")
    parser.add_argument("--output", help="Save plot to file (e.g., training_curves.png)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_path = Path(args.output) if args.output else None

    plot_metrics(args.versions, args.metrics, log_dir, output_path)


if __name__ == "__main__":
    main()
