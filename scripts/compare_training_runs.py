#!/usr/bin/env python3
"""
Training comparison tool - Compare metrics across multiple training runs
Usage: python scripts/compare_training_runs.py v16 v17 v18
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
import json


def parse_log_file(log_path: Path) -> Dict:
    """Extract key metrics from training log."""
    metrics = {
        "version": log_path.stem.split("_")[2] if "_" in log_path.stem else "unknown",
        "iterations": [],
        "success_rate": [],
        "collision_rate": [],
        "timeout_rate": [],
        "avg_return": [],
        "avg_len": [],
        "final_success": None,
        "final_collision": None,
        "peak_success": None,
        "peak_iteration": None,
    }

    with open(log_path, 'r') as f:
        for line in f:
            if '[METRIC][LidarNav]' in line:
                # Parse: iter=150 step=9618 success=33.7% collision=57.4% timeout=8.9%
                match = re.search(
                    r'iter=(\d+).*success=([\d.]+)%.*collision=([\d.]+)%.*timeout=([\d.]+)%'
                    r'.*avg_len=([\d.]+).*avg_return=([-\d.]+)',
                    line
                )
                if match:
                    iter_num = int(match.group(1))
                    success = float(match.group(2))
                    collision = float(match.group(3))
                    timeout = float(match.group(4))
                    avg_len = float(match.group(5))
                    avg_return = float(match.group(6))

                    metrics["iterations"].append(iter_num)
                    metrics["success_rate"].append(success)
                    metrics["collision_rate"].append(collision)
                    metrics["timeout_rate"].append(timeout)
                    metrics["avg_len"].append(avg_len)
                    metrics["avg_return"].append(avg_return)

    if metrics["success_rate"]:
        metrics["final_success"] = metrics["success_rate"][-1]
        metrics["final_collision"] = metrics["collision_rate"][-1]
        metrics["peak_success"] = max(metrics["success_rate"])
        metrics["peak_iteration"] = metrics["iterations"][
            metrics["success_rate"].index(metrics["peak_success"])
        ]

    return metrics


def print_comparison_table(all_metrics: List[Dict]):
    """Print formatted comparison table."""
    print("\n" + "="*80)
    print("Training Runs Comparison")
    print("="*80)

    # Header
    print(f"\n{'Version':<10} {'Final Success':<15} {'Peak Success':<15} "
          f"{'Final Collision':<18} {'Iterations':<12}")
    print("-"*80)

    # Rows
    for m in all_metrics:
        if m["final_success"] is not None:
            print(f"{m['version']:<10} "
                  f"{m['final_success']:>6.1f}%        "
                  f"{m['peak_success']:>6.1f}% (iter {m['peak_iteration']:<4})  "
                  f"{m['final_collision']:>6.1f}%           "
                  f"{len(m['iterations']):<12}")
        else:
            print(f"{m['version']:<10} {'No data':<15} {'No data':<15} "
                  f"{'No data':<18} {0:<12}")

    print("\n" + "="*80)

    # Best performer
    valid_metrics = [m for m in all_metrics if m["final_success"] is not None]
    if valid_metrics:
        best = max(valid_metrics, key=lambda x: x["final_success"])
        print(f"\n🏆 Best performer: {best['version']} "
              f"(Final: {best['final_success']:.1f}%, Peak: {best['peak_success']:.1f}%)")

        # Improvement analysis
        if len(valid_metrics) >= 2:
            baseline = valid_metrics[0]
            for m in valid_metrics[1:]:
                improvement = m["final_success"] - baseline["final_success"]
                print(f"   {m['version']} vs {baseline['version']}: "
                      f"{improvement:+.1f}% success rate")


def export_json(all_metrics: List[Dict], output_path: Path):
    """Export metrics to JSON for further analysis."""
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n📊 Detailed metrics exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare training runs")
    parser.add_argument("versions", nargs="+", help="Version names (e.g., v16 v17 v18)")
    parser.add_argument("--log-dir", default="logs/skrl", help="Log directory")
    parser.add_argument("--export", help="Export JSON to path")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    all_metrics = []

    for version in args.versions:
        # Find log files matching version
        pattern = f"short_nav_{version}_*.log"
        log_files = list(log_dir.glob(pattern))

        if not log_files:
            print(f"⚠️  No log files found for {version} (pattern: {pattern})")
            all_metrics.append({"version": version, "final_success": None})
            continue

        # Use most recent log
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        print(f"📄 Parsing {version}: {latest_log.name}")

        metrics = parse_log_file(latest_log)
        all_metrics.append(metrics)

    print_comparison_table(all_metrics)

    if args.export:
        export_json(all_metrics, Path(args.export))


if __name__ == "__main__":
    main()
