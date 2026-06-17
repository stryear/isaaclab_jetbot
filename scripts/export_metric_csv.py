#!/usr/bin/env python3
"""Export LidarNav training metrics from logs to CSV.

Examples:
  python scripts/export_metric_csv.py logs/train_jobs/mppi_stage2_prepare_restart_20260330_000824.log
  python scripts/export_metric_csv.py logs/train_jobs/dwb_*.log --output-dir logs/exports
  python scripts/export_metric_csv.py logs/train_jobs/*.log --summary-csv logs/exports/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


LINE_TAG = "[METRIC][LidarNav]"
KV_PATTERN = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


def parse_value(raw: str) -> Any:
    value = raw.rstrip(",")
    if value.endswith("%"):
        value = value[:-1]

    lower = value.lower()
    if lower == "nan":
        return math.nan

    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_metric_rows(log_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, start=1):
            if LINE_TAG not in line:
                continue

            row: Dict[str, Any] = {
                "source_log": str(log_path),
                "run_name": log_path.stem,
                "line_number": line_no,
            }
            for key, raw_value in KV_PATTERN.findall(line):
                row[key] = parse_value(raw_value)
            rows.append(row)
    return rows


def csv_fieldnames(rows: Iterable[Dict[str, Any]]) -> List[str]:
    preferred = [
        "source_log",
        "run_name",
        "line_number",
        "iter",
        "step",
        "ep_total",
        "window",
        "success",
        "collision",
        "coll_lidar",
        "coll_contact",
        "timeout",
        "avg_len",
        "avg_return",
        "avg_goal_end_dist",
        "avg_min_lidar",
        "phase",
    ]
    seen = set()
    extras: List[str] = []
    for row in rows:
        for key in row:
            if key not in preferred and key not in seen:
                seen.add(key)
                extras.append(key)
    return preferred + sorted(extras)


def write_detail_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = csv_fieldnames(rows)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_max(rows: List[Dict[str, Any]], key: str) -> Any:
    values = [row[key] for row in rows if key in row and isinstance(row[key], (int, float)) and not math.isnan(row[key])]
    return max(values) if values else ""


def final_value(rows: List[Dict[str, Any]], key: str) -> Any:
    for row in reversed(rows):
        value = row.get(key, "")
        if isinstance(value, float) and math.isnan(value):
            continue
        if value != "":
            return value
    return ""


def build_summary_row(log_path: Path, rows: List[Dict[str, Any]], detail_csv: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "source_log": str(log_path),
        "run_name": log_path.stem,
        "detail_csv": str(detail_csv),
        "num_metric_rows": len(rows),
        "first_step": rows[0].get("step", "") if rows else "",
        "last_step": rows[-1].get("step", "") if rows else "",
        "last_iter": final_value(rows, "iter"),
        "last_phase": final_value(rows, "phase"),
        "final_success": final_value(rows, "success"),
        "final_collision": final_value(rows, "collision"),
        "final_coll_lidar": final_value(rows, "coll_lidar"),
        "final_coll_contact": final_value(rows, "coll_contact"),
        "final_timeout": final_value(rows, "timeout"),
        "final_avg_len": final_value(rows, "avg_len"),
        "final_avg_return": final_value(rows, "avg_return"),
        "final_avg_goal_end_dist": final_value(rows, "avg_goal_end_dist"),
        "final_avg_min_lidar": final_value(rows, "avg_min_lidar"),
        "best_success": safe_max(rows, "success"),
        "best_avg_return": safe_max(rows, "avg_return"),
    }
    return summary


def write_summary_csv(summary_rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_log",
        "run_name",
        "detail_csv",
        "num_metric_rows",
        "first_step",
        "last_step",
        "last_iter",
        "last_phase",
        "final_success",
        "final_collision",
        "final_coll_lidar",
        "final_coll_contact",
        "final_timeout",
        "final_avg_len",
        "final_avg_return",
        "final_avg_goal_end_dist",
        "final_avg_min_lidar",
        "best_success",
        "best_avg_return",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def expand_inputs(inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in inputs:
        matched = [Path(p) for p in sorted(glob.glob(pattern))] if any(ch in pattern for ch in "*?[]") else [Path(pattern)]
        for path in matched:
            if path.is_file() and path not in paths:
                paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Export [METRIC][LidarNav] lines to CSV")
    parser.add_argument("inputs", nargs="+", help="Log files or glob patterns")
    parser.add_argument(
        "--output-dir",
        default="logs/exports/metrics_csv",
        help="Directory for per-log CSV files",
    )
    parser.add_argument(
        "--summary-csv",
        default="logs/exports/metrics_csv/summary.csv",
        help="Path for aggregate summary CSV",
    )
    args = parser.parse_args()

    logs = expand_inputs(args.inputs)
    if not logs:
        raise SystemExit("No log files matched the provided inputs")

    output_dir = Path(args.output_dir)
    summary_rows: List[Dict[str, Any]] = []

    for log_path in logs:
        rows = parse_metric_rows(log_path)
        if not rows:
            print(f"skip: no METRIC rows in {log_path}")
            continue

        detail_csv = output_dir / f"{log_path.stem}.csv"
        write_detail_csv(rows, detail_csv)
        summary_rows.append(build_summary_row(log_path, rows, detail_csv))
        print(f"wrote {detail_csv} ({len(rows)} rows)")

    if not summary_rows:
        raise SystemExit("No METRIC rows found in any input logs")

    summary_csv = Path(args.summary_csv)
    write_summary_csv(summary_rows, summary_csv)
    print(f"wrote {summary_csv} ({len(summary_rows)} runs)")


if __name__ == "__main__":
    main()
