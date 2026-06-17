#!/usr/bin/env python3
"""Summarize yaw diagnostics over step intervals from TensorBoard event files."""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from dataclasses import dataclass

import numpy as np

try:
    from tensorboard.backend.event_processing import event_accumulator
except ModuleNotFoundError as exc:
    raise SystemExit(
        "tensorboard is not installed in this Python environment.\n"
        "Run with IsaacLab Python instead, e.g.:\n"
        "  /home/cs/IsaacLab/_isaac_sim/python.sh scripts/skrl/analyze_yaw_stages.py --log-dir <run_dir>"
    ) from exc


@dataclass(frozen=True)
class Interval:
    name: str
    start: int
    end: int


TAG_CANDIDATES: dict[str, tuple[str, ...]] = {
    "yaw_rate_abs_mean": ("Info / yaw_rate_abs_mean", "Info/yaw_rate_abs_mean"),
    "yaw_rate_abs_p95": ("Info / yaw_rate_abs_p95", "Info/yaw_rate_abs_p95"),
    "filtered_omega_abs_mean": ("Info / filtered_omega_abs_mean", "Info/filtered_omega_abs_mean"),
    "yaw_sign_flip_rate": ("Info / yaw_sign_flip_rate", "Info/yaw_sign_flip_rate"),
    "yaw_sign_flip_rate_step": ("Info / yaw_sign_flip_rate_step", "Info/yaw_sign_flip_rate_step"),
    "yaw_persistence_reward_mean": (
        "Info / yaw_persistence_reward_mean",
        "Info/yaw_persistence_reward_mean",
    ),
    "yaw_flip_penalty_mean": (
        "Info / yaw_flip_penalty_mean",
        "Info/yaw_flip_penalty_mean",
    ),
    "success_rate": ("Info / success_rate", "Metrics/success_rate", "Info/success_rate"),
    "collision_rate": ("Info / collision_rate", "Metrics/collision_rate", "Info/collision_rate"),
    "timeout_rate": ("Info / timeout_rate", "Metrics/timeout_rate", "Info/timeout_rate"),
}


def _parse_interval(raw: str) -> Interval:
    # Format: "Stage name:start:end"
    parts = raw.rsplit(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid interval '{raw}'. Expected format: name:start:end")
    name = parts[0].strip()
    start = int(parts[1].replace("_", ""))
    end = int(parts[2].replace("_", ""))
    if start > end:
        raise ValueError(f"Invalid interval '{raw}': start > end")
    return Interval(name=name, start=start, end=end)


def _pick_event_file(log_dir: str, event_file: str | None) -> str:
    if event_file:
        if not os.path.isfile(event_file):
            raise FileNotFoundError(f"Event file not found: {event_file}")
        return event_file

    pattern = os.path.join(log_dir, "events.out.tfevents*")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No events file found in: {log_dir}")
    return max(files, key=os.path.getmtime)


def _resolve_tags(all_scalar_tags: list[str]) -> dict[str, str | None]:
    resolved: dict[str, str | None] = {}
    scalar_set = set(all_scalar_tags)
    for metric, candidates in TAG_CANDIDATES.items():
        resolved_tag = None
        for tag in candidates:
            if tag in scalar_set:
                resolved_tag = tag
                break
        resolved[metric] = resolved_tag
    return resolved


def _series_for_tag(
    ea: event_accumulator.EventAccumulator, tag: str | None, step_offset: int
) -> tuple[np.ndarray, np.ndarray] | None:
    if tag is None:
        return None
    events = ea.Scalars(tag)
    if not events:
        return None
    steps = np.asarray([ev.step for ev in events], dtype=np.int64) + int(step_offset)
    values = np.asarray([ev.value for ev in events], dtype=np.float64)
    return steps, values


def _summarize_interval(steps: np.ndarray, values: np.ndarray, start: int, end: int) -> tuple[int, float, float]:
    mask = (steps >= start) & (steps <= end)
    if not np.any(mask):
        return 0, math.nan, math.nan
    v = values[mask]
    return int(v.size), float(np.mean(v)), float(np.percentile(v, 95))


def _format(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, help="Run directory containing events.out.tfevents*")
    parser.add_argument("--event-file", default=None, help="Optional explicit event file path")
    parser.add_argument(
        "--interval",
        action="append",
        default=None,
        help="Interval in format 'name:start:end'. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output-csv",
        default="yaw_stage_comparison.csv",
        help="Path to output CSV summary (default: yaw_stage_comparison.csv)",
    )
    parser.add_argument(
        "--step-offset",
        type=int,
        default=0,
        help="Add this offset to TB steps before interval filtering (use resume common_step_counter).",
    )
    args = parser.parse_args()

    intervals = (
        [_parse_interval(raw) for raw in args.interval]
        if args.interval
        else [
            Interval("Stage 11 (low speed)", 2_400_000, 2_700_000),
            Interval("Late high collision", 3_300_000, 3_500_000),
        ]
    )

    event_file = _pick_event_file(args.log_dir, args.event_file)
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    scalar_tags = ea.Tags().get("scalars", [])
    resolved = _resolve_tags(scalar_tags)
    series = {metric: _series_for_tag(ea, tag, args.step_offset) for metric, tag in resolved.items()}

    for metric, tag in resolved.items():
        if tag is None:
            print(f"[WARN] Missing tag for metric '{metric}'. Candidates: {TAG_CANDIDATES[metric]}")

    metric_order = list(TAG_CANDIDATES.keys())
    rows: list[dict[str, str]] = []
    for interval in intervals:
        row: dict[str, str] = {
            "Stage": interval.name,
            "StepRange": f"{interval.start}-{interval.end}",
        }
        for metric in metric_order:
            data = series[metric]
            if data is None:
                count, mean_val, p95_val = 0, math.nan, math.nan
            else:
                count, mean_val, p95_val = _summarize_interval(data[0], data[1], interval.start, interval.end)
            row[f"{metric}_count"] = str(count)
            row[f"{metric}_mean"] = _format(mean_val)
            row[f"{metric}_p95"] = _format(p95_val)
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else ["Stage", "StepRange"]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Event file: {event_file}")
    print(f"[INFO] Output CSV: {args.output_csv}")
    print(f"[INFO] Step offset: {args.step_offset}")
    print("\n===== Yaw Stage Comparison =====")
    print(",".join(fieldnames))
    for row in rows:
        print(",".join(row[name] for name in fieldnames))


if __name__ == "__main__":
    main()
