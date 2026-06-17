#!/usr/bin/env python3
"""Monitor key skrl TensorBoard metrics and raise threshold alerts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as exc:  # pragma: no cover - runtime environment dependent
    print(
        "[ERROR] tensorboard is unavailable in this Python environment: "
        f"{exc}. Use /home/cs/IsaacLab/_isaac_sim/python.sh.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _normalize_tag(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _find_tag(all_tags: list[str], candidates: list[str]) -> str | None:
    normalized_candidates = [_normalize_tag(item) for item in candidates]
    normalized_map = {_normalize_tag(tag): tag for tag in all_tags}

    for item in normalized_candidates:
        if item in normalized_map:
            return normalized_map[item]

    for tag in all_tags:
        n_tag = _normalize_tag(tag)
        if any(item in n_tag or n_tag in item for item in normalized_candidates):
            return tag
    return None


def _resolve_latest_run(log_root: Path, algorithm: str, ml_framework: str) -> Path | None:
    pattern = f"*_{algorithm.lower()}_{ml_framework}"
    run_dirs = [path for path in log_root.glob(pattern) if path.is_dir()]
    if not run_dirs:
        run_dirs = [path for path in log_root.iterdir() if path.is_dir()]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def _resolve_latest_event(run_dir: Path) -> Path | None:
    candidates = list(run_dir.glob("events.out.tfevents.*"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _extract_metric(
    scalar_tags: list[str],
    accumulator: EventAccumulator,
    aliases: list[str],
    window: int,
) -> dict[str, Any] | None:
    tag = _find_tag(scalar_tags, aliases)
    if not tag:
        return None
    events = accumulator.Scalars(tag)
    if not events:
        return None
    recent = [item.value for item in events[-max(1, window) :]]
    return {
        "tag": tag,
        "latest": float(events[-1].value),
        "window_avg": float(sum(recent) / len(recent)),
        "step": int(events[-1].step),
    }


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def _write_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    fieldnames = list(row.keys())
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _append_jsonl(jsonl_path: Path, payload: dict[str, Any]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor skrl training metrics with threshold alerts.")
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("logs/skrl/lidar_short_nav_v17_direct"),
        help="Root directory containing skrl run folders.",
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Explicit run directory to monitor.")
    parser.add_argument("--algorithm", type=str, default="ppo", help="Algorithm suffix in run directory name.")
    parser.add_argument("--ml-framework", type=str, default="torch", help="Framework suffix in run directory name.")
    parser.add_argument("--window", type=int, default=10, help="Window size for rolling-average checks.")
    parser.add_argument("--poll-seconds", type=int, default=600, help="Polling interval for continuous mode.")
    parser.add_argument("--once", action="store_true", default=False, help="Run one check and exit.")
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("logs/auto_monitor_v17/metrics_summary.csv"),
        help="CSV output path for periodic summaries.",
    )
    parser.add_argument(
        "--jsonl-out",
        type=Path,
        default=Path("logs/auto_monitor_v17/metrics_summary.jsonl"),
        help="JSONL output path for periodic summaries.",
    )
    parser.add_argument("--entropy-min", type=float, default=0.10, help="Alert if entropy average is below this value.")
    parser.add_argument("--kl-min", type=float, default=0.005, help="Alert if KL average is below this value.")
    parser.add_argument("--kl-max", type=float, default=0.020, help="Alert if KL average is above this value.")
    parser.add_argument(
        "--explained-var-min",
        type=float,
        default=0.80,
        help="Alert if explained variance average is below this value.",
    )
    parser.add_argument(
        "--grad-norm-max",
        type=float,
        default=5.0,
        help="Alert if gradient norm average is above this value.",
    )
    parser.add_argument(
        "--fail-on-alert",
        action="store_true",
        default=False,
        help="Exit with status 1 if any alert is triggered in --once mode.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    metric_aliases = {
        "policy_loss": ["Loss / Policy loss", "Loss/Policy loss"],
        "value_loss": ["Loss / Value loss", "Loss/Value loss"],
        "entropy_loss": ["Loss / Entropy loss", "Loss/Entropy loss"],
        "policy_entropy": ["Policy / Entropy", "Policy/Entropy"],
        "explained_variance": ["Value / Explained variance", "Value/Explained variance"],
        "kl_divergence": ["Learning / KL divergence", "Learning/KL divergence"],
        "grad_norm": ["Optimization / Grad norm", "Optimization/Grad norm"],
    }

    while True:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_dir = args.run_dir.resolve() if args.run_dir else _resolve_latest_run(
            args.log_root.resolve(), args.algorithm, args.ml_framework
        )
        if run_dir is None:
            print(f"[MONITOR][{now}] no run directory found under {args.log_root.resolve()}")
            if args.once:
                return 1
            time.sleep(max(1, args.poll_seconds))
            continue

        event_path = _resolve_latest_event(run_dir)
        if event_path is None:
            print(f"[MONITOR][{now}] no TensorBoard event file found in {run_dir}")
            if args.once:
                return 1
            time.sleep(max(1, args.poll_seconds))
            continue

        accumulator = EventAccumulator(str(event_path))
        accumulator.Reload()
        scalar_tags = accumulator.Tags().get("scalars", [])

        metrics: dict[str, dict[str, Any] | None] = {}
        for key, aliases in metric_aliases.items():
            metrics[key] = _extract_metric(scalar_tags, accumulator, aliases, args.window)

        max_step = max((item["step"] for item in metrics.values() if item is not None), default=-1)
        alerts: list[str] = []

        entropy = metrics["policy_entropy"]
        if entropy and entropy["window_avg"] < args.entropy_min:
            alerts.append(f"policy_entropy_avg={entropy['window_avg']:.6f} < {args.entropy_min:.6f}")

        kl = metrics["kl_divergence"]
        if kl:
            if kl["window_avg"] < args.kl_min:
                alerts.append(f"kl_avg={kl['window_avg']:.6f} < {args.kl_min:.6f}")
            if kl["window_avg"] > args.kl_max:
                alerts.append(f"kl_avg={kl['window_avg']:.6f} > {args.kl_max:.6f}")

        explained = metrics["explained_variance"]
        if explained and explained["window_avg"] < args.explained_var_min:
            alerts.append(f"explained_var_avg={explained['window_avg']:.6f} < {args.explained_var_min:.6f}")

        grad = metrics["grad_norm"]
        if grad and grad["window_avg"] > args.grad_norm_max:
            alerts.append(f"grad_norm_avg={grad['window_avg']:.6f} > {args.grad_norm_max:.6f}")

        status = "ALERT" if alerts else "OK"
        print(
            (
                f"[MONITOR][{now}] status={status} run={run_dir.name} step={max_step} "
                f"policy_loss={_fmt_float(metrics['policy_loss']['latest'] if metrics['policy_loss'] else None)} "
                f"value_loss={_fmt_float(metrics['value_loss']['latest'] if metrics['value_loss'] else None)} "
                f"entropy={_fmt_float(entropy['latest'] if entropy else None)} "
                f"kl={_fmt_float(kl['latest'] if kl else None)} "
                f"explained_var={_fmt_float(explained['latest'] if explained else None)} "
                f"grad_norm={_fmt_float(grad['latest'] if grad else None)}"
            )
        )
        for alert in alerts:
            print(f"[MONITOR][ALERT] {alert}")

        row = {
            "timestamp": now,
            "status": status,
            "run_dir": str(run_dir),
            "event_file": str(event_path),
            "step": max_step,
            "policy_loss": metrics["policy_loss"]["latest"] if metrics["policy_loss"] else None,
            "value_loss": metrics["value_loss"]["latest"] if metrics["value_loss"] else None,
            "entropy_loss": metrics["entropy_loss"]["latest"] if metrics["entropy_loss"] else None,
            "policy_entropy": entropy["latest"] if entropy else None,
            "policy_entropy_avg": entropy["window_avg"] if entropy else None,
            "kl_divergence": kl["latest"] if kl else None,
            "kl_divergence_avg": kl["window_avg"] if kl else None,
            "explained_variance": explained["latest"] if explained else None,
            "explained_variance_avg": explained["window_avg"] if explained else None,
            "grad_norm": grad["latest"] if grad else None,
            "grad_norm_avg": grad["window_avg"] if grad else None,
            "alerts": " | ".join(alerts),
        }
        _write_csv_row(args.csv_out.resolve(), row)
        _append_jsonl(args.jsonl_out.resolve(), row)

        if args.once:
            if args.fail_on_alert and alerts:
                return 1
            return 0
        time.sleep(max(1, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
