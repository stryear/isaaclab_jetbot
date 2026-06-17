#!/usr/bin/env python3
"""Automatically evaluate new checkpoints and append evaluation summaries."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_latest_run(log_root: Path, algorithm: str, ml_framework: str) -> Path | None:
    pattern = f"*_{algorithm.lower()}_{ml_framework}"
    run_dirs = [path for path in log_root.glob(pattern) if path.is_dir()]
    if not run_dirs:
        run_dirs = [path for path in log_root.iterdir() if path.is_dir()]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def _checkpoint_step(path: Path) -> int:
    matched = re.fullmatch(r"agent_(\d+)\.pt", path.name)
    return int(matched.group(1)) if matched else -1


def _fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"processed_fingerprints": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"processed_fingerprints": []}
    if not isinstance(payload, dict):
        return {"processed_fingerprints": []}
    fingerprints = payload.get("processed_fingerprints", [])
    if not isinstance(fingerprints, list):
        fingerprints = []
    return {"processed_fingerprints": [str(item) for item in fingerprints]}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _write_summary_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _collect_pending_checkpoints(
    checkpoints_dir: Path,
    include_best: bool,
    processed_fingerprints: set[str],
    checkpoint_glob: str,
) -> list[tuple[str, int, Path, str]]:
    pending: list[tuple[str, int, Path, str]] = []

    for checkpoint in checkpoints_dir.glob(checkpoint_glob):
        if not checkpoint.is_file():
            continue
        step = _checkpoint_step(checkpoint)
        if step < 0:
            continue
        fp = _fingerprint(checkpoint)
        if fp in processed_fingerprints:
            continue
        pending.append(("agent", step, checkpoint.resolve(), fp))

    if include_best:
        best = checkpoints_dir / "best_agent.pt"
        if best.exists():
            fp = _fingerprint(best)
            if fp not in processed_fingerprints:
                pending.append(("best", -1, best.resolve(), fp))

    pending.sort(key=lambda item: (0 if item[0] == "agent" else 1, item[1], item[2].stat().st_mtime))
    return pending


def _collect_all_fingerprints(checkpoints_dir: Path, include_best: bool, checkpoint_glob: str) -> set[str]:
    fingerprints: set[str] = set()
    for checkpoint in checkpoints_dir.glob(checkpoint_glob):
        if checkpoint.is_file() and _checkpoint_step(checkpoint) >= 0:
            fingerprints.add(_fingerprint(checkpoint))
    if include_best:
        best = checkpoints_dir / "best_agent.pt"
        if best.exists():
            fingerprints.add(_fingerprint(best))
    return fingerprints


def _run_eval_once(
    project_root: Path,
    args: argparse.Namespace,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", checkpoint_path.stem)
    eval_log_dir = output_dir / "eval_logs"
    eval_report_dir = output_dir / "eval_reports"
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    eval_report_dir.mkdir(parents=True, exist_ok=True)
    log_path = eval_log_dir / f"{timestamp}_{stem}.log"
    report_path = eval_report_dir / f"{timestamp}_{stem}_state_report.json"

    isaaclab_sh = Path(args.isaaclab_sh).expanduser().resolve()
    cmd = [
        str(isaaclab_sh),
        "-p",
        str(project_root / "scripts/skrl/eval_lidar_nav.py"),
        "--task",
        args.task,
        "--algorithm",
        args.algorithm,
        "--ml_framework",
        args.ml_framework,
        "--checkpoint",
        str(checkpoint_path),
        "--num_envs",
        str(args.num_envs),
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--headless",
        "--state_report_json",
        str(report_path),
    ]
    for raw in args.eval_extra_arg:
        cmd.extend(shlex.split(raw))

    print(f"[AUTO-EVAL] command: {shlex.join(cmd)}")
    if args.dry_run:
        return {
            "exit_code": 0,
            "log_path": str(log_path),
            "report_path": str(report_path),
            "report": None,
            "dry_run": True,
        }

    # Prevent host user-site packages (e.g. ~/.local/lib/python*/site-packages/pxr)
    # from shadowing Isaac Sim bundled USD/PXR modules.
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = ""

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {shlex.join(cmd)}\n")
        handle.write(
            f"# env PYTHONNOUSERSITE={env.get('PYTHONNOUSERSITE', '')} PYTHONPATH={env.get('PYTHONPATH', '')}\n"
        )
        handle.flush()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(project_root),
            env=env,
        )
        if process.stdout is not None:
            for line in process.stdout:
                sys.stdout.write(line)
                handle.write(line)
        exit_code = process.wait()

    report = None
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report = None

    return {
        "exit_code": exit_code,
        "log_path": str(log_path),
        "report_path": str(report_path),
        "report": report,
        "dry_run": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automatically evaluate new checkpoints from a training run.")
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Lab-Tutorial-LidarShortNavV17-TurtleBot3-Direct-v0",
        help="Task name for evaluation script.",
    )
    parser.add_argument("--algorithm", type=str, default="PPO", help="Evaluation algorithm.")
    parser.add_argument("--ml-framework", type=str, default="torch", help="Evaluation ML framework.")
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("logs/skrl/lidar_short_nav_v17_direct"),
        help="Run root directory containing timestamped run folders.",
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Explicit run directory. Skips auto run discovery.")
    parser.add_argument("--checkpoints-dir", type=Path, default=None, help="Explicit checkpoints directory.")
    parser.add_argument("--checkpoint-glob", type=str, default="agent_*.pt", help="Glob for step checkpoints.")
    parser.add_argument("--include-best", action="store_true", default=False, help="Also evaluate best_agent.pt updates.")
    parser.add_argument("--episodes", type=int, default=50, help="Evaluation episodes per checkpoint.")
    parser.add_argument("--num-envs", type=int, default=16, help="Evaluation environments per checkpoint.")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed.")
    parser.add_argument("--poll-seconds", type=int, default=600, help="Polling interval in seconds.")
    parser.add_argument("--max-evals-per-cycle", type=int, default=1, help="Max pending checkpoints evaluated per poll.")
    parser.add_argument(
        "--catch-up-existing",
        action="store_true",
        default=False,
        help="On first startup, evaluate existing checkpoints instead of skipping directly to new checkpoints.",
    )
    parser.add_argument("--once", action="store_true", default=False, help="Run one poll iteration and exit.")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Do not execute eval command.")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        default=False,
        help="Do not mark failed checkpoints as processed. They will be retried next cycle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/auto_eval_v17"),
        help="Output directory for eval logs, reports and summaries.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="CSV output path. Default: <output-dir>/summary.csv",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="State file path. Default: <output-dir>/.auto_eval_state.json",
    )
    parser.add_argument(
        "--isaaclab-sh",
        type=str,
        default=str(Path.home() / "IsaacLab/isaaclab.sh"),
        help="Path to isaaclab.sh launcher.",
    )
    parser.add_argument(
        "--eval-extra-arg",
        action="append",
        default=[],
        help="Extra argument(s) appended to eval command. Repeatable. Example: --eval-extra-arg \"--print_every 10\"",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = _resolve_project_root()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = args.summary_csv.resolve() if args.summary_csv else (output_dir / "summary.csv")
    state_file = args.state_file.resolve() if args.state_file else (output_dir / ".auto_eval_state.json")
    state = _load_state(state_file)
    processed_fingerprints = set(state.get("processed_fingerprints", []))
    initialized = bool(state.get("initialized", False))

    while True:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_dir = args.run_dir.resolve() if args.run_dir else _resolve_latest_run(
            args.log_root.resolve(), args.algorithm, args.ml_framework
        )
        if run_dir is None:
            print(f"[AUTO-EVAL][{now}] no run directory found under {args.log_root.resolve()}")
            if args.once:
                return 1
            time.sleep(max(1, args.poll_seconds))
            continue

        checkpoints_dir = args.checkpoints_dir.resolve() if args.checkpoints_dir else (run_dir / "checkpoints")
        if not checkpoints_dir.exists():
            print(f"[AUTO-EVAL][{now}] checkpoints directory missing: {checkpoints_dir}")
            if args.once:
                return 1
            time.sleep(max(1, args.poll_seconds))
            continue

        if not initialized:
            if args.catch_up_existing:
                print(f"[AUTO-EVAL][{now}] catch-up enabled; existing checkpoints will be evaluated.")
            else:
                seed = _collect_all_fingerprints(
                    checkpoints_dir=checkpoints_dir,
                    include_best=args.include_best,
                    checkpoint_glob=args.checkpoint_glob,
                )
                processed_fingerprints.update(seed)
                print(f"[AUTO-EVAL][{now}] baseline initialized; skipping existing checkpoints count={len(seed)}")
            initialized = True
            state["initialized"] = True
            state["processed_fingerprints"] = sorted(processed_fingerprints)
            _save_state(state_file, state)

        pending = _collect_pending_checkpoints(
            checkpoints_dir=checkpoints_dir,
            include_best=args.include_best,
            processed_fingerprints=processed_fingerprints,
            checkpoint_glob=args.checkpoint_glob,
        )

        if not pending:
            print(f"[AUTO-EVAL][{now}] no new checkpoints in {checkpoints_dir}")
            if args.once:
                return 0
            time.sleep(max(1, args.poll_seconds))
            continue

        to_run = pending[: max(1, args.max_evals_per_cycle)]
        print(
            f"[AUTO-EVAL][{now}] pending={len(pending)} evaluating_now={len(to_run)} run={run_dir.name} "
            f"dir={checkpoints_dir}"
        )

        for ckpt_type, step, checkpoint, fingerprint in to_run:
            print(f"[AUTO-EVAL] evaluate type={ckpt_type} step={step} checkpoint={checkpoint}")
            result = _run_eval_once(project_root=project_root, args=args, checkpoint_path=checkpoint, output_dir=output_dir)
            report = result.get("report")
            overall = report.get("overall", {}) if isinstance(report, dict) else {}
            exit_code = int(result.get("exit_code", 1))

            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "run_dir": str(run_dir),
                "checkpoint_type": ckpt_type,
                "checkpoint_step": step if step >= 0 else "",
                "checkpoint_path": str(checkpoint),
                "exit_code": exit_code,
                "success_rate": overall.get("success_rate"),
                "collision_rate": overall.get("collision_rate"),
                "timeout_rate": overall.get("timeout_rate"),
                "return_mean": overall.get("return_mean"),
                "ep_len_mean": overall.get("ep_len_mean"),
                "episodes": report.get("episodes") if isinstance(report, dict) else None,
                "report_path": result.get("report_path"),
                "eval_log_path": result.get("log_path"),
                "dry_run": bool(result.get("dry_run", False)),
            }
            _write_summary_row(summary_csv, row)

            if not result.get("dry_run", False) and (exit_code == 0 or not args.retry_failed):
                processed_fingerprints.add(fingerprint)
                state["processed_fingerprints"] = sorted(processed_fingerprints)
                state["initialized"] = True
                _save_state(state_file, state)

            if exit_code != 0:
                print(
                    f"[AUTO-EVAL][WARN] evaluation failed (exit={exit_code}) for {checkpoint}. "
                    f"log={result.get('log_path')}"
                )

        if args.once:
            return 0
        time.sleep(max(1, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
