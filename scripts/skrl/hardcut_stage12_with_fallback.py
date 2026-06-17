#!/usr/bin/env python3
"""Run Stage-12 hard-cut training with an automatic rollback gate.

Workflow
1) Force Stage-12 entry by copying the resume checkpoint to agent_<force_step>.pt
2) Launch final-curriculum training (20k steps by default)
3) Evaluate first monitor_steps using [METRIC][LidarNav]
4) If collision > collision_threshold AND success < success_threshold: stop run and rollback
5) Rollback run trains a bridge-to-rear transition task (10k by default)
6) Optional: retry hard-cut Stage-12 from rollback's latest checkpoint
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from argparse import SUPPRESS
from typing import Iterable


METRIC_RE = re.compile(
    r"\[METRIC\]\[LidarNav\].*?step=(\d+).*?success=([\d.]+)%.*?collision=([\d.]+)%.*?timeout=([\d.]+)%"
)


@dataclass
class MetricPoint:
    step: int
    success: float
    collision: float
    timeout: float


@dataclass
class RunResult:
    name: str
    return_code: int
    log_path: Path
    gate_checked: bool
    gate_triggered: bool
    gate_summary: str
    run_dir: Path | None


def _now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _extract_step_from_name(path: Path) -> int:
    match = re.search(r"agent_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def _find_latest_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    candidates = [p for p in ckpt_dir.glob("agent_*.pt") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=_extract_step_from_name)


def _find_latest_run_dir(after_ts: float, logs_root: Path) -> Path | None:
    candidates = []
    if not logs_root.is_dir():
        return None
    for exp_dir in logs_root.glob("*/*"):
        if exp_dir.is_dir() and exp_dir.stat().st_mtime >= (after_ts - 2.0):
            candidates.append(exp_dir)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _stop_process(proc: subprocess.Popen[str], grace_s: float = 20.0) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=grace_s)
        return
    except Exception:
        pass
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10.0)
        return
    except Exception:
        pass
    if proc.poll() is not None:
        return
    proc.kill()
    proc.wait(timeout=5.0)


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return float("nan")
    return float(statistics.fmean(vals))


def run_training(
    *,
    name: str,
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    monitor_gate: bool,
    gate_start_step: int,
    monitor_steps: int,
    success_threshold: float,
    collision_threshold: float,
    logs_root: Path,
) -> RunResult:
    os.makedirs(log_path.parent, exist_ok=True)

    print(f"[RUN][{name}] cmd: {' '.join(cmd)}", flush=True)
    print(f"[RUN][{name}] log: {log_path}", flush=True)

    gate_checked = False
    gate_triggered = False
    gate_summary = ""
    points: list[MetricPoint] = []
    eval_end_step = gate_start_step + monitor_steps

    run_started_ts = time.time()

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert proc.stdout is not None

        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            log_file.write(raw_line)

            if (
                "[METRIC][LidarNav]" in line
                or "[DEBUG][LidarNav]" in line
                or "[INFO] Logging experiment in directory:" in line
                or "[INFO] Loading model checkpoint from:" in line
                or "[TRAIN]" in line
            ):
                print(line, flush=True)

            m = METRIC_RE.search(line)
            if m:
                step = int(m.group(1))
                success = float(m.group(2))
                collision = float(m.group(3))
                timeout = float(m.group(4))
                points.append(MetricPoint(step, success, collision, timeout))

                if monitor_gate and (not gate_checked) and step >= eval_end_step:
                    early = [p for p in points if p.step <= eval_end_step]
                    if not early:
                        early = points
                    avg_s = _mean(p.success for p in early)
                    avg_c = _mean(p.collision for p in early)
                    avg_t = _mean(p.timeout for p in early)
                    gate_summary = (
                        f"n={len(early)} step<= {eval_end_step} avg_success={avg_s:.2f}% "
                        f"avg_collision={avg_c:.2f}% avg_timeout={avg_t:.2f}%"
                    )
                    gate_checked = True
                    gate_triggered = bool(avg_c > collision_threshold and avg_s < success_threshold)
                    decision = "TRIGGER_ROLLBACK" if gate_triggered else "CONTINUE"
                    print(f"[GATE][{name}] {decision} :: {gate_summary}", flush=True)
                    if gate_triggered:
                        _stop_process(proc)
                        break

        if proc.poll() is None:
            proc.wait()
        rc = int(proc.returncode)

    run_dir = _find_latest_run_dir(run_started_ts, logs_root)
    return RunResult(
        name=name,
        return_code=rc,
        log_path=log_path,
        gate_checked=gate_checked,
        gate_triggered=gate_triggered,
        gate_summary=gate_summary,
        run_dir=run_dir,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-12 hard-cut run with automatic rollback gate.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("logs/resume_aliases/agent_2458400.pt"),
        help="Input checkpoint (bridge model).",
    )
    parser.add_argument("--robot", type=str, default="turtlebot3_burger")
    parser.add_argument("--algorithm", type=str, default="PPO")
    parser.add_argument("--hardcut-mode", type=str, default="single_obstacle_stage12_transition")
    parser.add_argument("--transition-mode", type=str, default="single_obstacle_rear_transition")
    parser.add_argument("--hardcut-iterations", type=int, default=80, help="~20k steps for PPO rollouts=256")
    parser.add_argument("--transition-iterations", type=int, default=40, help="~10k steps for PPO rollouts=256")
    parser.add_argument("--fallback-mode", type=str, default=None, help=SUPPRESS)
    parser.add_argument("--fallback-iterations", type=int, default=None, help=SUPPRESS)
    parser.add_argument(
        "--single-gap-force-goal-mode",
        type=str,
        choices=["rear", "rear_bridge", "fixed_local"],
        default="rear",
        help="Force single-obstacle goal mode for both hard-cut and fallback training calls.",
    )
    parser.add_argument("--force-step", type=int, default=2_700_000, help="Force Stage-12 entry step via alias filename.")
    parser.add_argument("--monitor-steps", type=int, default=2_000)
    parser.add_argument("--success-threshold", type=float, default=20.0)
    parser.add_argument("--collision-threshold", type=float, default=35.0)
    parser.add_argument(
        "--retry-after-fallback",
        action="store_true",
        help="After fallback 10k, retry hard-cut once from fallback latest checkpoint.",
    )
    parser.add_argument("--tag", type=str, default="stage12_hardcut")

    args = parser.parse_args()
    transition_mode = args.transition_mode if args.fallback_mode is None else args.fallback_mode
    transition_iterations = (
        int(args.transition_iterations) if args.fallback_iterations is None else int(args.fallback_iterations)
    )

    repo_root = Path(__file__).resolve().parents[2]
    logs_root = repo_root / "logs" / "skrl"
    tmp_root = repo_root / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    ckpt = (repo_root / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint
    if not ckpt.is_file():
        print(f"[ERROR] checkpoint not found: {ckpt}", file=sys.stderr)
        return 2

    alias_dir = repo_root / "logs" / "resume_aliases"
    alias_dir.mkdir(parents=True, exist_ok=True)
    hardcut_alias = alias_dir / f"agent_{int(args.force_step)}.pt"
    shutil.copy2(ckpt, hardcut_alias)
    print(f"[ALIAS] {ckpt} -> {hardcut_alias}", flush=True)

    tag = f"{args.tag}_{_now_tag()}"
    hardcut_log = tmp_root / f"train_hardcut_stage12_{tag}.log"

    hardcut_cmd = [
        "./launch.sh",
        "train",
        "--robot",
        args.robot,
        "--mode",
        args.hardcut_mode,
        "--algorithm",
        args.algorithm,
        "--max_iterations",
        str(int(args.hardcut_iterations)),
        "--single_gap_force_goal_mode",
        args.single_gap_force_goal_mode,
        "--checkpoint",
        str(hardcut_alias),
        "--headless",
    ]

    hardcut_result = run_training(
        name="hardcut_stage12",
        cmd=hardcut_cmd,
        cwd=repo_root,
        log_path=hardcut_log,
        monitor_gate=True,
        gate_start_step=int(args.force_step),
        monitor_steps=int(args.monitor_steps),
        success_threshold=float(args.success_threshold),
        collision_threshold=float(args.collision_threshold),
        logs_root=logs_root,
    )

    print(
        f"[RESULT][hardcut_stage12] rc={hardcut_result.return_code} "
        f"gate_checked={hardcut_result.gate_checked} gate_triggered={hardcut_result.gate_triggered} "
        f"{hardcut_result.gate_summary}",
        flush=True,
    )

    if not hardcut_result.gate_triggered:
        print("[DONE] Hard-cut run completed without rollback trigger.", flush=True)
        return 0 if hardcut_result.return_code == 0 else hardcut_result.return_code

    print("[ROLLBACK] Triggered. Launching fallback bridge-to-rear run.", flush=True)

    fallback_log = tmp_root / f"train_stage12_fallback_{tag}.log"
    fallback_cmd = [
        "./launch.sh",
        "train",
        "--robot",
        args.robot,
        "--mode",
        transition_mode,
        "--algorithm",
        args.algorithm,
        "--max_iterations",
        str(transition_iterations),
        "--single_gap_force_goal_mode",
        args.single_gap_force_goal_mode,
        "--checkpoint",
        str(ckpt),
        "--headless",
    ]

    fallback_result = run_training(
        name="fallback_bridge_to_rear",
        cmd=fallback_cmd,
        cwd=repo_root,
        log_path=fallback_log,
        monitor_gate=False,
        gate_start_step=0,
        monitor_steps=0,
        success_threshold=0.0,
        collision_threshold=0.0,
        logs_root=logs_root,
    )

    print(
        f"[RESULT][fallback_bridge_to_rear] rc={fallback_result.return_code} run_dir={fallback_result.run_dir}",
        flush=True,
    )

    if fallback_result.return_code != 0:
        return fallback_result.return_code

    if not args.retry_after_fallback:
        print("[DONE] Fallback run completed. Retry disabled.", flush=True)
        return 0

    if fallback_result.run_dir is None:
        print("[ERROR] Cannot determine fallback run directory for retry.", file=sys.stderr)
        return 3

    fallback_ckpt = _find_latest_checkpoint(fallback_result.run_dir)
    if fallback_ckpt is None:
        print(f"[ERROR] No checkpoint found under {fallback_result.run_dir}", file=sys.stderr)
        return 4

    retry_alias = alias_dir / f"agent_{int(args.force_step)}.pt"
    shutil.copy2(fallback_ckpt, retry_alias)
    retry_log = tmp_root / f"train_hardcut_stage12_retry_{tag}.log"

    print(f"[RETRY] Using fallback checkpoint: {fallback_ckpt}", flush=True)

    retry_result = run_training(
        name="hardcut_stage12_retry",
        cmd=hardcut_cmd[:-2] + [str(retry_alias), "--headless"],
        cwd=repo_root,
        log_path=retry_log,
        monitor_gate=True,
        gate_start_step=int(args.force_step),
        monitor_steps=int(args.monitor_steps),
        success_threshold=float(args.success_threshold),
        collision_threshold=float(args.collision_threshold),
        logs_root=logs_root,
    )

    print(
        f"[RESULT][hardcut_stage12_retry] rc={retry_result.return_code} "
        f"gate_checked={retry_result.gate_checked} gate_triggered={retry_result.gate_triggered} "
        f"{retry_result.gate_summary}",
        flush=True,
    )

    return 0 if retry_result.return_code == 0 else retry_result.return_code


if __name__ == "__main__":
    raise SystemExit(main())
