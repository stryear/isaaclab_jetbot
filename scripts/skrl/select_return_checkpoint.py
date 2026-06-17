#!/usr/bin/env python3
"""Select a checkpoint from return-state screen results with support-aware constraints."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ReturnCandidate:
    name: str
    checkpoint: str
    json_path: str
    ret_episodes: int
    ret_success: float
    ret_collision: float
    ret_timeout: float
    ret_stall_step: float
    ret_avg_goal_end_dist: float
    ret_avg_min_lidar: float


def _load_candidate(path: Path) -> ReturnCandidate | None:
    if path.name == "selection_summary.json":
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    ret = data.get("by_state", {}).get("return")
    ckpt = data.get("checkpoint")
    if not isinstance(ret, dict) or not ckpt:
        return None
    return ReturnCandidate(
        name=path.stem,
        checkpoint=str(ckpt),
        json_path=str(path),
        ret_episodes=int(ret.get("episodes", 0)),
        ret_success=float(ret.get("success_rate", 0.0)),
        ret_collision=float(ret.get("collision_rate", 1.0)),
        ret_timeout=float(ret.get("timeout_rate", 1.0)),
        ret_stall_step=float(ret.get("stall_step", 1.0)),
        ret_avg_goal_end_dist=float(ret.get("avg_goal_end_dist", 0.0)),
        ret_avg_min_lidar=float(ret.get("avg_min_lidar", 0.0)),
    )


def _constraint_tier(
    c: ReturnCandidate,
    min_return_episodes: int,
    target_collision: float,
    target_timeout: float,
    target_stall: float,
) -> int:
    enough_eps = c.ret_episodes >= min_return_episodes
    coll_ok = c.ret_collision <= target_collision
    timeout_ok = c.ret_timeout <= target_timeout
    stall_ok = c.ret_stall_step <= target_stall

    if enough_eps and coll_ok and timeout_ok and stall_ok:
        return 0
    if enough_eps and coll_ok and stall_ok:
        return 1
    if enough_eps and coll_ok and timeout_ok:
        return 2
    if enough_eps and coll_ok:
        return 3
    if enough_eps:
        return 4
    return 5


def _fallback_score(
    c: ReturnCandidate,
    min_return_episodes: int,
    target_collision: float,
    target_timeout: float,
    target_stall: float,
) -> float:
    support_deficit = max(0.0, float(min_return_episodes - c.ret_episodes) / max(1.0, float(min_return_episodes)))
    collision_excess = max(0.0, c.ret_collision - target_collision)
    timeout_excess = max(0.0, c.ret_timeout - target_timeout)
    stall_excess = max(0.0, c.ret_stall_step - target_stall)
    # Penalize overshooting targets first, then prefer lower timeout/stall and some success.
    return (
        4.0 * collision_excess
        + 2.0 * timeout_excess
        + 2.0 * stall_excess
        + 0.50 * c.ret_timeout
        + 0.35 * c.ret_stall_step
        + 0.25 * support_deficit
        - 0.50 * c.ret_success
    )


def _sort_key(
    c: ReturnCandidate,
    min_return_episodes: int,
    target_collision: float,
    target_timeout: float,
    target_stall: float,
) -> tuple[float, ...]:
    return (
        float(_constraint_tier(c, min_return_episodes, target_collision, target_timeout, target_stall)),
        _fallback_score(c, min_return_episodes, target_collision, target_timeout, target_stall),
        -c.ret_success,
        c.ret_timeout,
        c.ret_collision,
        c.ret_stall_step,
        -c.ret_episodes,
        c.name,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", required=True, help="Directory containing per-checkpoint JSON screen reports.")
    parser.add_argument("--target-collision", type=float, default=0.25)
    parser.add_argument("--target-timeout", type=float, default=0.70)
    parser.add_argument("--target-stall", type=float, default=0.50)
    parser.add_argument("--min-return-episodes", type=int, default=20)
    parser.add_argument("--summary-out", default="", help="Optional JSON summary output path.")
    parser.add_argument("--table-out", default="", help="Optional CSV table output path.")
    args = parser.parse_args()

    screen_dir = Path(args.screen_dir).expanduser().resolve()
    if not screen_dir.is_dir():
        raise SystemExit(f"[ERROR] screen dir not found: {screen_dir}")

    candidates = []
    for path in sorted(screen_dir.glob("*.json")):
        cand = _load_candidate(path)
        if cand is not None:
            candidates.append(cand)

    if not candidates:
        raise SystemExit(f"[ERROR] no per-checkpoint JSON files found in {screen_dir}")

    sorted_candidates = sorted(
        candidates,
        key=lambda c: _sort_key(
            c,
            args.min_return_episodes,
            args.target_collision,
            args.target_timeout,
            args.target_stall,
        ),
    )
    selected = sorted_candidates[0]
    selected_tier = _constraint_tier(
        selected,
        args.min_return_episodes,
        args.target_collision,
        args.target_timeout,
        args.target_stall,
    )

    summary = {
        "screen_dir": str(screen_dir),
        "selection_mode": "support_aware_multiconstraint",
        "constraint": {
            "ret_collision_lt": args.target_collision,
            "ret_timeout_lt": args.target_timeout,
            "ret_stall_step_lt": args.target_stall,
            "min_return_episodes": args.min_return_episodes,
        },
        "selected_tier": selected_tier,
        "selected": asdict(selected),
        "all_sorted": [asdict(c) for c in sorted_candidates],
    }

    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.table_out:
        with Path(args.table_out).open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "name",
                    "ret_episodes",
                    "ret_success",
                    "ret_collision",
                    "ret_timeout",
                    "ret_stall_step",
                    "ret_avg_goal_end_dist",
                    "ret_avg_min_lidar",
                    "tier",
                    "score",
                    "checkpoint",
                ]
            )
            for cand in sorted_candidates:
                writer.writerow(
                    [
                        cand.name,
                        cand.ret_episodes,
                        cand.ret_success,
                        cand.ret_collision,
                        cand.ret_timeout,
                        cand.ret_stall_step,
                        cand.ret_avg_goal_end_dist,
                        cand.ret_avg_min_lidar,
                        _constraint_tier(
                            cand,
                            args.min_return_episodes,
                            args.target_collision,
                            args.target_timeout,
                            args.target_stall,
                        ),
                        _fallback_score(
                            cand,
                            args.min_return_episodes,
                            args.target_collision,
                            args.target_timeout,
                            args.target_stall,
                        ),
                        cand.checkpoint,
                    ]
                )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
