#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a trained LiDAR-nav checkpoint with episode-level metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback
from typing import Any

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a trained skrl policy on LiDAR-nav task.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO", "SAC"],
    help="RL algorithm name used by the checkpoint.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="ML framework used by skrl runner.",
)
parser.add_argument("--num_envs", type=int, default=32, help="Number of parallel evaluation environments.")
parser.add_argument("--episodes", type=int, default=200, help="Number of episodes to evaluate.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. If omitted, auto-pick latest run.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--seed", type=int, default=42, help="Evaluation seed.")
parser.add_argument("--print_every", type=int, default=20, help="Print progress every N finished episodes.")
parser.add_argument(
    "--debug_print_enable",
    type=int,
    choices=[0, 1],
    default=None,
    help="Enable (1) or disable (0) lidar-nav debug prints if the env supports it.",
)
parser.add_argument(
    "--debug_print_interval_steps",
    type=int,
    default=None,
    help="Lidar-nav debug print interval in steps if the env supports it.",
)
parser.add_argument(
    "--debug_phase_tag",
    type=int,
    default=None,
    help="Optional phase tag appended to LiDAR debug/metric logs if the env supports it.",
)
parser.add_argument(
    "--num_obstacles",
    type=int,
    default=None,
    help="Override number of obstacles if the env supports it.",
)
parser.add_argument(
    "--scene_id",
    type=int,
    default=None,
    help="Force fixed-scene pool index if the env supports it (e.g. demo_multi_v1).",
)
parser.add_argument(
    "--scene_pool",
    type=str,
    default=None,
    help="Comma-separated fixed-scene indices to sample from if the env supports it (e.g. '3,4').",
)
parser.add_argument(
    "--scene_pool_weights",
    type=str,
    default=None,
    help="Comma-separated non-negative weights aligned with --scene_pool (e.g. '0.2,0.2,0.6').",
)
parser.add_argument(
    "--junction_half_width",
    type=float,
    default=None,
    help="Override junction lane half width for junction-enabled lidar-nav envs.",
)
parser.add_argument(
    "--junction_half_length",
    type=float,
    default=None,
    help="Override junction arm half length for junction-enabled lidar-nav envs.",
)
parser.add_argument(
    "--junction_goal_forward_max",
    type=float,
    default=None,
    help="Override max forward distance for junction goals if supported.",
)
parser.add_argument(
    "--junction_goal_forward_min",
    type=float,
    default=None,
    help="Override min forward distance for junction goals if supported.",
)
parser.add_argument(
    "--junction_robot_spawn_offset_min",
    type=float,
    default=None,
    help="Override minimum robot spawn offset on junction west arm if supported.",
)
parser.add_argument(
    "--junction_robot_spawn_offset_max",
    type=float,
    default=None,
    help="Override maximum robot spawn offset on junction west arm if supported.",
)
parser.add_argument(
    "--junction_robot_spawn_lateral_jitter",
    type=float,
    default=None,
    help="Override lateral robot spawn jitter in junction mode if supported.",
)
parser.add_argument(
    "--junction_x_sampling_prob",
    type=float,
    default=None,
    help="Override X-junction sampling probability in junction mode if supported.",
)
parser.add_argument(
    "--action_scale",
    type=float,
    default=None,
    help="Override action scale if the env supports it.",
)
parser.add_argument(
    "--near_goal_min_action_scale",
    type=float,
    default=None,
    help="Override near-goal minimum action scale if the env supports it.",
)
parser.add_argument(
    "--obstacle_slowdown_distance",
    type=float,
    default=None,
    help="Override obstacle slowdown distance if the env supports it.",
)
parser.add_argument(
    "--obstacle_min_action_scale",
    type=float,
    default=None,
    help="Override obstacle minimum action scale if the env supports it.",
)
parser.add_argument(
    "--goal_reach_threshold",
    type=float,
    default=None,
    help="Override goal reach threshold if the env supports it.",
)
parser.add_argument(
    "--goal_local_x",
    type=float,
    default=None,
    help="Override fixed-local goal x and matching defect_dwb curriculum fixed-local x values if supported.",
)
parser.add_argument(
    "--goal_local_y",
    type=float,
    default=None,
    help="Override fixed-local goal y and matching defect_dwb curriculum fixed-local y values if supported.",
)
parser.add_argument(
    "--goal_max_distance",
    type=float,
    default=None,
    help="Override goal distance normalization denominator if the env supports it.",
)
parser.add_argument(
    "--goal_spawn_range_min",
    type=float,
    default=None,
    help="Override goal spawn min distance if the env supports it.",
)
parser.add_argument(
    "--goal_spawn_range_max",
    type=float,
    default=None,
    help="Override goal spawn max distance if the env supports it.",
)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=None,
    help="Override episode length in seconds if the env supports it.",
)
parser.add_argument(
    "--v_max",
    type=float,
    default=None,
    help="Override maximum forward speed (m/s) for v-omega action mapping if the env supports it.",
)
parser.add_argument(
    "--omega_limit",
    type=float,
    default=None,
    help="Override maximum yaw rate (rad/s) for v-omega action mapping if the env supports it.",
)
parser.add_argument(
    "--tau_v",
    type=float,
    default=None,
    help="Override low-pass time constant tau_v (s) if the env supports it.",
)
parser.add_argument(
    "--tau_omega",
    type=float,
    default=None,
    help="Override low-pass time constant tau_omega (s) if the env supports it.",
)
parser.add_argument(
    "--wheel_linear_speed_max",
    type=float,
    default=None,
    help="Override wheel linear speed cap V_max (m/s) for diamond clamp if the env supports it.",
)
parser.add_argument(
    "--forward_only",
    type=int,
    choices=[0, 1],
    default=None,
    help="Use forward-only v mapping (1) or allow reverse (0) if the env supports it.",
)
parser.add_argument(
    "--wheel_target_delta_limit",
    type=float,
    default=None,
    help="Override per-step wheel target delta limit after v-omega mapping if the env supports it.",
)
parser.add_argument(
    "--defect_goal_distance_min",
    type=float,
    default=None,
    help="Override minimum sampled goal distance for defect scenes if supported.",
)
parser.add_argument(
    "--defect_goal_distance_max",
    type=float,
    default=None,
    help="Override maximum sampled goal distance for defect scenes if supported.",
)
parser.add_argument(
    "--defect_spawn_jitter_xy",
    type=float,
    default=None,
    help="Override spawn XY jitter for defect scenes if supported.",
)
parser.add_argument(
    "--defect_goal_jitter_xy",
    type=float,
    default=None,
    help="Override goal XY jitter for defect scenes if supported.",
)
parser.add_argument(
    "--defect_dwb_corridor_width",
    type=float,
    default=None,
    help="Override corridor width for dwb_oscillation defect scenes if supported.",
)
parser.add_argument(
    "--defect_dwb_corridor_half_length",
    type=float,
    default=None,
    help="Override half corridor length for dwb_oscillation defect scenes if supported.",
)
parser.add_argument(
    "--defect_dwb_obstacle_spacing",
    type=float,
    default=None,
    help="Override obstacle pair spacing for dwb_oscillation defect scenes if supported.",
)
parser.add_argument(
    "--defect_dwb_obstacle_spacing_jitter",
    type=float,
    default=None,
    help="Override obstacle spacing jitter for dwb_oscillation defect scenes if supported.",
)
parser.add_argument(
    "--defect_dwb_obstacle_pair_shift_y",
    type=float,
    default=None,
    help="Override obstacle pair lateral shift for dwb_oscillation defect scenes if supported.",
)
parser.add_argument(
    "--defect_dwb_obstacle_x_jitter",
    type=float,
    default=None,
    help="Override obstacle pair X jitter for dwb_oscillation defect scenes if supported.",
)
parser.add_argument(
    "--defect_door_corridor_width",
    type=float,
    default=None,
    help="Override corridor width for door_deadlock defect scenes if supported.",
)
parser.add_argument(
    "--defect_door_half_length",
    type=float,
    default=None,
    help="Override half corridor length for door_deadlock defect scenes if supported.",
)
parser.add_argument(
    "--defect_door_width",
    type=float,
    default=None,
    help="Override bottleneck door width for door_deadlock defect scenes if supported.",
)
parser.add_argument(
    "--defect_door_width_jitter",
    type=float,
    default=None,
    help="Override door width jitter magnitude for door_deadlock defect scenes if supported.",
)
parser.add_argument(
    "--defect_door_straight_distance",
    type=float,
    default=None,
    help="Override straight approach/exit distance around the door if supported.",
)
parser.add_argument(
    "--defect_door_line_angle_max_deg",
    type=float,
    default=None,
    help="Override max absolute line angle in degrees for door_deadlock collinear spawn-goal layout if supported.",
)
parser.add_argument(
    "--defect_door_goal_lateral_offset_max",
    type=float,
    default=None,
    help="Override max lateral goal offset after the doorway for door_deadlock scenes if supported.",
)
parser.add_argument(
    "--single_gap_goal_obstacle_debug",
    type=int,
    choices=[0, 1],
    default=None,
    help="Enable (1) or disable (0) single-obstacle scene reset prints including goal(x,y) and goal_mode.",
)
parser.add_argument(
    "--single_gap_goal_obstacle_debug_every",
    type=int,
    default=None,
    help="Print interval (in reset count) for single-obstacle goal debug logs.",
)
parser.add_argument(
    "--defect_dwb_collision_penalty",
    type=float,
    default=None,
    help="Override collision penalty and defect_dwb curriculum collision penalties if supported.",
)
parser.add_argument(
    "--defect_dwb_clearance_penalty_scale",
    type=float,
    default=None,
    help="Override DWB global clearance penalty scale. Use a negative value to penalize low LiDAR clearance.",
)
parser.add_argument(
    "--defect_dwb_clearance_penalty_distance",
    type=float,
    default=None,
    help="Override DWB global clearance penalty activation distance.",
)
parser.add_argument(
    "--defect_dwb_front_clearance_penalty_scale",
    type=float,
    default=None,
    help="Override DWB front-sector clearance penalty scale. Use a negative value to penalize close frontal obstacles.",
)
parser.add_argument(
    "--defect_dwb_front_clearance_penalty_distance",
    type=float,
    default=None,
    help="Override DWB front-sector clearance penalty activation distance.",
)
parser.add_argument(
    "--defect_dwb_clearance_penalty_power",
    type=float,
    default=None,
    help="Override exponent used by DWB clearance penalties.",
)
parser.add_argument(
    "--defect_dwb_gap_progress_reward_scale",
    type=float,
    default=None,
    help="Override DWB gap-progress reward scale if supported.",
)
parser.add_argument(
    "--defect_dwb_gap_alignment_reward_scale",
    type=float,
    default=None,
    help="Override DWB gap-alignment reward scale if supported.",
)
parser.add_argument(
    "--defect_dwb_gap_clear_bonus",
    type=float,
    default=None,
    help="Override DWB gap-clear bonus if supported.",
)
parser.add_argument(
    "--defect_dwb_side_commit_reward_scale",
    type=float,
    default=None,
    help="Override DWB side-commit reward scale if supported.",
)
parser.add_argument(
    "--defect_dwb_side_switch_penalty_scale",
    type=float,
    default=None,
    help="Override DWB side-switch penalty scale if supported.",
)
parser.add_argument(
    "--defect_dwb_side_commit_deadzone",
    type=float,
    default=None,
    help="Override DWB side-commit lateral deadzone if supported.",
)
parser.add_argument(
    "--bypass_reward_enable",
    type=int,
    choices=[0, 1],
    default=None,
    help="Enable (1) or disable (0) bypass reward shaping if supported.",
)
parser.add_argument(
    "--bypass_reward_scale",
    type=float,
    default=None,
    help="Override bypass reward scale if supported.",
)
parser.add_argument(
    "--bypass_open_side_min_lidar",
    type=float,
    default=None,
    help="Override bypass open-side LiDAR threshold if supported.",
)
parser.add_argument(
    "--bypass_blocked_side_max_lidar",
    type=float,
    default=None,
    help="Override bypass blocked-side LiDAR threshold if supported.",
)
parser.add_argument(
    "--bypass_near_obstacle_distance",
    type=float,
    default=None,
    help="Override bypass near-obstacle activation distance if supported.",
)
parser.add_argument(
    "--bypass_forward_speed_threshold",
    type=float,
    default=None,
    help="Override bypass minimum forward speed threshold if supported.",
)
parser.add_argument(
    "--single_gap_force_goal_mode",
    type=str,
    choices=["rear", "rear_bridge", "fixed_local"],
    default=None,
    help="Force single-obstacle goal mode and overwrite curriculum goal_modes if supported.",
)
parser.add_argument(
    "--rear_clearance",
    type=float,
    default=None,
    help="Override rear-goal clearance and defect_dwb curriculum rear-clearance values if supported.",
)
parser.add_argument(
    "--rear_lateral_center",
    type=float,
    default=None,
    help="Override rear-goal lateral center and defect_dwb curriculum rear-lateral-center values if supported.",
)
parser.add_argument(
    "--rear_offset_max",
    type=float,
    default=None,
    help="Override rear-goal lateral offset max and defect_dwb curriculum rear-offset-max values if supported.",
)
parser.add_argument(
    "--bridge_clearance",
    type=float,
    default=None,
    help="Override rear-bridge goal clearance and defect_dwb curriculum rear-bridge clearance values if supported.",
)
parser.add_argument(
    "--bridge_lateral_center",
    type=float,
    default=None,
    help="Override rear-bridge goal lateral center and defect_dwb curriculum rear-bridge lateral values if supported.",
)
parser.add_argument(
    "--bridge_offset_max",
    type=float,
    default=None,
    help="Override rear-bridge goal lateral offset max and defect_dwb curriculum rear-bridge offset values if supported.",
)
parser.add_argument(
    "--curriculum_start_step",
    type=int,
    default=None,
    help="Override the restored curriculum/common step counter when loading a checkpoint.",
)
parser.add_argument(
    "--freeze_curriculum_stage",
    type=int,
    default=None,
    help="Freeze the dwb-like curriculum at the specified stage index if the env supports it.",
)
parser.add_argument(
    "--state_report_json",
    type=str,
    default=None,
    help="Optional path to dump state-bucket evaluation report in JSON format.",
)
parser.add_argument(
    "--trajectory_json",
    type=str,
    default=None,
    help="Optional path to dump one environment trajectory trace in JSON format.",
)
parser.add_argument(
    "--trajectory_env_id",
    type=int,
    default=0,
    help="Environment index to trace when --trajectory_json is set.",
)
parser.add_argument(
    "--no_progress_min_delta_m",
    type=float,
    default=0.30,
    help="Timeout episode is no_progress if goal distance reduction is below this threshold (meters).",
)
parser.add_argument(
    "--no_progress_min_ratio",
    type=float,
    default=0.20,
    help="Timeout episode is no_progress if progress ratio is below this threshold.",
)
parser.add_argument(
    "--no_progress_stall_ratio",
    type=float,
    default=0.50,
    help="Timeout episode is no_progress if stall-step ratio is above this threshold.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

import skrl
from packaging import version

SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    raise SystemExit(1)

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
    from lstm_runner_patch import apply_lstm_runner_patch

    apply_lstm_runner_patch()
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner
else:
    raise ValueError(f"Unsupported ML framework: {args_cli.ml_framework}")

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import isaac_lab_tutorial.tasks  # noqa: F401
from checkpoint_step import infer_resume_step_from_checkpoint, restore_env_curriculum_step


def _find_latest_checkpoint(log_root_path: Path, algorithm: str, ml_framework: str) -> Path:
    run_dirs = sorted(
        [p for p in log_root_path.glob(f"*_{algorithm}_{ml_framework}") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No runs found under {log_root_path} for pattern '*_{algorithm}_{ml_framework}'.")

    latest_run = run_dirs[0]
    ckpt_dir = latest_run / "checkpoints"
    best_ckpt = ckpt_dir / "best_agent.pt"
    if best_ckpt.exists():
        return best_ckpt

    agent_ckpts = sorted(ckpt_dir.glob("agent_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not agent_ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}.")
    return agent_ckpts[0]


def _to_done_mask(x, num_envs: int) -> torch.Tensor:
    t = torch.as_tensor(x)
    if t.ndim == 2 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    return t.to(torch.bool).reshape(num_envs)


def _to_reward_vec(x, num_envs: int) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.ndim == 2 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    return t.reshape(num_envs)


def _to_vec(x: Any, num_envs: int, dtype: torch.dtype) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=dtype)
    if t.ndim == 2 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    return t.reshape(num_envs).detach().cpu()


def _safe_div(numer: float, denom: float) -> float:
    return float(numer) / float(denom) if float(denom) > 0.0 else 0.0


def _quat_wxyz_to_yaw(q: torch.Tensor) -> float:
    values = torch.as_tensor(q, dtype=torch.float32).detach().cpu().reshape(-1)
    if values.numel() < 4:
        return 0.0
    w, x, y, z = (float(values[i].item()) for i in range(4))
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _capture_trace_scene(unwrapped: Any, env_id: int) -> dict[str, Any]:
    origin = torch.zeros(2)
    if hasattr(unwrapped, "scene") and hasattr(unwrapped.scene, "env_origins"):
        origin = unwrapped.scene.env_origins[env_id, :2].detach().cpu()

    scene: dict[str, Any] = {"origin_xy": [float(origin[0].item()), float(origin[1].item())]}
    if hasattr(unwrapped, "_junction_wall_seg_valid"):
        valid = unwrapped._junction_wall_seg_valid[env_id].detach().cpu()
        starts = unwrapped._junction_wall_seg_start[env_id].detach().cpu()
        ends = unwrapped._junction_wall_seg_end[env_id].detach().cpu()
        segments = []
        for start, end, is_valid in zip(starts, ends, valid, strict=False):
            if not bool(is_valid.item()):
                continue
            s = start[:2] - origin
            e = end[:2] - origin
            segments.append(
                {
                    "start": [float(s[0].item()), float(s[1].item())],
                    "end": [float(e[0].item()), float(e[1].item())],
                }
            )
        scene["wall_segments"] = segments

    if hasattr(unwrapped, "obstacle_pos_w"):
        obstacles = []
        obs_xy = unwrapped.obstacle_pos_w[env_id, :, :2].detach().cpu() - origin
        for obs in obs_xy:
            x = float(obs[0].item())
            y = float(obs[1].item())
            if abs(x) > 20.0 or abs(y) > 20.0:
                continue
            obstacles.append([x, y])
        scene["obstacles_xy"] = obstacles

    if hasattr(unwrapped, "goal_pos_w"):
        goal = unwrapped.goal_pos_w[env_id, :2].detach().cpu() - origin
        scene["goal_xy"] = [float(goal[0].item()), float(goal[1].item())]
    return scene


def _capture_trace_point(unwrapped: Any, env_id: int, step: int) -> dict[str, Any]:
    origin = torch.zeros(2)
    if hasattr(unwrapped, "scene") and hasattr(unwrapped.scene, "env_origins"):
        origin = unwrapped.scene.env_origins[env_id, :2].detach().cpu()

    robot_xy = unwrapped.robot.data.root_pos_w[env_id, :2].detach().cpu() - origin
    goal_xy = torch.zeros(2)
    if hasattr(unwrapped, "goal_pos_w"):
        goal_xy = unwrapped.goal_pos_w[env_id, :2].detach().cpu() - origin

    point = {
        "step": int(step),
        "x": float(robot_xy[0].item()),
        "y": float(robot_xy[1].item()),
        "goal_x": float(goal_xy[0].item()),
        "goal_y": float(goal_xy[1].item()),
    }
    if hasattr(unwrapped.robot.data, "root_link_quat_w"):
        point["yaw"] = _quat_wxyz_to_yaw(unwrapped.robot.data.root_link_quat_w[env_id])
    if hasattr(unwrapped, "planner_state_index"):
        point["planner_state"] = int(unwrapped.planner_state_index[env_id].detach().cpu().item())
    if hasattr(unwrapped, "fixed_scene_active_index"):
        point["scene_index"] = int(unwrapped.fixed_scene_active_index[env_id].detach().cpu().item())
    if hasattr(unwrapped, "latest_lidar_ranges"):
        lidar = unwrapped.latest_lidar_ranges[env_id].detach().cpu()
        point["min_lidar"] = float(torch.min(lidar).item())
    return point


def _override_curriculum_tuple(env_cfg: object, attr_name: str, value: float) -> None:
    if not hasattr(env_cfg, attr_name):
        return
    raw = tuple(getattr(env_cfg, attr_name))
    if not raw:
        return
    setattr(env_cfg, attr_name, tuple(value for _ in raw))


def _parse_scene_pool_arg(scene_pool_arg: str | None) -> tuple[int, ...] | None:
    if scene_pool_arg is None:
        return None
    tokens = [token.strip() for token in str(scene_pool_arg).split(",")]
    indices: list[int] = []
    for token in tokens:
        if not token:
            continue
        try:
            index = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid scene index '{token}' in --scene_pool='{scene_pool_arg}'.") from exc
        if index < 0:
            raise ValueError(f"Scene indices in --scene_pool must be >= 0, got {index}.")
        if index not in indices:
            indices.append(index)
    return tuple(indices)


def _parse_scene_pool_weights_arg(scene_pool_weights_arg: str | None) -> tuple[float, ...] | None:
    if scene_pool_weights_arg is None:
        return None
    tokens = [token.strip() for token in str(scene_pool_weights_arg).split(",")]
    weights: list[float] = []
    for token in tokens:
        if not token:
            continue
        try:
            weight = float(token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid weight '{token}' in --scene_pool_weights='{scene_pool_weights_arg}'."
            ) from exc
        if weight < 0.0:
            raise ValueError(f"Scene weights in --scene_pool_weights must be >= 0, got {weight}.")
        weights.append(weight)
    return tuple(weights)


def _current_goal_dist(unwrapped, num_envs: int) -> torch.Tensor:
    if not (hasattr(unwrapped, "goal_pos_w") and hasattr(unwrapped, "robot")):
        return torch.zeros(num_envs, dtype=torch.float32)
    goal = getattr(unwrapped, "goal_pos_w")[:, :2]
    robot_xy = unwrapped.robot.data.root_pos_w[:, :2]
    return torch.linalg.norm(goal - robot_xy, dim=1).detach().cpu()


def _current_planner_state(unwrapped, num_envs: int) -> torch.Tensor:
    if not hasattr(unwrapped, "planner_state_index"):
        return torch.zeros(num_envs, dtype=torch.long)
    return _to_vec(getattr(unwrapped, "planner_state_index"), num_envs, torch.long)


def _current_fixed_scene_index(unwrapped, num_envs: int) -> torch.Tensor:
    if not hasattr(unwrapped, "fixed_scene_active_index"):
        return torch.full((num_envs,), -1, dtype=torch.long)
    return _to_vec(getattr(unwrapped, "fixed_scene_active_index"), num_envs, torch.long)


def main() -> None:
    algorithm = args_cli.algorithm.lower()

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.debug_print_enable is not None and hasattr(env_cfg, "debug_print_enable"):
        env_cfg.debug_print_enable = bool(args_cli.debug_print_enable)
    if args_cli.debug_print_interval_steps is not None and hasattr(env_cfg, "debug_print_interval_steps"):
        env_cfg.debug_print_interval_steps = max(1, int(args_cli.debug_print_interval_steps))
    if args_cli.debug_phase_tag is not None and hasattr(env_cfg, "debug_phase_tag"):
        env_cfg.debug_phase_tag = int(args_cli.debug_phase_tag)
    if args_cli.num_obstacles is not None and hasattr(env_cfg, "num_obstacles"):
        env_cfg.num_obstacles = max(0, int(args_cli.num_obstacles))
    parsed_scene_pool = _parse_scene_pool_arg(args_cli.scene_pool)
    if parsed_scene_pool is not None and hasattr(env_cfg, "fixed_scene_multi_index_pool"):
        env_cfg.fixed_scene_multi_index_pool = parsed_scene_pool
        if hasattr(env_cfg, "fixed_scene_multi_enable"):
            env_cfg.fixed_scene_multi_enable = True
    parsed_scene_pool_weights = _parse_scene_pool_weights_arg(args_cli.scene_pool_weights)
    if parsed_scene_pool_weights is not None and hasattr(env_cfg, "fixed_scene_multi_index_weights"):
        pool_for_weights = (
            parsed_scene_pool
            if parsed_scene_pool is not None
            else tuple(getattr(env_cfg, "fixed_scene_multi_index_pool", ()))
        )
        if len(pool_for_weights) > 0 and len(parsed_scene_pool_weights) != len(pool_for_weights):
            raise ValueError(
                "--scene_pool_weights length must match --scene_pool (or fixed_scene_multi_index_pool). "
                f"Got {len(parsed_scene_pool_weights)} vs {len(pool_for_weights)}."
            )
        env_cfg.fixed_scene_multi_index_weights = parsed_scene_pool_weights
        if hasattr(env_cfg, "fixed_scene_multi_enable"):
            env_cfg.fixed_scene_multi_enable = True
    if args_cli.scene_id is not None and hasattr(env_cfg, "fixed_scene_multi_index_override"):
        env_cfg.fixed_scene_multi_index_override = int(args_cli.scene_id)
        if hasattr(env_cfg, "fixed_scene_multi_enable"):
            env_cfg.fixed_scene_multi_enable = True
    if args_cli.episode_length_s is not None and hasattr(env_cfg, "episode_length_s"):
        env_cfg.episode_length_s = max(1.0, float(args_cli.episode_length_s))
    if args_cli.goal_max_distance is not None and hasattr(env_cfg, "goal_max_distance"):
        env_cfg.goal_max_distance = max(0.2, float(args_cli.goal_max_distance))
    if args_cli.goal_spawn_range_min is not None and hasattr(env_cfg, "goal_spawn_range_min"):
        env_cfg.goal_spawn_range_min = max(0.0, float(args_cli.goal_spawn_range_min))
    if args_cli.goal_spawn_range_max is not None and hasattr(env_cfg, "goal_spawn_range_max"):
        env_cfg.goal_spawn_range_max = max(0.1, float(args_cli.goal_spawn_range_max))
    if hasattr(env_cfg, "goal_spawn_range_min") and hasattr(env_cfg, "goal_spawn_range_max"):
        env_cfg.goal_spawn_range_max = max(float(env_cfg.goal_spawn_range_min), float(env_cfg.goal_spawn_range_max))
    if args_cli.junction_half_width is not None and hasattr(env_cfg, "junction_half_width"):
        env_cfg.junction_half_width = max(0.2, float(args_cli.junction_half_width))
    if args_cli.junction_half_length is not None and hasattr(env_cfg, "junction_half_length"):
        env_cfg.junction_half_length = max(1.0, float(args_cli.junction_half_length))
    if args_cli.junction_goal_forward_max is not None and hasattr(env_cfg, "junction_goal_forward_max"):
        env_cfg.junction_goal_forward_max = max(0.2, float(args_cli.junction_goal_forward_max))
    if args_cli.junction_goal_forward_min is not None and hasattr(env_cfg, "junction_goal_forward_min"):
        env_cfg.junction_goal_forward_min = max(0.0, float(args_cli.junction_goal_forward_min))
    if hasattr(env_cfg, "junction_goal_forward_min") and hasattr(env_cfg, "junction_goal_forward_max"):
        env_cfg.junction_goal_forward_max = max(
            float(env_cfg.junction_goal_forward_min),
            float(env_cfg.junction_goal_forward_max),
        )
    if args_cli.junction_robot_spawn_offset_min is not None and hasattr(env_cfg, "junction_robot_spawn_offset_min"):
        env_cfg.junction_robot_spawn_offset_min = max(0.0, float(args_cli.junction_robot_spawn_offset_min))
    if args_cli.junction_robot_spawn_offset_max is not None and hasattr(env_cfg, "junction_robot_spawn_offset_max"):
        env_cfg.junction_robot_spawn_offset_max = max(0.0, float(args_cli.junction_robot_spawn_offset_max))
    if hasattr(env_cfg, "junction_robot_spawn_offset_min") and hasattr(env_cfg, "junction_robot_spawn_offset_max"):
        env_cfg.junction_robot_spawn_offset_max = max(
            float(env_cfg.junction_robot_spawn_offset_min),
            float(env_cfg.junction_robot_spawn_offset_max),
        )
    if (
        args_cli.junction_robot_spawn_lateral_jitter is not None
        and hasattr(env_cfg, "junction_robot_spawn_lateral_jitter")
    ):
        env_cfg.junction_robot_spawn_lateral_jitter = max(0.0, float(args_cli.junction_robot_spawn_lateral_jitter))
    if args_cli.junction_x_sampling_prob is not None and hasattr(env_cfg, "junction_x_sampling_prob"):
        env_cfg.junction_x_sampling_prob = min(1.0, max(0.0, float(args_cli.junction_x_sampling_prob)))
    if args_cli.action_scale is not None and hasattr(env_cfg, "action_scale"):
        env_cfg.action_scale = max(0.1, float(args_cli.action_scale))
    if args_cli.near_goal_min_action_scale is not None and hasattr(env_cfg, "near_goal_min_action_scale"):
        env_cfg.near_goal_min_action_scale = min(1.0, max(0.0, float(args_cli.near_goal_min_action_scale)))
    if args_cli.obstacle_slowdown_distance is not None and hasattr(env_cfg, "obstacle_slowdown_distance"):
        env_cfg.obstacle_slowdown_distance = max(0.05, float(args_cli.obstacle_slowdown_distance))
    if args_cli.obstacle_min_action_scale is not None and hasattr(env_cfg, "obstacle_min_action_scale"):
        env_cfg.obstacle_min_action_scale = min(1.0, max(0.0, float(args_cli.obstacle_min_action_scale)))
    if args_cli.goal_reach_threshold is not None and hasattr(env_cfg, "goal_reach_threshold"):
        goal_reach_threshold = max(0.05, float(args_cli.goal_reach_threshold))
        env_cfg.goal_reach_threshold = goal_reach_threshold
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_reach_thresholds", goal_reach_threshold)
        print(f"[EVAL][OVERRIDE] goal_reach_threshold={goal_reach_threshold:.3f}")
    if args_cli.goal_local_x is not None:
        goal_local_x = float(args_cli.goal_local_x)
        if hasattr(env_cfg, "defect_single_gap_goal_local_x"):
            env_cfg.defect_single_gap_goal_local_x = goal_local_x
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_local_xs", goal_local_x)
        print(f"[EVAL][OVERRIDE] goal_local_x={goal_local_x:.3f}")
    if args_cli.goal_local_y is not None:
        goal_local_y = float(args_cli.goal_local_y)
        if hasattr(env_cfg, "defect_single_gap_goal_local_y"):
            env_cfg.defect_single_gap_goal_local_y = goal_local_y
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_local_ys", goal_local_y)
        print(f"[EVAL][OVERRIDE] goal_local_y={goal_local_y:.3f}")
    if args_cli.v_max is not None and hasattr(env_cfg, "v_max"):
        env_cfg.v_max = max(0.01, float(args_cli.v_max))
    if args_cli.omega_limit is not None and hasattr(env_cfg, "omega_limit"):
        env_cfg.omega_limit = max(0.05, float(args_cli.omega_limit))
    if args_cli.tau_v is not None and hasattr(env_cfg, "tau_v"):
        env_cfg.tau_v = max(1.0e-3, float(args_cli.tau_v))
    if args_cli.tau_omega is not None and hasattr(env_cfg, "tau_omega"):
        env_cfg.tau_omega = max(1.0e-3, float(args_cli.tau_omega))
    if args_cli.wheel_linear_speed_max is not None and hasattr(env_cfg, "wheel_linear_speed_max"):
        env_cfg.wheel_linear_speed_max = max(0.01, float(args_cli.wheel_linear_speed_max))
    if args_cli.forward_only is not None and hasattr(env_cfg, "forward_only"):
        env_cfg.forward_only = bool(args_cli.forward_only)
    if args_cli.wheel_target_delta_limit is not None and hasattr(env_cfg, "wheel_target_delta_limit"):
        env_cfg.wheel_target_delta_limit = max(0.0, float(args_cli.wheel_target_delta_limit))
    if args_cli.defect_goal_distance_min is not None and hasattr(env_cfg, "defect_goal_distance_min"):
        env_cfg.defect_goal_distance_min = max(0.05, float(args_cli.defect_goal_distance_min))
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_distance_mins", env_cfg.defect_goal_distance_min)
    if args_cli.defect_goal_distance_max is not None and hasattr(env_cfg, "defect_goal_distance_max"):
        env_cfg.defect_goal_distance_max = max(
            max(0.05, float(getattr(env_cfg, "defect_goal_distance_min", 0.05))),
            float(args_cli.defect_goal_distance_max),
        )
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_distance_maxs", env_cfg.defect_goal_distance_max)
    if args_cli.defect_spawn_jitter_xy is not None and hasattr(env_cfg, "defect_spawn_jitter_xy"):
        env_cfg.defect_spawn_jitter_xy = max(0.0, float(args_cli.defect_spawn_jitter_xy))
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_spawn_jitters", env_cfg.defect_spawn_jitter_xy)
    if args_cli.defect_goal_jitter_xy is not None and hasattr(env_cfg, "defect_goal_jitter_xy"):
        env_cfg.defect_goal_jitter_xy = max(0.0, float(args_cli.defect_goal_jitter_xy))
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_jitters", env_cfg.defect_goal_jitter_xy)
    if args_cli.defect_dwb_corridor_width is not None and hasattr(env_cfg, "defect_dwb_corridor_width"):
        env_cfg.defect_dwb_corridor_width = max(0.6, float(args_cli.defect_dwb_corridor_width))
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_corridor_widths", env_cfg.defect_dwb_corridor_width)
    if args_cli.defect_dwb_corridor_half_length is not None and hasattr(env_cfg, "defect_dwb_corridor_half_length"):
        env_cfg.defect_dwb_corridor_half_length = max(1.2, float(args_cli.defect_dwb_corridor_half_length))
    if args_cli.defect_dwb_obstacle_spacing is not None and hasattr(env_cfg, "defect_dwb_obstacle_spacing"):
        env_cfg.defect_dwb_obstacle_spacing = max(0.45, float(args_cli.defect_dwb_obstacle_spacing))
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_obstacle_spacings", env_cfg.defect_dwb_obstacle_spacing)
    if (
        args_cli.defect_dwb_obstacle_spacing_jitter is not None
        and hasattr(env_cfg, "defect_dwb_obstacle_spacing_jitter")
    ):
        env_cfg.defect_dwb_obstacle_spacing_jitter = max(0.0, float(args_cli.defect_dwb_obstacle_spacing_jitter))
        _override_curriculum_tuple(
            env_cfg,
            "defect_dwb_curriculum_obstacle_spacing_jitters",
            env_cfg.defect_dwb_obstacle_spacing_jitter,
        )
    if args_cli.defect_dwb_obstacle_pair_shift_y is not None and hasattr(env_cfg, "defect_dwb_obstacle_pair_shift_y"):
        env_cfg.defect_dwb_obstacle_pair_shift_y = max(0.0, float(args_cli.defect_dwb_obstacle_pair_shift_y))
        _override_curriculum_tuple(
            env_cfg,
            "defect_dwb_curriculum_obstacle_pair_shift_ys",
            env_cfg.defect_dwb_obstacle_pair_shift_y,
        )
    if args_cli.defect_dwb_obstacle_x_jitter is not None and hasattr(env_cfg, "defect_dwb_obstacle_x_jitter"):
        env_cfg.defect_dwb_obstacle_x_jitter = max(0.0, float(args_cli.defect_dwb_obstacle_x_jitter))
        _override_curriculum_tuple(
            env_cfg,
            "defect_dwb_curriculum_obstacle_x_jitters",
            env_cfg.defect_dwb_obstacle_x_jitter,
        )
    if args_cli.defect_door_corridor_width is not None and hasattr(env_cfg, "defect_door_corridor_width"):
        env_cfg.defect_door_corridor_width = max(0.6, float(args_cli.defect_door_corridor_width))
    if args_cli.defect_door_half_length is not None and hasattr(env_cfg, "defect_door_half_length"):
        env_cfg.defect_door_half_length = max(1.0, float(args_cli.defect_door_half_length))
    if args_cli.defect_door_width is not None and hasattr(env_cfg, "defect_door_width"):
        env_cfg.defect_door_width = max(0.3, float(args_cli.defect_door_width))
    if args_cli.defect_door_width_jitter is not None and hasattr(env_cfg, "defect_door_width_jitter"):
        env_cfg.defect_door_width_jitter = max(0.0, float(args_cli.defect_door_width_jitter))
    if args_cli.defect_door_straight_distance is not None and hasattr(env_cfg, "defect_door_straight_distance"):
        env_cfg.defect_door_straight_distance = max(0.2, float(args_cli.defect_door_straight_distance))
    if args_cli.defect_door_line_angle_max_deg is not None and hasattr(env_cfg, "defect_door_line_angle_max_deg"):
        env_cfg.defect_door_line_angle_max_deg = max(0.0, float(args_cli.defect_door_line_angle_max_deg))
    if args_cli.defect_door_goal_lateral_offset_max is not None and hasattr(
        env_cfg, "defect_door_goal_lateral_offset_max"
    ):
        env_cfg.defect_door_goal_lateral_offset_max = max(0.0, float(args_cli.defect_door_goal_lateral_offset_max))
    if args_cli.single_gap_force_goal_mode is not None:
        forced_goal_mode = str(args_cli.single_gap_force_goal_mode).strip().lower()
        try:
            setattr(env_cfg, "defect_single_gap_goal_mode", forced_goal_mode)
        except Exception:
            pass
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_modes", forced_goal_mode)
        print(f"[EVAL][OVERRIDE] single_gap_goal_mode={forced_goal_mode}")
    if args_cli.rear_clearance is not None:
        rear_clearance = max(0.0, float(args_cli.rear_clearance))
        if hasattr(env_cfg, "defect_single_gap_goal_rear_clearance_min"):
            env_cfg.defect_single_gap_goal_rear_clearance_min = rear_clearance
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_rear_clearance_mins", rear_clearance)
        print(f"[EVAL][OVERRIDE] rear_clearance={rear_clearance:.3f}")
    if args_cli.rear_lateral_center is not None:
        rear_lateral_center = float(args_cli.rear_lateral_center)
        if hasattr(env_cfg, "defect_single_gap_goal_rear_lateral_center"):
            env_cfg.defect_single_gap_goal_rear_lateral_center = rear_lateral_center
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_rear_lateral_centers", rear_lateral_center)
        print(f"[EVAL][OVERRIDE] rear_lateral_center={rear_lateral_center:.3f}")
    if args_cli.rear_offset_max is not None:
        rear_offset_max = max(0.0, float(args_cli.rear_offset_max))
        if hasattr(env_cfg, "defect_single_gap_goal_rear_lateral_offset_max"):
            env_cfg.defect_single_gap_goal_rear_lateral_offset_max = rear_offset_max
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_rear_lateral_offset_maxs", rear_offset_max)
        print(f"[EVAL][OVERRIDE] rear_offset_max={rear_offset_max:.3f}")
    if args_cli.bridge_clearance is not None:
        bridge_clearance = max(0.0, float(args_cli.bridge_clearance))
        if hasattr(env_cfg, "defect_single_gap_goal_bridge_clearance_min"):
            env_cfg.defect_single_gap_goal_bridge_clearance_min = bridge_clearance
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_bridge_clearance_mins", bridge_clearance)
        print(f"[EVAL][OVERRIDE] bridge_clearance={bridge_clearance:.3f}")
    if args_cli.bridge_lateral_center is not None:
        bridge_lateral_center = float(args_cli.bridge_lateral_center)
        if hasattr(env_cfg, "defect_single_gap_goal_bridge_lateral_center"):
            env_cfg.defect_single_gap_goal_bridge_lateral_center = bridge_lateral_center
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_bridge_lateral_centers", bridge_lateral_center)
        print(f"[EVAL][OVERRIDE] bridge_lateral_center={bridge_lateral_center:.3f}")
    if args_cli.bridge_offset_max is not None:
        bridge_offset_max = max(0.0, float(args_cli.bridge_offset_max))
        if hasattr(env_cfg, "defect_single_gap_goal_bridge_lateral_offset_max"):
            env_cfg.defect_single_gap_goal_bridge_lateral_offset_max = bridge_offset_max
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_bridge_lateral_offset_maxs", bridge_offset_max)
        print(f"[EVAL][OVERRIDE] bridge_offset_max={bridge_offset_max:.3f}")
    if args_cli.defect_dwb_collision_penalty is not None:
        collision_penalty = float(args_cli.defect_dwb_collision_penalty)
        if hasattr(env_cfg, "collision_penalty"):
            env_cfg.collision_penalty = collision_penalty
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_collision_penalties", collision_penalty)
        print(f"[EVAL][OVERRIDE] defect_dwb_collision_penalty={collision_penalty:.3f}")
    if args_cli.defect_dwb_clearance_penalty_scale is not None and hasattr(
        env_cfg, "defect_dwb_clearance_penalty_scale"
    ):
        env_cfg.defect_dwb_clearance_penalty_scale = float(args_cli.defect_dwb_clearance_penalty_scale)
        print(
            "[EVAL][OVERRIDE] "
            f"defect_dwb_clearance_penalty_scale={env_cfg.defect_dwb_clearance_penalty_scale:.3f}"
        )
    if args_cli.defect_dwb_clearance_penalty_distance is not None and hasattr(
        env_cfg, "defect_dwb_clearance_penalty_distance"
    ):
        env_cfg.defect_dwb_clearance_penalty_distance = max(
            0.0,
            float(args_cli.defect_dwb_clearance_penalty_distance),
        )
        print(
            "[EVAL][OVERRIDE] "
            f"defect_dwb_clearance_penalty_distance={env_cfg.defect_dwb_clearance_penalty_distance:.3f}"
        )
    if args_cli.defect_dwb_front_clearance_penalty_scale is not None and hasattr(
        env_cfg, "defect_dwb_front_clearance_penalty_scale"
    ):
        env_cfg.defect_dwb_front_clearance_penalty_scale = float(
            args_cli.defect_dwb_front_clearance_penalty_scale
        )
        print(
            "[EVAL][OVERRIDE] "
            f"defect_dwb_front_clearance_penalty_scale={env_cfg.defect_dwb_front_clearance_penalty_scale:.3f}"
        )
    if args_cli.defect_dwb_front_clearance_penalty_distance is not None and hasattr(
        env_cfg, "defect_dwb_front_clearance_penalty_distance"
    ):
        env_cfg.defect_dwb_front_clearance_penalty_distance = max(
            0.0,
            float(args_cli.defect_dwb_front_clearance_penalty_distance),
        )
        print(
            "[EVAL][OVERRIDE] "
            f"defect_dwb_front_clearance_penalty_distance={env_cfg.defect_dwb_front_clearance_penalty_distance:.3f}"
        )
    if args_cli.defect_dwb_clearance_penalty_power is not None and hasattr(
        env_cfg, "defect_dwb_clearance_penalty_power"
    ):
        env_cfg.defect_dwb_clearance_penalty_power = max(
            1.0,
            float(args_cli.defect_dwb_clearance_penalty_power),
        )
        print(
            "[EVAL][OVERRIDE] "
            f"defect_dwb_clearance_penalty_power={env_cfg.defect_dwb_clearance_penalty_power:.3f}"
        )
    if args_cli.defect_dwb_gap_progress_reward_scale is not None and hasattr(
        env_cfg, "defect_dwb_gap_progress_reward_scale"
    ):
        env_cfg.defect_dwb_gap_progress_reward_scale = max(
            0.0,
            float(args_cli.defect_dwb_gap_progress_reward_scale),
        )
        print(
            "[EVAL][OVERRIDE] "
            f"defect_dwb_gap_progress_reward_scale={env_cfg.defect_dwb_gap_progress_reward_scale:.3f}"
        )
    if args_cli.defect_dwb_gap_alignment_reward_scale is not None and hasattr(
        env_cfg, "defect_dwb_gap_alignment_reward_scale"
    ):
        env_cfg.defect_dwb_gap_alignment_reward_scale = max(
            0.0,
            float(args_cli.defect_dwb_gap_alignment_reward_scale),
        )
        print(
            "[EVAL][OVERRIDE] "
            f"defect_dwb_gap_alignment_reward_scale={env_cfg.defect_dwb_gap_alignment_reward_scale:.3f}"
        )
    if args_cli.defect_dwb_gap_clear_bonus is not None and hasattr(env_cfg, "defect_dwb_gap_clear_bonus"):
        env_cfg.defect_dwb_gap_clear_bonus = max(0.0, float(args_cli.defect_dwb_gap_clear_bonus))
        print(f"[EVAL][OVERRIDE] defect_dwb_gap_clear_bonus={env_cfg.defect_dwb_gap_clear_bonus:.3f}")
    if args_cli.defect_dwb_side_commit_reward_scale is not None and hasattr(
        env_cfg, "defect_dwb_side_commit_reward_scale"
    ):
        env_cfg.defect_dwb_side_commit_reward_scale = max(
            0.0,
            float(args_cli.defect_dwb_side_commit_reward_scale),
        )
        print(
            "[EVAL][OVERRIDE] "
            f"defect_dwb_side_commit_reward_scale={env_cfg.defect_dwb_side_commit_reward_scale:.3f}"
        )
    if args_cli.defect_dwb_side_switch_penalty_scale is not None and hasattr(
        env_cfg, "defect_dwb_side_switch_penalty_scale"
    ):
        env_cfg.defect_dwb_side_switch_penalty_scale = max(
            0.0,
            float(args_cli.defect_dwb_side_switch_penalty_scale),
        )
        print(
            "[EVAL][OVERRIDE] "
            f"defect_dwb_side_switch_penalty_scale={env_cfg.defect_dwb_side_switch_penalty_scale:.3f}"
        )
    if args_cli.defect_dwb_side_commit_deadzone is not None and hasattr(env_cfg, "defect_dwb_side_commit_deadzone"):
        env_cfg.defect_dwb_side_commit_deadzone = max(0.0, float(args_cli.defect_dwb_side_commit_deadzone))
        print(f"[EVAL][OVERRIDE] defect_dwb_side_commit_deadzone={env_cfg.defect_dwb_side_commit_deadzone:.3f}")
    if args_cli.bypass_reward_enable is not None and hasattr(env_cfg, "bypass_reward_enable"):
        env_cfg.bypass_reward_enable = bool(int(args_cli.bypass_reward_enable))
        print(f"[EVAL][OVERRIDE] bypass_reward_enable={int(env_cfg.bypass_reward_enable)}")
    if args_cli.bypass_reward_scale is not None and hasattr(env_cfg, "bypass_reward_scale"):
        env_cfg.bypass_reward_scale = max(0.0, float(args_cli.bypass_reward_scale))
        print(f"[EVAL][OVERRIDE] bypass_reward_scale={env_cfg.bypass_reward_scale:.3f}")
    if args_cli.bypass_open_side_min_lidar is not None and hasattr(env_cfg, "bypass_open_side_min_lidar"):
        env_cfg.bypass_open_side_min_lidar = max(0.0, float(args_cli.bypass_open_side_min_lidar))
        print(f"[EVAL][OVERRIDE] bypass_open_side_min_lidar={env_cfg.bypass_open_side_min_lidar:.3f}")
    if args_cli.bypass_blocked_side_max_lidar is not None and hasattr(env_cfg, "bypass_blocked_side_max_lidar"):
        env_cfg.bypass_blocked_side_max_lidar = max(0.0, float(args_cli.bypass_blocked_side_max_lidar))
        print(f"[EVAL][OVERRIDE] bypass_blocked_side_max_lidar={env_cfg.bypass_blocked_side_max_lidar:.3f}")
    if args_cli.bypass_near_obstacle_distance is not None and hasattr(env_cfg, "bypass_near_obstacle_distance"):
        env_cfg.bypass_near_obstacle_distance = max(0.0, float(args_cli.bypass_near_obstacle_distance))
        print(f"[EVAL][OVERRIDE] bypass_near_obstacle_distance={env_cfg.bypass_near_obstacle_distance:.3f}")
    if args_cli.bypass_forward_speed_threshold is not None and hasattr(env_cfg, "bypass_forward_speed_threshold"):
        env_cfg.bypass_forward_speed_threshold = max(0.0, float(args_cli.bypass_forward_speed_threshold))
        print(f"[EVAL][OVERRIDE] bypass_forward_speed_threshold={env_cfg.bypass_forward_speed_threshold:.3f}")
    if args_cli.freeze_curriculum_stage is not None and hasattr(env_cfg, "defect_dwb_curriculum_freeze_stage"):
        env_cfg.defect_dwb_curriculum_freeze_stage = int(args_cli.freeze_curriculum_stage)
        print(f"[EVAL][OVERRIDE] freeze_curriculum_stage={int(args_cli.freeze_curriculum_stage)}")
    if args_cli.single_gap_goal_obstacle_debug is not None:
        try:
            setattr(env_cfg, "defect_single_gap_goal_obstacle_debug", bool(args_cli.single_gap_goal_obstacle_debug))
        except Exception:
            pass
    if args_cli.single_gap_goal_obstacle_debug_every is not None:
        try:
            setattr(
                env_cfg,
                "defect_single_gap_goal_obstacle_debug_every",
                max(1, int(args_cli.single_gap_goal_obstacle_debug_every)),
            )
        except Exception:
            pass
    env_cfg.seed = args_cli.seed

    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    log_root_path = Path("logs") / "skrl" / experiment_cfg["agent"]["experiment"]["directory"]
    log_root_path = log_root_path.resolve()

    if args_cli.checkpoint:
        resume_path = Path(args_cli.checkpoint).expanduser().resolve()
    else:
        resume_path = _find_latest_checkpoint(log_root_path, algorithm, args_cli.ml_framework)
    if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    print(f"[EVAL] task={args_cli.task}")
    print(f"[EVAL] checkpoint={resume_path}")
    print(f"[EVAL] num_envs={args_cli.num_envs}, episodes={args_cli.episodes}")

    # Create base environment for terminal-state inspection.
    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(base_env.unwrapped, DirectMARLEnv) and algorithm in ["ppo", "sac"]:
        base_env = multi_agent_to_single_agent(base_env)

    resume_step, resume_step_source = infer_resume_step_from_checkpoint(resume_path)
    if resume_step is not None or args_cli.curriculum_start_step is not None:
        restore_env_curriculum_step(
            base_env,
            resume_step,
            curriculum_start_step=args_cli.curriculum_start_step,
            source=resume_step_source,
            log_prefix="[EVAL]",
        )
    else:
        print(f"[EVAL] Could not infer curriculum step from checkpoint: {resume_path.name}")

    env = SkrlVecEnvWrapper(base_env, ml_framework=args_cli.ml_framework)
    runner = Runner(env, experiment_cfg)
    runner.agent.load(str(resume_path))
    runner.agent.set_running_mode("eval")

    obs, _ = env.reset()
    num_envs = args_cli.num_envs
    ep_return = torch.zeros(num_envs, dtype=torch.float32)
    ep_length = torch.zeros(num_envs, dtype=torch.int32)
    ep_stall_steps = torch.zeros(num_envs, dtype=torch.int32)
    ep_sat_steps = torch.zeros(num_envs, dtype=torch.int32)
    ep_turn_cmd_reward_sum = torch.zeros(num_envs, dtype=torch.float32)
    ep_turn_penalty_sum = torch.zeros(num_envs, dtype=torch.float32)

    finished = 0
    success = 0
    collision = 0
    timeout = 0
    returns: list[float] = []
    lengths: list[int] = []

    unwrapped = base_env.unwrapped
    state_names = tuple(getattr(unwrapped, "PLANNER_STATE_NAMES", ("explore", "fallback", "junction", "return")))
    state_stats: dict[str, dict[str, float | int]] = {
        state_name: {
            "episodes": 0,
            "success": 0,
            "collision": 0,
            "timeout": 0,
            "goal_end_dist_sum": 0.0,
            "min_lidar_sum": 0.0,
            "steps_total": 0,
            "stall_steps_total": 0,
            "sat_steps_total": 0,
            "turn_cmd_reward_sum": 0.0,
            "turn_penalty_sum": 0.0,
            "no_progress": 0,
            "local_timeout": 0,
        }
        for state_name in state_names
    }
    scene_stats: dict[int, dict[str, float | int]] = {}

    episode_state = _current_planner_state(unwrapped, num_envs)
    episode_scene = _current_fixed_scene_index(unwrapped, num_envs)
    episode_start_goal_dist = _current_goal_dist(unwrapped, num_envs)
    trace_env_id = max(0, min(int(args_cli.trajectory_env_id), num_envs - 1))
    trace_enabled = bool(args_cli.trajectory_json)
    trace_done = False
    trace_points: list[dict[str, Any]] = []
    trace_scene: dict[str, Any] = {}
    trace_result: dict[str, Any] = {}
    if trace_enabled:
        trace_scene = _capture_trace_scene(unwrapped, trace_env_id)
        trace_points.append(_capture_trace_point(unwrapped, trace_env_id, 0))

    loop_error: BaseException | None = None
    warned_app_not_running = False
    try:
        while finished < args_cli.episodes:
            if not simulation_app.is_running() and not warned_app_not_running:
                print(
                    (
                        "[EVAL] warning: simulation_app.is_running() is False; "
                        "continuing until requested episode count is reached."
                    ),
                    flush=True,
                )
                warned_app_not_running = True
            with torch.inference_mode():
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                if hasattr(env, "possible_agents"):
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])
                obs, reward, terminated, truncated, _ = env.step(actions)

            reward_vec = _to_reward_vec(reward, num_envs)
            term_mask = _to_done_mask(terminated, num_envs)
            trunc_mask = _to_done_mask(truncated, num_envs)
            done_mask = torch.logical_or(term_mask, trunc_mask)

            step_turn_cmd_reward = _to_vec(
                getattr(unwrapped, "_last_step_turn_cmd_reward", torch.zeros(num_envs)),
                num_envs,
                torch.float32,
            )
            step_turn_penalty = _to_vec(
                getattr(unwrapped, "_last_step_turn_penalty", torch.zeros(num_envs)),
                num_envs,
                torch.float32,
            )
            step_stall = _to_vec(getattr(unwrapped, "_last_step_stall", torch.zeros(num_envs)), num_envs, torch.bool)
            step_sat = _to_vec(
                getattr(unwrapped, "_last_step_action_saturated", torch.zeros(num_envs)),
                num_envs,
                torch.bool,
            )

            ep_return += reward_vec.cpu()
            ep_length += 1
            ep_stall_steps += step_stall.to(torch.int32)
            ep_sat_steps += step_sat.to(torch.int32)
            ep_turn_cmd_reward_sum += step_turn_cmd_reward
            ep_turn_penalty_sum += step_turn_penalty
            if trace_enabled and not trace_done:
                trace_points.append(_capture_trace_point(unwrapped, trace_env_id, int(ep_length[trace_env_id].item())))

            if torch.any(done_mask):
                done_ids = torch.where(done_mask)[0]
                done_ids_cpu = done_ids.detach().cpu()
                done_success_t = _to_vec(
                    getattr(unwrapped, "_last_done_success", torch.zeros(num_envs)),
                    num_envs,
                    torch.bool,
                )
                done_collision_t = _to_vec(
                    getattr(unwrapped, "_last_done_collision", torch.zeros(num_envs)),
                    num_envs,
                    torch.bool,
                )
                done_timeout_t = _to_vec(
                    getattr(unwrapped, "_last_done_timeout", torch.zeros(num_envs)),
                    num_envs,
                    torch.bool,
                )
                done_goal_dist_t = _to_vec(
                    getattr(unwrapped, "_last_done_goal_dist", torch.zeros(num_envs)),
                    num_envs,
                    torch.float32,
                )
                done_min_lidar_t = _to_vec(
                    getattr(unwrapped, "_last_done_min_lidar", torch.zeros(num_envs)),
                    num_envs,
                    torch.float32,
                )
                done_state_t = _to_vec(
                    getattr(unwrapped, "_last_done_planner_state", episode_state),
                    num_envs,
                    torch.long,
                )
                done_scene_t = _to_vec(
                    getattr(unwrapped, "_last_done_fixed_scene_index", episode_scene),
                    num_envs,
                    torch.long,
                )

                for env_id in done_ids_cpu.tolist():
                    if finished >= args_cli.episodes:
                        break
                    finished += 1
                    ep_len = int(ep_length[env_id].item())
                    ep_stall = int(ep_stall_steps[env_id].item())
                    ep_sat = int(ep_sat_steps[env_id].item())
                    returns.append(float(ep_return[env_id].item()))
                    lengths.append(ep_len)

                    done_success = bool(done_success_t[env_id].item())
                    done_collision = bool(done_collision_t[env_id].item())
                    done_timeout = bool(done_timeout_t[env_id].item())

                    if done_success:
                        success += 1
                    elif done_collision:
                        collision += 1
                    elif done_timeout:
                        timeout += 1
                    else:
                        # fallback to step outputs
                        if bool(trunc_mask[env_id].item()):
                            timeout += 1
                            done_timeout = True
                        else:
                            collision += 1
                            done_collision = True

                    if trace_enabled and not trace_done and env_id == trace_env_id:
                        if done_success:
                            result = "success"
                        elif done_collision:
                            result = "collision"
                        elif done_timeout:
                            result = "timeout"
                        elif bool(trunc_mask[env_id].item()):
                            result = "timeout"
                        else:
                            result = "collision"
                        trace_result = {
                            "result": result,
                            "episode_index": int(finished),
                            "env_id": int(env_id),
                            "episode_length": int(ep_len),
                            "return": float(ep_return[env_id].item()),
                            "goal_end_dist": float(done_goal_dist_t[env_id].item()),
                            "min_lidar": float(done_min_lidar_t[env_id].item()),
                        }
                        trace_done = True

                    state_idx = int(done_state_t[env_id].item())
                    if state_idx < 0 or state_idx >= len(state_names):
                        state_idx = 0
                    state_name = state_names[state_idx]
                    state_bucket = state_stats[state_name]
                    state_bucket["episodes"] += 1
                    if done_success:
                        state_bucket["success"] += 1
                    elif done_collision:
                        state_bucket["collision"] += 1
                    else:
                        state_bucket["timeout"] += 1

                    goal_end_dist = float(done_goal_dist_t[env_id].item())
                    min_lidar = float(done_min_lidar_t[env_id].item())
                    state_bucket["goal_end_dist_sum"] += goal_end_dist
                    state_bucket["min_lidar_sum"] += min_lidar
                    state_bucket["steps_total"] += ep_len
                    state_bucket["stall_steps_total"] += ep_stall
                    state_bucket["sat_steps_total"] += ep_sat
                    state_bucket["turn_cmd_reward_sum"] += float(ep_turn_cmd_reward_sum[env_id].item())
                    state_bucket["turn_penalty_sum"] += float(ep_turn_penalty_sum[env_id].item())

                    scene_idx = int(done_scene_t[env_id].item())
                    if scene_idx >= 0:
                        scene_bucket = scene_stats.setdefault(
                            scene_idx,
                            {
                                "episodes": 0,
                                "success": 0,
                                "collision": 0,
                                "timeout": 0,
                                "goal_end_dist_sum": 0.0,
                                "min_lidar_sum": 0.0,
                            },
                        )
                        scene_bucket["episodes"] += 1
                        if done_success:
                            scene_bucket["success"] += 1
                        elif done_collision:
                            scene_bucket["collision"] += 1
                        else:
                            scene_bucket["timeout"] += 1
                        scene_bucket["goal_end_dist_sum"] += goal_end_dist
                        scene_bucket["min_lidar_sum"] += min_lidar

                    if done_timeout:
                        start_dist = max(1.0e-6, float(episode_start_goal_dist[env_id].item()))
                        progress_delta = max(0.0, start_dist - goal_end_dist)
                        progress_ratio = progress_delta / start_dist
                        stall_ratio = _safe_div(float(ep_stall), float(ep_len))
                        no_progress = (
                            (
                                progress_delta < float(args_cli.no_progress_min_delta_m)
                                and progress_ratio < float(args_cli.no_progress_min_ratio)
                            )
                            or stall_ratio >= float(args_cli.no_progress_stall_ratio)
                        )
                        if no_progress:
                            state_bucket["no_progress"] += 1
                        else:
                            state_bucket["local_timeout"] += 1

                    ep_return[env_id] = 0.0
                    ep_length[env_id] = 0
                    ep_stall_steps[env_id] = 0
                    ep_sat_steps[env_id] = 0
                    ep_turn_cmd_reward_sum[env_id] = 0.0
                    ep_turn_penalty_sum[env_id] = 0.0

                    if finished % max(1, args_cli.print_every) == 0 or finished == args_cli.episodes:
                        print(
                            (
                                f"[EVAL] finished={finished}/{args_cli.episodes} "
                                f"success={success/finished:.1%} "
                                f"collision={collision/finished:.1%} "
                                f"timeout={timeout/finished:.1%} "
                                f"mean_return={np.mean(returns):.2f} "
                                f"mean_len={np.mean(lengths):.1f}"
                            ),
                            flush=True,
                        )

                # Track context of newly reset episodes.
                next_state = _current_planner_state(unwrapped, num_envs)
                next_scene = _current_fixed_scene_index(unwrapped, num_envs)
                next_goal_dist = _current_goal_dist(unwrapped, num_envs)
                episode_state[done_ids_cpu] = next_state[done_ids_cpu]
                episode_scene[done_ids_cpu] = next_scene[done_ids_cpu]
                episode_start_goal_dist[done_ids_cpu] = next_goal_dist[done_ids_cpu]
    except BaseException as exc:
        loop_error = exc

    try:
        print("[EVAL] loop complete; preparing summary", flush=True)
        total = max(1, finished)
        return_mean = float(np.mean(returns)) if returns else 0.0
        return_std = float(np.std(returns)) if returns else 0.0
        ep_len_mean = float(np.mean(lengths)) if lengths else 0.0
        ep_len_std = float(np.std(lengths)) if lengths else 0.0

        print("=" * 88, flush=True)
        print(f"[EVAL][RESULT] episodes={finished}", flush=True)
        print(f"[EVAL][RESULT] success_rate={success/total:.4f} ({success}/{total})", flush=True)
        print(f"[EVAL][RESULT] collision_rate={collision/total:.4f} ({collision}/{total})", flush=True)
        print(f"[EVAL][RESULT] timeout_rate={timeout/total:.4f} ({timeout}/{total})", flush=True)
        if returns:
            print(f"[EVAL][RESULT] return_mean={return_mean:.4f}, return_std={return_std:.4f}", flush=True)
        if lengths:
            print(f"[EVAL][RESULT] ep_len_mean={ep_len_mean:.4f}, ep_len_std={ep_len_std:.4f}", flush=True)

        by_state_report: dict[str, dict[str, Any]] = {}
        for state_name in state_names:
            state_bucket = state_stats[state_name]
            state_episodes = int(state_bucket["episodes"])
            state_total = max(1, state_episodes)
            timeout_eps = int(state_bucket["timeout"])
            timeout_total = max(1, timeout_eps)
            steps_total = int(state_bucket["steps_total"])
            steps_denom = max(1, steps_total)
            success_rate = _safe_div(float(state_bucket["success"]), float(state_total))
            collision_rate = _safe_div(float(state_bucket["collision"]), float(state_total))
            timeout_rate = _safe_div(float(state_bucket["timeout"]), float(state_total))
            avg_goal_end_dist = _safe_div(float(state_bucket["goal_end_dist_sum"]), float(state_total))
            avg_min_lidar = _safe_div(float(state_bucket["min_lidar_sum"]), float(state_total))
            stall_step = _safe_div(float(state_bucket["stall_steps_total"]), float(steps_denom))
            sat_step = _safe_div(float(state_bucket["sat_steps_total"]), float(steps_denom))
            turn_cmd_reward = _safe_div(float(state_bucket["turn_cmd_reward_sum"]), float(steps_denom))
            turn_penalty = _safe_div(float(state_bucket["turn_penalty_sum"]), float(steps_denom))
            no_progress_count = int(state_bucket["no_progress"])
            local_timeout_count = int(state_bucket["local_timeout"])
            no_progress_rate = _safe_div(float(no_progress_count), float(timeout_total))
            local_timeout_rate = _safe_div(float(local_timeout_count), float(timeout_total))
            by_state_report[state_name] = {
                "episodes": state_episodes,
                "success": int(state_bucket["success"]),
                "collision": int(state_bucket["collision"]),
                "timeout": timeout_eps,
                "success_rate": success_rate,
                "collision_rate": collision_rate,
                "timeout_rate": timeout_rate,
                "avg_goal_end_dist": avg_goal_end_dist,
                "avg_min_lidar": avg_min_lidar,
                "stall_step": stall_step,
                "sat_step": sat_step,
                "turn_cmd_reward": turn_cmd_reward,
                "turn_penalty": turn_penalty,
                "no_progress": no_progress_count,
                "local_timeout": local_timeout_count,
                "no_progress_rate_in_timeout": no_progress_rate,
                "local_timeout_rate_in_timeout": local_timeout_rate,
                "timeout_episodes": timeout_eps,
            }
            print(
                (
                    f"[EVAL][STATE] state={state_name} episodes={state_episodes} "
                    f"success={success_rate:.1%} collision={collision_rate:.1%} timeout={timeout_rate:.1%} "
                    f"avg_goal_end_dist={avg_goal_end_dist:.3f} avg_min_lidar={avg_min_lidar:.3f} "
                    f"stall_step={stall_step:.1%} sat_step={sat_step:.1%} "
                    f"turn_cmd_reward={turn_cmd_reward:.4f} turn_penalty={turn_penalty:.4f} "
                    f"no_progress={no_progress_count}/{timeout_eps}({no_progress_rate:.1%}) "
                    f"local_timeout={local_timeout_count}/{timeout_eps}({local_timeout_rate:.1%})"
                ),
                flush=True,
            )
        by_scene_report: dict[str, dict[str, Any]] = {}
        for scene_idx in sorted(scene_stats.keys()):
            scene_bucket = scene_stats[scene_idx]
            scene_episodes = int(scene_bucket["episodes"])
            scene_total = max(1, scene_episodes)
            scene_success = int(scene_bucket["success"])
            scene_collision = int(scene_bucket["collision"])
            scene_timeout = int(scene_bucket["timeout"])
            success_rate = _safe_div(float(scene_success), float(scene_total))
            collision_rate = _safe_div(float(scene_collision), float(scene_total))
            timeout_rate = _safe_div(float(scene_timeout), float(scene_total))
            avg_goal_end_dist = _safe_div(float(scene_bucket["goal_end_dist_sum"]), float(scene_total))
            avg_min_lidar = _safe_div(float(scene_bucket["min_lidar_sum"]), float(scene_total))
            scene_key = f"scene_{scene_idx}"
            by_scene_report[scene_key] = {
                "scene_index": scene_idx,
                "episodes": scene_episodes,
                "success": scene_success,
                "collision": scene_collision,
                "timeout": scene_timeout,
                "success_rate": success_rate,
                "collision_rate": collision_rate,
                "timeout_rate": timeout_rate,
                "avg_goal_end_dist": avg_goal_end_dist,
                "avg_min_lidar": avg_min_lidar,
            }
            print(
                (
                    f"[EVAL][SCENE] scene={scene_idx} episodes={scene_episodes} "
                    f"success={success_rate:.1%} collision={collision_rate:.1%} timeout={timeout_rate:.1%} "
                    f"avg_goal_end_dist={avg_goal_end_dist:.3f} avg_min_lidar={avg_min_lidar:.3f}"
                ),
                flush=True,
            )
        print("=" * 88, flush=True)

        if args_cli.state_report_json:
            state_report_path = Path(args_cli.state_report_json).expanduser().resolve()
            state_report_path.parent.mkdir(parents=True, exist_ok=True)
            report = {
                "task": args_cli.task,
                "checkpoint": str(resume_path),
                "episodes": finished,
                "overall": {
                    "success_rate": _safe_div(float(success), float(total)),
                    "collision_rate": _safe_div(float(collision), float(total)),
                    "timeout_rate": _safe_div(float(timeout), float(total)),
                    "success": success,
                    "collision": collision,
                    "timeout": timeout,
                    "return_mean": return_mean,
                    "return_std": return_std,
                    "ep_len_mean": ep_len_mean,
                    "ep_len_std": ep_len_std,
                },
                "by_state": by_state_report,
                "by_scene": by_scene_report,
                "no_progress_rule": {
                    "min_delta_m": float(args_cli.no_progress_min_delta_m),
                    "min_ratio": float(args_cli.no_progress_min_ratio),
                    "stall_ratio": float(args_cli.no_progress_stall_ratio),
                },
            }
            state_report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            print(f"[EVAL][RESULT] state_report_json={state_report_path}", flush=True)
        if trace_enabled:
            trajectory_path = Path(args_cli.trajectory_json).expanduser().resolve()
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trace_payload = {
                "task": args_cli.task,
                "checkpoint": str(resume_path),
                "seed": int(args_cli.seed),
                "trace_env_id": int(trace_env_id),
                "scene": trace_scene,
                "result": trace_result,
                "points": trace_points,
            }
            trajectory_path.write_text(json.dumps(trace_payload, indent=2, sort_keys=True), encoding="utf-8")
            print(f"[EVAL][RESULT] trajectory_json={trajectory_path}", flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        env.close()
    if loop_error is not None:
        print(f"[EVAL] loop_error={type(loop_error).__name__}: {loop_error}", flush=True)
        raise loop_error


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
