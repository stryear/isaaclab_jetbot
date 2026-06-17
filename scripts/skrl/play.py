# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--video_full_episode",
    action="store_true",
    default=False,
    help="Record one full episode and stop when terminated/truncated.",
)
parser.add_argument(
    "--stop_on_success",
    action="store_true",
    default=False,
    help="Keep rolling over episodes and stop as soon as a successful episode is observed.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Optional hard step budget for play loop (0 disables the limit).",
)
parser.add_argument(
    "--video_folder",
    type=str,
    default=None,
    help="Output folder for recorded videos. Defaults to <log_dir>/videos/play.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO", "SAC"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
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
    help="Override fixed-local single-gap goal x and defect_dwb curriculum goal-local-x values if supported.",
)
parser.add_argument(
    "--goal_local_y",
    type=float,
    default=None,
    help="Override fixed-local single-gap goal y and defect_dwb curriculum goal-local-y values if supported.",
)
parser.add_argument(
    "--goal_max_distance",
    type=float,
    default=None,
    help="Override goal distance normalization denominator if the env supports it.",
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

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
from pathlib import Path
import numpy as np
import torch

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
    from lstm_runner_patch import apply_lstm_runner_patch

    apply_lstm_runner_patch()
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict

try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    # Isaac Lab versions without pretrained checkpoint helper.
    get_published_pretrained_checkpoint = None

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import isaac_lab_tutorial.tasks  # noqa: F401
from checkpoint_step import infer_resume_step_from_checkpoint, restore_env_curriculum_step

# config shortcuts
algorithm = args_cli.algorithm.lower()


def _any_true(value) -> bool:
    """Best-effort truth reduction for bool/tensor/array/container done flags."""
    if isinstance(value, dict):
        return any(_any_true(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_any_true(v) for v in value)
    if torch.is_tensor(value):
        return bool(torch.any(value).item())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    try:
        return bool(value)
    except Exception:
        return False


def _to_numpy_bool(value, *, default_size: int | None = None) -> np.ndarray:
    """Convert bool-like containers to a flat numpy bool array."""
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (list, tuple, set)):
        arr = np.asarray(list(value))
    else:
        arr = np.asarray([bool(value)])
    arr = np.asarray(arr, dtype=np.bool_).reshape(-1)
    if default_size is not None and arr.size == 1 and default_size > 1:
        arr = np.repeat(arr, default_size)
    return arr


def _override_curriculum_tuple(env_cfg: object, attr_name: str, value) -> None:
    if not hasattr(env_cfg, attr_name):
        return
    raw = tuple(getattr(env_cfg, attr_name))
    if not raw:
        return
    setattr(env_cfg, attr_name, tuple(value for _ in raw))


def main():
    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    if args_cli.debug_print_enable is not None and hasattr(env_cfg, "debug_print_enable"):
        env_cfg.debug_print_enable = bool(args_cli.debug_print_enable)
    if args_cli.debug_print_interval_steps is not None and hasattr(env_cfg, "debug_print_interval_steps"):
        env_cfg.debug_print_interval_steps = max(1, int(args_cli.debug_print_interval_steps))
    if args_cli.debug_phase_tag is not None and hasattr(env_cfg, "debug_phase_tag"):
        env_cfg.debug_phase_tag = int(args_cli.debug_phase_tag)
    if args_cli.num_obstacles is not None and hasattr(env_cfg, "num_obstacles"):
        env_cfg.num_obstacles = max(0, int(args_cli.num_obstacles))
    if args_cli.scene_id is not None and hasattr(env_cfg, "fixed_scene_multi_index_override"):
        env_cfg.fixed_scene_multi_index_override = int(args_cli.scene_id)
        if hasattr(env_cfg, "fixed_scene_multi_enable"):
            env_cfg.fixed_scene_multi_enable = True
    if args_cli.episode_length_s is not None and hasattr(env_cfg, "episode_length_s"):
        env_cfg.episode_length_s = max(1.0, float(args_cli.episode_length_s))
    if args_cli.goal_max_distance is not None and hasattr(env_cfg, "goal_max_distance"):
        env_cfg.goal_max_distance = max(0.2, float(args_cli.goal_max_distance))
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
    if args_cli.defect_goal_distance_min is not None and hasattr(env_cfg, "defect_goal_distance_min"):
        env_cfg.defect_goal_distance_min = max(0.05, float(args_cli.defect_goal_distance_min))
    if args_cli.defect_goal_distance_max is not None and hasattr(env_cfg, "defect_goal_distance_max"):
        env_cfg.defect_goal_distance_max = max(0.05, float(args_cli.defect_goal_distance_max))
    if hasattr(env_cfg, "defect_goal_distance_min") and hasattr(env_cfg, "defect_goal_distance_max"):
        env_cfg.defect_goal_distance_max = max(
            float(env_cfg.defect_goal_distance_min),
            float(env_cfg.defect_goal_distance_max),
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
        if hasattr(env_cfg, "defect_single_gap_goal_mode"):
            env_cfg.defect_single_gap_goal_mode = forced_goal_mode
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_modes", forced_goal_mode)
        print(f"[PLAY][OVERRIDE] single_gap_goal_mode={forced_goal_mode}")
    if args_cli.rear_clearance is not None:
        rear_clearance = max(0.0, float(args_cli.rear_clearance))
        if hasattr(env_cfg, "defect_single_gap_goal_rear_clearance_min"):
            env_cfg.defect_single_gap_goal_rear_clearance_min = rear_clearance
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_rear_clearance_mins", rear_clearance)
        print(f"[PLAY][OVERRIDE] rear_clearance={rear_clearance:.3f}")
    if args_cli.rear_lateral_center is not None:
        rear_lateral_center = float(args_cli.rear_lateral_center)
        if hasattr(env_cfg, "defect_single_gap_goal_rear_lateral_center"):
            env_cfg.defect_single_gap_goal_rear_lateral_center = rear_lateral_center
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_rear_lateral_centers", rear_lateral_center)
        print(f"[PLAY][OVERRIDE] rear_lateral_center={rear_lateral_center:.3f}")
    if args_cli.rear_offset_max is not None:
        rear_offset_max = max(0.0, float(args_cli.rear_offset_max))
        if hasattr(env_cfg, "defect_single_gap_goal_rear_lateral_offset_max"):
            env_cfg.defect_single_gap_goal_rear_lateral_offset_max = rear_offset_max
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_rear_lateral_offset_maxs", rear_offset_max)
        print(f"[PLAY][OVERRIDE] rear_offset_max={rear_offset_max:.3f}")
    if args_cli.bridge_clearance is not None:
        bridge_clearance = max(0.0, float(args_cli.bridge_clearance))
        if hasattr(env_cfg, "defect_single_gap_goal_bridge_clearance_min"):
            env_cfg.defect_single_gap_goal_bridge_clearance_min = bridge_clearance
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_bridge_clearance_mins", bridge_clearance)
        print(f"[PLAY][OVERRIDE] bridge_clearance={bridge_clearance:.3f}")
    if args_cli.bridge_lateral_center is not None:
        bridge_lateral_center = float(args_cli.bridge_lateral_center)
        if hasattr(env_cfg, "defect_single_gap_goal_bridge_lateral_center"):
            env_cfg.defect_single_gap_goal_bridge_lateral_center = bridge_lateral_center
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_bridge_lateral_centers", bridge_lateral_center)
        print(f"[PLAY][OVERRIDE] bridge_lateral_center={bridge_lateral_center:.3f}")
    if args_cli.bridge_offset_max is not None:
        bridge_offset_max = max(0.0, float(args_cli.bridge_offset_max))
        if hasattr(env_cfg, "defect_single_gap_goal_bridge_lateral_offset_max"):
            env_cfg.defect_single_gap_goal_bridge_lateral_offset_max = bridge_offset_max
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_bridge_lateral_offset_maxs", bridge_offset_max)
        print(f"[PLAY][OVERRIDE] bridge_offset_max={bridge_offset_max:.3f}")
    if args_cli.freeze_curriculum_stage is not None and hasattr(env_cfg, "defect_dwb_curriculum_freeze_stage"):
        env_cfg.defect_dwb_curriculum_freeze_stage = int(args_cli.freeze_curriculum_stage)
        print(f"[PLAY][OVERRIDE] freeze_curriculum_stage={int(args_cli.freeze_curriculum_stage)}")
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
        print(f"[PLAY][OVERRIDE] goal_reach_threshold={goal_reach_threshold:.3f}")
    if args_cli.goal_local_x is not None:
        goal_local_x = float(args_cli.goal_local_x)
        if hasattr(env_cfg, "defect_single_gap_goal_local_x"):
            env_cfg.defect_single_gap_goal_local_x = goal_local_x
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_local_xs", goal_local_x)
        print(f"[PLAY][OVERRIDE] goal_local_x={goal_local_x:.3f}")
    if args_cli.goal_local_y is not None:
        goal_local_y = float(args_cli.goal_local_y)
        if hasattr(env_cfg, "defect_single_gap_goal_local_y"):
            env_cfg.defect_single_gap_goal_local_y = goal_local_y
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_local_ys", goal_local_y)
        print(f"[PLAY][OVERRIDE] goal_local_y={goal_local_y:.3f}")
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
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        if get_published_pretrained_checkpoint is None:
            print(
                "[ERROR] --use_pretrained_checkpoint is not supported by this Isaac Lab version. "
                "Please provide --checkpoint <path>."
            )
            return
        resume_path = get_published_pretrained_checkpoint("skrl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    resume_path = Path(resume_path).expanduser().resolve()
    log_dir = str(resume_path.parent.parent)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo", "sac"]:
        env = multi_agent_to_single_agent(env)

    resume_step, resume_step_source = infer_resume_step_from_checkpoint(resume_path)
    if resume_step is not None or args_cli.curriculum_start_step is not None:
        restore_env_curriculum_step(
            env,
            resume_step,
            curriculum_start_step=args_cli.curriculum_start_step,
            source=resume_step_source,
        )
    else:
        print(f"[INFO] Could not infer curriculum step from checkpoint: {resume_path.name}")

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_length = 0 if (args_cli.video_full_episode or args_cli.stop_on_success) else args_cli.video_length
        video_folder = args_cli.video_folder or os.path.join(log_dir, "videos", "play")
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        if args_cli.video_full_episode:
            print("[INFO] Video mode: full episode")
        if args_cli.stop_on_success:
            print("[INFO] Video mode: stop on first successful episode")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(str(resume_path))
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    episode_counter = 0
    num_envs_runtime = max(1, int(getattr(env.unwrapped, "num_envs", 1)))
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, terminated, truncated, _ = env.step(actions)

        term_mask = _to_numpy_bool(terminated, default_size=num_envs_runtime)
        trunc_mask = _to_numpy_bool(truncated, default_size=num_envs_runtime)
        done_mask = np.logical_or(term_mask, trunc_mask)
        done_indices = np.where(done_mask)[0]
        success_this_step = False
        if done_indices.size > 0:
            done_success = _to_numpy_bool(
                getattr(env.unwrapped, "_last_done_success", np.zeros(num_envs_runtime, dtype=np.bool_)),
                default_size=num_envs_runtime,
            )
            done_collision = _to_numpy_bool(
                getattr(env.unwrapped, "_last_done_collision", np.zeros(num_envs_runtime, dtype=np.bool_)),
                default_size=num_envs_runtime,
            )
            done_timeout = _to_numpy_bool(
                getattr(env.unwrapped, "_last_done_timeout", np.zeros(num_envs_runtime, dtype=np.bool_)),
                default_size=num_envs_runtime,
            )
            for env_id in done_indices.tolist():
                episode_counter += 1
                if bool(done_success[env_id]):
                    result = "success"
                    success_this_step = True
                elif bool(done_collision[env_id]):
                    result = "collision"
                elif bool(done_timeout[env_id]):
                    result = "timeout"
                elif bool(trunc_mask[env_id]):
                    result = "timeout"
                else:
                    result = "collision"
                print(
                    f"[PLAY][EPISODE] idx={episode_counter} env={env_id} step={timestep + 1} result={result}",
                    flush=True,
                )

        timestep += 1
        if args_cli.video:
            # exit the play loop after recording one video
            if args_cli.video_full_episode and not args_cli.stop_on_success:
                if _any_true(terminated) or _any_true(truncated):
                    break
            elif not args_cli.stop_on_success and timestep == args_cli.video_length:
                break
        if args_cli.stop_on_success and success_this_step:
            print(f"[PLAY] Stopping after first successful episode at step={timestep}", flush=True)
            break
        if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:
            print(f"[PLAY] Reached max_steps={args_cli.max_steps}; stopping.", flush=True)
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
