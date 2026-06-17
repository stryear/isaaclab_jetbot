# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--env_spacing",
    type=float,
    default=None,
    help="Override spacing between cloned environments (meters).",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
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
    "--reset_debug_interval_resets",
    type=int,
    default=None,
    help="Reset-sampling debug print interval in completed reset samples if the env supports it.",
)
parser.add_argument(
    "--debug_phase_tag",
    type=int,
    default=None,
    help="Optional phase tag appended to LiDAR debug/metric logs if the env supports it.",
)
parser.add_argument(
    "--metrics_print_interval_episodes",
    type=int,
    default=None,
    help="Terminal metrics print interval in completed episodes if the env supports it.",
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
    help="Override fixed-scene pool index if the env supports it.",
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
    "--goal_spawn_range_max",
    type=float,
    default=None,
    help="Override goal spawn max distance if the env supports it.",
)
parser.add_argument(
    "--goal_spawn_range_min",
    type=float,
    default=None,
    help="Override goal spawn min distance if the env supports it.",
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
    "--corridor_half_width",
    type=float,
    default=None,
    help="Override corridor half width for corridor-enabled lidar-nav envs.",
)
parser.add_argument(
    "--corridor_half_length",
    type=float,
    default=None,
    help="Override corridor half length for corridor-enabled lidar-nav envs.",
)
parser.add_argument(
    "--corridor_goal_lateral_range",
    type=float,
    default=None,
    help="Override goal lateral jitter range in corridor mode if supported.",
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
    "--defect_dwb_corridor_width_jitter",
    type=float,
    default=None,
    help="Override corridor width jitter for dwb_oscillation defect scenes if supported.",
)
parser.add_argument(
    "--defect_dwb_corridor_half_length",
    type=float,
    default=None,
    help="Override half corridor length for dwb_oscillation defect scenes if supported.",
)
parser.add_argument(
    "--defect_dwb_corridor_half_length_jitter",
    type=float,
    default=None,
    help="Override half corridor length jitter for dwb_oscillation defect scenes if supported.",
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
    "--defect_dwb_collision_penalty",
    type=float,
    default=None,
    help="Override collision penalty and defect_dwb curriculum collision penalties if supported.",
)
parser.add_argument(
    "--defect_dwb_danger_speed_penalty_scale",
    type=float,
    default=None,
    help="Override danger-speed penalty scale and defect_dwb curriculum danger scales if supported.",
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
    "--defect_dwb_speed_cap_narrow",
    type=float,
    default=None,
    help="Override narrow-pass speed cap and defect_dwb curriculum cap_narrow values if supported.",
)
parser.add_argument(
    "--defect_dwb_speed_surge_scale",
    type=float,
    default=None,
    help="Override forward speed surge reward scale and defect_dwb curriculum surge scales if supported.",
)
parser.add_argument(
    "--defect_dwb_speed_deficit_scale",
    type=float,
    default=None,
    help="Override forward speed deficit penalty scale and defect_dwb curriculum deficit scales if supported.",
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
    help="Override the restored curriculum/common step counter when resuming from checkpoint.",
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
    help="Override minimum near-goal action scale if the env supports it.",
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
    help="Override minimum obstacle action scale if the env supports it.",
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

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
from datetime import datetime

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
    if args_cli.algorithm.lower() == "ppo":
        try:
            from ppo_metrics_patch import apply_ppo_metrics_patch

            apply_ppo_metrics_patch()
        except Exception as err:
            print(f"[WARN] Failed to apply PPO metrics patch: {err}")
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
try:
    from isaaclab.utils.io import dump_pickle
except ImportError:
    import pickle

    def dump_pickle(filename: str, data: object) -> None:
        """Dump data to a pickle file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(data, f)

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaac_lab_tutorial.tasks  # noqa: F401
from checkpoint_step import infer_resume_step_from_checkpoint, restore_env_curriculum_step

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


def _override_curriculum_tuple(env_cfg: object, attr_name: str, value) -> None:
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


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.env_spacing is not None and hasattr(env_cfg.scene, "env_spacing"):
        env_cfg.scene.env_spacing = max(0.5, float(args_cli.env_spacing))
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.debug_print_enable is not None and hasattr(env_cfg, "debug_print_enable"):
        env_cfg.debug_print_enable = bool(args_cli.debug_print_enable)
    if args_cli.debug_print_interval_steps is not None and hasattr(env_cfg, "debug_print_interval_steps"):
        env_cfg.debug_print_interval_steps = max(1, int(args_cli.debug_print_interval_steps))
    if args_cli.reset_debug_interval_resets is not None and hasattr(env_cfg, "reset_debug_interval_resets"):
        env_cfg.reset_debug_interval_resets = max(1, int(args_cli.reset_debug_interval_resets))
    if args_cli.debug_phase_tag is not None and hasattr(env_cfg, "debug_phase_tag"):
        env_cfg.debug_phase_tag = int(args_cli.debug_phase_tag)
    if args_cli.metrics_print_interval_episodes is not None and hasattr(env_cfg, "metrics_print_interval_episodes"):
        env_cfg.metrics_print_interval_episodes = max(1, int(args_cli.metrics_print_interval_episodes))
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
    if args_cli.corridor_half_width is not None and hasattr(env_cfg, "corridor_half_width"):
        env_cfg.corridor_half_width = max(0.2, float(args_cli.corridor_half_width))
    if args_cli.corridor_half_length is not None and hasattr(env_cfg, "corridor_half_length"):
        env_cfg.corridor_half_length = max(1.0, float(args_cli.corridor_half_length))
    if args_cli.corridor_goal_lateral_range is not None and hasattr(env_cfg, "corridor_goal_lateral_range"):
        env_cfg.corridor_goal_lateral_range = max(0.0, float(args_cli.corridor_goal_lateral_range))
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
    if (
        hasattr(env_cfg, "junction_robot_spawn_offset_min")
        and hasattr(env_cfg, "junction_robot_spawn_offset_max")
    ):
        env_cfg.junction_robot_spawn_offset_max = max(
            float(env_cfg.junction_robot_spawn_offset_min),
            float(env_cfg.junction_robot_spawn_offset_max),
        )
    if args_cli.junction_robot_spawn_lateral_jitter is not None and hasattr(env_cfg, "junction_robot_spawn_lateral_jitter"):
        env_cfg.junction_robot_spawn_lateral_jitter = max(0.0, float(args_cli.junction_robot_spawn_lateral_jitter))
    if args_cli.junction_x_sampling_prob is not None and hasattr(env_cfg, "junction_x_sampling_prob"):
        env_cfg.junction_x_sampling_prob = min(1.0, max(0.0, float(args_cli.junction_x_sampling_prob)))
    if args_cli.defect_goal_distance_min is not None and hasattr(env_cfg, "defect_goal_distance_min"):
        env_cfg.defect_goal_distance_min = max(0.05, float(args_cli.defect_goal_distance_min))
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_distance_mins", env_cfg.defect_goal_distance_min)
    if args_cli.defect_goal_distance_max is not None and hasattr(env_cfg, "defect_goal_distance_max"):
        env_cfg.defect_goal_distance_max = max(0.05, float(args_cli.defect_goal_distance_max))
    if hasattr(env_cfg, "defect_goal_distance_min") and hasattr(env_cfg, "defect_goal_distance_max"):
        env_cfg.defect_goal_distance_max = max(
            float(env_cfg.defect_goal_distance_min),
            float(env_cfg.defect_goal_distance_max),
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
    if args_cli.defect_dwb_corridor_width_jitter is not None and hasattr(
        env_cfg, "defect_dwb_corridor_width_jitter"
    ):
        env_cfg.defect_dwb_corridor_width_jitter = max(0.0, float(args_cli.defect_dwb_corridor_width_jitter))
    if args_cli.defect_dwb_corridor_half_length is not None and hasattr(
        env_cfg, "defect_dwb_corridor_half_length"
    ):
        env_cfg.defect_dwb_corridor_half_length = max(1.2, float(args_cli.defect_dwb_corridor_half_length))
    if args_cli.defect_dwb_corridor_half_length_jitter is not None and hasattr(
        env_cfg, "defect_dwb_corridor_half_length_jitter"
    ):
        env_cfg.defect_dwb_corridor_half_length_jitter = max(
            0.0,
            float(args_cli.defect_dwb_corridor_half_length_jitter),
        )
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
    if args_cli.defect_dwb_collision_penalty is not None:
        collision_penalty = float(args_cli.defect_dwb_collision_penalty)
        if hasattr(env_cfg, "collision_penalty"):
            env_cfg.collision_penalty = collision_penalty
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_collision_penalties", collision_penalty)
    if args_cli.defect_dwb_danger_speed_penalty_scale is not None:
        danger_scale = float(args_cli.defect_dwb_danger_speed_penalty_scale)
        if hasattr(env_cfg, "danger_speed_penalty_scale"):
            env_cfg.danger_speed_penalty_scale = danger_scale
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_danger_speed_penalty_scales", danger_scale)
    if args_cli.defect_dwb_clearance_penalty_scale is not None and hasattr(
        env_cfg, "defect_dwb_clearance_penalty_scale"
    ):
        env_cfg.defect_dwb_clearance_penalty_scale = float(args_cli.defect_dwb_clearance_penalty_scale)
    if args_cli.defect_dwb_clearance_penalty_distance is not None and hasattr(
        env_cfg, "defect_dwb_clearance_penalty_distance"
    ):
        env_cfg.defect_dwb_clearance_penalty_distance = max(
            0.0,
            float(args_cli.defect_dwb_clearance_penalty_distance),
        )
    if args_cli.defect_dwb_front_clearance_penalty_scale is not None and hasattr(
        env_cfg, "defect_dwb_front_clearance_penalty_scale"
    ):
        env_cfg.defect_dwb_front_clearance_penalty_scale = float(
            args_cli.defect_dwb_front_clearance_penalty_scale
        )
    if args_cli.defect_dwb_front_clearance_penalty_distance is not None and hasattr(
        env_cfg, "defect_dwb_front_clearance_penalty_distance"
    ):
        env_cfg.defect_dwb_front_clearance_penalty_distance = max(
            0.0,
            float(args_cli.defect_dwb_front_clearance_penalty_distance),
        )
    if args_cli.defect_dwb_clearance_penalty_power is not None and hasattr(
        env_cfg, "defect_dwb_clearance_penalty_power"
    ):
        env_cfg.defect_dwb_clearance_penalty_power = max(
            1.0,
            float(args_cli.defect_dwb_clearance_penalty_power),
        )
    if args_cli.defect_dwb_gap_progress_reward_scale is not None and hasattr(
        env_cfg, "defect_dwb_gap_progress_reward_scale"
    ):
        env_cfg.defect_dwb_gap_progress_reward_scale = max(
            0.0,
            float(args_cli.defect_dwb_gap_progress_reward_scale),
        )
        print(
            "[TRAIN][OVERRIDE] "
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
            "[TRAIN][OVERRIDE] "
            f"defect_dwb_gap_alignment_reward_scale={env_cfg.defect_dwb_gap_alignment_reward_scale:.3f}"
        )
    if args_cli.defect_dwb_gap_clear_bonus is not None and hasattr(env_cfg, "defect_dwb_gap_clear_bonus"):
        env_cfg.defect_dwb_gap_clear_bonus = max(0.0, float(args_cli.defect_dwb_gap_clear_bonus))
        print(f"[TRAIN][OVERRIDE] defect_dwb_gap_clear_bonus={env_cfg.defect_dwb_gap_clear_bonus:.3f}")
    if args_cli.defect_dwb_side_commit_reward_scale is not None and hasattr(
        env_cfg, "defect_dwb_side_commit_reward_scale"
    ):
        env_cfg.defect_dwb_side_commit_reward_scale = max(
            0.0,
            float(args_cli.defect_dwb_side_commit_reward_scale),
        )
        print(
            "[TRAIN][OVERRIDE] "
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
            "[TRAIN][OVERRIDE] "
            f"defect_dwb_side_switch_penalty_scale={env_cfg.defect_dwb_side_switch_penalty_scale:.3f}"
        )
    if args_cli.defect_dwb_side_commit_deadzone is not None and hasattr(env_cfg, "defect_dwb_side_commit_deadzone"):
        env_cfg.defect_dwb_side_commit_deadzone = max(0.0, float(args_cli.defect_dwb_side_commit_deadzone))
        print(f"[TRAIN][OVERRIDE] defect_dwb_side_commit_deadzone={env_cfg.defect_dwb_side_commit_deadzone:.3f}")
    if args_cli.bypass_reward_enable is not None and hasattr(env_cfg, "bypass_reward_enable"):
        env_cfg.bypass_reward_enable = bool(int(args_cli.bypass_reward_enable))
        print(f"[TRAIN][OVERRIDE] bypass_reward_enable={int(env_cfg.bypass_reward_enable)}")
    if args_cli.bypass_reward_scale is not None and hasattr(env_cfg, "bypass_reward_scale"):
        env_cfg.bypass_reward_scale = max(0.0, float(args_cli.bypass_reward_scale))
        print(f"[TRAIN][OVERRIDE] bypass_reward_scale={env_cfg.bypass_reward_scale:.3f}")
    if args_cli.bypass_open_side_min_lidar is not None and hasattr(env_cfg, "bypass_open_side_min_lidar"):
        env_cfg.bypass_open_side_min_lidar = max(0.0, float(args_cli.bypass_open_side_min_lidar))
        print(f"[TRAIN][OVERRIDE] bypass_open_side_min_lidar={env_cfg.bypass_open_side_min_lidar:.3f}")
    if args_cli.bypass_blocked_side_max_lidar is not None and hasattr(env_cfg, "bypass_blocked_side_max_lidar"):
        env_cfg.bypass_blocked_side_max_lidar = max(0.0, float(args_cli.bypass_blocked_side_max_lidar))
        print(f"[TRAIN][OVERRIDE] bypass_blocked_side_max_lidar={env_cfg.bypass_blocked_side_max_lidar:.3f}")
    if args_cli.bypass_near_obstacle_distance is not None and hasattr(env_cfg, "bypass_near_obstacle_distance"):
        env_cfg.bypass_near_obstacle_distance = max(0.0, float(args_cli.bypass_near_obstacle_distance))
        print(f"[TRAIN][OVERRIDE] bypass_near_obstacle_distance={env_cfg.bypass_near_obstacle_distance:.3f}")
    if args_cli.bypass_forward_speed_threshold is not None and hasattr(env_cfg, "bypass_forward_speed_threshold"):
        env_cfg.bypass_forward_speed_threshold = max(0.0, float(args_cli.bypass_forward_speed_threshold))
        print(f"[TRAIN][OVERRIDE] bypass_forward_speed_threshold={env_cfg.bypass_forward_speed_threshold:.3f}")
    if args_cli.defect_dwb_speed_cap_narrow is not None:
        cap_narrow = max(0.01, float(args_cli.defect_dwb_speed_cap_narrow))
        if hasattr(env_cfg, "speed_cap_narrow"):
            env_cfg.speed_cap_narrow = cap_narrow
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_speed_cap_narrows", cap_narrow)
    if args_cli.defect_dwb_speed_surge_scale is not None:
        surge_scale = max(0.0, float(args_cli.defect_dwb_speed_surge_scale))
        if hasattr(env_cfg, "defect_dwb_speed_surge_scale"):
            env_cfg.defect_dwb_speed_surge_scale = surge_scale
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_speed_surge_scales", surge_scale)
    if args_cli.defect_dwb_speed_deficit_scale is not None:
        deficit_scale = max(0.0, float(args_cli.defect_dwb_speed_deficit_scale))
        if hasattr(env_cfg, "defect_dwb_speed_deficit_scale"):
            env_cfg.defect_dwb_speed_deficit_scale = deficit_scale
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_speed_deficit_scales", deficit_scale)
    if args_cli.single_gap_force_goal_mode is not None:
        forced_goal_mode = str(args_cli.single_gap_force_goal_mode).strip().lower()
        if hasattr(env_cfg, "defect_single_gap_goal_mode"):
            env_cfg.defect_single_gap_goal_mode = forced_goal_mode
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_modes", forced_goal_mode)
        print(f"[TRAIN][OVERRIDE] single_gap_goal_mode={forced_goal_mode}")
    if args_cli.rear_clearance is not None:
        rear_clearance = max(0.0, float(args_cli.rear_clearance))
        if hasattr(env_cfg, "defect_single_gap_goal_rear_clearance_min"):
            env_cfg.defect_single_gap_goal_rear_clearance_min = rear_clearance
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_rear_clearance_mins", rear_clearance)
        print(f"[TRAIN][OVERRIDE] rear_clearance={rear_clearance:.3f}")
    if args_cli.rear_lateral_center is not None:
        rear_lateral_center = float(args_cli.rear_lateral_center)
        if hasattr(env_cfg, "defect_single_gap_goal_rear_lateral_center"):
            env_cfg.defect_single_gap_goal_rear_lateral_center = rear_lateral_center
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_rear_lateral_centers", rear_lateral_center)
        print(f"[TRAIN][OVERRIDE] rear_lateral_center={rear_lateral_center:.3f}")
    if args_cli.rear_offset_max is not None:
        rear_offset_max = max(0.0, float(args_cli.rear_offset_max))
        if hasattr(env_cfg, "defect_single_gap_goal_rear_lateral_offset_max"):
            env_cfg.defect_single_gap_goal_rear_lateral_offset_max = rear_offset_max
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_rear_lateral_offset_maxs", rear_offset_max)
        print(f"[TRAIN][OVERRIDE] rear_offset_max={rear_offset_max:.3f}")
    if args_cli.bridge_clearance is not None:
        bridge_clearance = max(0.0, float(args_cli.bridge_clearance))
        if hasattr(env_cfg, "defect_single_gap_goal_bridge_clearance_min"):
            env_cfg.defect_single_gap_goal_bridge_clearance_min = bridge_clearance
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_bridge_clearance_mins", bridge_clearance)
        print(f"[TRAIN][OVERRIDE] bridge_clearance={bridge_clearance:.3f}")
    if args_cli.bridge_lateral_center is not None:
        bridge_lateral_center = float(args_cli.bridge_lateral_center)
        if hasattr(env_cfg, "defect_single_gap_goal_bridge_lateral_center"):
            env_cfg.defect_single_gap_goal_bridge_lateral_center = bridge_lateral_center
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_bridge_lateral_centers", bridge_lateral_center)
        print(f"[TRAIN][OVERRIDE] bridge_lateral_center={bridge_lateral_center:.3f}")
    if args_cli.bridge_offset_max is not None:
        bridge_offset_max = max(0.0, float(args_cli.bridge_offset_max))
        if hasattr(env_cfg, "defect_single_gap_goal_bridge_lateral_offset_max"):
            env_cfg.defect_single_gap_goal_bridge_lateral_offset_max = bridge_offset_max
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_bridge_lateral_offset_maxs", bridge_offset_max)
        print(f"[TRAIN][OVERRIDE] bridge_offset_max={bridge_offset_max:.3f}")
    if args_cli.freeze_curriculum_stage is not None and hasattr(env_cfg, "defect_dwb_curriculum_freeze_stage"):
        env_cfg.defect_dwb_curriculum_freeze_stage = int(args_cli.freeze_curriculum_stage)
        print(f"[TRAIN][OVERRIDE] freeze_curriculum_stage={int(args_cli.freeze_curriculum_stage)}")
    if args_cli.action_scale is not None and hasattr(env_cfg, "action_scale"):
        env_cfg.action_scale = max(0.1, float(args_cli.action_scale))
    if args_cli.near_goal_min_action_scale is not None and hasattr(env_cfg, "near_goal_min_action_scale"):
        env_cfg.near_goal_min_action_scale = min(1.0, max(0.0, float(args_cli.near_goal_min_action_scale)))
    if args_cli.obstacle_slowdown_distance is not None and hasattr(env_cfg, "obstacle_slowdown_distance"):
        env_cfg.obstacle_slowdown_distance = max(0.0, float(args_cli.obstacle_slowdown_distance))
    if args_cli.obstacle_min_action_scale is not None and hasattr(env_cfg, "obstacle_min_action_scale"):
        env_cfg.obstacle_min_action_scale = min(1.0, max(0.0, float(args_cli.obstacle_min_action_scale)))
    if args_cli.goal_reach_threshold is not None and hasattr(env_cfg, "goal_reach_threshold"):
        goal_reach_threshold = max(0.05, float(args_cli.goal_reach_threshold))
        env_cfg.goal_reach_threshold = goal_reach_threshold
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_reach_thresholds", goal_reach_threshold)
        print(f"[TRAIN][OVERRIDE] goal_reach_threshold={goal_reach_threshold:.3f}")
    if args_cli.goal_local_x is not None:
        goal_local_x = float(args_cli.goal_local_x)
        if hasattr(env_cfg, "defect_single_gap_goal_local_x"):
            env_cfg.defect_single_gap_goal_local_x = goal_local_x
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_local_xs", goal_local_x)
        print(f"[TRAIN][OVERRIDE] goal_local_x={goal_local_x:.3f}")
    if args_cli.goal_local_y is not None:
        goal_local_y = float(args_cli.goal_local_y)
        if hasattr(env_cfg, "defect_single_gap_goal_local_y"):
            env_cfg.defect_single_gap_goal_local_y = goal_local_y
        _override_curriculum_tuple(env_cfg, "defect_dwb_curriculum_goal_local_ys", goal_local_y)
        print(f"[TRAIN][OVERRIDE] goal_local_y={goal_local_y:.3f}")
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
    if hasattr(env_cfg, "debug_iteration_rollouts"):
        env_cfg.debug_iteration_rollouts = max(1, int(agent_cfg.get("agent", {}).get("rollouts", 64)))

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    print(f"Exact experiment name requested from command line {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo", "sac"]:
        env = multi_agent_to_single_agent(env)

    # Recover environment step counter from checkpoint filename (e.g. agent_176000.pt) for curriculum continuity.
    if resume_path:
        resume_step, resume_step_source = infer_resume_step_from_checkpoint(resume_path)
        if resume_step is not None or args_cli.curriculum_start_step is not None:
            restore_env_curriculum_step(
                env,
                resume_step,
                curriculum_start_step=args_cli.curriculum_start_step,
                source=resume_step_source,
            )
        else:
            print(f"[INFO] Could not infer curriculum step from checkpoint: {resume_path}")
    elif args_cli.curriculum_start_step is not None:
        restore_env_curriculum_step(
            env,
            None,
            curriculum_start_step=args_cli.curriculum_start_step,
            source="CLI override",
        )

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # run training
    runner.run()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
