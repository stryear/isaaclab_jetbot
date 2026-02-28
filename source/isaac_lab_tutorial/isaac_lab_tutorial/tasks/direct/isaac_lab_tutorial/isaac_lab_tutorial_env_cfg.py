# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaac_lab_tutorial.robots.jetbot import JETBOT_CONFIG
from isaac_lab_tutorial.robots.turtlebot3_burger import TURTLEBOT3_BURGER_CONFIG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

@configclass
class IsaacLabTutorialEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 2
    # observation_space = 9
    observation_space = 3
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # robot(s)
    robot_cfg: ArticulationCfg = JETBOT_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=2.0, replicate_physics=True)
    dof_names = ["left_wheel_joint", "right_wheel_joint"]


@configclass
class SphereFollowEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 20.0
    # - spaces definition
    action_space = 2
    observation_space = 4
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # robot(s)
    robot_cfg: ArticulationCfg = JETBOT_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=4.0, replicate_physics=True)
    dof_names = ["left_wheel_joint", "right_wheel_joint"]
    # sphere parameters
    sphere_radius: float = 0.1
    sphere_reach_threshold: float = 0.3
    sphere_spawn_range_min: float = 0.5
    sphere_spawn_range_max: float = 1.5
    sphere_max_distance: float = 3.0
    # reward scales
    reach_bonus: float = 5.0
    time_penalty: float = -0.01
    alignment_reward_scale: float = 0.5
    approach_reward_scale: float = 1.0


@configclass
class SphereFollowTurtleBot3EnvCfg(SphereFollowEnvCfg):
    # robot(s)
    robot_cfg: ArticulationCfg = TURTLEBOT3_BURGER_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    dof_names = ["wheel_left_joint", "wheel_right_joint"]
