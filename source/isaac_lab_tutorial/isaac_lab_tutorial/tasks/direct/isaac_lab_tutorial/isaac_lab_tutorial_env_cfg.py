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
from isaaclab.sensors import ContactSensorCfg
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


@configclass
class LidarNavTurtleBot3EnvCfg(DirectRLEnvCfg):
    """TurtleBot3 LiDAR obstacle-avoidance navigation environment config."""

    # env
    # Keep physics at 120 Hz and run policy/control at 10 Hz (120 / 12).
    decimation = 12
    episode_length_s = 30.0
    action_space = 2
    lidar_num_beams = 35
    # Default observation contract (58D with privileged observation flags enabled):
    # [base_lin_vel_3, base_ang_vel_3, projected_gravity_3, goal_xy_2, last_action_2,
    #  lidar_beams, turn_cmd_3, speed_cap, planner_state_4, goal_changed, goal_age_norm]
    observation_space = 23 + lidar_num_beams
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # robot(s)
    robot_cfg: ArticulationCfg = TURTLEBOT3_BURGER_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=5.0, replicate_physics=True)
    dof_names = ["wheel_left_joint", "wheel_right_joint"]

    # action mapping (policy outputs in [-1, 1] -> v/omega)
    v_max: float = 1.0
    omega_limit: float = 6.0
    forward_only: bool = True

    # topological guidance contract aligned with ROS runtime v2
    speed_cap_nominal: float = 0.6
    speed_cap_explore: float = 0.35
    speed_cap_turn: float = 0.25
    speed_cap_fallback: float = 0.18
    speed_cap_return: float = 0.30
    return_speed_cap_high_clearance: float = 0.30
    return_speed_cap_mid_clearance: float = 0.30
    return_speed_cap_low_clearance: float = 0.22
    return_speed_cap_high_clearance_threshold: float = 0.80
    return_speed_cap_mid_clearance_threshold: float = 0.40
    speed_cap_narrow: float = 0.20
    speed_cap_narrow_clearance: float = 0.45
    goal_changed_pulse_sec: float = 0.75
    goal_age_norm_window_sec: float = 10.0
    use_privileged_obs: bool = True
    use_turn_cmd_obs: bool = True
    use_planner_state_obs: bool = True
    # Normalize lidar beams by lidar_max_range before feeding policy observation.
    use_lidar_normalization: bool = False
    near_goal_slowdown_distance: float = 0.8
    near_goal_min_action_scale: float = 0.30
    obstacle_slowdown_distance: float = 1.0
    obstacle_min_action_scale: float = 0.15
    heading_speed_cos_min: float = 0.30
    heading_speed_power: float = 1.5
    heading_speed_min_scale: float = 0.20
    planner_state_mixed_training_enable: bool = False
    planner_state_prob_explore: float = 1.0
    planner_state_prob_fallback: float = 0.0
    planner_state_prob_junction: float = 0.0
    planner_state_prob_return: float = 0.0
    relay_goal_enable: bool = False
    explore_relay_goal_distance_min: float = 1.5
    explore_relay_goal_distance_max: float = 2.5
    fallback_relay_goal_distance_min: float = 1.5
    fallback_relay_goal_distance_max: float = 2.5
    relay_goal_distance_min: float = 1.5
    relay_goal_distance_max: float = 2.5
    junction_relay_goal_distance_min: float = 1.5
    junction_relay_goal_distance_max: float = 2.5
    junction_turn_relay_goal_distance_min: float = 1.5
    junction_turn_relay_goal_distance_max: float = 2.5
    relay_goal_lateral_range: float = 0.18
    return_relay_goal_distance_min: float = 1.5
    return_relay_goal_distance_max: float = 2.5
    return_relay_goal_lateral_range: float = 0.18
    return_relay_goal_lateral_min_abs: float = 0.0
    return_obstacle_centerline_keepout: float = 0.0
    return_obstacle_force_opposite_goal_side: bool = False
    linear_state_spawn_distance_min: float = 1.8
    linear_state_spawn_distance_max: float = 3.2
    linear_state_spawn_lateral_jitter: float = 0.08
    linear_corridor_half_width_choices: tuple[float, ...] = (0.50, 0.80, 1.50)
    linear_open_area_prob: float = 0.0
    linear_open_area_half_extent: float = 2.50
    junction_half_width_choices: tuple[float, ...] = (0.85,)
    junction_state_spawn_distance_min: float = 0.8
    junction_state_spawn_distance_max: float = 1.4
    junction_relay_entry_offset: float = 0.45
    fallback_blocker_distance_min: float = 0.55
    fallback_blocker_distance_max: float = 0.85
    fallback_blocker_lateral_range: float = 0.16

    # command smoothing and safety constraints
    tau_v: float = 0.30
    tau_omega: float = 0.15
    wheel_linear_speed_max: float = 1.0
    diamond_clamp_eps: float = 1.0e-9
    wheel_target_delta_limit: float = 0.0

    # differential-drive parameters (for ROS2 bridge)
    wheel_radius: float = 0.033
    wheel_base: float = 0.160

    # goal and obstacle setup
    goal_radius: float = 0.10
    goal_reach_threshold: float = 0.40
    goal_spawn_range_min: float = 2.0
    goal_spawn_range_max: float = 8.0
    goal_max_distance: float = 15.0

    num_obstacles: int = 4
    obstacle_radius: float = 0.18
    obstacle_height: float = 0.45
    obstacle_spawn_range_min: float = 0.7
    obstacle_spawn_range_max: float = 2.2
    obstacle_min_separation: float = 0.60
    # physical obstacle collision properties
    physical_obstacles_enable: bool = True
    obstacle_kinematic: bool = True
    obstacle_static_friction: float = 1.0
    obstacle_dynamic_friction: float = 0.8
    obstacle_restitution: float = 0.0
    obstacle_contact_offset: float = 0.01
    obstacle_rest_offset: float = 0.0
    contact_collision_force_threshold: float = 2.0
    contact_force_penalty_scale: float = -0.05
    robot_radius: float = 0.12

    # virtual 2D LiDAR model
    lidar_fov_deg: float = 270.0
    lidar_min_range: float = 0.05
    lidar_max_range: float = 3.0
    lidar_noise_std: float = 0.03
    lidar_frame_stack: int = 1
    collision_threshold: float = 0.25
    # Optional collision-lidar gating for rear-goal single-gap scenes.
    # When enabled, lidar-based collision uses only a front-centered sector.
    defect_single_gap_rear_collision_front_sector_only: bool = False
    defect_single_gap_rear_collision_front_sector_deg: float = 180.0
    # Contact sensor bound to a single rigid body (base_footprint).
    # NOTE: Isaac Lab filtered contact reporting requires a single sensor primitive per environment.
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base_footprint",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        filter_prim_paths_expr=["/World/envs/env_.*/Obstacle_0"],
    )
    # optional corridor geometry (virtual walls used by LiDAR/rules only)
    corridor_enable: bool = False
    corridor_half_width: float = 0.85
    corridor_half_length: float = 6.0
    corridor_goal_lateral_range: float = 0.15
    corridor_obstacle_lateral_margin: float = 0.15
    corridor_obstacle_forward_min_offset: float = 0.60
    corridor_obstacle_goal_margin: float = 0.70
    corridor_visualize_walls: bool = False
    corridor_wall_height: float = 0.45
    corridor_wall_thickness: float = 0.04
    # optional junction topology (L/T/X) with command-conditioned navigation
    junction_enable: bool = False
    junction_half_width: float = 0.85
    junction_half_length: float = 6.0
    junction_robot_spawn_offset_min: float = 0.8
    junction_robot_spawn_offset_max: float = 1.6
    junction_robot_spawn_lateral_jitter: float = 0.08
    junction_goal_forward_min: float = 1.2
    junction_goal_forward_max: float = 4.8
    junction_goal_lateral_range: float = 0.12
    junction_obstacle_lateral_margin: float = 0.18
    junction_obstacle_forward_min_offset: float = 0.75
    junction_x_sampling_prob: float = 0.33
    junction_visualize_walls: bool = False
    junction_wall_height: float = 0.45
    junction_wall_thickness: float = 0.04
    # optional deterministic demo scene override (fixed robot / goal / obstacle placement)
    fixed_scene_enable: bool = False
    fixed_scene_use_linear_layout: bool = True
    fixed_scene_robot_spawn_xy: tuple[float, float] = (-2.0, 0.0)
    fixed_scene_robot_spawn_yaw: float = 0.0
    fixed_scene_goal_xy: tuple[float, float] = (2.0, 0.0)
    fixed_scene_obstacle_xy: tuple[tuple[float, float], ...] = ((0.0, 0.0),)
    # optional fixed-scene pool for deterministic multi-scene training/evaluation
    fixed_scene_multi_enable: bool = False
    # -1: random sample at each reset; >=0: force specific scene index
    fixed_scene_multi_index_override: int = -1
    # Optional subset of multi-scene indices sampled during reset. Empty tuple means all scenes.
    fixed_scene_multi_index_pool: tuple[int, ...] = ()
    # Optional per-index sampling weights aligned with `fixed_scene_multi_index_pool`.
    # If provided, values should be non-negative and sum can be arbitrary (> 0 after sanitization).
    fixed_scene_multi_index_weights: tuple[float, ...] = ()
    # (robot_x, robot_y, robot_yaw, goal_x, goal_y, corridor_half_width)
    fixed_scene_multi_scene_params: tuple[tuple[float, float, float, float, float, float], ...] = ()
    # Per-scene obstacle XY tuples. Outer index must align with fixed_scene_multi_scene_params.
    fixed_scene_multi_obstacle_xy: tuple[tuple[tuple[float, float], ...], ...] = ()
    # place unused obstacles far away from the playable area
    fixed_scene_inactive_obstacle_offset_x: float = 50.0
    fixed_scene_inactive_obstacle_offset_y: float = 50.0
    # defect-scene override for geometry-locked scenario training
    defect_scene_enable: bool = False
    defect_scene_mode: str = "dwb_oscillation"
    defect_scene_visualize_walls: bool = True
    defect_wall_segment_capacity: int = 96
    defect_spawn_jitter_xy: float = 0.2
    defect_goal_jitter_xy: float = 0.0
    defect_goal_distance_min: float = 1.0
    defect_goal_distance_max: float = 3.0
    # dwb_oscillation geometry
    defect_dwb_corridor_width: float = 2.0
    defect_dwb_corridor_width_jitter: float = 0.0
    defect_dwb_corridor_half_length: float = 3.0
    defect_dwb_corridor_half_length_jitter: float = 0.0
    defect_dwb_obstacle_spacing: float = 0.6
    defect_dwb_obstacle_spacing_jitter: float = 0.0
    defect_dwb_obstacle_pair_shift_y: float = 0.0
    defect_dwb_obstacle_x_jitter: float = 0.0
    # single_obstacle_symmetric_gap geometry
    defect_single_gap_obstacle_y_jitter: float = 0.0
    defect_single_gap_target_lateral_margin: float = 0.06
    defect_single_gap_target_lateral_tolerance: float = 0.12
    # mppi_corner_fail geometry
    defect_corner_corridor_width: float = 0.8
    defect_corner_inner_radius: float = 0.5
    defect_corner_inner_radius_jitter: float = 0.05
    defect_corner_straight_pre_length: float = 2.2
    defect_corner_straight_post_length: float = 2.2
    # door_deadlock geometry
    defect_door_corridor_width: float = 2.0
    defect_door_half_length: float = 3.0
    defect_door_width: float = 0.8
    defect_door_width_jitter: float = 0.05
    defect_door_straight_distance: float = 1.5
    defect_door_line_angle_max_deg: float = 15.0
    # u_shape_trap geometry
    defect_u_opening_width: float = 0.6
    defect_u_opening_width_jitter: float = 0.0
    defect_u_depth: float = 2.7
    defect_u_inner_width: float = 2.0
    defect_u_goal_outside_x: float = 1.8
    defect_u_arena_half_width: float = 2.6
    # defect-scene reward extensions
    defect_lateral_velocity_penalty_scale: float = -0.2
    defect_narrow_passage_bonus: float = 2.0
    defect_narrow_passage_lateral_tolerance: float = 0.1
    defect_narrow_passage_x_window: float = 0.25
    defect_dwb_forward_speed_reward_scale: float = 0.0
    defect_dwb_forward_speed_min_lidar: float = 0.0
    defect_dwb_forward_speed_heading_min: float = 0.0
    defect_dwb_speed_surge_scale: float = 0.0
    defect_dwb_speed_surge_threshold: float = 0.12
    defect_dwb_speed_deficit_scale: float = 0.0
    defect_dwb_speed_deficit_target: float = 0.10
    defect_dwb_clearance_penalty_scale: float = 0.0
    defect_dwb_clearance_penalty_distance: float = 0.30
    defect_dwb_front_clearance_penalty_scale: float = 0.0
    defect_dwb_front_clearance_penalty_distance: float = 0.45
    defect_dwb_clearance_penalty_power: float = 2.0
    defect_dwb_gap_alignment_reward_scale: float = 0.0
    defect_dwb_post_gap_forward_reward_scale: float = 0.0
    defect_dwb_post_gap_progress_reward_scale: float = 0.0
    defect_dwb_side_commit_reward_scale: float = 0.0
    defect_dwb_side_switch_penalty_scale: float = 0.0
    defect_dwb_side_commit_deadzone: float = 0.0
    defect_dwb_commit_forward_speed_min: float = 0.0
    defect_dwb_commit_min_lidar: float = 0.0
    defect_dwb_commit_heading_cos_min: float = 0.0
    defect_dwb_commit_x_margin_before_gap: float = 0.0
    defect_dwb_commit_x_margin_after_gap: float = 0.0
    defect_dwb_commit_lateral_tolerance: float = 0.0
    defect_dwb_observation_enhance_enable: bool = False
    defect_dwb_no_progress_speed_threshold: float = 0.0
    defect_dwb_no_progress_delta_threshold: float = 0.0
    defect_dwb_no_progress_moving_speed_threshold: float = 0.0
    defect_dwb_no_progress_moving_delta_threshold: float = 0.0
    defect_dwb_no_progress_clearance_threshold: float = 0.0
    defect_dwb_no_progress_penalty: float = 0.0
    defect_dwb_no_progress_terminate_steps: int = 0
    defect_turn_completion_bonus: float = 5.0
    defect_u_escape_bonus: float = 20.0
    defect_u_exploration_reward_scale: float = 0.01
    defect_u_backward_encouragement: float = 0.0
    defect_u_backward_speed_threshold: float = -0.1
    defect_u_exploration_grid_size: float = 0.2

    # optional scene-local subgoal shaping (useful for junction-like staged navigation)
    subgoal_reward_enable: bool = False
    subgoal_scene_index: int = 4
    subgoal_xy: tuple[float, float] = (0.0, 0.5)
    subgoal_radius: float = 0.30
    subgoal_progress_reward_scale: float = 8.0
    subgoal_bonus: float = 5.0
    # optional bypass shaping: reward moving through asymmetric free-space near obstacles
    bypass_reward_enable: bool = False
    # -1 applies to all scenes; >=0 applies only to fixed-scene index
    bypass_scene_index: int = -1
    bypass_open_side_min_lidar: float = 0.30
    bypass_blocked_side_max_lidar: float = 0.30
    bypass_near_obstacle_distance: float = 0.90
    bypass_forward_speed_threshold: float = 0.05
    bypass_reward_scale: float = 0.10

    # reward weights
    progress_reward_scale: float = 8.0
    heading_gate_power: float = 2.0
    heading_gate_min: float = 0.0
    heading_reward_scale: float = 0.0
    forward_goal_reward_scale: float = 0.45
    clearance_reward_scale: float = 0.20
    smooth_action_reward_scale: float = -0.03
    turn_penalty_scale: float = -0.08
    time_penalty: float = -0.02
    stall_speed_threshold: float = 0.02
    stall_distance_threshold: float = 0.8
    stall_penalty: float = -0.15
    danger_speed_distance: float = 0.80
    danger_speed_penalty_scale: float = -0.40
    action_saturation_threshold: float = 0.85
    action_saturation_penalty_scale: float = -0.10
    action_saturation_penalty_power: float = 1.5
    turn_cmd_reward_scale: float = 0.0
    turn_cmd_yaw_reward_scale: float = 0.0
    turn_cmd_yaw_max_abs: float = 1.2
    # recovery-state shaping (fallback/return only; neutral by default)
    recovery_state_reward_enable: bool = False
    recovery_forward_goal_scale_mult_fallback: float = 1.0
    recovery_forward_goal_scale_mult_return: float = 1.0
    recovery_clearance_scale_mult_fallback: float = 1.0
    recovery_clearance_scale_mult_return: float = 1.0
    recovery_danger_speed_penalty_mult_fallback: float = 1.0
    recovery_danger_speed_penalty_mult_return: float = 1.0
    recovery_turn_penalty_mult_fallback: float = 1.0
    recovery_turn_penalty_mult_return: float = 1.0
    recovery_action_saturation_penalty_mult_fallback: float = 1.0
    recovery_action_saturation_penalty_mult_return: float = 1.0
    recovery_stall_speed_threshold_mult_fallback: float = 1.0
    recovery_stall_speed_threshold_mult_return: float = 1.0
    recovery_heading_gate_min_return: float = 0.0
    recovery_heading_gate_min_return_mid_clearance: float = 0.0
    recovery_heading_gate_min_return_low_clearance: float = 0.0
    recovery_heading_gate_min_return_high_clearance_threshold: float = 0.80
    recovery_heading_gate_min_return_clearance_threshold: float = 0.60
    return_no_progress_delta_threshold: float = 0.0015
    return_no_progress_speed_threshold: float = 0.08
    return_no_progress_clearance_threshold: float = 0.55
    return_no_progress_moving_speed_threshold: float = 0.14
    return_no_progress_moving_delta_threshold: float = 0.004
    return_no_progress_moving_penalty_mult: float = 1.0
    return_forward_goal_scale_high_clearance: float = 1.0
    return_forward_goal_scale_mid_clearance: float = 1.0
    return_forward_goal_scale_low_clearance: float = 1.0
    return_forward_goal_scale_high_clearance_threshold: float = 0.80
    return_forward_goal_scale_mid_clearance_threshold: float = 0.60
    return_turn_penalty_clearance_boost_high: float = 1.0
    return_turn_penalty_clearance_boost_mid: float = 1.0
    return_turn_penalty_clearance_boost_low: float = 1.0
    return_turn_penalty_boost_high_clearance_threshold: float = 0.80
    return_turn_penalty_boost_mid_clearance_threshold: float = 0.60
    return_action_sat_clearance_boost_high: float = 1.0
    return_action_sat_clearance_boost_mid: float = 1.0
    return_action_sat_clearance_boost_low: float = 1.0
    return_action_sat_boost_high_clearance_threshold: float = 0.80
    return_action_sat_boost_mid_clearance_threshold: float = 0.60
    return_no_progress_penalty_step: float = 0.0
    return_no_progress_penalty_cap: float = 0.0
    return_forward_goal_bad_progress_scale: float = 1.0
    return_progress_age_mult_gain: float = 0.0
    return_progress_age_mult_cap: float = 1.0
    recovery_clearance_gain_reward_scale: float = 0.0
    recovery_yaw_oscillation_penalty_scale: float = 0.0
    success_bonus: float = 40.0
    collision_penalty: float = -60.0

    # terminal metrics printing
    metrics_print_interval_episodes: int = 100

    # step-level debug printing
    debug_print_enable: bool = True
    debug_print_interval_steps: int = 500
    debug_iteration_rollouts: int = 64
    reset_debug_interval_resets: int = 128
    debug_phase_tag: int = 0

    def __post_init__(self):
        super().__post_init__()
        # Expand obstacle filter paths explicitly to avoid PhysX wildcard count mismatch.
        # Example: ["/World/envs/env_.*/Obstacle_0", ..., "/World/envs/env_.*/Obstacle_{N-1}"]
        num_obs = max(1, int(self.num_obstacles))
        self.contact_sensor.filter_prim_paths_expr = [f"/World/envs/env_.*/Obstacle_{i}" for i in range(num_obs)]


@configclass
class LidarCorridorTurtleBot3EnvCfg(LidarNavTurtleBot3EnvCfg):
    """TurtleBot3 LiDAR corridor-navigation environment config."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=14.0, replicate_physics=True)

    corridor_enable: bool = True
    corridor_half_width: float = 0.80
    corridor_half_length: float = 8.0
    corridor_goal_lateral_range: float = 0.12
    corridor_obstacle_lateral_margin: float = 0.18
    corridor_obstacle_forward_min_offset: float = 0.65
    corridor_obstacle_goal_margin: float = 0.75
    corridor_visualize_walls: bool = True
    corridor_wall_height: float = 0.45
    corridor_wall_thickness: float = 0.04

    goal_spawn_range_min: float = 2.0
    goal_spawn_range_max: float = 8.0
    num_obstacles: int = 5
    obstacle_spawn_range_min: float = 1.5
    obstacle_spawn_range_max: float = 7.0


@configclass
class LidarJunctionTurtleBot3EnvCfg(LidarNavTurtleBot3EnvCfg):
    """TurtleBot3 LiDAR junction-navigation environment config (L/T/X intersections)."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=16.0, replicate_physics=True)
    # Same v2 observation contract as deployment: 58D = 23 base/context dims + 35 lidar beams.
    observation_space = 58

    junction_enable: bool = True
    planner_state_mixed_training_enable: bool = True
    planner_state_prob_explore: float = 0.40
    planner_state_prob_fallback: float = 0.00
    planner_state_prob_junction: float = 0.60
    planner_state_prob_return: float = 0.00
    speed_cap_return: float = 0.34
    return_speed_cap_high_clearance: float = 0.40
    return_speed_cap_mid_clearance: float = 0.34
    return_speed_cap_low_clearance: float = 0.22
    return_speed_cap_high_clearance_threshold: float = 0.80
    return_speed_cap_mid_clearance_threshold: float = 0.40
    relay_goal_enable: bool = True
    explore_relay_goal_distance_min: float = 0.3
    explore_relay_goal_distance_max: float = 2.5
    explore_relay_goal_close_sampling_prob: float = 0.30
    explore_relay_goal_close_distance_min: float = 0.30
    explore_relay_goal_close_distance_max: float = 0.60
    fallback_relay_goal_distance_min: float = 1.0
    fallback_relay_goal_distance_max: float = 2.0
    relay_goal_distance_min: float = 1.8
    relay_goal_distance_max: float = 3.5
    junction_relay_goal_distance_min: float = 0.3
    junction_relay_goal_distance_max: float = 2.0
    junction_relay_goal_close_sampling_prob: float = 0.30
    junction_relay_goal_close_distance_min: float = 0.30
    junction_relay_goal_close_distance_max: float = 0.60
    junction_turn_relay_goal_distance_min: float = 0.3
    junction_turn_relay_goal_distance_max: float = 1.8
    junction_turn_relay_goal_close_sampling_prob: float = 0.30
    junction_turn_relay_goal_close_distance_min: float = 0.30
    junction_turn_relay_goal_close_distance_max: float = 0.60
    relay_goal_lateral_range: float = 0.12
    return_relay_goal_distance_min: float = 1.0
    return_relay_goal_distance_max: float = 1.8
    return_relay_goal_lateral_range: float = 0.18
    return_relay_goal_lateral_min_abs: float = 0.10
    return_obstacle_centerline_keepout: float = 0.18
    return_obstacle_force_opposite_goal_side: bool = True
    linear_state_spawn_distance_min: float = 1.8
    linear_state_spawn_distance_max: float = 3.0
    linear_state_spawn_lateral_jitter: float = 0.06
    linear_corridor_half_width_choices: tuple[float, ...] = (0.50, 0.80, 1.50)
    linear_open_area_prob: float = 0.20
    linear_open_area_half_extent: float = 2.50
    junction_half_width_choices: tuple[float, ...] = (0.50, 0.80, 1.50)
    junction_state_spawn_distance_min: float = 0.7
    junction_state_spawn_distance_max: float = 1.2
    junction_relay_entry_offset: float = 0.40
    fallback_blocker_distance_min: float = 0.50
    fallback_blocker_distance_max: float = 0.75
    fallback_blocker_lateral_range: float = 0.14
    junction_half_width: float = 0.80
    junction_half_length: float = 8.0
    junction_robot_spawn_offset_min: float = 0.8
    junction_robot_spawn_offset_max: float = 1.8
    junction_robot_spawn_lateral_jitter: float = 0.06
    junction_goal_forward_min: float = 1.5
    junction_goal_forward_max: float = 2.5
    junction_goal_lateral_range: float = 0.10
    junction_obstacle_lateral_margin: float = 0.20
    junction_obstacle_forward_min_offset: float = 0.85
    junction_x_sampling_prob: float = 0.25
    junction_visualize_walls: bool = True
    junction_wall_height: float = 0.45
    junction_wall_thickness: float = 0.04

    # keep corridor disabled in this mode
    corridor_enable: bool = False
    corridor_visualize_walls: bool = False

    num_obstacles: int = 4
    v_max: float = 0.6
    omega_limit: float = 2.0
    tau_v: float = 0.12
    tau_omega: float = 0.08
    wheel_linear_speed_max: float = 0.6
    near_goal_min_action_scale: float = 0.40
    obstacle_slowdown_distance: float = 0.75
    obstacle_min_action_scale: float = 0.35
    heading_speed_cos_min: float = 0.30
    heading_speed_power: float = 1.5
    heading_speed_min_scale: float = 0.20
    goal_reach_threshold: float = 0.45
    goal_spawn_range_min: float = 1.5
    goal_spawn_range_max: float = 2.5
    obstacle_spawn_range_min: float = 1.5
    obstacle_spawn_range_max: float = 7.0

    heading_reward_scale: float = 0.10
    heading_gate_min: float = 0.10
    forward_goal_reward_scale: float = 0.45
    turn_penalty_scale: float = -0.02
    time_penalty: float = -0.04
    stall_speed_threshold: float = 0.05
    stall_progress_threshold: float = 0.001
    # Relay pass-through shaping: do not demand perfect heading inside the success halo.
    pass_through_heading_fade_distance: float = 0.60
    front_lidar_sector_deg: float = 60.0
    front_danger_distance: float = 0.80
    # Near-goal multiplier branch is neutralized in v9; pass-through shaping handles the last meter.
    near_goal_dist_mid: float = 2.0
    near_goal_dist_close: float = 1.2
    near_goal_clearance_min: float = 0.6  # min_lidar gate: multiplier=1.0 when below this

    # progress_reward multiplier tiers
    near_goal_progress_mult_mid: float = 1.0
    near_goal_progress_mult_close: float = 1.0
    # forward_goal_reward multiplier tiers (same distance bins)
    near_goal_fwd_mult_mid: float = 1.0         # goal_d < near_goal_dist_mid (disabled)
    near_goal_fwd_mult_close: float = 1.0       # goal_d < near_goal_dist_close (disabled)
    speed_cap_fallback: float = 0.26
    stall_penalty: float = -0.12
    clearance_reward_scale: float = 0.12
    danger_speed_penalty_scale: float = -5.0
    action_saturation_threshold: float = 0.80
    action_saturation_penalty_scale: float = -0.06
    # fallback/return-only recovery shaping
    recovery_state_reward_enable: bool = False
    recovery_forward_goal_scale_mult_fallback: float = 0.35
    recovery_forward_goal_scale_mult_return: float = 1.18
    recovery_clearance_scale_mult_fallback: float = 2.20
    recovery_clearance_scale_mult_return: float = 1.20
    recovery_danger_speed_penalty_mult_fallback: float = 2.50
    recovery_danger_speed_penalty_mult_return: float = 2.40
    recovery_turn_penalty_mult_fallback: float = 1.90
    recovery_turn_penalty_mult_return: float = 0.70
    recovery_action_saturation_penalty_mult_fallback: float = 2.20
    recovery_action_saturation_penalty_mult_return: float = 1.10
    recovery_stall_speed_threshold_mult_fallback: float = 0.45
    recovery_stall_speed_threshold_mult_return: float = 0.60
    recovery_heading_gate_min_return: float = 0.22
    recovery_heading_gate_min_return_mid_clearance: float = 0.10
    recovery_heading_gate_min_return_low_clearance: float = 0.05
    recovery_heading_gate_min_return_high_clearance_threshold: float = 0.80
    recovery_heading_gate_min_return_clearance_threshold: float = 0.40
    return_no_progress_delta_threshold: float = 0.0015
    return_no_progress_speed_threshold: float = 0.08
    return_no_progress_clearance_threshold: float = 0.28
    return_no_progress_moving_speed_threshold: float = 0.14
    return_no_progress_moving_delta_threshold: float = 0.004
    return_no_progress_moving_penalty_mult: float = 1.30
    return_forward_goal_scale_high_clearance: float = 1.00
    return_forward_goal_scale_mid_clearance: float = 0.75
    return_forward_goal_scale_low_clearance: float = 0.35
    return_forward_goal_scale_high_clearance_threshold: float = 0.80
    return_forward_goal_scale_mid_clearance_threshold: float = 0.40
    return_turn_penalty_clearance_boost_high: float = 1.00
    return_turn_penalty_clearance_boost_mid: float = 1.00
    return_turn_penalty_clearance_boost_low: float = 0.55
    return_turn_penalty_boost_high_clearance_threshold: float = 0.80
    return_turn_penalty_boost_mid_clearance_threshold: float = 0.40
    return_action_sat_clearance_boost_high: float = 1.00
    return_action_sat_clearance_boost_mid: float = 1.00
    return_action_sat_clearance_boost_low: float = 1.30
    return_action_sat_boost_high_clearance_threshold: float = 0.80
    return_action_sat_boost_mid_clearance_threshold: float = 0.40
    return_no_progress_penalty_step: float = 0.002
    return_no_progress_penalty_cap: float = 0.06
    return_forward_goal_bad_progress_scale: float = 0.35
    return_progress_age_mult_gain: float = 0.80
    return_progress_age_mult_cap: float = 1.80
    recovery_clearance_gain_reward_scale: float = 0.20
    recovery_yaw_oscillation_penalty_scale: float = -0.06
    turn_cmd_reward_scale: float = 0.22
    turn_cmd_yaw_reward_scale: float = 0.08
    success_bonus: float = 100.0
    collision_penalty: float = -80.0


@configclass
class LidarShortNavJunctionTurtleBot3EnvCfg(LidarJunctionTurtleBot3EnvCfg):
    """TurtleBot3 short-range junction-navigation preset for local avoidance."""

    # Increase inter-environment spacing to avoid visual/physics crowding.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=24.0, replicate_physics=True)

    # v2: keep proactive behavior but avoid over-aggressive explore collisions.
    speed_cap_explore: float = 0.30
    speed_cap_turn: float = 0.25
    speed_cap_narrow: float = 0.20

    linear_state_spawn_distance_min: float = 1.0
    linear_state_spawn_distance_max: float = 2.4
    # v3: ease explore/linear curriculum only (keep junction topology difficulty unchanged).
    linear_open_area_prob: float = 0.50
    linear_open_area_half_extent: float = 3.00
    linear_corridor_half_width_choices: tuple[float, ...] = (0.80, 1.20, 1.80)
    # Used by _spawn_linear_obstacle_positions as goal clearance in linear/explore sampling.
    corridor_obstacle_goal_margin: float = 0.95

    explore_relay_goal_distance_min: float = 0.3
    explore_relay_goal_distance_max: float = 2.0
    fallback_relay_goal_distance_min: float = 0.8
    fallback_relay_goal_distance_max: float = 1.5
    relay_goal_distance_min: float = 1.0
    relay_goal_distance_max: float = 2.5
    junction_relay_goal_distance_min: float = 0.3
    junction_relay_goal_distance_max: float = 1.8
    junction_turn_relay_goal_distance_min: float = 0.3
    junction_turn_relay_goal_distance_max: float = 1.5
    return_relay_goal_distance_min: float = 0.8
    return_relay_goal_distance_max: float = 1.5

    goal_reach_threshold: float = 0.30
    goal_spawn_range_min: float = 1.0
    goal_spawn_range_max: float = 2.5
    junction_goal_forward_min: float = 1.0
    junction_goal_forward_max: float = 2.5
    junction_x_sampling_prob: float = 0.25

    # Reward/termination shaping for short-range navigation (v2 balance):
    # reduce idling without incentivizing blind acceleration.
    progress_reward_scale: float = 8.8
    forward_goal_reward_scale: float = 0.50
    time_penalty: float = -0.03
    stall_speed_threshold: float = 0.04
    stall_penalty: float = -0.15
    collision_penalty: float = -100.0
    danger_speed_penalty_scale: float = -6.0
    front_danger_distance: float = 0.90
    obstacle_slowdown_distance: float = 0.85
    obstacle_min_action_scale: float = 0.30


@configclass
class LidarShortNavUnifiedTurtleBot3EnvCfg(LidarShortNavJunctionTurtleBot3EnvCfg):
    """Short-range unified local-navigation preset without privileged observations."""

    # 51D = 58D default - turn_cmd_3 - planner_state_4.
    observation_space = 51

    # Remove privileged state/context inputs from policy observations.
    use_privileged_obs: bool = False
    use_turn_cmd_obs: bool = False
    use_planner_state_obs: bool = False

    # Unified speed caps across straight/intersection contexts.
    speed_cap_explore: float = 0.22
    speed_cap_turn: float = 0.22
    speed_cap_narrow: float = 0.18

    # Reward balancing for generic local navigation.
    progress_reward_scale: float = 6.0
    forward_goal_reward_scale: float = 0.40
    time_penalty: float = -0.03
    stall_penalty: float = -0.12
    collision_penalty: float = -100.0

    # Remove explicit turn-command shaping; direction is encoded by relative goal.
    turn_cmd_reward_scale: float = 0.0
    turn_cmd_yaw_reward_scale: float = 0.0

    # Unified geometry curriculum (keep both linear and junction topology sampling).
    planner_state_prob_explore: float = 0.50
    planner_state_prob_junction: float = 0.50
    linear_open_area_prob: float = 0.70
    linear_corridor_half_width_choices: tuple[float, ...] = (1.00, 1.50, 2.00)
    corridor_obstacle_goal_margin: float = 1.20
    junction_x_sampling_prob: float = 0.25


@configclass
class LidarShortNavUnifiedCurriculumTurtleBot3EnvCfg(LidarShortNavUnifiedTurtleBot3EnvCfg):
    """Phase-1 curriculum preset: explore-heavy sampling with lower speed caps."""

    # Keep the same 51D pure-perception observation contract as unified.
    observation_space = 51
    use_privileged_obs: bool = False
    use_turn_cmd_obs: bool = False
    use_planner_state_obs: bool = False
    use_lidar_normalization: bool = True

    # Phase-1 safer action envelope.
    speed_cap_explore: float = 0.18
    speed_cap_turn: float = 0.18
    speed_cap_narrow: float = 0.16
    omega_limit: float = 1.5
    use_action_smoothing: bool = True
    action_smoothing_alpha: float = 0.7

    # Explore-heavy curriculum distribution.
    planner_state_prob_explore: float = 0.80
    planner_state_prob_fallback: float = 0.00
    planner_state_prob_junction: float = 0.20
    planner_state_prob_return: float = 0.00
    # Keep high open-area ratio for diversity regularization.
    linear_open_area_prob: float = 0.90
    linear_corridor_half_width_choices: tuple[float, ...] = (1.50, 2.00, 2.50)
    junction_x_sampling_prob: float = 0.25

    # v13 task-level shaping: shorten target distances and strengthen dense progress signals.
    goal_spawn_range_min: float = 0.8
    goal_spawn_range_max: float = 1.5
    junction_goal_forward_min: float = 0.8
    junction_goal_forward_max: float = 1.5
    progress_reward_scale: float = 10.0
    forward_goal_reward_scale: float = 0.60
    goal_reach_threshold: float = 0.35
    collision_penalty: float = -150.0
    stall_penalty: float = -0.12


@configclass
class LidarShortNavV14TurtleBot3EnvCfg(LidarShortNavUnifiedCurriculumTurtleBot3EnvCfg):
    """v14 preset: shorter local goals with v13-style dense shaping."""

    # Keep pure-perception observation contract.
    observation_space = 51
    use_privileged_obs: bool = False
    use_turn_cmd_obs: bool = False
    use_planner_state_obs: bool = False
    use_lidar_normalization: bool = True

    # Restore v13 action envelope (avoid over-conservative v13b caps).
    speed_cap_explore: float = 0.20
    speed_cap_turn: float = 0.20
    speed_cap_narrow: float = 0.18
    omega_limit: float = 1.5
    use_action_smoothing: bool = True
    action_smoothing_alpha: float = 0.7

    # Keep phase-1 curriculum distribution and scene setup.
    planner_state_prob_explore: float = 0.80
    planner_state_prob_fallback: float = 0.00
    planner_state_prob_junction: float = 0.20
    planner_state_prob_return: float = 0.00
    linear_open_area_prob: float = 0.90
    linear_corridor_half_width_choices: tuple[float, ...] = (1.50, 2.00, 2.50)
    junction_x_sampling_prob: float = 0.25

    # v14 difficulty reduction: shorten goals further and loosen reach radius.
    goal_spawn_range_min: float = 0.5
    goal_spawn_range_max: float = 1.0
    junction_goal_forward_min: float = 0.5
    junction_goal_forward_max: float = 1.0
    goal_reach_threshold: float = 0.40

    # Keep v13 dense shaping and penalties.
    progress_reward_scale: float = 12.0
    forward_goal_reward_scale: float = 0.80
    collision_penalty: float = -120.0
    stall_penalty: float = -0.15


@configclass
class LidarShortNavV15TurtleBot3EnvCfg(LidarShortNavV14TurtleBot3EnvCfg):
    """v15 preset: longer PPO horizon + stronger dense safety shaping."""

    # Keep v14 task geometry and action envelope; adjust safety shaping only.
    progress_reward_scale: float = 12.0
    forward_goal_reward_scale: float = 0.80

    # Dense safety signal: penalize risky forward motion earlier and reward clearance.
    front_danger_distance: float = 1.00
    danger_speed_penalty_scale: float = -12.0
    clearance_reward_scale: float = 0.24

    # Reduce terminal penalty magnitude to lower critic target variance.
    collision_penalty: float = -50.0
    stall_penalty: float = -0.15


@configclass
class LidarShortNavV15ATurtleBot3EnvCfg(LidarShortNavV14TurtleBot3EnvCfg):
    """v15a ablation: PPO horizon/batch only (env/reward identical to v14)."""


@configclass
class LidarShortNavV15BTurtleBot3EnvCfg(LidarShortNavV14TurtleBot3EnvCfg):
    """v15b ablation: dense safety shaping only (PPO settings identical to v14)."""

    # Keep v14 target/progress shaping; strengthen dense safety term.
    progress_reward_scale: float = 12.0
    forward_goal_reward_scale: float = 0.80
    collision_penalty: float = -120.0
    stall_penalty: float = -0.15

    front_danger_distance: float = 1.00
    danger_speed_penalty_scale: float = -12.0
    clearance_reward_scale: float = 0.24


@configclass
class LidarShortNavV16TurtleBot3EnvCfg(LidarShortNavV15BTurtleBot3EnvCfg):
    """v16 preset: 360deg / 90-beam LiDAR + 3-frame LiDAR stacking."""

    # Keep v15b reward/safety shaping and PPO recipe; upgrade perception only.
    lidar_fov_deg: float = 360.0
    lidar_num_beams: int = 90
    lidar_frame_stack: int = 3
    observation_space = 286  # 16 base dims + (90 beams * 3 frames)


@configclass
class LidarShortNavV17TurtleBot3EnvCfg(LidarShortNavV16TurtleBot3EnvCfg):
    """v17 preset: v16 perception + increased network capacity + longer rollout."""

    # Keep v16 perception (360deg, 90 beams, 3-frame stack).
    # Improvements target network architecture and training dynamics only.


@configclass
class LidarShortNavV18TurtleBot3EnvCfg(LidarShortNavV17TurtleBot3EnvCfg):
    """v18 preset: 360deg / 90-beam single-frame LiDAR for CNN-based policies."""

    # Keep v17 geometry/reward/training dynamics and switch perception to single-frame input.
    # Allow reverse maneuvers for trap escape.
    forward_only: bool = False
    v_max: float = 0.5
    lidar_fov_deg: float = 360.0
    lidar_num_beams: int = 90
    lidar_frame_stack: int = 1
    # High-range perception setting for v18/v18a.
    lidar_max_range: float = 10.0
    observation_space = 106  # 16 base dims + 90-beam LiDAR


@configclass
class LidarShortNavV18ATurtleBot3EnvCfg(LidarShortNavV18TurtleBot3EnvCfg):
    """v18a (v18.1): v18 CNN perception with stronger anti-timeout shaping."""

    # Keep reverse motion enabled in v18a.
    forward_only: bool = False
    # Encourage faster episode completion to reduce timeout-heavy behavior.
    time_penalty: float = -0.05


@configclass
class LidarShortNavDemoTurtleBot3EnvCfg(LidarShortNavV18ATurtleBot3EnvCfg):
    """Deterministic demo preset for high-success scene-specific overfitting."""

    # Keep one environment for repeatable demo runs.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=24.0, replicate_physics=True)
    # Use deterministic explore-only straight corridor layout.
    planner_state_mixed_training_enable: bool = True
    planner_state_prob_explore: float = 1.0
    planner_state_prob_fallback: float = 0.0
    planner_state_prob_junction: float = 0.0
    planner_state_prob_return: float = 0.0
    linear_open_area_prob: float = 0.0
    linear_corridor_half_width_choices: tuple[float, ...] = (2.00,)
    junction_half_width_choices: tuple[float, ...] = (2.00,)
    junction_half_width: float = 2.00
    junction_half_length: float = 5.0
    linear_state_spawn_distance_min: float = 2.0
    linear_state_spawn_distance_max: float = 2.0
    linear_state_spawn_lateral_jitter: float = 0.0
    speed_cap_explore: float = 0.30
    speed_cap_turn: float = 0.30
    speed_cap_narrow: float = 0.24
    obstacle_slowdown_distance: float = 0.40
    obstacle_min_action_scale: float = 0.45
    front_danger_distance: float = 0.75
    danger_speed_penalty_scale: float = -2.0
    stall_penalty: float = -0.10
    time_penalty: float = -0.03
    goal_reach_threshold: float = 0.45
    relay_goal_enable: bool = False
    relay_goal_lateral_range: float = 0.0
    # reduce sensor stochasticity for deterministic playback
    lidar_noise_std: float = 0.0
    # default demo placement: robot -> obstacle -> goal on centerline
    num_obstacles: int = 1
    fixed_scene_enable: bool = True
    fixed_scene_use_linear_layout: bool = True
    fixed_scene_robot_spawn_xy: tuple[float, float] = (-1.5, 0.0)
    fixed_scene_robot_spawn_yaw: float = 0.0
    fixed_scene_goal_xy: tuple[float, float] = (1.5, 0.0)
    fixed_scene_obstacle_xy: tuple[tuple[float, float], ...] = ((0.0, 0.60),)


@configclass
class LidarShortNavDemoHardTurtleBot3EnvCfg(LidarShortNavDemoTurtleBot3EnvCfg):
    """Deterministic harder demo preset with narrower lane and two fixed obstacles."""

    # Keep deterministic explore-only regime.
    planner_state_prob_explore: float = 1.0
    planner_state_prob_fallback: float = 0.0
    planner_state_prob_junction: float = 0.0
    planner_state_prob_return: float = 0.0
    linear_open_area_prob: float = 0.0

    # Stage-1 medium difficulty: narrower corridor and longer fixed route.
    linear_corridor_half_width_choices: tuple[float, ...] = (1.20,)
    junction_half_width_choices: tuple[float, ...] = (1.20,)
    junction_half_width: float = 1.20
    num_obstacles: int = 2
    fixed_scene_robot_spawn_xy: tuple[float, float] = (-2.0, 0.0)
    fixed_scene_robot_spawn_yaw: float = 0.0
    fixed_scene_goal_xy: tuple[float, float] = (2.0, 0.0)
    fixed_scene_obstacle_xy: tuple[tuple[float, float], ...] = ((-0.5, 0.55), (0.8, -0.55))


@configclass
class LidarShortNavDemoHardV1TurtleBot3EnvCfg(LidarShortNavDemoHardTurtleBot3EnvCfg):
    """Softened stage-1 hard demo preset for curriculum warm-start."""

    # Keep the same corridor/task layout as demo-hard and widen obstacle lateral offsets.
    fixed_scene_obstacle_xy: tuple[tuple[float, float], ...] = ((-0.5, 0.75), (0.8, -0.75))


@configclass
class LidarShortNavDemoMultiV1TurtleBot3EnvCfg(LidarShortNavDemoHardTurtleBot3EnvCfg):
    """Stage-3 deterministic scene-pool preset (multi fixed scenes)."""

    # Keep deterministic explore-only regime.
    planner_state_prob_explore: float = 1.0
    planner_state_prob_fallback: float = 0.0
    planner_state_prob_junction: float = 0.0
    planner_state_prob_return: float = 0.0
    linear_open_area_prob: float = 0.0

    # Keep two obstacles for all scenes; missing obstacles are auto-parked out-of-bounds.
    num_obstacles: int = 2
    fixed_scene_enable: bool = True
    fixed_scene_use_linear_layout: bool = True
    fixed_scene_multi_enable: bool = True
    fixed_scene_multi_index_override: int = -1

    # (robot_x, robot_y, robot_yaw, goal_x, goal_y, corridor_half_width)
    fixed_scene_multi_scene_params: tuple[tuple[float, float, float, float, float, float], ...] = (
        (-1.5, 0.0, 0.0, 1.5, 0.0, 2.00),   # scene 0: demo baseline
        (-2.0, 0.0, 0.0, 2.0, 0.0, 1.20),   # scene 1: hard-v1 style straight dual-obstacle
        (-2.0, 0.0, 0.0, 2.0, 0.0, 1.00),   # scene 2: narrow corridor single-obstacle
        (-2.0, 0.0, 0.0, 2.0, 0.0, 1.20),   # scene 3: S-shaped obstacle channel
        (-1.8, 0.0, 0.0, 1.2, 0.85, 1.40),  # scene 4: off-center goal (turning-like demo)
    )
    fixed_scene_multi_obstacle_xy: tuple[tuple[tuple[float, float], ...], ...] = (
        ((0.0, 0.60),),                     # scene 0
        ((-0.5, 0.75), (0.8, -0.75)),       # scene 1
        ((0.0, 0.50),),                     # scene 2
        ((-0.6, -0.45), (0.6, 0.45)),       # scene 3
        ((-0.2, 0.35), (0.6, -0.30)),       # scene 4
    )


@configclass
class LidarShortNavDemoMultiV1BridgeTurtleBot3EnvCfg(LidarShortNavDemoMultiV1TurtleBot3EnvCfg):
    """Bridge preset for stage-3 curriculum before tightening to MultiV1."""

    # Loosen action bottlenecks in chicane-like layouts to avoid high-stall local minima.
    # Enable reverse motion to learn "back off then bypass" behavior in local deadlocks.
    forward_only: bool = False
    obstacle_slowdown_distance: float = 0.30
    obstacle_min_action_scale: float = 0.55
    action_saturation_penalty_scale: float = -0.03
    # Keep curriculum stable: mix scene0-3 first, then handle scene4 in a dedicated phase.
    fixed_scene_multi_index_pool: tuple[int, ...] = (0, 1, 2, 3)

    # Keep scene 0/1/2 unchanged and soften scene 3/4 geometry.
    fixed_scene_multi_scene_params: tuple[tuple[float, float, float, float, float, float], ...] = (
        (-1.5, 0.0, 0.0, 1.5, 0.0, 2.00),   # scene 0: demo baseline
        (-2.0, 0.0, 0.0, 2.0, 0.0, 1.20),   # scene 1: hard-v1 style straight dual-obstacle
        (-2.0, 0.0, 0.0, 2.0, 0.0, 1.00),   # scene 2: narrow corridor single-obstacle
        (-2.0, 0.0, 0.0, 2.0, 0.0, 1.30),   # scene 3 bridge: wider S-channel
        (-1.8, 0.0, 0.0, 1.6, 0.45, 1.80),  # scene 4 bridge: mild off-center goal
    )
    fixed_scene_multi_obstacle_xy: tuple[tuple[tuple[float, float], ...], ...] = (
        ((0.0, 0.60),),                     # scene 0
        ((-0.5, 0.75), (0.8, -0.75)),       # scene 1
        ((0.0, 0.50),),                     # scene 2
        ((-0.6, -0.55), (0.6, 0.55)),       # scene 3 bridge
        ((-0.1, 0.20), (0.85, -0.15)),      # scene 4 bridge
    )

    # Enable scene4-specific subgoal shaping to break the "wait at junction" local minimum.
    subgoal_reward_enable: bool = True
    subgoal_scene_index: int = 4
    subgoal_xy: tuple[float, float] = (0.0, 0.5)
    subgoal_radius: float = 0.30
    subgoal_progress_reward_scale: float = 8.0
    subgoal_bonus: float = 5.0
    # Encourage selecting a side and moving past obstacles in scene 4.
    bypass_reward_enable: bool = True
    bypass_scene_index: int = 4
    bypass_open_side_min_lidar: float = 0.30
    bypass_blocked_side_max_lidar: float = 0.30
    bypass_near_obstacle_distance: float = 0.95
    bypass_forward_speed_threshold: float = 0.05
    bypass_reward_scale: float = 0.10


@configclass
class LidarShortNavDemoScene4EasyTurtleBot3EnvCfg(LidarShortNavDemoMultiV1BridgeTurtleBot3EnvCfg):
    """Curriculum stage-A preset: obstacle-free side-goal scene for scene4 warm-up."""

    # Keep deterministic fixed reset and isolate a single easy scene.
    fixed_scene_multi_enable: bool = True
    fixed_scene_multi_index_override: int = 0
    fixed_scene_multi_index_pool: tuple[int, ...] = (0,)
    fixed_scene_use_linear_layout: bool = False
    planner_state_prob_explore: float = 1.0
    planner_state_prob_fallback: float = 0.0
    planner_state_prob_junction: float = 0.0
    planner_state_prob_return: float = 0.0

    # Remove virtual wall constraints so policy can learn side-goal steering first.
    junction_enable: bool = False
    corridor_enable: bool = False
    linear_open_area_prob: float = 1.0
    linear_open_area_half_extent: float = 3.5

    # Keep obstacle tensor shape stable; park all obstacles out of the playable area.
    num_obstacles: int = 2
    fixed_scene_multi_scene_params: tuple[tuple[float, float, float, float, float, float], ...] = (
        (-1.8, 0.0, 0.0, 1.6, 0.45, 3.50),
    )
    fixed_scene_multi_obstacle_xy: tuple[tuple[tuple[float, float], ...], ...] = (
        (),
    )


@configclass
class LidarShortNavDemoScene4MidTurtleBot3EnvCfg(LidarShortNavDemoMultiV1BridgeTurtleBot3EnvCfg):
    """Curriculum stage-A2 preset: keep wall constraints and remove scene4 obstacles."""

    # Keep deterministic fixed reset and isolate the mid-difficulty scene.
    fixed_scene_multi_enable: bool = True
    fixed_scene_multi_index_override: int = 0
    fixed_scene_multi_index_pool: tuple[int, ...] = (0,)
    fixed_scene_use_linear_layout: bool = True
    planner_state_prob_explore: float = 1.0
    planner_state_prob_fallback: float = 0.0
    planner_state_prob_junction: float = 0.0
    planner_state_prob_return: float = 0.0

    # Preserve corridor/junction walls and disable open-area sampling.
    junction_enable: bool = True
    corridor_enable: bool = True
    linear_open_area_prob: float = 0.0

    # Keep obstacle tensor shape stable; park all obstacles out of the playable area.
    num_obstacles: int = 2
    fixed_scene_multi_scene_params: tuple[tuple[float, float, float, float, float, float], ...] = (
        (-1.8, 0.0, 0.0, 1.6, 0.45, 2.20),
    )
    fixed_scene_multi_obstacle_xy: tuple[tuple[tuple[float, float], ...], ...] = (
        (),
    )


@configclass
class LidarShortNavDemoScene4Mid2TurtleBot3EnvCfg(LidarShortNavDemoScene4MidTurtleBot3EnvCfg):
    """Curriculum stage-A3 preset: narrower wall-constrained scene between mid and full."""

    # Keep the same geometry/constraints as scene4_mid and tighten corridor half-width.
    fixed_scene_multi_scene_params: tuple[tuple[float, float, float, float, float, float], ...] = (
        (-1.8, 0.0, 0.0, 1.6, 0.45, 2.00),
    )


@configclass
class LidarShortNavDemoScene4Mid2Obs1TurtleBot3EnvCfg(LidarShortNavDemoScene4Mid2TurtleBot3EnvCfg):
    """Curriculum stage-A4 preset: mid2 geometry + single effective obstacle."""

    # Keep obstacle tensor shape stable and activate only one obstacle in scene.
    num_obstacles: int = 2
    fixed_scene_multi_obstacle_xy: tuple[tuple[tuple[float, float], ...], ...] = (
        ((0.0, 0.20),),
    )


@configclass
class LidarShortNavDemoScene4Mid2Obs1OpenTurtleBot3EnvCfg(LidarShortNavDemoScene4Mid2Obs1TurtleBot3EnvCfg):
    """Curriculum stage-A4-open preset: open area + single obstacle (no wall constraints)."""

    # Keep deterministic fixed reset and isolate this curriculum scene.
    fixed_scene_multi_enable: bool = True
    fixed_scene_multi_index_override: int = 0
    fixed_scene_multi_index_pool: tuple[int, ...] = (0,)
    fixed_scene_use_linear_layout: bool = False

    # Remove wall constraints and force open-area layout.
    junction_enable: bool = False
    corridor_enable: bool = False
    linear_open_area_prob: float = 1.0
    linear_open_area_half_extent: float = 3.5


@configclass
class LidarShortNavDemoScene4WallOnlyTurtleBot3EnvCfg(LidarShortNavDemoMultiV1BridgeTurtleBot3EnvCfg):
    """Curriculum preset: full scene4 wall geometry with no internal obstacles."""

    # Keep deterministic fixed reset and isolate the wall-only scene.
    fixed_scene_multi_enable: bool = True
    fixed_scene_multi_index_override: int = 0
    fixed_scene_multi_index_pool: tuple[int, ...] = (0,)
    fixed_scene_use_linear_layout: bool = True

    # Keep wall constraints (same as bridge scene4) and remove obstacles.
    junction_enable: bool = True
    corridor_enable: bool = True
    linear_open_area_prob: float = 0.0
    num_obstacles: int = 2
    fixed_scene_multi_scene_params: tuple[tuple[float, float, float, float, float, float], ...] = (
        (-1.8, 0.0, 0.0, 1.6, 0.45, 1.80),
    )
    fixed_scene_multi_obstacle_xy: tuple[tuple[tuple[float, float], ...], ...] = (
        (),
    )


@configclass
class LidarShortNavDefectDwbOscillationTurtleBot3EnvCfg(LidarShortNavV18ATurtleBot3EnvCfg):
    """Defect scene: 2 m corridor with two central cylinders for DWB oscillation reproduction."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=20.0, replicate_physics=True)
    episode_length_s = 18.0
    planner_state_mixed_training_enable: bool = False
    junction_enable: bool = False
    corridor_enable: bool = False
    linear_open_area_prob: float = 0.0
    relay_goal_enable: bool = False

    forward_only: bool = False
    defect_scene_enable: bool = True
    defect_scene_mode: str = "dwb_oscillation"
    defect_scene_visualize_walls: bool = True
    defect_wall_segment_capacity: int = 64
    defect_spawn_jitter_xy: float = 0.2
    defect_goal_distance_min: float = 0.8
    defect_goal_distance_max: float = 3.0

    num_obstacles: int = 2
    obstacle_radius: float = 0.2
    obstacle_min_separation: float = 0.55
    defect_dwb_corridor_width: float = 2.0
    defect_dwb_corridor_width_jitter: float = 0.0
    defect_dwb_corridor_half_length: float = 2.0
    defect_dwb_corridor_half_length_jitter: float = 0.0
    defect_dwb_obstacle_spacing: float = 0.6
    defect_dwb_obstacle_spacing_jitter: float = 0.05
    defect_dwb_obstacle_pair_shift_y: float = 0.16
    defect_dwb_obstacle_x_jitter: float = 0.12
    # Curriculum for gradual hardening in dwb_oscillation defect scene.
    defect_dwb_curriculum_enable: bool = True
    # When enabled, stage_steps are interpreted relative to the first observed training step
    # of the current run (useful for resume-only transition bridges).
    defect_dwb_curriculum_use_relative_steps: bool = False
    # When >= 0, freeze the dwb-like curriculum at the specified stage index.
    defect_dwb_curriculum_freeze_stage: int = -1
    defect_dwb_curriculum_stage_steps: tuple[int, ...] = (0, 2_200_000, 4_000_000)
    defect_dwb_curriculum_collision_thresholds: tuple[float, ...] = (0.08, 0.09, 0.10)
    defect_dwb_curriculum_contact_force_thresholds: tuple[float, ...] = (10.0, 9.0, 8.0)
    defect_dwb_curriculum_goal_distance_mins: tuple[float, ...] = (0.8, 0.9, 1.0)
    defect_dwb_curriculum_goal_distance_maxs: tuple[float, ...] = (1.3, 1.8, 2.2)
    defect_dwb_curriculum_goal_reach_thresholds: tuple[float, ...] = (0.60, 0.50, 0.45)
    defect_dwb_curriculum_time_penalties: tuple[float, ...] = (-0.01, -0.012, -0.015)
    defect_dwb_curriculum_lateral_penalty_weights: tuple[float, ...] = (-0.4, -0.45, -0.5)
    defect_dwb_curriculum_passage_bonuses: tuple[float, ...] = (12.0, 10.0, 8.0)
    defect_dwb_curriculum_corridor_widths: tuple[float, ...] = (3.5, 3.0, 2.6)
    defect_dwb_curriculum_obstacle_spacings: tuple[float, ...] = (1.2, 1.0, 0.9)
    defect_dwb_curriculum_v_maxes: tuple[float, ...] = (0.26, 0.28, 0.35)
    defect_dwb_curriculum_progress_reward_scales: tuple[float, ...] = (12.0, 10.0, 10.0)
    defect_dwb_curriculum_forward_goal_reward_scales: tuple[float, ...] = (1.20, 1.05, 0.95)
    defect_dwb_curriculum_stall_penalties: tuple[float, ...] = (-0.15, -0.15, -0.14)
    defect_dwb_curriculum_spawn_jitters: tuple[float, ...] = (0.08, 0.08, 0.10)
    defect_dwb_curriculum_goal_jitters: tuple[float, ...] = (0.18, 0.16, 0.12)
    defect_dwb_curriculum_obstacle_pair_shift_ys: tuple[float, ...] = (0.10, 0.08, 0.05)
    defect_dwb_curriculum_obstacle_x_jitters: tuple[float, ...] = (0.08, 0.06, 0.04)
    defect_dwb_curriculum_obstacle_spacing_jitters: tuple[float, ...] = (0.05, 0.03, 0.02)
    defect_dwb_gap_progress_reward_scale: float = 8.0
    defect_dwb_gap_alignment_reward_scale: float = 4.0
    defect_dwb_gap_clear_bonus: float = 20.0
    defect_dwb_gap_clear_x_margin: float = 0.14
    # Keep dense progress shaping scale consistent for 1~3 m goals.
    progress_reward_scale: float = 12.0
    forward_goal_reward_scale: float = 1.20
    goal_reach_threshold: float = 0.50
    smooth_action_reward_scale: float = -0.01
    time_penalty: float = -0.01
    stall_speed_threshold: float = 0.08
    stall_progress_threshold: float = 0.0035
    stall_penalty: float = -0.15
    danger_speed_penalty_scale: float = -0.15
    obstacle_min_action_scale: float = 0.45
    heading_speed_min_scale: float = 0.45
    tau_v: float = 0.14
    tau_omega: float = 0.10
    # Align sparse terminal bonus with door-scene training target.
    success_bonus: float = 40.0
    collision_penalty: float = -40.0
    # Milestone: reward passing the central narrow gap.
    defect_narrow_passage_bonus: float = 10.0
    defect_dwb_forward_speed_reward_scale: float = 1.00
    defect_dwb_forward_speed_min_lidar: float = 0.16
    defect_dwb_forward_speed_heading_min: float = 0.15
    defect_dwb_post_gap_forward_reward_scale: float = 0.15
    defect_dwb_post_gap_progress_reward_scale: float = 2.0
    # Encourage early side commitment through the symmetric bottleneck and penalize late switches.
    # Preset (active baseline): commit=1.5, switch_penalty=0.20, deadzone=0.06
    # Preset (stable conservative): commit=1.0, switch_penalty=0.30, deadzone=0.10
    # Preset (aggressive): commit=2.2, switch_penalty=0.14, deadzone=0.03
    defect_dwb_side_commit_reward_scale: float = 1.5
    defect_dwb_side_switch_penalty_scale: float = 0.20
    defect_dwb_side_commit_deadzone: float = 0.06
    defect_dwb_commit_forward_speed_min: float = 0.08
    defect_dwb_commit_min_lidar: float = 0.20
    defect_dwb_commit_heading_cos_min: float = 0.45
    defect_dwb_commit_x_margin_before_gap: float = 0.32
    defect_dwb_commit_x_margin_after_gap: float = 0.28
    defect_dwb_commit_lateral_tolerance: float = 0.18
    defect_dwb_observation_enhance_enable: bool = True
    defect_dwb_no_progress_speed_threshold: float = 0.08
    defect_dwb_no_progress_delta_threshold: float = 0.0035
    defect_dwb_no_progress_moving_speed_threshold: float = 0.12
    defect_dwb_no_progress_moving_delta_threshold: float = 0.0020
    defect_dwb_no_progress_clearance_threshold: float = 0.16
    defect_dwb_no_progress_penalty: float = -0.12
    defect_dwb_no_progress_terminate_steps: int = 20
    bypass_reward_enable: bool = True
    bypass_scene_index: int = -1
    bypass_open_side_min_lidar: float = 0.30
    bypass_blocked_side_max_lidar: float = 0.30
    bypass_near_obstacle_distance: float = 0.95
    bypass_forward_speed_threshold: float = 0.05
    bypass_reward_scale: float = 0.14


@configclass
class LidarShortNavDefectSingleObstacleSymmetricGapTurtleBot3EnvCfg(LidarShortNavDefectDwbOscillationTurtleBot3EnvCfg):
    """Defect scene: single centered obstacle with symmetric upper/lower bypass gaps."""

    defect_scene_mode: str = "single_obstacle_symmetric_gap"
    num_obstacles: int = 1
    obstacle_radius: float = 0.38
    defect_single_gap_obstacle_size_x: float = 0.60
    defect_single_gap_obstacle_size_y: float = 1.00
    defect_single_gap_goal_mode: str = "rear"
    defect_single_gap_goal_local_x: float = 0.0
    defect_single_gap_goal_local_y: float = 0.0
    defect_single_gap_goal_local_jitter_x: float = 0.0
    defect_single_gap_goal_local_jitter_y: float = 0.0
    defect_single_gap_goal_rear_clearance_min: float = 0.80
    defect_single_gap_goal_rear_lateral_center: float = 0.0
    defect_single_gap_goal_rear_lateral_offset_max: float = 0.12
    defect_single_gap_goal_bridge_clearance_min: float = 0.30
    defect_single_gap_goal_bridge_lateral_center: float = 0.20
    defect_single_gap_goal_bridge_lateral_offset_max: float = 0.02
    defect_single_gap_spawn_obstacle_clearance: float = 0.70
    defect_single_gap_goal_clearance_margin: float = 0.04
    defect_dwb_corridor_width: float = 2.0
    defect_dwb_corridor_half_length: float = 2.2
    defect_dwb_obstacle_spacing: float = 0.0
    defect_dwb_obstacle_spacing_jitter: float = 0.0
    defect_dwb_obstacle_pair_shift_y: float = 0.0
    defect_dwb_obstacle_x_jitter: float = 0.08
    defect_single_gap_obstacle_x_center: float = 0.0
    defect_single_gap_obstacle_y_jitter: float = 0.04
    defect_single_gap_target_lateral_margin: float = 0.06
    defect_single_gap_target_lateral_tolerance: float = 0.12
    defect_spawn_jitter_xy: float = 0.08
    defect_goal_jitter_xy: float = 0.06
    defect_goal_distance_min: float = 1.0
    defect_goal_distance_max: float = 3.0

    # Start with a milder corridor and converge to the paper-like 2.0 m setup.
    defect_dwb_curriculum_stage_steps: tuple[int, ...] = (0, 2_200_000, 4_000_000)
    defect_dwb_curriculum_collision_thresholds: tuple[float, ...] = (0.08, 0.09, 0.10)
    defect_dwb_curriculum_contact_force_thresholds: tuple[float, ...] = (10.0, 9.0, 8.0)
    # Keep curriculum aligned with the enforced single-obstacle front/rear clearances.
    defect_dwb_curriculum_goal_distance_mins: tuple[float, ...] = (2.1, 2.2, 2.3)
    defect_dwb_curriculum_goal_distance_maxs: tuple[float, ...] = (3.0, 3.2, 3.4)
    defect_dwb_curriculum_goal_reach_thresholds: tuple[float, ...] = (0.60, 0.50, 0.45)
    defect_dwb_curriculum_time_penalties: tuple[float, ...] = (-0.01, -0.012, -0.015)
    defect_dwb_curriculum_lateral_penalty_weights: tuple[float, ...] = (-0.35, -0.40, -0.45)
    defect_dwb_curriculum_passage_bonuses: tuple[float, ...] = (10.0, 9.0, 8.0)
    defect_dwb_curriculum_corridor_widths: tuple[float, ...] = (3.0, 2.5, 2.0)
    defect_dwb_curriculum_obstacle_spacings: tuple[float, ...] = (0.0, 0.0, 0.0)
    defect_dwb_curriculum_v_maxes: tuple[float, ...] = (0.35, 0.38, 0.42)
    defect_dwb_curriculum_progress_reward_scales: tuple[float, ...] = (9.5, 9.5, 10.0)
    defect_dwb_curriculum_forward_goal_reward_scales: tuple[float, ...] = (1.00, 0.95, 0.95)
    defect_dwb_curriculum_stall_penalties: tuple[float, ...] = (-0.14, -0.14, -0.12)
    defect_dwb_curriculum_spawn_jitters: tuple[float, ...] = (0.03, 0.05, 0.08)
    defect_dwb_curriculum_goal_jitters: tuple[float, ...] = (0.03, 0.05, 0.06)
    defect_dwb_curriculum_obstacle_pair_shift_ys: tuple[float, ...] = (0.0, 0.0, 0.0)
    defect_dwb_curriculum_obstacle_x_jitters: tuple[float, ...] = (0.04, 0.06, 0.08)
    defect_dwb_curriculum_obstacle_x_centers: tuple[float, ...] = (0.0, 0.0, 0.0)
    defect_dwb_curriculum_obstacle_spacing_jitters: tuple[float, ...] = (0.0, 0.0, 0.0)

    # Single-obstacle shaping: reward clean lateral commitment around the blocker instead of center-gap alignment.
    defect_narrow_passage_bonus: float = 8.0
    defect_dwb_gap_progress_reward_scale: float = 6.0
    defect_dwb_gap_alignment_reward_scale: float = 8.0
    defect_dwb_gap_clear_bonus: float = 20.0
    defect_dwb_gap_clear_x_margin: float = 0.14
    # Penalize lingering on the blocker centerline once the robot is close enough that
    # straight-line progress is no longer the desired behavior.
    defect_single_gap_centerline_penalty_scale: float = -0.10
    defect_single_gap_centerline_penalty_forward_dist: float = 0.55
    defect_dwb_forward_speed_reward_scale: float = 0.80
    defect_dwb_forward_speed_min_lidar: float = 0.18
    defect_dwb_forward_speed_heading_min: float = 0.15
    defect_dwb_post_gap_forward_reward_scale: float = 0.15
    defect_dwb_post_gap_progress_reward_scale: float = 2.0
    defect_dwb_side_commit_reward_scale: float = 1.5
    defect_dwb_side_switch_penalty_scale: float = 0.20
    defect_dwb_side_commit_deadzone: float = 0.06
    defect_dwb_commit_forward_speed_min: float = 0.08
    defect_dwb_commit_min_lidar: float = 0.22
    defect_dwb_commit_heading_cos_min: float = 0.45
    defect_dwb_commit_x_margin_before_gap: float = 0.40
    defect_dwb_commit_x_margin_after_gap: float = 0.30
    defect_dwb_commit_lateral_tolerance: float = 0.10
    defect_dwb_observation_enhance_enable: bool = True
    defect_dwb_no_progress_speed_threshold: float = 0.08
    defect_dwb_no_progress_delta_threshold: float = 0.0035
    defect_dwb_no_progress_moving_speed_threshold: float = 0.12
    defect_dwb_no_progress_moving_delta_threshold: float = 0.0020
    defect_dwb_no_progress_clearance_threshold: float = 0.18
    defect_dwb_no_progress_penalty: float = -0.12
    defect_dwb_no_progress_terminate_steps: int = 50

    # Allow the policy to keep enough forward velocity after obstacle and near-clearance scaling.
    speed_cap_explore: float = 0.32
    speed_cap_turn: float = 0.30
    speed_cap_narrow: float = 0.26
    bypass_reward_scale: float = 0.30


@configclass
class LidarShortNavDefectSmallObstaclePretrainTurtleBot3EnvCfg(
    LidarShortNavDefectSingleObstacleSymmetricGapTurtleBot3EnvCfg
):
    """Easier single-obstacle pretrain task that teaches the first successful side bypass."""

    defect_scene_mode: str = "single_obstacle_symmetric_gap"
    defect_single_gap_obstacle_size_x: float = 0.55
    defect_single_gap_obstacle_size_y: float = 0.60
    defect_single_gap_goal_mode: str = "fixed_local"
    defect_single_gap_goal_local_x: float = -0.05
    defect_single_gap_goal_local_y: float = 0.36
    defect_single_gap_goal_local_jitter_x: float = 0.03
    defect_single_gap_goal_local_jitter_y: float = 0.03
    defect_single_gap_goal_rear_clearance_min: float = 0.55
    defect_single_gap_goal_rear_lateral_center: float = 0.34
    defect_single_gap_goal_rear_lateral_offset_max: float = 0.04
    defect_single_gap_spawn_obstacle_clearance: float = 0.50
    defect_dwb_obstacle_x_jitter: float = 0.0
    defect_single_gap_obstacle_y_jitter: float = 0.0
    defect_single_gap_target_lateral_margin: float = 0.04
    defect_single_gap_target_lateral_tolerance: float = 0.16
    defect_spawn_jitter_xy: float = 0.06
    defect_goal_jitter_xy: float = 0.05

    # Keep this task materially easier so PPO can discover the first successful bypasses.
    defect_dwb_curriculum_goal_distance_mins: tuple[float, ...] = (1.5, 1.7, 1.9)
    defect_dwb_curriculum_goal_distance_maxs: tuple[float, ...] = (2.1, 2.4, 2.7)
    defect_dwb_curriculum_goal_reach_thresholds: tuple[float, ...] = (0.70, 0.65, 0.60)
    defect_dwb_curriculum_corridor_widths: tuple[float, ...] = (3.0, 2.8, 2.6)
    defect_dwb_curriculum_v_maxes: tuple[float, ...] = (0.38, 0.40, 0.42)
    defect_dwb_curriculum_progress_reward_scales: tuple[float, ...] = (9.0, 9.5, 10.0)
    defect_dwb_curriculum_forward_goal_reward_scales: tuple[float, ...] = (1.05, 1.00, 0.95)
    defect_dwb_curriculum_spawn_jitters: tuple[float, ...] = (0.02, 0.04, 0.06)
    defect_dwb_curriculum_goal_jitters: tuple[float, ...] = (0.02, 0.04, 0.05)
    defect_dwb_gap_alignment_reward_scale: float = 9.0
    defect_dwb_gap_clear_bonus: float = 24.0
    defect_single_gap_centerline_penalty_scale: float = -0.08
    defect_single_gap_centerline_penalty_forward_dist: float = 0.45
    defect_dwb_side_commit_reward_scale: float = 1.8
    defect_dwb_no_progress_terminate_steps: int = 60
    subgoal_reward_enable: bool = False


@configclass
class LidarShortNavDefectSingleObstacleWaypointBridgeTurtleBot3EnvCfg(
    LidarShortNavDefectSingleObstacleSymmetricGapTurtleBot3EnvCfg
):
    """Bridge task: final obstacle geometry, but waypoint-style goal to transfer the bypass prior."""

    defect_scene_mode: str = "single_obstacle_symmetric_gap"
    defect_single_gap_obstacle_size_x: float = 0.60
    defect_single_gap_obstacle_size_y: float = 1.00
    defect_single_gap_goal_mode: str = "fixed_local"
    defect_single_gap_goal_local_x: float = -0.05
    defect_single_gap_goal_local_y: float = 0.62
    defect_single_gap_goal_local_jitter_x: float = 0.03
    defect_single_gap_goal_local_jitter_y: float = 0.04
    defect_dwb_obstacle_x_jitter: float = 0.0
    defect_single_gap_obstacle_y_jitter: float = 0.0
    defect_goal_distance_min: float = 1.6
    defect_goal_distance_max: float = 2.4

    defect_dwb_curriculum_goal_distance_mins: tuple[float, ...] = (1.8, 2.0, 2.2)
    defect_dwb_curriculum_goal_distance_maxs: tuple[float, ...] = (2.4, 2.6, 2.8)
    defect_dwb_curriculum_goal_reach_thresholds: tuple[float, ...] = (0.70, 0.65, 0.60)
    defect_dwb_curriculum_corridor_widths: tuple[float, ...] = (3.0, 2.8, 2.6)
    defect_dwb_curriculum_v_maxes: tuple[float, ...] = (0.36, 0.38, 0.40)
    defect_dwb_curriculum_progress_reward_scales: tuple[float, ...] = (9.0, 9.5, 10.0)
    defect_dwb_curriculum_forward_goal_reward_scales: tuple[float, ...] = (1.00, 1.00, 0.95)
    defect_dwb_gap_alignment_reward_scale: float = 9.0
    defect_dwb_gap_clear_bonus: float = 24.0
    defect_single_gap_centerline_penalty_scale: float = -0.08
    defect_single_gap_centerline_penalty_forward_dist: float = 0.45
    defect_dwb_side_commit_reward_scale: float = 1.8
    defect_dwb_no_progress_terminate_steps: int = 60
    subgoal_reward_enable: bool = False


@configclass
class LidarShortNavDefectSingleObstacleRearOffsetBridgeTurtleBot3EnvCfg(
    LidarShortNavDefectSingleObstacleSymmetricGapTurtleBot3EnvCfg
):
    """Bridge task: final obstacle geometry with a waypoint fixed just behind the upper bypass side."""

    defect_scene_mode: str = "single_obstacle_symmetric_gap"
    defect_single_gap_obstacle_size_x: float = 0.60
    defect_single_gap_obstacle_size_y: float = 1.00
    defect_single_gap_goal_mode: str = "fixed_local"
    defect_single_gap_goal_local_x: float = 0.42
    defect_single_gap_goal_local_y: float = 0.62
    defect_single_gap_goal_local_jitter_x: float = 0.03
    defect_single_gap_goal_local_jitter_y: float = 0.04
    defect_single_gap_spawn_obstacle_clearance: float = 0.65
    defect_dwb_obstacle_x_jitter: float = 0.0
    defect_single_gap_obstacle_y_jitter: float = 0.0
    defect_goal_distance_min: float = 1.9
    defect_goal_distance_max: float = 2.6

    defect_dwb_curriculum_goal_distance_mins: tuple[float, ...] = (2.0, 2.1, 2.2)
    defect_dwb_curriculum_goal_distance_maxs: tuple[float, ...] = (2.6, 2.8, 3.0)
    defect_dwb_curriculum_goal_reach_thresholds: tuple[float, ...] = (0.70, 0.65, 0.60)
    defect_dwb_curriculum_corridor_widths: tuple[float, ...] = (3.0, 2.8, 2.6)
    defect_dwb_curriculum_v_maxes: tuple[float, ...] = (0.36, 0.38, 0.40)
    defect_dwb_curriculum_progress_reward_scales: tuple[float, ...] = (9.0, 9.5, 10.0)
    defect_dwb_curriculum_forward_goal_reward_scales: tuple[float, ...] = (1.00, 1.00, 0.95)
    defect_dwb_gap_alignment_reward_scale: float = 9.0
    defect_dwb_gap_clear_bonus: float = 24.0
    defect_single_gap_centerline_penalty_scale: float = -0.08
    defect_single_gap_centerline_penalty_forward_dist: float = 0.45
    defect_dwb_side_commit_reward_scale: float = 1.8
    defect_dwb_no_progress_terminate_steps: int = 60
    # Keep Stage-1 bypass stabilization from Stage 11, but with reduced strength for exploration.
    yaw_persistence_stage_indices: tuple[int, ...] = (1,)
    yaw_persistence_reward_scale: float = 0.02
    yaw_flip_penalty_scale: float = 0.005
    subgoal_reward_enable: bool = False


@configclass
class LidarShortNavDefectSingleObstacleRearCenteringBridgeTurtleBot3EnvCfg(
    LidarShortNavDefectSingleObstacleSymmetricGapTurtleBot3EnvCfg
):
    """Bridge task: keep the final obstacle geometry while shrinking the rear-side target back toward center."""

    defect_scene_mode: str = "single_obstacle_symmetric_gap"
    defect_single_gap_obstacle_size_x: float = 0.60
    defect_single_gap_obstacle_size_y: float = 1.00
    defect_single_gap_goal_mode: str = "fixed_local"
    defect_single_gap_goal_local_x: float = 0.42
    defect_single_gap_goal_local_y: float = 0.35
    defect_single_gap_goal_local_jitter_x: float = 0.03
    defect_single_gap_goal_local_jitter_y: float = 0.02
    defect_single_gap_spawn_obstacle_clearance: float = 0.65
    defect_dwb_obstacle_x_jitter: float = 0.0
    defect_single_gap_obstacle_y_jitter: float = 0.0
    defect_goal_distance_min: float = 1.9
    defect_goal_distance_max: float = 2.6

    defect_dwb_curriculum_goal_distance_mins: tuple[float, ...] = (2.0, 2.1, 2.2)
    defect_dwb_curriculum_goal_distance_maxs: tuple[float, ...] = (2.6, 2.8, 3.0)
    defect_dwb_curriculum_goal_reach_thresholds: tuple[float, ...] = (0.70, 0.65, 0.60)
    defect_dwb_curriculum_corridor_widths: tuple[float, ...] = (3.0, 2.8, 2.6)
    defect_dwb_curriculum_v_maxes: tuple[float, ...] = (0.36, 0.38, 0.40)
    defect_dwb_curriculum_progress_reward_scales: tuple[float, ...] = (9.0, 9.5, 10.0)
    defect_dwb_curriculum_forward_goal_reward_scales: tuple[float, ...] = (1.00, 1.00, 0.95)
    defect_dwb_curriculum_goal_local_xs: tuple[float, ...] = (0.42, 0.42, 0.42)
    defect_dwb_curriculum_goal_local_ys: tuple[float, ...] = (0.35, 0.18, 0.00)
    defect_dwb_curriculum_goal_local_jitter_xs: tuple[float, ...] = (0.03, 0.02, 0.01)
    defect_dwb_curriculum_goal_local_jitter_ys: tuple[float, ...] = (0.03, 0.02, 0.01)
    defect_dwb_gap_alignment_reward_scale: float = 9.0
    defect_dwb_gap_clear_bonus: float = 24.0
    defect_single_gap_centerline_penalty_scale: float = -0.08
    defect_single_gap_centerline_penalty_forward_dist: float = 0.45
    defect_dwb_side_commit_reward_scale: float = 1.8
    defect_dwb_no_progress_terminate_steps: int = 60
    # Keep turn-direction persistence active during bridge adaptation (stage-1 for this checkpoint range).
    yaw_persistence_stage_indices: tuple[int, ...] = (1,)
    yaw_persistence_reward_scale: float = 0.02
    yaw_flip_penalty_scale: float = 0.005
    subgoal_reward_enable: bool = False


@configclass
class LidarShortNavDefectSingleObstacleRearTransitionTurtleBot3EnvCfg(
    LidarShortNavDefectSingleObstacleSymmetricGapTurtleBot3EnvCfg
):
    """Bridge task: stage from easy rear-relative goals into the true rear-goal geometry."""

    defect_scene_mode: str = "single_obstacle_symmetric_gap"
    defect_single_gap_obstacle_size_x: float = 0.60
    defect_single_gap_obstacle_size_y: float = 1.00
    defect_single_gap_goal_mode: str = "rear_bridge"
    defect_single_gap_goal_bridge_clearance_min: float = 0.30
    defect_single_gap_goal_bridge_lateral_center: float = 0.20
    defect_single_gap_goal_bridge_lateral_offset_max: float = 0.02
    defect_single_gap_goal_rear_clearance_min: float = 0.80
    defect_single_gap_goal_rear_lateral_center: float = 0.62
    defect_single_gap_goal_rear_lateral_offset_max: float = 0.04
    # Reduce false lidar collisions from rear-side obstacle corners during exit turns.
    defect_single_gap_rear_collision_front_sector_only: bool = True
    defect_single_gap_rear_collision_front_sector_deg: float = 180.0
    defect_single_gap_goal_obstacle_debug: bool = True
    defect_single_gap_goal_obstacle_debug_every: int = 200
    defect_single_gap_spawn_obstacle_clearance: float = 0.68
    defect_dwb_obstacle_x_jitter: float = 0.02
    defect_single_gap_obstacle_y_jitter: float = 0.02
    defect_goal_distance_min: float = 2.0
    defect_goal_distance_max: float = 3.2

    defect_dwb_curriculum_use_relative_steps: bool = True
    defect_dwb_curriculum_stage_steps: tuple[int, ...] = (0, 10_000, 20_000, 30_000, 40_000, 50_000)
    defect_dwb_curriculum_goal_distance_mins: tuple[float, ...] = (2.0, 2.0, 2.0, 2.0, 2.1, 2.2)
    defect_dwb_curriculum_goal_distance_maxs: tuple[float, ...] = (2.7, 2.8, 2.8, 2.9, 2.9, 3.0)
    defect_dwb_curriculum_goal_reach_thresholds: tuple[float, ...] = (0.70, 0.69, 0.68, 0.67, 0.66, 0.64)
    defect_dwb_curriculum_corridor_widths: tuple[float, ...] = (3.0, 3.0, 3.0, 3.0, 2.9, 2.8)
    defect_dwb_curriculum_v_maxes: tuple[float, ...] = (0.35, 0.35, 0.35, 0.35, 0.35, 0.34)
    defect_dwb_curriculum_time_penalties: tuple[float, ...] = (-0.07, -0.07, -0.07, -0.07, -0.07, -0.07)
    defect_dwb_curriculum_progress_reward_scales: tuple[float, ...] = (4.5, 4.45, 4.4, 4.3, 4.2, 4.0)
    defect_dwb_curriculum_heading_reward_scales: tuple[float, ...] = (0.03, 0.03, 0.03, 0.03, 0.03, 0.03)
    defect_dwb_curriculum_forward_goal_reward_scales: tuple[float, ...] = (0.25, 0.245, 0.24, 0.23, 0.22, 0.20)
    defect_dwb_curriculum_forward_speed_reward_scales: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    defect_dwb_curriculum_speed_surge_scales: tuple[float, ...] = (5.0, 5.0, 5.0, 5.0, 5.0, 5.0)
    defect_dwb_curriculum_speed_deficit_scales: tuple[float, ...] = (5.0, 5.0, 5.0, 5.0, 5.0, 5.0)
    defect_dwb_curriculum_stall_penalties: tuple[float, ...] = (-0.80, -0.80, -0.80, -0.80, -0.80, -0.80)
    defect_dwb_curriculum_stall_speed_thresholds: tuple[float, ...] = (0.15, 0.15, 0.15, 0.15, 0.15, 0.15)
    defect_dwb_curriculum_success_bonuses: tuple[float, ...] = (100.0, 100.0, 100.0, 100.0, 100.0, 100.0)
    defect_dwb_curriculum_collision_penalties: tuple[float, ...] = (-30.0, -30.0, -30.0, -30.0, -30.0, -30.0)
    defect_dwb_curriculum_goal_modes: tuple[str, ...] = (
        "rear_bridge",
        "rear_bridge",
        "rear_bridge",
        "rear_bridge",
        "rear",
        "rear",
    )
    defect_dwb_curriculum_goal_bridge_clearance_mins: tuple[float, ...] = (0.30, 0.35, 0.40, 0.45, 0.45, 0.45)
    defect_dwb_curriculum_goal_bridge_lateral_centers: tuple[float, ...] = (0.20, 0.22, 0.25, 0.28, 0.28, 0.28)
    defect_dwb_curriculum_goal_bridge_lateral_offset_maxs: tuple[float, ...] = (0.02, 0.02, 0.02, 0.02, 0.02, 0.02)
    defect_dwb_curriculum_goal_rear_clearance_mins: tuple[float, ...] = (0.40, 0.40, 0.40, 0.45, 0.50, 0.60)
    defect_dwb_curriculum_goal_rear_lateral_centers: tuple[float, ...] = (0.30, 0.30, 0.30, 0.28, 0.30, 0.45)
    defect_dwb_curriculum_goal_rear_lateral_offset_maxs: tuple[float, ...] = (0.02, 0.02, 0.02, 0.02, 0.02, 0.03)
    defect_dwb_gap_alignment_reward_scale: float = 9.0
    defect_dwb_gap_clear_bonus: float = 24.0
    defect_single_gap_centerline_penalty_scale: float = -0.08
    defect_single_gap_centerline_penalty_forward_dist: float = 0.45
    defect_dwb_side_commit_reward_scale: float = 1.8
    defect_dwb_no_progress_terminate_steps: int = 60
    yaw_persistence_stage_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    yaw_persistence_reward_scale: float = 0.02
    yaw_flip_penalty_scale: float = 0.005
    subgoal_reward_enable: bool = False


@configclass
class LidarShortNavDefectSingleObstacleRearFixedTransitionTurtleBot3EnvCfg(
    LidarShortNavDefectSingleObstacleSymmetricGapTurtleBot3EnvCfg
):
    """Bridge task: fixed-local rear-side waypoints before switching to dynamic rear goals."""

    episode_length_s = 22.0
    defect_scene_mode: str = "single_obstacle_symmetric_gap"
    defect_single_gap_obstacle_size_x: float = 0.60
    defect_single_gap_obstacle_size_y: float = 1.00
    defect_single_gap_goal_mode: str = "fixed_local"
    defect_single_gap_goal_local_x: float = 0.45
    defect_single_gap_goal_local_y: float = 0.80
    defect_single_gap_goal_local_jitter_x: float = 0.03
    defect_single_gap_goal_local_jitter_y: float = 0.02
    defect_single_gap_spawn_obstacle_clearance: float = 0.68
    defect_dwb_obstacle_x_jitter: float = 0.02
    defect_single_gap_obstacle_y_jitter: float = 0.00
    defect_goal_distance_min: float = 2.0
    defect_goal_distance_max: float = 3.2
    defect_single_gap_goal_obstacle_debug: bool = True
    defect_single_gap_goal_obstacle_debug_every: int = 200

    # Start from the stage11_final fixed_local depth (goal_local_x=0.45), push the target deeper,
    # then probe two branches: first co-retreat x/y to find the centering boundary, then hold the
    # stable goal_local=(0.66, 0.38) while expanding the goal-distance range. After confirming the
    # far-distance branch up to goal_d=[2.8, 3.8], use a priority-x bridge: keep the same y-offset,
    # temporarily relax the distance range to [2.6, 3.6], ramp goal_local_x from 0.74 to 0.80, then
    # recover the distance range back to [2.8, 3.8]. Once x=0.80 is stabilized at the restored
    # distance range, repeat the same strategy for the 1.00 target: first ramp x from 0.82 to 1.00
    # at [2.6, 3.6], then restore the goal-distance range back to [2.8, 3.8] while keeping x=1.00.
    defect_dwb_curriculum_use_relative_steps: bool = True
    defect_dwb_curriculum_stage_steps: tuple[int, ...] = (
        0,
        10_000,
        20_000,
        30_000,
        40_000,
        50_000,
        60_000,
        64_000,
        68_000,
        72_000,
        76_000,
        80_000,
        84_000,
        88_000,
        92_000,
        96_000,
        100_000,
        104_000,
        108_000,
        112_000,
        116_000,
        120_000,
        124_000,
        128_000,
        132_000,
        136_000,
        140_000,
        144_000,
        148_000,
        152_000,
        156_000,
        160_000,
        164_000,
        168_000,
        172_000,
        176_000,
        180_000,
        184_000,
        188_000,
        192_000,
        196_000,
        200_000,
        204_000,
        208_000,
        212_000,
        216_000,
        220_000,
        224_000,
        228_000,
    )
    defect_dwb_curriculum_goal_distance_mins: tuple[float, ...] = (
        2.0,
        2.0,
        2.05,
        2.10,
        2.15,
        2.20,
        2.20,
        2.20,
        2.20,
        2.20,
        2.20,
        2.20,
        2.20,
        2.20,
        2.20,
        2.30,
        2.40,
        2.50,
        2.60,
        2.70,
        2.80,
        2.80,
        2.80,
        2.80,
        2.80,
        2.80,
        2.60,
        2.60,
        2.60,
        2.60,
        2.60,
        2.60,
        2.60,
        2.70,
        2.80,
        2.80,
        2.80,
        2.80,
        2.80,
        2.80,
        2.80,
        2.80,
        2.80,
        2.80,
        2.80,
        2.80,
        2.80,
        2.80,
        2.80,
    )
    defect_dwb_curriculum_goal_distance_maxs: tuple[float, ...] = (
        2.7,
        2.8,
        2.9,
        3.0,
        3.05,
        3.10,
        3.20,
        3.20,
        3.20,
        3.20,
        3.20,
        3.20,
        3.20,
        3.20,
        3.20,
        3.30,
        3.40,
        3.50,
        3.60,
        3.70,
        3.80,
        3.80,
        3.80,
        3.80,
        3.80,
        3.80,
        3.60,
        3.60,
        3.60,
        3.60,
        3.60,
        3.60,
        3.60,
        3.70,
        3.80,
        3.80,
        3.80,
        3.80,
        3.80,
        3.80,
        3.80,
        3.80,
        3.80,
        3.80,
        3.80,
        3.80,
        3.80,
        3.80,
        3.80,
    )
    defect_dwb_curriculum_goal_reach_thresholds: tuple[float, ...] = (
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
        0.70,
    )
    defect_dwb_curriculum_corridor_widths: tuple[float, ...] = (3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0)
    defect_dwb_curriculum_v_maxes: tuple[float, ...] = (0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35)
    defect_dwb_curriculum_time_penalties: tuple[float, ...] = (
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.07,
        -0.06,
        -0.06,
        -0.06,
        -0.055,
        -0.055,
        -0.05,
        -0.05,
        -0.05,
        -0.05,
    )
    defect_dwb_curriculum_progress_reward_scales: tuple[float, ...] = (4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5)
    defect_dwb_curriculum_heading_reward_scales: tuple[float, ...] = (0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03)
    defect_dwb_curriculum_forward_goal_reward_scales: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25)
    defect_dwb_curriculum_forward_speed_reward_scales: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    defect_dwb_curriculum_speed_surge_scales: tuple[float, ...] = (
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        3.0,
        3.0,
        2.8,
        2.6,
        2.4,
        2.2,
        2.0,
        2.0,
        2.0,
    )
    defect_dwb_curriculum_speed_deficit_scales: tuple[float, ...] = (
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        2.5,
        2.3,
        2.1,
        1.9,
        1.8,
        1.7,
        1.6,
        1.5,
        1.5,
    )
    defect_dwb_curriculum_stall_penalties: tuple[float, ...] = (
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.80,
        -0.55,
        -0.50,
        -0.48,
        -0.45,
        -0.42,
        -0.40,
        -0.38,
        -0.35,
        -0.35,
    )
    defect_dwb_curriculum_stall_speed_thresholds: tuple[float, ...] = (
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.14,
        0.14,
        0.14,
        0.13,
        0.13,
        0.12,
        0.12,
        0.12,
        0.12,
    )
    defect_dwb_curriculum_success_bonuses: tuple[float, ...] = (100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0)
    defect_dwb_curriculum_collision_penalties: tuple[float, ...] = (-25.0, -25.0, -25.0, -25.0, -25.0, -25.0, -25.0)
    defect_dwb_curriculum_goal_modes: tuple[str, ...] = (
        "fixed_local",
        "fixed_local",
        "fixed_local",
        "fixed_local",
        "fixed_local",
        "fixed_local",
        "fixed_local",
    )
    defect_dwb_curriculum_goal_local_xs: tuple[float, ...] = (
        0.45,
        0.50,
        0.55,
        0.60,
        0.66,
        0.75,
        0.85,
        0.78,
        0.71,
        0.64,
        0.58,
        0.53,
        0.58,
        0.62,
        0.66,
        0.66,
        0.66,
        0.66,
        0.66,
        0.66,
        0.66,
        0.70,
        0.71,
        0.72,
        0.73,
        0.74,
        0.74,
        0.75,
        0.76,
        0.77,
        0.78,
        0.79,
        0.80,
        0.80,
        0.80,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
    )
    defect_dwb_curriculum_goal_local_ys: tuple[float, ...] = (
        0.80,
        0.80,
        0.80,
        0.80,
        0.80,
        0.80,
        0.80,
        0.68,
        0.56,
        0.45,
        0.35,
        0.26,
        0.30,
        0.34,
        0.38,
        0.38,
        0.38,
        0.38,
        0.38,
        0.38,
        0.38,
        0.38,
        0.38,
        0.38,
        0.38,
        0.38,
        0.38,
        0.38,
        0.38,
        0.38,
        0.38,
        0.38,
        0.38,
        0.38,
        0.38,
        0.50,
        0.50,
        0.48,
        0.48,
        0.46,
        0.44,
        0.435,
        0.43,
        0.425,
        0.42,
        0.415,
        0.41,
        0.405,
        0.40,
    )
    defect_dwb_curriculum_goal_local_jitter_xs: tuple[float, ...] = (0.03, 0.03, 0.03, 0.025, 0.02, 0.02, 0.02)
    defect_dwb_curriculum_goal_local_jitter_ys: tuple[float, ...] = (0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02)
    defect_dwb_gap_alignment_reward_scale: float = 9.0
    defect_dwb_post_gap_lateral_align_reward_scale: float = 0.0
    defect_dwb_post_gap_lateral_align_deadband: float = 0.04
    defect_dwb_post_gap_lateral_align_x_margin: float = 0.10
    defect_dwb_curriculum_post_gap_lateral_align_reward_scales: tuple[float, ...] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        5.5,
        6.0,
    )
    defect_dwb_gap_clear_bonus: float = 24.0
    defect_single_gap_centerline_penalty_scale: float = -0.08
    defect_single_gap_centerline_penalty_forward_dist: float = 0.45
    defect_dwb_side_commit_reward_scale: float = 1.8
    defect_dwb_no_progress_terminate_steps: int = 60
    yaw_persistence_stage_indices: tuple[int, ...] = (0, 1, 2)
    yaw_persistence_reward_scale: float = 0.02
    yaw_flip_penalty_scale: float = 0.005
    subgoal_reward_enable: bool = False


@configclass
class LidarShortNavDefectSingleObstacleFinalCurriculumTurtleBot3EnvCfg(
    LidarShortNavDefectSingleObstacleSymmetricGapTurtleBot3EnvCfg
):
    """Final-task finetune: keep final geometry, then anneal from rear-side waypoint to true rear goal."""

    episode_length_s = 22.0
    defect_scene_mode: str = "single_obstacle_symmetric_gap"
    defect_single_gap_obstacle_size_x: float = 0.60
    defect_single_gap_obstacle_size_y: float = 1.00
    defect_single_gap_goal_mode: str = "fixed_local"
    defect_single_gap_goal_local_x: float = 0.42
    defect_single_gap_goal_local_y: float = 0.35
    defect_single_gap_goal_local_jitter_x: float = 0.03
    defect_single_gap_goal_local_jitter_y: float = 0.03
    defect_single_gap_goal_rear_clearance_min: float = 0.80
    defect_single_gap_spawn_obstacle_clearance: float = 0.70
    defect_dwb_obstacle_x_jitter: float = 0.04
    defect_single_gap_obstacle_y_jitter: float = 0.02
    defect_goal_distance_min: float = 2.1
    defect_goal_distance_max: float = 3.0

    # Late-stage fixed-local shifts still collapse when geometry and reward shaping change together.
    # Keep adding shorter bridges near the tail so the policy first learns the reward reshaping,
    # then the farther rearward local goal, and only then the final rear target.
    defect_dwb_curriculum_stage_steps: tuple[int, ...] = (
        0,
        80_000,
        180_000,
        360_000,
        520_000,
        760_000,
        900_000,
        1_200_000,
        1_500_000,
        1_800_000,
        2_100_000,
        2_400_000,
        2_700_000,
        3_000_000,
        3_300_000,
    )
    defect_dwb_curriculum_goal_distance_mins: tuple[float, ...] = (2.1, 2.1, 2.1, 2.15, 2.15, 2.15, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2)
    defect_dwb_curriculum_goal_distance_maxs: tuple[float, ...] = (2.8, 2.9, 3.0, 3.0, 3.0, 3.0, 3.10, 3.15, 3.16, 3.18, 3.18, 3.20, 3.22, 3.24, 3.26)
    defect_dwb_curriculum_goal_reach_thresholds: tuple[float, ...] = (
        0.70,
        0.68,
        0.66,
        0.67,
        0.70,
        0.72,
        0.72,
        0.72,
        0.72,
        0.80,
        0.76,
        0.90,
        0.70,
        0.68,
        0.64,
    )
    # Stage 11+ is a survival curriculum: slow or dithering behavior must go net negative.
    defect_dwb_curriculum_time_penalties: tuple[float, ...] = (-0.01, -0.01, -0.01, -0.012, -0.012, -0.012, -0.012, -0.012, -0.012, -0.012, -0.012, -0.05, -0.07, -0.07, -0.07)
    # Push yaw usage cost high enough that lingering in oscillatory steering becomes a losing strategy.
    defect_dwb_curriculum_turn_penalty_scales: tuple[float, ...] = (
        -0.08,
        -0.08,
        -0.08,
        -0.08,
        -0.08,
        -0.08,
        -0.08,
        -0.08,
        -0.08,
        -0.08,
        -0.08,
        -0.35,
        -0.35,
        -0.35,
        -0.35,
    )
    # Stage 11+ also raises the effective speed floor: crawling below 0.15 m/s counts as stalling.
    defect_dwb_curriculum_stall_speed_thresholds: tuple[float, ...] = (
        0.08,
        0.08,
        0.08,
        0.08,
        0.08,
        0.08,
        0.08,
        0.08,
        0.08,
        0.08,
        0.08,
        0.15,
        0.15,
        0.15,
        0.15,
    )
    defect_dwb_curriculum_lateral_penalty_weights: tuple[float, ...] = (-0.35, -0.40, -0.40, -0.38, -0.36, -0.30, -0.22, -0.22, -0.20, -0.20, -0.20, -0.19, -0.18, -0.18, -0.18)
    defect_dwb_curriculum_passage_bonuses: tuple[float, ...] = (10.0, 9.0, 9.0, 8.9, 8.9, 8.9, 8.7, 8.7, 8.7, 8.7, 8.7, 8.7, 8.5, 8.4, 8.3)
    defect_dwb_curriculum_gap_alignment_reward_scales: tuple[float, ...] = (9.0, 9.0, 8.0, 7.5, 7.0, 4.5, 4.5, 4.5, 4.0, 4.0, 4.0, 3.8, 3.0, 2.0, 0.0)
    defect_dwb_curriculum_corridor_widths: tuple[float, ...] = (3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.95, 2.9)
    defect_dwb_curriculum_v_maxes: tuple[float, ...] = (0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.34, 0.34)
    defect_dwb_curriculum_progress_reward_scales: tuple[float, ...] = (9.0, 9.1, 9.2, 9.25, 9.35, 9.35, 9.40, 9.40, 9.40, 9.42, 9.42, 4.0, 4.0, 4.0, 4.0)
    defect_dwb_curriculum_heading_reward_scales: tuple[float, ...] = (0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.03, 0.03, 0.03, 0.03)
    defect_dwb_curriculum_forward_goal_reward_scales: tuple[float, ...] = (1.00, 1.00, 1.00, 0.97, 0.97, 0.96, 0.94, 0.94, 0.94, 0.94, 0.94, 0.20, 0.20, 0.20, 0.20)
    defect_dwb_curriculum_forward_speed_reward_scales: tuple[float, ...] = (0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.00, 0.00, 0.00, 0.00)
    defect_dwb_curriculum_speed_surge_scales: tuple[float, ...] = (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 5.00, 7.50, 6.00, 4.00)
    defect_dwb_curriculum_speed_deficit_scales: tuple[float, ...] = (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 2.00, 5.00, 5.00, 5.00)
    defect_dwb_curriculum_stall_penalties: tuple[float, ...] = (-0.14, -0.14, -0.13, -0.13, -0.12, -0.12, -0.12, -0.12, -0.12, -0.12, -0.12, -0.80, -0.80, -0.80, -0.80)
    defect_dwb_curriculum_success_bonuses: tuple[float, ...] = (40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 100.0, 100.0, 100.0, 100.0)
    defect_dwb_curriculum_collision_penalties: tuple[float, ...] = (-40.0, -40.0, -40.0, -40.0, -40.0, -40.0, -40.0, -40.0, -40.0, -40.0, -40.0, -45.0, -15.0, -25.0, -40.0)
    defect_dwb_curriculum_speed_cap_explores: tuple[float, ...] = (
        0.32,
        0.32,
        0.32,
        0.32,
        0.32,
        0.32,
        0.32,
        0.32,
        0.32,
        0.32,
        0.32,
        0.32,
        0.32,
        0.31,
        0.30,
    )
    defect_dwb_curriculum_speed_cap_narrows: tuple[float, ...] = (
        0.26,
        0.26,
        0.26,
        0.26,
        0.26,
        0.26,
        0.26,
        0.26,
        0.26,
        0.32,
        0.32,
        0.28,
        0.32,
        0.28,
        0.26,
    )
    defect_dwb_curriculum_front_danger_distances: tuple[float, ...] = (
        0.80,
        0.80,
        0.80,
        0.80,
        0.80,
        0.80,
        0.80,
        0.80,
        0.80,
        0.80,
        0.80,
        0.80,
        0.80,
        0.90,
        1.00,
    )
    defect_dwb_curriculum_danger_speed_penalty_scales: tuple[float, ...] = (
        -0.15,
        -0.15,
        -0.15,
        -0.15,
        -0.15,
        -0.15,
        -0.15,
        -0.15,
        -0.15,
        -0.08,
        -0.08,
        -0.16,
        -0.12,
        -0.18,
        -0.25,
    )
    defect_dwb_curriculum_spawn_jitters: tuple[float, ...] = (0.03, 0.04, 0.04, 0.05, 0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.07, 0.07, 0.08)
    defect_dwb_curriculum_goal_jitters: tuple[float, ...] = (0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.05, 0.05, 0.06)
    defect_dwb_curriculum_obstacle_x_jitters: tuple[float, ...] = (0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.035, 0.035, 0.04)
    defect_dwb_curriculum_obstacle_x_centers: tuple[float, ...] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    defect_dwb_curriculum_goal_modes: tuple[str, ...] = (
        "fixed_local",
        "fixed_local",
        "fixed_local",
        "fixed_local",
        "fixed_local",
        "fixed_local",
        "fixed_local",
        "fixed_local",
        "fixed_local",
        "fixed_local",
        "fixed_local",
        "fixed_local",
        "rear",
        "rear",
        "rear",
    )
    defect_dwb_curriculum_goal_local_xs: tuple[float, ...] = (0.42, 0.42, 0.42, 0.42, 0.42, 0.44, 0.44, 0.44, 0.44, 0.45, 0.45, 0.45, 0.48, 0.52, 0.66)
    defect_dwb_curriculum_goal_local_ys: tuple[float, ...] = (0.35, 0.24, 0.18, 0.16, 0.15, 0.14, 0.14, 0.13, 0.13, 0.13, 0.13, 0.12, 0.06, 0.05, 0.03)
    defect_dwb_curriculum_goal_local_jitter_xs: tuple[float, ...] = (0.03, 0.025, 0.02, 0.02, 0.015, 0.012, 0.012, 0.010, 0.010, 0.010, 0.008, 0.008, 0.006, 0.004, 0.00)
    defect_dwb_curriculum_goal_local_jitter_ys: tuple[float, ...] = (0.03, 0.025, 0.02, 0.02, 0.015, 0.012, 0.012, 0.010, 0.010, 0.010, 0.008, 0.008, 0.006, 0.004, 0.00)

    defect_dwb_gap_alignment_reward_scale: float = 9.0
    defect_dwb_gap_clear_bonus: float = 24.0
    defect_single_gap_centerline_penalty_scale: float = -0.08
    defect_single_gap_centerline_penalty_forward_dist: float = 0.45
    defect_dwb_side_commit_reward_scale: float = 1.8
    defect_dwb_no_progress_terminate_steps: int = 60
    # Keep directional commitment active at late fixed-local (stage 11) and rear-goal stages.
    yaw_persistence_stage_indices: tuple[int, ...] = (11, 12, 13, 14)
    yaw_persistence_reward_scale: float = 0.02
    yaw_flip_penalty_scale: float = 0.005
    subgoal_reward_enable: bool = False


@configclass
class LidarShortNavDefectSingleObstacleStage12TransitionTurtleBot3EnvCfg(
    LidarShortNavDefectSingleObstacleFinalCurriculumTurtleBot3EnvCfg
):
    """Safer Stage-12 hard-cut variant with stronger collision cost and milder surge incentive."""

    defect_dwb_curriculum_collision_penalties: tuple[float, ...] = (
        -40.0,
        -40.0,
        -40.0,
        -40.0,
        -40.0,
        -40.0,
        -40.0,
        -40.0,
        -40.0,
        -40.0,
        -40.0,
        -15.0,
        -30.0,
        -30.0,
        -40.0,
    )
    defect_dwb_curriculum_speed_surge_scales: tuple[float, ...] = (
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        7.50,
        5.00,
        5.00,
        4.00,
    )


@configclass
class LidarShortNavDefectMppiCornerFailTurtleBot3EnvCfg(LidarShortNavV18ATurtleBot3EnvCfg):
    """Defect scene: L-shaped narrow corner with rounded inner radius and inner-corner blocker."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=20.0, replicate_physics=True)
    planner_state_mixed_training_enable: bool = False
    junction_enable: bool = False
    corridor_enable: bool = False
    linear_open_area_prob: float = 0.0
    relay_goal_enable: bool = False

    forward_only: bool = False
    defect_scene_enable: bool = True
    defect_scene_mode: str = "mppi_corner_fail"
    defect_scene_visualize_walls: bool = True
    defect_wall_segment_capacity: int = 128
    defect_spawn_jitter_xy: float = 0.2
    defect_goal_jitter_xy: float = 0.2
    defect_goal_distance_min: float = 1.0
    defect_goal_distance_max: float = 3.0

    num_obstacles: int = 1
    obstacle_radius: float = 0.125
    defect_corner_corridor_width: float = 0.8
    defect_corner_inner_radius: float = 0.5
    defect_corner_inner_radius_jitter: float = 0.05
    defect_corner_straight_pre_length: float = 1.0
    defect_corner_straight_post_length: float = 1.5
    defect_corner_inner_block_enable: bool = True
    defect_corner_inner_block_size: float = 0.25
    defect_corner_inner_block_offset: float = 0.17
    # Curriculum stages ease geometry first, then recover target difficulty.
    defect_mppi_curriculum_enable: bool = True
    defect_mppi_curriculum_stage_steps: tuple[int, ...] = (0, 2_000_000, 3_000_000)
    # Lower threshold is easier here because collision is triggered by min_lidar < threshold.
    defect_mppi_curriculum_collision_thresholds: tuple[float, ...] = (0.14, 0.15, 0.20)
    defect_mppi_curriculum_corridor_widths: tuple[float, ...] = (1.15, 1.10, 0.95)
    defect_mppi_curriculum_inner_radii: tuple[float, ...] = (0.80, 0.76, 0.65)
    defect_mppi_curriculum_inner_radius_jitters: tuple[float, ...] = (0.01, 0.015, 0.03)
    defect_mppi_curriculum_spawn_jitters: tuple[float, ...] = (0.03, 0.04, 0.10)
    defect_mppi_curriculum_goal_jitters: tuple[float, ...] = (0.03, 0.04, 0.10)
    defect_mppi_curriculum_straight_pre_lengths: tuple[float, ...] = (1.5, 1.45, 1.3)
    defect_mppi_curriculum_straight_post_lengths: tuple[float, ...] = (2.0, 1.95, 1.8)
    defect_mppi_curriculum_goal_distance_mins: tuple[float, ...] = (1.2, 1.15, 1.0)
    defect_mppi_curriculum_goal_distance_maxs: tuple[float, ...] = (2.0, 2.1, 2.5)
    defect_mppi_curriculum_goal_y_min_margins: tuple[float, ...] = (0.10, 0.12, 0.20)
    defect_mppi_curriculum_goal_y_max_margins: tuple[float, ...] = (0.05, 0.06, 0.10)
    defect_mppi_curriculum_goal_reach_thresholds: tuple[float, ...] = (0.50, 0.50, 0.47)
    defect_mppi_curriculum_forward_goal_reward_scales: tuple[float, ...] = (0.60, 0.58, 0.52)
    defect_mppi_curriculum_near_goal_progress_mult_mids: tuple[float, ...] = (1.20, 1.18, 1.10)
    defect_mppi_curriculum_near_goal_progress_mult_closes: tuple[float, ...] = (1.50, 1.45, 1.25)
    defect_mppi_curriculum_near_goal_fwd_mult_mids: tuple[float, ...] = (1.20, 1.18, 1.10)
    defect_mppi_curriculum_near_goal_fwd_mult_closes: tuple[float, ...] = (1.60, 1.50, 1.30)
    defect_mppi_curriculum_inner_block_sizes: tuple[float, ...] = (0.18, 0.19, 0.22)
    defect_mppi_curriculum_inner_block_offsets: tuple[float, ...] = (0.20, 0.20, 0.18)
    # Stage 0 uses higher actionable speed; front-sector slowdown keeps turn safety.
    defect_mppi_curriculum_v_maxes: tuple[float, ...] = (0.25, 0.27, 0.35)
    defect_mppi_curriculum_speed_cap_explores: tuple[float, ...] = (0.22, 0.22, 0.24)
    defect_mppi_curriculum_speed_cap_turns: tuple[float, ...] = (0.20, 0.20, 0.22)
    defect_mppi_curriculum_speed_cap_narrows: tuple[float, ...] = (0.18, 0.18, 0.20)
    defect_mppi_curriculum_speed_cap_narrow_clearances: tuple[float, ...] = (0.26, 0.28, 0.34)
    defect_mppi_curriculum_front_danger_distances: tuple[float, ...] = (0.50, 0.52, 0.60)
    defect_mppi_curriculum_danger_speed_penalty_scales: tuple[float, ...] = (-0.5, -0.6, -1.0)
    defect_mppi_curriculum_progress_reward_scales: tuple[float, ...] = (14.0, 13.0, 10.0)
    defect_mppi_curriculum_time_penalties: tuple[float, ...] = (-0.015, -0.017, -0.025)
    defect_mppi_curriculum_turn_completion_bonuses: tuple[float, ...] = (18.0, 16.0, 12.0)
    # Keep dense progress shaping scale consistent for 1~3 m goals.
    progress_reward_scale: float = 8.0
    # Align sparse terminal bonus with door-scene training target.
    success_bonus: float = 40.0
    # Milestone: one-shot bonus after corner completion.
    defect_turn_completion_bonus: float = 15.0
    # Goal placement margins in post-corner branch (from center_radius).
    defect_corner_goal_y_min_margin: float = 0.30
    defect_corner_goal_y_max_margin: float = 0.20
    # In corner scenes, speed slowdown should look at front-sector clearance instead of global min_lidar.
    defect_mppi_front_obstacle_slowdown_enable: bool = True
    defect_mppi_front_obstacle_slowdown_half_fov_deg: float = 45.0
    # Dense progress hint for the corner-exit section (guarded in reward code to avoid loitering exploits).
    defect_corner_mid_progress_reward: float = 0.10
    defect_corner_mid_progress_x_min: float = 1.0
    defect_corner_mid_progress_x_max: float = 2.0


@configclass
class LidarShortNavDefectDoorDeadlockTurtleBot3EnvCfg(LidarShortNavV18ATurtleBot3EnvCfg):
    """Defect scene: door bottleneck with 0.8 m opening and 1.5 m pre/post straight segments."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=20.0, replicate_physics=True)
    planner_state_mixed_training_enable: bool = False
    junction_enable: bool = False
    corridor_enable: bool = False
    linear_open_area_prob: float = 0.0
    relay_goal_enable: bool = False

    forward_only: bool = False
    defect_scene_enable: bool = True
    defect_scene_mode: str = "door_deadlock"
    defect_scene_visualize_walls: bool = True
    defect_wall_segment_capacity: int = 80
    defect_spawn_jitter_xy: float = 0.2

    num_obstacles: int = 2
    defect_door_corridor_width: float = 2.0
    defect_door_half_length: float = 3.0
    defect_door_width: float = 0.8
    defect_door_width_jitter: float = 0.05
    defect_door_straight_distance: float = 1.5
    defect_door_line_angle_max_deg: float = 15.0
    # Allow the goal to be laterally offset after the robot passes the doorway.
    # This breaks the strict collinear layout and trains the policy to turn after exiting.
    defect_door_goal_lateral_offset_max: float = 0.0
    # After clearing the doorway, cap forward speed if the goal is still laterally offset.
    # This reduces post-door overshoot when the robot needs to bend back toward the goal.
    defect_door_post_pass_lateral_slowdown_enable: bool = True
    defect_door_post_pass_x_min: float = 0.10
    defect_door_post_pass_goal_lateral_threshold: float = 0.16
    defect_door_post_pass_speed_cap: float = 0.11
    # Keep dense progress shaping scale consistent for 1~3 m goals.
    progress_reward_scale: float = 8.0
    # Sparse terminal bonus baseline for defect tasks.
    success_bonus: float = 40.0
    # Milestone: reward successful door-slot traversal.
    defect_narrow_passage_bonus: float = 2.0


@configclass
class LidarShortNavDefectUShapeTrapTurtleBot3EnvCfg(LidarShortNavV18ATurtleBot3EnvCfg):
    """Defect scene: U-shape trap with narrow opening, deep pocket, and outside goal."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=22.0, replicate_physics=True)
    planner_state_mixed_training_enable: bool = False
    junction_enable: bool = False
    corridor_enable: bool = False
    linear_open_area_prob: float = 0.0
    relay_goal_enable: bool = False

    forward_only: bool = False
    defect_scene_enable: bool = True
    defect_scene_mode: str = "u_shape_trap"
    defect_scene_visualize_walls: bool = True
    defect_wall_segment_capacity: int = 96
    defect_spawn_jitter_xy: float = 0.2
    defect_goal_jitter_xy: float = 0.2
    defect_goal_distance_min: float = 1.0
    defect_goal_distance_max: float = 3.0

    num_obstacles: int = 2
    defect_u_opening_width: float = 0.6
    defect_u_opening_width_jitter: float = 0.05
    defect_u_depth: float = 2.7
    defect_u_inner_width: float = 2.0
    defect_u_goal_outside_x: float = 1.8
    defect_u_arena_half_width: float = 2.6
    # Curriculum for gradual hardening in u_shape_trap defect scene.
    defect_ushape_curriculum_enable: bool = True
    defect_ushape_curriculum_stage_steps: tuple[int, ...] = (0, 1_000_000, 2_500_000)
    defect_ushape_curriculum_collision_thresholds: tuple[float, ...] = (0.20, 0.23, 0.25)
    defect_ushape_curriculum_goal_distance_maxs: tuple[float, ...] = (2.5, 2.8, 3.0)
    defect_ushape_curriculum_time_penalties: tuple[float, ...] = (-0.02, -0.03, -0.04)
    defect_ushape_curriculum_escape_bonuses: tuple[float, ...] = (20.0, 20.0, 20.0)
    defect_ushape_curriculum_exploration_bonuses: tuple[float, ...] = (0.01, 0.005, 0.002)
    defect_ushape_curriculum_backward_encouragements: tuple[float, ...] = (0.05, 0.03, 0.01)
    defect_ushape_curriculum_opening_widths: tuple[float, ...] = (0.70, 0.65, 0.60)
    defect_ushape_curriculum_inner_widths: tuple[float, ...] = (2.20, 2.10, 2.00)
    defect_ushape_curriculum_v_maxes: tuple[float, ...] = (0.30, 0.40, 0.50)
    # Keep dense progress shaping scale consistent for 1~3 m goals.
    progress_reward_scale: float = 8.0
    # Align sparse terminal bonus with door-scene training target.
    success_bonus: float = 40.0
    # Milestone: reward leaving U-shape trap.
    defect_u_escape_bonus: float = 20.0
