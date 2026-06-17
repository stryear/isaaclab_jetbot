# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from .isaac_lab_tutorial_env_cfg import IsaacLabTutorialEnvCfg, LidarNavTurtleBot3EnvCfg, SphereFollowEnvCfg

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)

class IsaacLabTutorialEnv(DirectRLEnv):
    cfg: IsaacLabTutorialEnvCfg

    def __init__(self, cfg: IsaacLabTutorialEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.visualization_markers = define_markers()

        self.up_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()  
        self.yaws = torch.zeros((self.cfg.scene.num_envs, 1)).cuda()
        self.commands = torch.randn((self.cfg.scene.num_envs, 3)).cuda()
        self.commands[:,-1] = 0.0
        self.commands = self.commands/torch.linalg.norm(self.commands, dim=1, keepdim=True)
        
        # offsets to account for atan range and keep things on [-pi, pi]
        ratio = self.commands[:,1]/(self.commands[:,0]+1E-8)
        gzero = torch.where(self.commands > 0, True, False)
        lzero = torch.where(self.commands < 0, True, False)
        plus = lzero[:,0]*gzero[:,1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        self.yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

        self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset[:,-1] = 0.5
        self.forward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
        self.command_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
        

    def _visualize_markers(self):
        self.marker_locations = self.robot.data.root_pos_w
        self.forward_marker_orientations = self.robot.data.root_quat_w
        self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()

        loc = self.marker_locations + self.marker_offset
        loc = torch.vstack((loc, loc))
        rots = torch.vstack((self.forward_marker_orientations, self.command_marker_orientations))

        all_envs = torch.arange(self.cfg.scene.num_envs)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))

        self.visualization_markers.visualize(loc, rots, marker_indices=indices)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()# + torch.ones_like(actions)
        self._visualize_markers()

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        self.velocity = self.robot.data.root_com_vel_w 
        self.forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)
        # obs = torch.hstack((self.velocity, self.commands))

        dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        cross = torch.cross(self.forwards, self.commands, dim=-1)[:,-1].reshape(-1,1)
        forward_speed = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
        obs = torch.hstack((dot, cross, forward_speed))
        
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        forward_reward = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
        alignment_reward = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        total_reward = forward_reward + alignment_reward
        # total_reward = forward_reward*alignment_reward
        # total_reward = forward_reward*alignment_reward + forward_reward
        # total_reward = forward_reward*torch.exp(alignment_reward)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return False, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self.commands[env_ids] = torch.randn((len(env_ids), 3)).cuda()
        self.commands[env_ids,-1] = 0.0
        self.commands[env_ids] = self.commands[env_ids]/torch.linalg.norm(self.commands[env_ids], dim=1, keepdim=True)
        
        ratio = self.commands[env_ids][:,1]/(self.commands[env_ids][:,0]+1E-8)
        gzero = torch.where(self.commands[env_ids] > 0, True, False)
        lzero = torch.where(self.commands[env_ids]< 0, True, False)
        plus = lzero[:,0]*gzero[:,1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        self.yaws[env_ids] = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_state_to_sim(default_root_state, env_ids)
        self._visualize_markers()


def define_sphere_marker(
    radius: float = 0.1,
    color: tuple[float, float, float] = (0.0, 1.0, 0.0),
    prim_path: str = "/Visuals/sphereMarkers",
) -> VisualizationMarkers:
    """Define a colored sphere marker."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


def define_cylinder_marker(
    radius: float = 0.1,
    height: float = 0.45,
    color: tuple[float, float, float] = (1.0, 0.2, 0.2),
    prim_path: str = "/Visuals/cylinderMarkers",
) -> VisualizationMarkers:
    """Define a colored cylinder marker."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "cylinder": sim_utils.CylinderCfg(
                radius=radius,
                height=height,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


def define_cuboid_marker(
    size: tuple[float, float, float] = (0.2, 0.2, 0.45),
    color: tuple[float, float, float] = (1.0, 0.2, 0.2),
    prim_path: str = "/Visuals/cuboidMarkers",
) -> VisualizationMarkers:
    """Define a colored cuboid marker."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "cuboid": sim_utils.CuboidCfg(
                size=size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


def define_corridor_wall_marker(
    color: tuple[float, float, float] = (0.55, 0.55, 0.60),
    prim_path: str = "/Visuals/lidarNavCorridorWallMarkers",
) -> VisualizationMarkers:
    """Define a cuboid marker used to visualize corridor walls."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "wall": sim_utils.CuboidCfg(
                size=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


class SphereFollowEnv(DirectRLEnv):
    cfg: SphereFollowEnvCfg

    def __init__(self, cfg: SphereFollowEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # sphere marker
        self.sphere_markers = define_sphere_marker(self.cfg.sphere_radius)

        # sphere positions in world frame (one per env)
        self.sphere_pos_w = torch.zeros((self.cfg.scene.num_envs, 3), device=self.device)
        self.prev_dist_to_sphere = torch.zeros((self.cfg.scene.num_envs,), device=self.device)

    def _spawn_sphere_positions(self, env_ids: torch.Tensor, robot_pos_xy: torch.Tensor | None = None) -> None:
        """Spawn spheres at random positions relative to the robot.

        Args:
            env_ids: Environment indices to respawn spheres for.
            robot_pos_xy: Optional (N, 2) tensor of robot XY positions. If None,
                reads from sim data (only valid mid-episode, NOT right after reset).
        """
        num = len(env_ids)
        # random angle and distance
        angles = torch.rand(num, device=self.device) * 2.0 * math.pi
        dists = (
            torch.rand(num, device=self.device)
            * (self.cfg.sphere_spawn_range_max - self.cfg.sphere_spawn_range_min)
            + self.cfg.sphere_spawn_range_min
        )
        # use provided position or read from sim
        if robot_pos_xy is None:
            robot_pos_xy = self.robot.data.root_pos_w[env_ids, :2]
        offset_x = dists * torch.cos(angles)
        offset_y = dists * torch.sin(angles)
        self.sphere_pos_w[env_ids, 0] = robot_pos_xy[:, 0] + offset_x
        self.sphere_pos_w[env_ids, 1] = robot_pos_xy[:, 1] + offset_y
        self.sphere_pos_w[env_ids, 2] = self.cfg.sphere_radius

    def _visualize_sphere(self):
        """Update sphere marker positions."""
        orientations = torch.zeros((self.cfg.scene.num_envs, 4), device=self.device)
        orientations[:, 0] = 1.0  # identity quaternion (w, x, y, z)
        self.sphere_markers.visualize(self.sphere_pos_w, orientations)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self._visualize_sphere()

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        # forward direction of robot
        forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)
        # direction to sphere (XY plane)
        dir_to_sphere = self.sphere_pos_w - self.robot.data.root_pos_w
        dir_to_sphere[:, 2] = 0.0
        dist = torch.linalg.norm(dir_to_sphere[:, :2], dim=1, keepdim=True).clamp(min=1e-6)
        dir_to_sphere_norm = dir_to_sphere / dist

        # dot product: +1 facing sphere, -1 facing away
        dot = torch.sum(forwards * dir_to_sphere_norm, dim=-1, keepdim=True)
        # cross product Z component: sign indicates turn direction
        cross_z = (forwards[:, 0] * dir_to_sphere_norm[:, 1] - forwards[:, 1] * dir_to_sphere_norm[:, 0]).unsqueeze(-1)
        # normalized distance
        dist_norm = (dist / self.cfg.sphere_max_distance).clamp(0.0, 1.0)
        # forward speed in body frame
        forward_speed = self.robot.data.root_com_lin_vel_b[:, 0].unsqueeze(-1)

        obs = torch.cat((dot, cross_z, dist_norm, forward_speed), dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # current distance to sphere
        diff = self.sphere_pos_w[:, :2] - self.robot.data.root_pos_w[:, :2]
        curr_dist = torch.linalg.norm(diff, dim=1)

        # approach reward: positive when getting closer
        approach = (self.prev_dist_to_sphere - curr_dist) * self.cfg.approach_reward_scale

        # alignment reward
        forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)
        dir_to_sphere = self.sphere_pos_w - self.robot.data.root_pos_w
        dir_to_sphere[:, 2] = 0.0
        dist_for_norm = torch.linalg.norm(dir_to_sphere[:, :2], dim=1, keepdim=True).clamp(min=1e-6)
        dir_to_sphere_norm = dir_to_sphere / dist_for_norm
        dot = torch.sum(forwards * dir_to_sphere_norm, dim=-1)
        alignment = dot * self.cfg.alignment_reward_scale

        # reach bonus: sphere reached, reposition it
        reached = curr_dist < self.cfg.sphere_reach_threshold
        reach = reached.float() * self.cfg.reach_bonus

        # reposition sphere for envs that reached it
        reached_ids = torch.where(reached)[0]
        if len(reached_ids) > 0:
            self._spawn_sphere_positions(reached_ids)
            # reset prev_dist for repositioned spheres to avoid reward spike
            new_diff = self.sphere_pos_w[reached_ids, :2] - self.robot.data.root_pos_w[reached_ids, :2]
            curr_dist[reached_ids] = torch.linalg.norm(new_diff, dim=1)

        # time penalty
        time_pen = torch.full_like(curr_dist, self.cfg.time_penalty)

        # update prev_dist
        self.prev_dist_to_sphere = curr_dist.clone()

        total_reward = approach + alignment + reach + time_pen
        return total_reward.unsqueeze(-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return False, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # reset robot pose
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_state_to_sim(default_root_state, env_ids)

        # use the known reset position (sim data is stale until next step)
        env_ids_tensor = torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids
        robot_reset_pos_xy = default_root_state[:, :2]

        # spawn new sphere positions relative to the reset position
        self._spawn_sphere_positions(env_ids_tensor, robot_pos_xy=robot_reset_pos_xy)

        # init prev_dist using the known reset position
        diff = self.sphere_pos_w[env_ids, :2] - robot_reset_pos_xy
        self.prev_dist_to_sphere[env_ids] = torch.linalg.norm(diff, dim=1)

        self._visualize_sphere()


class LidarNavTurtleBot3Env(DirectRLEnv):
    """TurtleBot3 obstacle-avoidance navigation with virtual 2D LiDAR observations."""

    PLANNER_STATE_NAMES = ("explore", "fallback", "junction", "return")

    TURN_LEFT = 0
    TURN_STRAIGHT = 1
    TURN_RIGHT = 2

    STATE_EXPLORE = 0
    STATE_FALLBACK = 1
    STATE_JUNCTION = 2
    STATE_RETURN = 3

    TOPO_L = 0
    TOPO_T = 1
    TOPO_X = 2

    ARM_W = 0
    ARM_E = 1
    ARM_N = 2
    ARM_S = 3

    DEFECT_DWB_OSCILLATION = 0
    DEFECT_MPPI_CORNER_FAIL = 1
    DEFECT_DOOR_DEADLOCK = 2
    DEFECT_U_SHAPE_TRAP = 3
    DEFECT_SINGLE_OBSTACLE_SYMMETRIC_GAP = 4

    cfg: LidarNavTurtleBot3EnvCfg

    def __init__(self, cfg: LidarNavTurtleBot3EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)

    def _get_defect_scene_mode(self) -> str:
        return str(getattr(self.cfg, "defect_scene_mode", "")).strip().lower()

    def _is_dwb_like_scene_mode(self, scene_mode: str | None = None) -> bool:
        mode = self._get_defect_scene_mode() if scene_mode is None else str(scene_mode).strip().lower()
        return mode in ("dwb_oscillation", "single_obstacle_symmetric_gap")

    def _use_box_obstacles(self) -> bool:
        return self._get_defect_scene_mode() == "single_obstacle_symmetric_gap"

    def _get_dwb_like_scene_mask(self, scene_idx: torch.Tensor) -> torch.Tensor:
        return torch.logical_or(
            scene_idx == self.DEFECT_DWB_OSCILLATION,
            scene_idx == self.DEFECT_SINGLE_OBSTACLE_SYMMETRIC_GAP,
        )

    def _get_box_obstacle_half_extents(self) -> tuple[float, float]:
        default_half_extent = float(getattr(self.cfg, "obstacle_radius", 0.2))
        size_x = float(getattr(self.cfg, "defect_single_gap_obstacle_size_x", 2.0 * default_half_extent))
        size_y = float(getattr(self.cfg, "defect_single_gap_obstacle_size_y", 2.0 * default_half_extent))
        return 0.5 * max(1.0e-3, size_x), 0.5 * max(1.0e-3, size_y)

    def _get_box_obstacle_half_extent_x(self) -> float:
        return self._get_box_obstacle_half_extents()[0]

    def _get_box_obstacle_half_extent_y(self) -> float:
        return self._get_box_obstacle_half_extents()[1]

    def _get_single_obstacle_symmetric_gap_target_abs_y(self) -> float:
        half_width = 0.5 * max(0.6, float(getattr(self.cfg, "defect_dwb_corridor_width", 2.0)))
        obstacle_radius = self._get_box_obstacle_half_extent_y() if self._use_box_obstacles() else float(
            getattr(self.cfg, "obstacle_radius", 0.2)
        )
        target_margin = max(0.02, float(getattr(self.cfg, "defect_single_gap_target_lateral_margin", 0.12)))
        return min(max(obstacle_radius + target_margin, 0.05), half_width - 0.05)

    def _project_single_gap_goal_outside_obstacle(
        self,
        goal_x: float,
        goal_y: float,
        obstacle_x: float,
        obstacle_y: float,
        half_extent_x: float,
        half_extent_y: float,
        goal_x_min: float,
        goal_x_max: float,
        goal_y_min: float,
        goal_y_max: float,
    ) -> tuple[float, float]:
        goal_clearance = float(getattr(self.cfg, "goal_radius", 0.10)) + max(
            0.02,
            float(getattr(self.cfg, "defect_single_gap_goal_clearance_margin", 0.04)),
        )
        inflated_half_x = half_extent_x + goal_clearance
        inflated_half_y = half_extent_y + goal_clearance

        if abs(goal_x - obstacle_x) > inflated_half_x or abs(goal_y - obstacle_y) > inflated_half_y:
            return goal_x, goal_y

        clamped_goal_x = min(max(goal_x, goal_x_min), goal_x_max)
        clamped_goal_y = min(max(goal_y, goal_y_min), goal_y_max)
        candidates: list[tuple[float, float, float]] = []

        def maybe_add(candidate_x: float, candidate_y: float):
            candidate_x = min(max(candidate_x, goal_x_min), goal_x_max)
            candidate_y = min(max(candidate_y, goal_y_min), goal_y_max)
            if abs(candidate_x - obstacle_x) <= inflated_half_x and abs(candidate_y - obstacle_y) <= inflated_half_y:
                return
            delta_sq = (candidate_x - clamped_goal_x) ** 2 + (candidate_y - clamped_goal_y) ** 2
            candidates.append((delta_sq, candidate_x, candidate_y))

        # Prefer keeping the goal in front of the obstacle; if that is blocked by the corridor bound,
        # move it to the closest free side in y.
        maybe_add(obstacle_x + inflated_half_x, clamped_goal_y)
        maybe_add(clamped_goal_x, obstacle_y + inflated_half_y)
        maybe_add(clamped_goal_x, obstacle_y - inflated_half_y)
        maybe_add(obstacle_x + inflated_half_x, obstacle_y + inflated_half_y)
        maybe_add(obstacle_x + inflated_half_x, obstacle_y - inflated_half_y)

        if candidates:
            _, projected_x, projected_y = min(candidates, key=lambda item: item[0])
            return projected_x, projected_y

        fallback_y = goal_y_max if abs(goal_y_max - obstacle_y) >= abs(goal_y_min - obstacle_y) else goal_y_min
        return clamped_goal_x, fallback_y

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add physical obstacles as rigid kinematic colliders (optional)
        self.obstacle_rigid_collection: RigidObjectCollection | None = None
        if bool(getattr(self.cfg, "physical_obstacles_enable", False)):
            rigid_props = sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=bool(getattr(self.cfg, "obstacle_kinematic", True)),
                disable_gravity=True,
                linear_damping=0.0,
                angular_damping=0.0,
            )
            collision_props = sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=float(getattr(self.cfg, "obstacle_contact_offset", 0.01)),
                rest_offset=float(getattr(self.cfg, "obstacle_rest_offset", 0.0)),
            )
            physics_material = sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=float(getattr(self.cfg, "obstacle_static_friction", 1.0)),
                dynamic_friction=float(getattr(self.cfg, "obstacle_dynamic_friction", 0.8)),
                restitution=float(getattr(self.cfg, "obstacle_restitution", 0.0)),
            )
            if self._use_box_obstacles():
                half_extent_x, half_extent_y = self._get_box_obstacle_half_extents()
                obstacle_spawn_cfg = sim_utils.CuboidCfg(
                    size=(2.0 * half_extent_x, 2.0 * half_extent_y, float(self.cfg.obstacle_height)),
                    rigid_props=rigid_props,
                    collision_props=collision_props,
                    physics_material=physics_material,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
                )
            else:
                obstacle_spawn_cfg = sim_utils.CylinderCfg(
                    radius=float(self.cfg.obstacle_radius),
                    height=float(self.cfg.obstacle_height),
                    axis="Z",
                    rigid_props=rigid_props,
                    collision_props=collision_props,
                    physics_material=physics_material,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
                )
            obstacle_cfgs: dict[str, RigidObjectCfg] = {}
            for obs_idx in range(int(self.cfg.num_obstacles)):
                obstacle_cfgs[f"obstacle_{obs_idx}"] = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/Obstacle_{obs_idx}",
                    spawn=obstacle_spawn_cfg,
                    # keep them out of view until first reset writes sampled positions
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
                )
            self.obstacle_rigid_collection = RigidObjectCollection(
                cfg=RigidObjectCollectionCfg(rigid_objects=obstacle_cfgs)
            )
            self.scene.rigid_object_collections["obstacles"] = self.obstacle_rigid_collection

        # contact sensor for physical collision force (filtered to obstacle rigid bodies)
        self._contact_sensor: ContactSensor | None = None
        if getattr(self.cfg, "contact_sensor", None) is not None:
            self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
            self.scene.sensors["contact_sensor"] = self._contact_sensor
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # markers
        self.goal_markers = define_sphere_marker(
            radius=self.cfg.goal_radius,
            color=(0.0, 1.0, 0.0),
            prim_path="/Visuals/lidarNavGoalMarkers",
        )
        if self._use_box_obstacles():
            half_extent_x, half_extent_y = self._get_box_obstacle_half_extents()
            self.obstacle_markers = define_cuboid_marker(
                size=(2.0 * half_extent_x, 2.0 * half_extent_y, float(self.cfg.obstacle_height)),
                color=(1.0, 0.2, 0.2),
                prim_path="/Visuals/lidarNavObstacleMarkers",
            )
        else:
            self.obstacle_markers = define_cylinder_marker(
                radius=self.cfg.obstacle_radius,
                height=self.cfg.obstacle_height,
                color=(1.0, 0.2, 0.2),
                prim_path="/Visuals/lidarNavObstacleMarkers",
            )
        self.corridor_wall_markers = None
        if bool(getattr(self.cfg, "corridor_enable", False)) and bool(
            getattr(self.cfg, "corridor_visualize_walls", False)
        ):
            self.corridor_wall_markers = define_corridor_wall_marker()
        self.junction_wall_markers = None
        junction_marker_enable = bool(getattr(self.cfg, "junction_enable", False)) and bool(
            getattr(self.cfg, "junction_visualize_walls", False)
        )
        defect_marker_enable = bool(getattr(self.cfg, "defect_scene_enable", False)) and bool(
            getattr(self.cfg, "defect_scene_visualize_walls", False)
        )
        if junction_marker_enable or defect_marker_enable:
            self.junction_wall_markers = define_corridor_wall_marker(
                color=(0.45, 0.50, 0.55),
                prim_path="/Visuals/lidarNavJunctionWallMarkers",
            )

        # state tensors
        num_envs = self.cfg.scene.num_envs
        self.goal_pos_w = torch.zeros((num_envs, 3), device=self.device)
        self.obstacle_pos_w = torch.zeros((num_envs, self.cfg.num_obstacles, 3), device=self.device)
        self.prev_goal_dist = torch.zeros((num_envs,), device=self.device)
        self.subgoal_pos_w = torch.zeros((num_envs, 3), device=self.device)
        self.prev_subgoal_dist = torch.zeros((num_envs,), device=self.device)
        self._subgoal_reached = torch.ones((num_envs,), dtype=torch.bool, device=self.device)
        self._subgoal_reward_enable = bool(getattr(self.cfg, "subgoal_reward_enable", False))
        subgoal_xy_cfg = getattr(self.cfg, "subgoal_xy", (0.0, 0.5))
        if not isinstance(subgoal_xy_cfg, (list, tuple)) or len(subgoal_xy_cfg) < 2:
            subgoal_xy_cfg = (0.0, 0.5)
        self._subgoal_local_xy = torch.tensor(
            [float(subgoal_xy_cfg[0]), float(subgoal_xy_cfg[1])],
            dtype=torch.float32,
            device=self.device,
        )
        self.prev_actions = torch.zeros((num_envs, self.cfg.action_space), device=self.device)
        self.actions = torch.zeros((num_envs, self.cfg.action_space), device=self.device)
        self.filtered_v_cmd = torch.zeros((num_envs,), device=self.device)
        self.filtered_omega_cmd = torch.zeros((num_envs,), device=self.device)
        self.prev_wheel_targets = torch.zeros((num_envs, self.cfg.action_space), device=self.device)
        self.latest_lidar_ranges = torch.full(
            (num_envs, self.cfg.lidar_num_beams),
            self.cfg.lidar_max_range,
            device=self.device,
        )
        self._lidar_frame_stack = max(1, int(getattr(self.cfg, "lidar_frame_stack", 1)))
        lidar_obs_reset_value = 1.0 if bool(getattr(self.cfg, "use_lidar_normalization", False)) else float(self.cfg.lidar_max_range)
        self.lidar_frame_buffer = torch.full(
            (num_envs, self._lidar_frame_stack, self.cfg.lidar_num_beams),
            lidar_obs_reset_value,
            device=self.device,
        )
        self.latest_contact_force = torch.zeros((num_envs,), device=self.device)
        self.episode_return = torch.zeros((num_envs,), device=self.device)
        self.prev_min_lidar = torch.full((num_envs,), self.cfg.lidar_max_range, device=self.device)
        self.prev_yaw_rate = torch.zeros((num_envs,), device=self.device)
        self._prev_yaw_sign = torch.zeros((num_envs,), device=self.device)
        self._yaw_flip_counter = torch.zeros((num_envs,), device=self.device)
        self._yaw_flip_step_counter = torch.zeros((num_envs,), device=self.device)
        self.return_no_progress_steps = torch.zeros((num_envs,), dtype=torch.int32, device=self.device)
        self._defect_dwb_no_progress_steps = torch.zeros((num_envs,), dtype=torch.int32, device=self.device)
        # Used by ROS2 bridge and diagnostics.
        self.latest_goal_pos_w = self.goal_pos_w

        # command-conditioned navigation state (junction mode).
        self.turn_cmd_index = torch.full((num_envs,), self.TURN_STRAIGHT, dtype=torch.long, device=self.device)
        self.turn_cmd_onehot = torch.zeros((num_envs, 3), device=self.device)
        self.turn_cmd_dir_b = torch.zeros((num_envs, 2), device=self.device)
        self.turn_cmd_dir_b[:, 0] = 1.0
        self.planner_state_index = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        self.planner_state_onehot = torch.zeros((num_envs, 4), device=self.device)
        self.planner_state_onehot[:, 0] = 1.0
        self.gravity_vec_w = torch.zeros((num_envs, 3), device=self.device)
        self.gravity_vec_w[:, 2] = -1.0
        self.control_dt = max(1.0e-6, float(self.cfg.decimation) * float(self.cfg.sim.dt))
        self.junction_type = torch.full((num_envs,), self.TOPO_X, dtype=torch.long, device=self.device)
        self.junction_active_arms = torch.zeros((num_envs, 4), dtype=torch.bool, device=self.device)
        self.junction_active_arms[:, self.ARM_W] = True
        self.junction_active_arms[:, self.ARM_E] = True
        self.junction_active_arms[:, self.ARM_N] = True
        self.junction_active_arms[:, self.ARM_S] = True
        self._junction_max_segments = max(24, int(getattr(self.cfg, "defect_wall_segment_capacity", 24)))
        self._junction_wall_seg_start = torch.zeros((num_envs, self._junction_max_segments, 2), device=self.device)
        self._junction_wall_seg_end = torch.zeros((num_envs, self._junction_max_segments, 2), device=self.device)
        self._junction_wall_seg_valid = torch.zeros((num_envs, self._junction_max_segments), dtype=torch.bool, device=self.device)
        default_half_width = float(
            self.cfg.junction_half_width if bool(getattr(self.cfg, "junction_enable", False)) else self.cfg.corridor_half_width
        )
        default_half_length = float(
            self.cfg.junction_half_length if bool(getattr(self.cfg, "junction_enable", False)) else self.cfg.corridor_half_length
        )
        self.layout_half_width = torch.full((num_envs,), default_half_width, device=self.device)
        self.layout_half_length = torch.full((num_envs,), default_half_length, device=self.device)
        self.layout_open_area = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)

        fov_deg = float(self.cfg.lidar_fov_deg)
        if fov_deg >= 359.999:
            # Avoid duplicated rear-facing endpoint at -pi/+pi for full 360° scans.
            self.lidar_beam_angles = torch.linspace(
                -math.pi,
                math.pi,
                self.cfg.lidar_num_beams + 1,
                device=self.device,
            )[:-1]
        else:
            self.lidar_beam_angles = torch.linspace(
                -math.radians(fov_deg) / 2.0,
                math.radians(fov_deg) / 2.0,
                self.cfg.lidar_num_beams,
                device=self.device,
            )
        front_sector_half_rad = math.radians(float(getattr(self.cfg, "front_lidar_sector_deg", 60.0))) / 2.0
        self.front_lidar_beam_mask = torch.abs(self.lidar_beam_angles) <= front_sector_half_rad
        if not bool(torch.any(self.front_lidar_beam_mask).item()):
            self.front_lidar_beam_mask = torch.zeros_like(self.lidar_beam_angles, dtype=torch.bool)
            self.front_lidar_beam_mask[self.cfg.lidar_num_beams // 2] = True
        self.left_lidar_beam_mask = self.lidar_beam_angles > front_sector_half_rad
        if not bool(torch.any(self.left_lidar_beam_mask).item()):
            self.left_lidar_beam_mask = torch.zeros_like(self.lidar_beam_angles, dtype=torch.bool)
            self.left_lidar_beam_mask[min(self.cfg.lidar_num_beams - 1, (self.cfg.lidar_num_beams * 3) // 4)] = True
        self.right_lidar_beam_mask = self.lidar_beam_angles < -front_sector_half_rad
        if not bool(torch.any(self.right_lidar_beam_mask).item()):
            self.right_lidar_beam_mask = torch.zeros_like(self.lidar_beam_angles, dtype=torch.bool)
            self.right_lidar_beam_mask[max(0, self.cfg.lidar_num_beams // 4)] = True

        # per-env latest termination context (used for terminal metrics in _reset_idx)
        self._last_done_success = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        self._last_done_collision = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        self._last_done_collision_lidar = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        self._last_done_collision_contact = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        self._last_done_timeout = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        self._last_done_goal_dist = torch.zeros((num_envs,), device=self.device)
        self._last_done_min_lidar = torch.full((num_envs,), self.cfg.lidar_max_range, device=self.device)
        self._last_done_planner_state = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        self.fixed_scene_active_index = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self._last_done_fixed_scene_index = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self._defect_narrow_zone_prev_inside = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        self._defect_turn_bonus_given = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        self._defect_corner_seen_turn = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        self._defect_escape_bonus_given = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        self._defect_u_was_inside = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        explore_grid_size = max(0.05, float(getattr(self.cfg, "defect_u_exploration_grid_size", 0.2)))
        u_depth = max(0.5, float(getattr(self.cfg, "defect_u_depth", 2.7)))
        u_goal_x = float(getattr(self.cfg, "defect_u_goal_outside_x", 1.8))
        u_half_w = max(0.5, float(getattr(self.cfg, "defect_u_arena_half_width", 2.6)))
        self._defect_u_grid_cell_size = explore_grid_size
        self._defect_u_grid_x_min = -u_depth - 1.5
        self._defect_u_grid_x_max = max(2.5, u_goal_x + 1.5)
        self._defect_u_grid_y_min = -u_half_w - 0.5
        self._defect_u_grid_y_max = u_half_w + 0.5
        grid_nx = max(
            8,
            int(math.ceil((self._defect_u_grid_x_max - self._defect_u_grid_x_min) / self._defect_u_grid_cell_size)) + 1,
        )
        grid_ny = max(
            8,
            int(math.ceil((self._defect_u_grid_y_max - self._defect_u_grid_y_min) / self._defect_u_grid_cell_size)) + 1,
        )
        self._defect_u_explore_visited = torch.zeros((num_envs, grid_nx, grid_ny), dtype=torch.bool, device=self.device)
        # Latest per-step reward terms and predicates for external evaluation scripts.
        self._last_step_turn_cmd_reward = torch.zeros((num_envs,), device=self.device)
        self._last_step_turn_penalty = torch.zeros((num_envs,), device=self.device)
        self._last_step_stall = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        self._last_step_action_saturated = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        self._last_step_progress_delta = torch.zeros((num_envs,), device=self.device)

        # rolling terminal metrics for human-readable training diagnostics
        self._metrics_total_episodes = 0
        self._metrics_window_episodes = 0
        self._metrics_window_success = 0
        self._metrics_window_collision = 0
        self._metrics_window_collision_lidar = 0
        self._metrics_window_collision_contact = 0
        self._metrics_window_timeout = 0
        self._metrics_window_episode_length_sum = 0.0
        self._metrics_window_return_sum = 0.0
        self._metrics_window_goal_dist_sum = 0.0
        self._metrics_window_min_lidar_sum = 0.0
        self._metrics_print_interval_episodes = max(1, int(self.cfg.metrics_print_interval_episodes))

        # step-level debug statistics (rolling average)
        self._debug_print_enable = bool(getattr(self.cfg, "debug_print_enable", False))
        self._debug_print_interval_steps = max(1, int(getattr(self.cfg, "debug_print_interval_steps", 500)))
        self._debug_iteration_rollouts = max(1, int(getattr(self.cfg, "debug_iteration_rollouts", 64)))
        self._reset_debug_interval_resets = max(1, int(getattr(self.cfg, "reset_debug_interval_resets", 128)))
        self._debug_phase_tag = int(getattr(self.cfg, "debug_phase_tag", 0))
        self._debug_window_steps = 0
        self._debug_reward_sum = 0.0
        self._debug_progress_sum = 0.0
        self._debug_heading_sum = 0.0
        self._debug_forward_goal_sum = 0.0
        self._debug_speed_surge_sum = 0.0
        self._debug_speed_deficit_sum = 0.0
        self._debug_clearance_sum = 0.0
        self._debug_smooth_sum = 0.0
        self._debug_turn_penalty_sum = 0.0
        self._debug_time_penalty_sum = 0.0
        self._debug_stall_penalty_sum = 0.0
        self._debug_danger_speed_penalty_sum = 0.0
        self._debug_dwb_clearance_penalty_sum = 0.0
        self._debug_action_sat_penalty_sum = 0.0
        self._debug_single_gap_center_penalty_sum = 0.0
        self._debug_post_gap_lateral_align_sum = 0.0
        self._debug_success_bonus_sum = 0.0
        self._debug_collision_penalty_sum = 0.0
        self._debug_goal_dist_sum = 0.0
        self._debug_min_lidar_sum = 0.0
        self._debug_action_abs_sum = 0.0
        self._debug_action_eff_abs_sum = 0.0
        self._debug_forward_speed_sum = 0.0
        self._debug_yaw_rate_abs_mean_sum = 0.0
        self._debug_yaw_rate_abs_p95_sum = 0.0
        self._debug_filtered_omega_abs_mean_sum = 0.0
        self._debug_yaw_sign_flip_frac_sum = 0.0
        self._debug_success_frac_sum = 0.0
        self._debug_collision_frac_sum = 0.0
        self._debug_collision_lidar_frac_sum = 0.0
        self._debug_collision_contact_frac_sum = 0.0
        self._debug_stall_frac_sum = 0.0
        self._debug_action_sat_frac_sum = 0.0
        self._reset_debug_samples = 0
        self._reset_debug_state_counts = [0, 0, 0, 0]
        self._reset_debug_goal_dist_sum = [0.0, 0.0, 0.0, 0.0]
        self._reset_debug_goal_dist_min = [float("inf"), float("inf"), float("inf"), float("inf")]
        self._reset_debug_goal_dist_max = [0.0, 0.0, 0.0, 0.0]
        self._reset_debug_speed_cap_sum = [0.0, 0.0, 0.0, 0.0]
        self._reset_debug_goal_x_b_sum = [0.0, 0.0, 0.0, 0.0]
        self._reset_debug_goal_y_abs_sum = [0.0, 0.0, 0.0, 0.0]
        self._reset_debug_yaw_deg_sum = [0.0, 0.0, 0.0, 0.0]
        self._reset_debug_door_angle_count = 0
        self._reset_debug_door_angle_deg_sum = 0.0
        self._reset_debug_door_angle_deg_min = float("inf")
        self._reset_debug_door_angle_deg_max = float("-inf")
        self._reset_debug_door_goal_lat_sum = 0.0
        self._reset_debug_door_goal_lat_min = float("inf")
        self._reset_debug_door_goal_lat_max = float("-inf")
        self._single_gap_goal_debug_enable = bool(getattr(self.cfg, "defect_single_gap_goal_obstacle_debug", False))
        self._single_gap_goal_debug_every = max(
            1, int(getattr(self.cfg, "defect_single_gap_goal_obstacle_debug_every", 200))
        )
        self._single_gap_goal_debug_counter = 0
        self._defect_dwb_curriculum_stage = -1
        self._defect_dwb_curriculum_step_origin: int | None = None
        self._defect_dwb_curriculum_step_origin_override: int | None = None
        self._defect_ushape_curriculum_stage = -1
        self._defect_mppi_curriculum_stage = -1
        self._defect_dwb_gap_center_local_x = torch.zeros((num_envs,), device=self.device)
        self._defect_dwb_gap_center_local_y = torch.zeros((num_envs,), device=self.device)
        self._defect_dwb_prev_gap_dist = torch.zeros((num_envs,), device=self.device)
        self._defect_dwb_prev_gap_abs_y = torch.zeros((num_envs,), device=self.device)
        self._defect_dwb_prev_gap_signed_y = torch.zeros((num_envs,), device=self.device)
        self._defect_dwb_prev_post_gap_goal_abs_y = torch.zeros((num_envs,), device=self.device)
        self._defect_dwb_gap_reached = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        self._defect_dwb_gap_clear_bonus_given = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        self._defect_dwb_gap_commit_latched = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        self._apply_defect_dwb_curriculum(force=True)
        self._apply_defect_ushape_curriculum(force=True)
        self._apply_defect_mppi_curriculum(force=True)

        self._visualize_corridor_walls()
        self._visualize_junction_walls()

    def _env_ids_tensor(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.cfg.scene.num_envs, device=self.device)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.tensor(env_ids, device=self.device, dtype=torch.long)

    def _set_planner_state(self, env_ids: torch.Tensor, state: str) -> None:
        state_idx = {
            "explore": self.STATE_EXPLORE,
            "fallback": self.STATE_FALLBACK,
            "junction": self.STATE_JUNCTION,
            "return": self.STATE_RETURN,
        }[state]
        self.planner_state_index[env_ids] = state_idx
        self.planner_state_onehot[env_ids] = 0.0
        self.planner_state_onehot[env_ids, state_idx] = 1.0

    def _use_mixed_planner_state_training(self) -> bool:
        return bool(getattr(self.cfg, "junction_enable", False)) and bool(
            getattr(self.cfg, "planner_state_mixed_training_enable", False)
        )

    def _sample_planner_state_indices(self, env_ids: torch.Tensor) -> torch.Tensor:
        if env_ids.numel() == 0:
            return torch.zeros((0,), dtype=torch.long, device=self.device)
        if not self._use_mixed_planner_state_training():
            default_state = self.STATE_JUNCTION if bool(getattr(self.cfg, "junction_enable", False)) else self.STATE_EXPLORE
            return torch.full((env_ids.numel(),), default_state, dtype=torch.long, device=self.device)

        weights = torch.tensor(
            [
                float(getattr(self.cfg, "planner_state_prob_explore", 0.0)),
                float(getattr(self.cfg, "planner_state_prob_fallback", 0.0)),
                float(getattr(self.cfg, "planner_state_prob_junction", 0.0)),
                float(getattr(self.cfg, "planner_state_prob_return", 0.0)),
            ],
            dtype=torch.float32,
            device=self.device,
        ).clamp(min=0.0)
        weight_sum = float(weights.sum().item())
        if weight_sum <= 0.0:
            weights[self.STATE_JUNCTION] = 1.0
            weight_sum = 1.0
        return torch.multinomial(weights / weight_sum, env_ids.numel(), replacement=True)

    def _clear_turn_cmd(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.turn_cmd_index[env_ids] = self.TURN_STRAIGHT
        self.turn_cmd_onehot[env_ids] = 0.0
        self.turn_cmd_dir_b[env_ids, 0] = 1.0
        self.turn_cmd_dir_b[env_ids, 1] = 0.0

    def _use_fixed_scene_multi_reset(self) -> bool:
        if not bool(getattr(self.cfg, "fixed_scene_multi_enable", False)):
            return False
        return len(self._parse_fixed_scene_multi_scene_params()) > 0

    def _use_fixed_scene_reset(self) -> bool:
        return bool(getattr(self.cfg, "fixed_scene_enable", False)) or self._use_fixed_scene_multi_reset()

    def _use_defect_scene_reset(self) -> bool:
        return bool(getattr(self.cfg, "defect_scene_enable", False))

    def _sample_signed_uniform(self, magnitude: float) -> float:
        mag = max(0.0, float(magnitude))
        if mag <= 0.0:
            return 0.0
        return float((torch.rand(1, device=self.device).item() * 2.0 - 1.0) * mag)

    def _sample_uniform(self, low: float, high: float) -> float:
        lo = float(min(low, high))
        hi = float(max(low, high))
        if hi - lo <= 1.0e-6:
            return lo
        return float(torch.rand(1, device=self.device).item() * (hi - lo) + lo)

    def _sample_defect_goal_distance(self) -> float:
        dist_min = max(0.05, float(getattr(self.cfg, "defect_goal_distance_min", 1.0)))
        dist_max = max(dist_min, float(getattr(self.cfg, "defect_goal_distance_max", dist_min)))
        return self._sample_uniform(dist_min, dist_max)

    def _get_curriculum_value(self, attr_name: str, stage_idx: int, fallback: float) -> float:
        values = getattr(self.cfg, attr_name, ())
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            return float(fallback)
        clamped_stage = min(max(int(stage_idx), 0), len(values) - 1)
        return float(values[clamped_stage])

    def _get_curriculum_choice(self, attr_name: str, stage_idx: int, fallback):
        values = getattr(self.cfg, attr_name, ())
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            return fallback
        clamped_stage = min(max(int(stage_idx), 0), len(values) - 1)
        return values[clamped_stage]

    def _apply_defect_dwb_curriculum(self, force: bool = False) -> None:
        if not bool(getattr(self.cfg, "defect_dwb_curriculum_enable", False)):
            return
        scene_mode = self._get_defect_scene_mode()
        if not self._is_dwb_like_scene_mode(scene_mode):
            return
        stage_steps = getattr(self.cfg, "defect_dwb_curriculum_stage_steps", (0,))
        if not isinstance(stage_steps, (list, tuple)) or len(stage_steps) == 0:
            stage_steps = (0,)
        current_step = int(max(0, int(getattr(self, "common_step_counter", 0))))
        use_relative_steps = bool(getattr(self.cfg, "defect_dwb_curriculum_use_relative_steps", False))
        if use_relative_steps:
            step_origin_override = getattr(self, "_defect_dwb_curriculum_step_origin_override", None)
            if step_origin_override is not None:
                self._defect_dwb_curriculum_step_origin = int(step_origin_override)
                self._defect_dwb_curriculum_step_origin_override = None
            elif self._defect_dwb_curriculum_step_origin is None:
                self._defect_dwb_curriculum_step_origin = current_step
            effective_step = max(0, current_step - int(self._defect_dwb_curriculum_step_origin))
        else:
            if force:
                self._defect_dwb_curriculum_step_origin = None
                self._defect_dwb_curriculum_step_origin_override = None
            effective_step = current_step
        freeze_stage_cfg = int(getattr(self.cfg, "defect_dwb_curriculum_freeze_stage", -1))
        if freeze_stage_cfg >= 0:
            stage_idx = min(max(freeze_stage_cfg, 0), len(stage_steps) - 1)
        else:
            stage_idx = 0
            for idx, start_step in enumerate(stage_steps):
                if effective_step >= int(start_step):
                    stage_idx = idx
                else:
                    break
        if not force and stage_idx == self._defect_dwb_curriculum_stage:
            return

        self.cfg.collision_threshold = self._get_curriculum_value(
            "defect_dwb_curriculum_collision_thresholds", stage_idx, self.cfg.collision_threshold
        )
        self.cfg.contact_collision_force_threshold = self._get_curriculum_value(
            "defect_dwb_curriculum_contact_force_thresholds",
            stage_idx,
            getattr(self.cfg, "contact_collision_force_threshold", 2.0),
        )
        self.cfg.goal_reach_threshold = self._get_curriculum_value(
            "defect_dwb_curriculum_goal_reach_thresholds",
            stage_idx,
            getattr(self.cfg, "goal_reach_threshold", 0.45),
        )
        goal_distance_min = self._get_curriculum_value(
            "defect_dwb_curriculum_goal_distance_mins",
            stage_idx,
            getattr(self.cfg, "defect_goal_distance_min", 1.0),
        )
        goal_distance_min = max(0.05, float(goal_distance_min))
        goal_distance_max = self._get_curriculum_value(
            "defect_dwb_curriculum_goal_distance_maxs",
            stage_idx,
            getattr(self.cfg, "defect_goal_distance_max", goal_distance_min),
        )
        self.cfg.defect_goal_distance_min = goal_distance_min
        self.cfg.defect_goal_distance_max = max(goal_distance_min, goal_distance_max)
        self.cfg.time_penalty = self._get_curriculum_value(
            "defect_dwb_curriculum_time_penalties",
            stage_idx,
            getattr(self.cfg, "time_penalty", -0.03),
        )
        self.cfg.progress_reward_scale = self._get_curriculum_value(
            "defect_dwb_curriculum_progress_reward_scales",
            stage_idx,
            getattr(self.cfg, "progress_reward_scale", 10.0),
        )
        self.cfg.heading_reward_scale = self._get_curriculum_value(
            "defect_dwb_curriculum_heading_reward_scales",
            stage_idx,
            getattr(self.cfg, "heading_reward_scale", 0.10),
        )
        self.cfg.forward_goal_reward_scale = self._get_curriculum_value(
            "defect_dwb_curriculum_forward_goal_reward_scales",
            stage_idx,
            getattr(self.cfg, "forward_goal_reward_scale", 0.70),
        )
        self.cfg.stall_penalty = self._get_curriculum_value(
            "defect_dwb_curriculum_stall_penalties",
            stage_idx,
            getattr(self.cfg, "stall_penalty", -0.06),
        )
        self.cfg.stall_speed_threshold = self._get_curriculum_value(
            "defect_dwb_curriculum_stall_speed_thresholds",
            stage_idx,
            getattr(self.cfg, "stall_speed_threshold", 0.05),
        )
        self.cfg.turn_penalty_scale = self._get_curriculum_value(
            "defect_dwb_curriculum_turn_penalty_scales",
            stage_idx,
            getattr(self.cfg, "turn_penalty_scale", -0.02),
        )
        self.cfg.defect_dwb_forward_speed_reward_scale = self._get_curriculum_value(
            "defect_dwb_curriculum_forward_speed_reward_scales",
            stage_idx,
            getattr(self.cfg, "defect_dwb_forward_speed_reward_scale", 0.0),
        )
        self.cfg.defect_dwb_speed_surge_scale = self._get_curriculum_value(
            "defect_dwb_curriculum_speed_surge_scales",
            stage_idx,
            getattr(self.cfg, "defect_dwb_speed_surge_scale", 0.0),
        )
        self.cfg.defect_dwb_speed_deficit_scale = self._get_curriculum_value(
            "defect_dwb_curriculum_speed_deficit_scales",
            stage_idx,
            getattr(self.cfg, "defect_dwb_speed_deficit_scale", 0.0),
        )
        self.cfg.success_bonus = self._get_curriculum_value(
            "defect_dwb_curriculum_success_bonuses",
            stage_idx,
            getattr(self.cfg, "success_bonus", 40.0),
        )
        self.cfg.collision_penalty = self._get_curriculum_value(
            "defect_dwb_curriculum_collision_penalties",
            stage_idx,
            getattr(self.cfg, "collision_penalty", -40.0),
        )
        self.cfg.speed_cap_explore = self._get_curriculum_value(
            "defect_dwb_curriculum_speed_cap_explores",
            stage_idx,
            getattr(self.cfg, "speed_cap_explore", self.cfg.v_max),
        )
        self.cfg.speed_cap_narrow = self._get_curriculum_value(
            "defect_dwb_curriculum_speed_cap_narrows",
            stage_idx,
            getattr(self.cfg, "speed_cap_narrow", self.cfg.v_max),
        )
        self.cfg.front_danger_distance = self._get_curriculum_value(
            "defect_dwb_curriculum_front_danger_distances",
            stage_idx,
            getattr(self.cfg, "front_danger_distance", getattr(self.cfg, "danger_speed_distance", 0.75)),
        )
        self.cfg.danger_speed_penalty_scale = self._get_curriculum_value(
            "defect_dwb_curriculum_danger_speed_penalty_scales",
            stage_idx,
            getattr(self.cfg, "danger_speed_penalty_scale", -2.0),
        )
        self.cfg.defect_lateral_velocity_penalty_scale = self._get_curriculum_value(
            "defect_dwb_curriculum_lateral_penalty_weights",
            stage_idx,
            getattr(self.cfg, "defect_lateral_velocity_penalty_scale", -0.2),
        )
        self.cfg.defect_narrow_passage_bonus = self._get_curriculum_value(
            "defect_dwb_curriculum_passage_bonuses",
            stage_idx,
            getattr(self.cfg, "defect_narrow_passage_bonus", 2.0),
        )
        self.cfg.defect_dwb_gap_alignment_reward_scale = self._get_curriculum_value(
            "defect_dwb_curriculum_gap_alignment_reward_scales",
            stage_idx,
            getattr(self.cfg, "defect_dwb_gap_alignment_reward_scale", 0.0),
        )
        self.cfg.defect_dwb_post_gap_lateral_align_reward_scale = self._get_curriculum_value(
            "defect_dwb_curriculum_post_gap_lateral_align_reward_scales",
            stage_idx,
            getattr(self.cfg, "defect_dwb_post_gap_lateral_align_reward_scale", 0.0),
        )
        self.cfg.defect_dwb_corridor_width = self._get_curriculum_value(
            "defect_dwb_curriculum_corridor_widths",
            stage_idx,
            getattr(self.cfg, "defect_dwb_corridor_width", 2.0),
        )
        self.cfg.defect_dwb_obstacle_spacing = self._get_curriculum_value(
            "defect_dwb_curriculum_obstacle_spacings",
            stage_idx,
            getattr(self.cfg, "defect_dwb_obstacle_spacing", 0.6),
        )
        self.cfg.defect_spawn_jitter_xy = self._get_curriculum_value(
            "defect_dwb_curriculum_spawn_jitters",
            stage_idx,
            getattr(self.cfg, "defect_spawn_jitter_xy", 0.2),
        )
        self.cfg.defect_goal_jitter_xy = self._get_curriculum_value(
            "defect_dwb_curriculum_goal_jitters",
            stage_idx,
            getattr(self.cfg, "defect_goal_jitter_xy", 0.2),
        )
        self.cfg.defect_dwb_obstacle_pair_shift_y = self._get_curriculum_value(
            "defect_dwb_curriculum_obstacle_pair_shift_ys",
            stage_idx,
            getattr(self.cfg, "defect_dwb_obstacle_pair_shift_y", 0.0),
        )
        self.cfg.defect_dwb_obstacle_x_jitter = self._get_curriculum_value(
            "defect_dwb_curriculum_obstacle_x_jitters",
            stage_idx,
            getattr(self.cfg, "defect_dwb_obstacle_x_jitter", 0.0),
        )
        self.cfg.defect_single_gap_obstacle_x_center = self._get_curriculum_value(
            "defect_dwb_curriculum_obstacle_x_centers",
            stage_idx,
            getattr(self.cfg, "defect_single_gap_obstacle_x_center", 0.0),
        )
        self.cfg.defect_dwb_obstacle_spacing_jitter = self._get_curriculum_value(
            "defect_dwb_curriculum_obstacle_spacing_jitters",
            stage_idx,
            getattr(self.cfg, "defect_dwb_obstacle_spacing_jitter", 0.0),
        )
        self.cfg.v_max = self._get_curriculum_value(
            "defect_dwb_curriculum_v_maxes",
            stage_idx,
            getattr(self.cfg, "v_max", 0.5),
        )
        self.cfg.defect_single_gap_goal_local_x = self._get_curriculum_value(
            "defect_dwb_curriculum_goal_local_xs",
            stage_idx,
            getattr(self.cfg, "defect_single_gap_goal_local_x", 0.0),
        )
        self.cfg.defect_single_gap_goal_local_y = self._get_curriculum_value(
            "defect_dwb_curriculum_goal_local_ys",
            stage_idx,
            getattr(self.cfg, "defect_single_gap_goal_local_y", 0.0),
        )
        self.cfg.defect_single_gap_goal_local_jitter_x = self._get_curriculum_value(
            "defect_dwb_curriculum_goal_local_jitter_xs",
            stage_idx,
            getattr(self.cfg, "defect_single_gap_goal_local_jitter_x", 0.0),
        )
        self.cfg.defect_single_gap_goal_local_jitter_y = self._get_curriculum_value(
            "defect_dwb_curriculum_goal_local_jitter_ys",
            stage_idx,
            getattr(self.cfg, "defect_single_gap_goal_local_jitter_y", 0.0),
        )
        self.cfg.defect_single_gap_goal_mode = str(
            self._get_curriculum_choice(
                "defect_dwb_curriculum_goal_modes",
                stage_idx,
                getattr(self.cfg, "defect_single_gap_goal_mode", "rear"),
            )
        )
        self.cfg.defect_single_gap_goal_rear_clearance_min = self._get_curriculum_value(
            "defect_dwb_curriculum_goal_rear_clearance_mins",
            stage_idx,
            getattr(self.cfg, "defect_single_gap_goal_rear_clearance_min", 0.50),
        )
        self.cfg.defect_single_gap_goal_rear_lateral_center = self._get_curriculum_value(
            "defect_dwb_curriculum_goal_rear_lateral_centers",
            stage_idx,
            getattr(self.cfg, "defect_single_gap_goal_rear_lateral_center", 0.0),
        )
        self.cfg.defect_single_gap_goal_rear_lateral_offset_max = self._get_curriculum_value(
            "defect_dwb_curriculum_goal_rear_lateral_offset_maxs",
            stage_idx,
            getattr(self.cfg, "defect_single_gap_goal_rear_lateral_offset_max", 0.12),
        )
        self.cfg.defect_single_gap_goal_bridge_clearance_min = self._get_curriculum_value(
            "defect_dwb_curriculum_goal_bridge_clearance_mins",
            stage_idx,
            getattr(self.cfg, "defect_single_gap_goal_bridge_clearance_min", 0.30),
        )
        self.cfg.defect_single_gap_goal_bridge_lateral_center = self._get_curriculum_value(
            "defect_dwb_curriculum_goal_bridge_lateral_centers",
            stage_idx,
            getattr(self.cfg, "defect_single_gap_goal_bridge_lateral_center", 0.20),
        )
        self.cfg.defect_single_gap_goal_bridge_lateral_offset_max = self._get_curriculum_value(
            "defect_dwb_curriculum_goal_bridge_lateral_offset_maxs",
            stage_idx,
            getattr(self.cfg, "defect_single_gap_goal_bridge_lateral_offset_max", 0.02),
        )

        self._defect_dwb_curriculum_stage = stage_idx
        print(
            "[CURRICULUM][LidarNav] "
            f"mode={scene_mode} stage={stage_idx} step={current_step} "
            f"curr_step={effective_step} "
            f"freeze_stage={freeze_stage_cfg} "
            f"collision_th={self.cfg.collision_threshold:.2f} "
            f"contact_th={self.cfg.contact_collision_force_threshold:.2f} "
            f"goal_d=[{goal_distance_min:.2f},{self.cfg.defect_goal_distance_max:.2f}] "
            f"goal_reach={self.cfg.goal_reach_threshold:.2f} "
            f"goal_mode={self.cfg.defect_single_gap_goal_mode} "
            f"goal_local=({getattr(self.cfg, 'defect_single_gap_goal_local_x', 0.0):.2f},"
            f"{getattr(self.cfg, 'defect_single_gap_goal_local_y', 0.0):.2f}) "
            f"goal_rear=(clear={getattr(self.cfg, 'defect_single_gap_goal_rear_clearance_min', 0.0):.2f},"
            f"lat={getattr(self.cfg, 'defect_single_gap_goal_rear_lateral_center', 0.0):.2f}±"
            f"{getattr(self.cfg, 'defect_single_gap_goal_rear_lateral_offset_max', 0.0):.2f}) "
            f"goal_bridge=(clear={getattr(self.cfg, 'defect_single_gap_goal_bridge_clearance_min', 0.0):.2f},"
            f"lat={getattr(self.cfg, 'defect_single_gap_goal_bridge_lateral_center', 0.0):.2f}±"
            f"{getattr(self.cfg, 'defect_single_gap_goal_bridge_lateral_offset_max', 0.0):.2f}) "
            f"width={self.cfg.defect_dwb_corridor_width:.2f} "
            f"spacing={self.cfg.defect_dwb_obstacle_spacing:.2f} "
            f"spawn_jitter={self.cfg.defect_spawn_jitter_xy:.2f} "
            f"goal_jitter={self.cfg.defect_goal_jitter_xy:.2f} "
            f"pair_shift={self.cfg.defect_dwb_obstacle_pair_shift_y:.2f} "
            f"x_jitter={self.cfg.defect_dwb_obstacle_x_jitter:.2f} "
            f"x_center={self.cfg.defect_single_gap_obstacle_x_center:.2f} "
            f"spacing_jitter={self.cfg.defect_dwb_obstacle_spacing_jitter:.2f} "
            f"v_max={self.cfg.v_max:.2f} "
            f"cap_explore={self.cfg.speed_cap_explore:.2f} "
            f"cap_narrow={self.cfg.speed_cap_narrow:.2f} "
            f"front_danger={self.cfg.front_danger_distance:.2f} "
            f"danger_scale={self.cfg.danger_speed_penalty_scale:.2f} "
            f"progress_scale={self.cfg.progress_reward_scale:.2f} "
            f"head_scale={self.cfg.heading_reward_scale:.2f} "
            f"fwd_scale={self.cfg.forward_goal_reward_scale:.2f} "
            f"time_pen={self.cfg.time_penalty:.2f} "
            f"turn_pen={self.cfg.turn_penalty_scale:.2f} "
            f"stall_speed={self.cfg.stall_speed_threshold:.2f} "
            f"stall_pen={self.cfg.stall_penalty:.2f} "
            f"succ_cfg={self.cfg.success_bonus:.1f} "
            f"coll_pen={self.cfg.collision_penalty:.1f} "
            f"lat_pen={self.cfg.defect_lateral_velocity_penalty_scale:.2f} "
            f"passage_bonus={self.cfg.defect_narrow_passage_bonus:.2f} "
            f"gap_align={self.cfg.defect_dwb_gap_alignment_reward_scale:.2f} "
            f"dwb_fwd_bonus={float(getattr(self.cfg, 'defect_dwb_forward_speed_reward_scale', 0.0)):.2f} "
            f"surge_scale={float(getattr(self.cfg, 'defect_dwb_speed_surge_scale', 0.0)):.2f} "
            f"deficit_scale={float(getattr(self.cfg, 'defect_dwb_speed_deficit_scale', 0.0)):.2f} "
            f"dwb_np_pen={float(getattr(self.cfg, 'defect_dwb_no_progress_penalty', 0.0)):.2f} "
            f"dwb_np_term={int(getattr(self.cfg, 'defect_dwb_no_progress_terminate_steps', 0))}"
        )

    def _compute_defect_dwb_no_progress_mask(
        self,
        curr_goal_dist: torch.Tensor,
        min_lidar: torch.Tensor,
        forward_speed: torch.Tensor,
        progress_delta: torch.Tensor,
    ) -> torch.Tensor:
        dwb_mask = self._get_dwb_like_scene_mask(self.fixed_scene_active_index)
        if not bool(torch.any(dwb_mask).item()):
            return torch.zeros_like(curr_goal_dist, dtype=torch.bool)

        in_success_halo = curr_goal_dist < self.cfg.goal_reach_threshold
        clearance_threshold = float(getattr(self.cfg, "defect_dwb_no_progress_clearance_threshold", 0.0))
        clearance_ok = min_lidar > clearance_threshold
        speed_threshold = max(0.0, float(getattr(self.cfg, "defect_dwb_no_progress_speed_threshold", 0.0)))
        delta_threshold = max(0.0, float(getattr(self.cfg, "defect_dwb_no_progress_delta_threshold", 0.0)))
        moving_speed_threshold = max(
            speed_threshold,
            float(getattr(self.cfg, "defect_dwb_no_progress_moving_speed_threshold", speed_threshold)),
        )
        moving_delta_threshold = max(
            0.0,
            float(getattr(self.cfg, "defect_dwb_no_progress_moving_delta_threshold", delta_threshold)),
        )

        dwb_stuck = torch.logical_and(forward_speed < speed_threshold, progress_delta < delta_threshold)
        dwb_ineffective_motion = torch.logical_and(
            forward_speed >= moving_speed_threshold,
            progress_delta < moving_delta_threshold,
        )
        return torch.logical_and(
            torch.logical_and(dwb_mask, clearance_ok),
            torch.logical_and(~in_success_halo, torch.logical_or(dwb_stuck, dwb_ineffective_motion)),
        )

    def _use_front_sector_collision_lidar(self) -> bool:
        if not bool(getattr(self.cfg, "defect_single_gap_rear_collision_front_sector_only", False)):
            return False
        if self._get_defect_scene_mode() != "single_obstacle_symmetric_gap":
            return False
        goal_mode = str(getattr(self.cfg, "defect_single_gap_goal_mode", "")).strip().lower()
        return goal_mode in {"rear", "rear_bridge"}

    def _compute_collision_lidar(
        self, lidar_ranges: torch.Tensor, min_lidar: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return lidar-collision mask and the range used for lidar-collision thresholding."""
        if min_lidar is None:
            min_lidar = torch.min(lidar_ranges, dim=1).values

        collision_lidar_min = min_lidar
        if self._use_front_sector_collision_lidar():
            sector_deg = float(getattr(self.cfg, "defect_single_gap_rear_collision_front_sector_deg", 180.0))
            sector_deg = max(1.0, min(359.0, sector_deg))
            collision_half_rad = math.radians(sector_deg) / 2.0
            collision_beam_mask = torch.abs(self.lidar_beam_angles) <= collision_half_rad
            if not bool(torch.any(collision_beam_mask).item()):
                collision_beam_mask = self.front_lidar_beam_mask
            collision_lidar_ranges = torch.where(
                collision_beam_mask.unsqueeze(0),
                lidar_ranges,
                torch.full_like(lidar_ranges, self.cfg.lidar_max_range),
            )
            collision_lidar_min = torch.min(collision_lidar_ranges, dim=1).values

        collision_lidar = collision_lidar_min < self.cfg.collision_threshold
        return collision_lidar, collision_lidar_min

    def _apply_defect_ushape_curriculum(self, force: bool = False) -> None:
        if not bool(getattr(self.cfg, "defect_ushape_curriculum_enable", False)):
            return
        if str(getattr(self.cfg, "defect_scene_mode", "")).strip().lower() != "u_shape_trap":
            return
        stage_steps = getattr(self.cfg, "defect_ushape_curriculum_stage_steps", (0,))
        if not isinstance(stage_steps, (list, tuple)) or len(stage_steps) == 0:
            stage_steps = (0,)
        current_step = int(max(0, int(getattr(self, "common_step_counter", 0))))
        stage_idx = 0
        for idx, start_step in enumerate(stage_steps):
            if current_step >= int(start_step):
                stage_idx = idx
            else:
                break
        if not force and stage_idx == self._defect_ushape_curriculum_stage:
            return

        self.cfg.collision_threshold = self._get_curriculum_value(
            "defect_ushape_curriculum_collision_thresholds", stage_idx, self.cfg.collision_threshold
        )
        goal_distance_min = max(0.05, float(getattr(self.cfg, "defect_goal_distance_min", 1.0)))
        goal_distance_max = self._get_curriculum_value(
            "defect_ushape_curriculum_goal_distance_maxs",
            stage_idx,
            getattr(self.cfg, "defect_goal_distance_max", goal_distance_min),
        )
        self.cfg.defect_goal_distance_max = max(goal_distance_min, goal_distance_max)
        self.cfg.time_penalty = self._get_curriculum_value(
            "defect_ushape_curriculum_time_penalties",
            stage_idx,
            getattr(self.cfg, "time_penalty", -0.04),
        )
        self.cfg.defect_u_escape_bonus = self._get_curriculum_value(
            "defect_ushape_curriculum_escape_bonuses",
            stage_idx,
            getattr(self.cfg, "defect_u_escape_bonus", 20.0),
        )
        self.cfg.defect_u_exploration_reward_scale = self._get_curriculum_value(
            "defect_ushape_curriculum_exploration_bonuses",
            stage_idx,
            getattr(self.cfg, "defect_u_exploration_reward_scale", 0.01),
        )
        self.cfg.defect_u_backward_encouragement = self._get_curriculum_value(
            "defect_ushape_curriculum_backward_encouragements",
            stage_idx,
            getattr(self.cfg, "defect_u_backward_encouragement", 0.0),
        )
        self.cfg.defect_u_opening_width = self._get_curriculum_value(
            "defect_ushape_curriculum_opening_widths",
            stage_idx,
            getattr(self.cfg, "defect_u_opening_width", 0.6),
        )
        inner_width = self._get_curriculum_value(
            "defect_ushape_curriculum_inner_widths",
            stage_idx,
            getattr(self.cfg, "defect_u_inner_width", 2.0),
        )
        self.cfg.defect_u_inner_width = max(self.cfg.defect_u_opening_width + 0.2, inner_width)
        self.cfg.v_max = self._get_curriculum_value(
            "defect_ushape_curriculum_v_maxes", stage_idx, getattr(self.cfg, "v_max", 0.5)
        )

        self._defect_ushape_curriculum_stage = stage_idx
        print(
            "[CURRICULUM][LidarNav] "
            f"mode=u_shape_trap stage={stage_idx} step={current_step} "
            f"collision_th={self.cfg.collision_threshold:.2f} "
            f"goal_d=[{goal_distance_min:.2f},{self.cfg.defect_goal_distance_max:.2f}] "
            f"opening={self.cfg.defect_u_opening_width:.2f} "
            f"inner_w={self.cfg.defect_u_inner_width:.2f} "
            f"v_max={self.cfg.v_max:.2f} "
            f"time_pen={self.cfg.time_penalty:.2f} "
            f"escape_bonus={self.cfg.defect_u_escape_bonus:.2f} "
            f"explore_bonus={self.cfg.defect_u_exploration_reward_scale:.3f} "
            f"backward_bonus={self.cfg.defect_u_backward_encouragement:.3f}"
        )

    def _apply_defect_mppi_curriculum(self, force: bool = False) -> None:
        if not bool(getattr(self.cfg, "defect_mppi_curriculum_enable", False)):
            return
        if str(getattr(self.cfg, "defect_scene_mode", "")).strip().lower() != "mppi_corner_fail":
            return
        stage_steps = getattr(self.cfg, "defect_mppi_curriculum_stage_steps", (0,))
        if not isinstance(stage_steps, (list, tuple)) or len(stage_steps) == 0:
            stage_steps = (0,)
        current_step = int(max(0, int(getattr(self, "common_step_counter", 0))))
        stage_idx = 0
        for idx, start_step in enumerate(stage_steps):
            if current_step >= int(start_step):
                stage_idx = idx
            else:
                break
        if not force and stage_idx == self._defect_mppi_curriculum_stage:
            return

        self.cfg.collision_threshold = self._get_curriculum_value(
            "defect_mppi_curriculum_collision_thresholds", stage_idx, self.cfg.collision_threshold
        )
        self.cfg.defect_corner_corridor_width = self._get_curriculum_value(
            "defect_mppi_curriculum_corridor_widths",
            stage_idx,
            getattr(self.cfg, "defect_corner_corridor_width", 0.8),
        )
        self.cfg.defect_corner_inner_radius = self._get_curriculum_value(
            "defect_mppi_curriculum_inner_radii",
            stage_idx,
            getattr(self.cfg, "defect_corner_inner_radius", 0.5),
        )
        self.cfg.defect_corner_inner_radius_jitter = self._get_curriculum_value(
            "defect_mppi_curriculum_inner_radius_jitters",
            stage_idx,
            getattr(self.cfg, "defect_corner_inner_radius_jitter", 0.05),
        )
        self.cfg.defect_spawn_jitter_xy = self._get_curriculum_value(
            "defect_mppi_curriculum_spawn_jitters",
            stage_idx,
            getattr(self.cfg, "defect_spawn_jitter_xy", 0.2),
        )
        self.cfg.defect_goal_jitter_xy = self._get_curriculum_value(
            "defect_mppi_curriculum_goal_jitters",
            stage_idx,
            getattr(self.cfg, "defect_goal_jitter_xy", 0.2),
        )
        self.cfg.defect_corner_straight_pre_length = self._get_curriculum_value(
            "defect_mppi_curriculum_straight_pre_lengths",
            stage_idx,
            getattr(self.cfg, "defect_corner_straight_pre_length", 1.0),
        )
        self.cfg.defect_corner_straight_post_length = self._get_curriculum_value(
            "defect_mppi_curriculum_straight_post_lengths",
            stage_idx,
            getattr(self.cfg, "defect_corner_straight_post_length", 1.5),
        )
        self.cfg.defect_goal_distance_min = self._get_curriculum_value(
            "defect_mppi_curriculum_goal_distance_mins",
            stage_idx,
            getattr(self.cfg, "defect_goal_distance_min", 1.0),
        )
        goal_distance_max = self._get_curriculum_value(
            "defect_mppi_curriculum_goal_distance_maxs",
            stage_idx,
            getattr(self.cfg, "defect_goal_distance_max", self.cfg.defect_goal_distance_min),
        )
        self.cfg.defect_goal_distance_max = max(self.cfg.defect_goal_distance_min, goal_distance_max)
        self.cfg.defect_corner_goal_y_min_margin = self._get_curriculum_value(
            "defect_mppi_curriculum_goal_y_min_margins",
            stage_idx,
            getattr(self.cfg, "defect_corner_goal_y_min_margin", 0.3),
        )
        self.cfg.defect_corner_goal_y_max_margin = self._get_curriculum_value(
            "defect_mppi_curriculum_goal_y_max_margins",
            stage_idx,
            getattr(self.cfg, "defect_corner_goal_y_max_margin", 0.2),
        )
        self.cfg.goal_reach_threshold = self._get_curriculum_value(
            "defect_mppi_curriculum_goal_reach_thresholds",
            stage_idx,
            getattr(self.cfg, "goal_reach_threshold", 0.45),
        )
        self.cfg.forward_goal_reward_scale = self._get_curriculum_value(
            "defect_mppi_curriculum_forward_goal_reward_scales",
            stage_idx,
            getattr(self.cfg, "forward_goal_reward_scale", 0.45),
        )
        self.cfg.near_goal_progress_mult_mid = self._get_curriculum_value(
            "defect_mppi_curriculum_near_goal_progress_mult_mids",
            stage_idx,
            getattr(self.cfg, "near_goal_progress_mult_mid", 1.0),
        )
        self.cfg.near_goal_progress_mult_close = self._get_curriculum_value(
            "defect_mppi_curriculum_near_goal_progress_mult_closes",
            stage_idx,
            getattr(self.cfg, "near_goal_progress_mult_close", 1.0),
        )
        self.cfg.near_goal_fwd_mult_mid = self._get_curriculum_value(
            "defect_mppi_curriculum_near_goal_fwd_mult_mids",
            stage_idx,
            getattr(self.cfg, "near_goal_fwd_mult_mid", 1.0),
        )
        self.cfg.near_goal_fwd_mult_close = self._get_curriculum_value(
            "defect_mppi_curriculum_near_goal_fwd_mult_closes",
            stage_idx,
            getattr(self.cfg, "near_goal_fwd_mult_close", 1.0),
        )
        self.cfg.defect_corner_inner_block_size = self._get_curriculum_value(
            "defect_mppi_curriculum_inner_block_sizes",
            stage_idx,
            getattr(self.cfg, "defect_corner_inner_block_size", 0.25),
        )
        self.cfg.defect_corner_inner_block_offset = self._get_curriculum_value(
            "defect_mppi_curriculum_inner_block_offsets",
            stage_idx,
            getattr(self.cfg, "defect_corner_inner_block_offset", 0.17),
        )
        self.cfg.v_max = self._get_curriculum_value(
            "defect_mppi_curriculum_v_maxes", stage_idx, getattr(self.cfg, "v_max", 0.5)
        )
        self.cfg.speed_cap_explore = self._get_curriculum_value(
            "defect_mppi_curriculum_speed_cap_explores",
            stage_idx,
            getattr(self.cfg, "speed_cap_explore", self.cfg.v_max),
        )
        self.cfg.speed_cap_turn = self._get_curriculum_value(
            "defect_mppi_curriculum_speed_cap_turns",
            stage_idx,
            getattr(self.cfg, "speed_cap_turn", self.cfg.v_max),
        )
        self.cfg.speed_cap_narrow = self._get_curriculum_value(
            "defect_mppi_curriculum_speed_cap_narrows",
            stage_idx,
            getattr(self.cfg, "speed_cap_narrow", self.cfg.v_max),
        )
        self.cfg.speed_cap_narrow_clearance = self._get_curriculum_value(
            "defect_mppi_curriculum_speed_cap_narrow_clearances",
            stage_idx,
            getattr(self.cfg, "speed_cap_narrow_clearance", 0.45),
        )
        self.cfg.front_danger_distance = self._get_curriculum_value(
            "defect_mppi_curriculum_front_danger_distances",
            stage_idx,
            getattr(self.cfg, "front_danger_distance", getattr(self.cfg, "danger_speed_distance", 0.75)),
        )
        self.cfg.danger_speed_penalty_scale = self._get_curriculum_value(
            "defect_mppi_curriculum_danger_speed_penalty_scales",
            stage_idx,
            getattr(self.cfg, "danger_speed_penalty_scale", -2.0),
        )
        self.cfg.progress_reward_scale = self._get_curriculum_value(
            "defect_mppi_curriculum_progress_reward_scales",
            stage_idx,
            getattr(self.cfg, "progress_reward_scale", 8.0),
        )
        self.cfg.time_penalty = self._get_curriculum_value(
            "defect_mppi_curriculum_time_penalties",
            stage_idx,
            getattr(self.cfg, "time_penalty", -0.05),
        )
        self.cfg.defect_turn_completion_bonus = self._get_curriculum_value(
            "defect_mppi_curriculum_turn_completion_bonuses",
            stage_idx,
            getattr(self.cfg, "defect_turn_completion_bonus", 5.0),
        )
        self._defect_mppi_curriculum_stage = stage_idx
        print(
            "[CURRICULUM][LidarNav] "
            f"mode=mppi_corner_fail stage={stage_idx} step={current_step} "
            f"collision_th={self.cfg.collision_threshold:.2f} "
            f"width={self.cfg.defect_corner_corridor_width:.2f} "
            f"inner_r={self.cfg.defect_corner_inner_radius:.2f} "
            f"pre={self.cfg.defect_corner_straight_pre_length:.2f} "
            f"post={self.cfg.defect_corner_straight_post_length:.2f} "
            f"radius_jitter={self.cfg.defect_corner_inner_radius_jitter:.2f} "
            f"spawn_jitter={self.cfg.defect_spawn_jitter_xy:.2f} "
            f"goal_jitter={self.cfg.defect_goal_jitter_xy:.2f} "
            f"goal_d=[{self.cfg.defect_goal_distance_min:.2f},{self.cfg.defect_goal_distance_max:.2f}] "
            f"goal_y_margin=[{self.cfg.defect_corner_goal_y_min_margin:.2f},{self.cfg.defect_corner_goal_y_max_margin:.2f}] "
            f"goal_reach={self.cfg.goal_reach_threshold:.2f} "
            f"fwd_scale={self.cfg.forward_goal_reward_scale:.2f} "
            f"near_goal_prog=[{self.cfg.near_goal_progress_mult_mid:.2f},{self.cfg.near_goal_progress_mult_close:.2f}] "
            f"near_goal_fwd=[{self.cfg.near_goal_fwd_mult_mid:.2f},{self.cfg.near_goal_fwd_mult_close:.2f}] "
            f"block_size={self.cfg.defect_corner_inner_block_size:.2f} "
            f"block_offset={self.cfg.defect_corner_inner_block_offset:.2f} "
            f"v_max={self.cfg.v_max:.2f} "
            f"cap_explore={self.cfg.speed_cap_explore:.2f} "
            f"cap_turn={self.cfg.speed_cap_turn:.2f} "
            f"cap_narrow={self.cfg.speed_cap_narrow:.2f} "
            f"narrow_clear={self.cfg.speed_cap_narrow_clearance:.2f} "
            f"front_danger={self.cfg.front_danger_distance:.2f} "
            f"danger_scale={self.cfg.danger_speed_penalty_scale:.2f} "
            f"progress_scale={self.cfg.progress_reward_scale:.2f} "
            f"time_pen={self.cfg.time_penalty:.2f} "
            f"turn_bonus={self.cfg.defect_turn_completion_bonus:.2f}"
        )

    def _build_box_segments_local(self, x_min: float, x_max: float, y_min: float, y_max: float) -> torch.Tensor:
        segments = [
            [[x_min, y_min], [x_max, y_min]],
            [[x_max, y_min], [x_max, y_max]],
            [[x_max, y_max], [x_min, y_max]],
            [[x_min, y_max], [x_min, y_min]],
        ]
        return torch.tensor(segments, dtype=torch.float32, device=self.device)

    def _build_arc_segments_local(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        theta_start: float,
        theta_end: float,
        num_segments: int,
    ) -> torch.Tensor:
        seg_count = max(2, int(num_segments))
        angles = torch.linspace(theta_start, theta_end, seg_count + 1, dtype=torch.float32, device=self.device)
        pts = torch.zeros((seg_count + 1, 2), dtype=torch.float32, device=self.device)
        pts[:, 0] = float(center_x) + float(radius) * torch.cos(angles)
        pts[:, 1] = float(center_y) + float(radius) * torch.sin(angles)
        return torch.stack((pts[:-1], pts[1:]), dim=1)

    def _set_custom_wall_segments(self, env_id: int, segments_local: torch.Tensor) -> None:
        self._junction_wall_seg_valid[env_id] = False
        self._junction_wall_seg_start[env_id] = 0.0
        self._junction_wall_seg_end[env_id] = 0.0
        if segments_local.numel() == 0:
            return
        origin_xy = self.scene.env_origins[env_id, :2]
        segments_world = segments_local + origin_xy.view(1, 1, 2)
        seg_count = min(int(segments_world.shape[0]), int(self._junction_max_segments))
        if seg_count <= 0:
            return
        self._junction_wall_seg_start[env_id, :seg_count] = segments_world[:seg_count, 0, :]
        self._junction_wall_seg_end[env_id, :seg_count] = segments_world[:seg_count, 1, :]
        self._junction_wall_seg_valid[env_id, :seg_count] = True

    def _set_obstacles_from_local_xy(self, env_id: int, obstacle_xy: list[tuple[float, float]]) -> None:
        num_obstacles = int(self.cfg.num_obstacles)
        inactive_offset_x = float(getattr(self.cfg, "fixed_scene_inactive_obstacle_offset_x", 50.0))
        inactive_offset_y = float(getattr(self.cfg, "fixed_scene_inactive_obstacle_offset_y", 50.0))
        obstacle_z = 0.5 * float(self.cfg.obstacle_height)
        env_origin = self.scene.env_origins[env_id, :2]
        for obs_idx in range(num_obstacles):
            if obs_idx < len(obstacle_xy):
                ox, oy = obstacle_xy[obs_idx]
                self.obstacle_pos_w[env_id, obs_idx, 0] = env_origin[0] + float(ox)
                self.obstacle_pos_w[env_id, obs_idx, 1] = env_origin[1] + float(oy)
            else:
                self.obstacle_pos_w[env_id, obs_idx, 0] = env_origin[0] + inactive_offset_x + float(obs_idx)
                self.obstacle_pos_w[env_id, obs_idx, 1] = env_origin[1] + inactive_offset_y
            self.obstacle_pos_w[env_id, obs_idx, 2] = obstacle_z

    def _u_explore_grid_indices(self, local_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_idx = torch.floor((local_xy[:, 0] - self._defect_u_grid_x_min) / self._defect_u_grid_cell_size).to(torch.long)
        y_idx = torch.floor((local_xy[:, 1] - self._defect_u_grid_y_min) / self._defect_u_grid_cell_size).to(torch.long)
        valid_x = torch.logical_and(x_idx >= 0, x_idx < self._defect_u_explore_visited.shape[1])
        valid_y = torch.logical_and(y_idx >= 0, y_idx < self._defect_u_explore_visited.shape[2])
        valid = torch.logical_and(valid_x, valid_y)
        return x_idx, y_idx, valid

    def _mark_u_explore_cells(self, env_ids: torch.Tensor, local_xy: torch.Tensor) -> torch.Tensor:
        new_cells = torch.zeros((env_ids.numel(),), dtype=torch.bool, device=self.device)
        if env_ids.numel() == 0:
            return new_cells
        x_idx, y_idx, valid = self._u_explore_grid_indices(local_xy)
        if not bool(torch.any(valid).item()):
            return new_cells
        valid_env_ids = env_ids[valid]
        valid_x_idx = x_idx[valid]
        valid_y_idx = y_idx[valid]
        visited = self._defect_u_explore_visited[valid_env_ids, valid_x_idx, valid_y_idx]
        unvisited = ~visited
        if bool(torch.any(unvisited).item()):
            self._defect_u_explore_visited[
                valid_env_ids[unvisited], valid_x_idx[unvisited], valid_y_idx[unvisited]
            ] = True
        new_cells[valid] = unvisited
        return new_cells

    def _setup_scene_dwb_oscillation(
        self,
    ) -> tuple[tuple[float, float], float, tuple[float, float], list[tuple[float, float]], torch.Tensor]:
        corridor_width = max(
            0.6,
            float(getattr(self.cfg, "defect_dwb_corridor_width", 2.0))
            + self._sample_signed_uniform(float(getattr(self.cfg, "defect_dwb_corridor_width_jitter", 0.0))),
        )
        half_width = 0.5 * corridor_width
        half_length = max(
            1.2,
            float(getattr(self.cfg, "defect_dwb_corridor_half_length", 3.0))
            + self._sample_signed_uniform(float(getattr(self.cfg, "defect_dwb_corridor_half_length_jitter", 0.0))),
        )
        spacing_nominal = float(getattr(self.cfg, "defect_dwb_obstacle_spacing", 0.6))
        spacing_jitter = float(getattr(self.cfg, "defect_dwb_obstacle_spacing_jitter", 0.0))
        spacing = max(0.45, spacing_nominal + self._sample_signed_uniform(spacing_jitter))
        spawn_jitter = float(getattr(self.cfg, "defect_spawn_jitter_xy", 0.2))
        goal_jitter = float(getattr(self.cfg, "defect_goal_jitter_xy", 0.0))
        goal_dist_min = max(0.05, float(getattr(self.cfg, "defect_goal_distance_min", 1.0)))
        goal_dist_max = max(goal_dist_min, float(getattr(self.cfg, "defect_goal_distance_max", goal_dist_min)))
        target_goal_dist = min(max(self._sample_defect_goal_distance(), goal_dist_min), goal_dist_max)

        goal_x_min = 0.20
        goal_x_max = half_length - 0.35
        spawn_x_min = max(-half_length + 0.35, goal_x_min - target_goal_dist)
        spawn_x_max = min(-0.20, goal_x_max - target_goal_dist)
        if spawn_x_max < spawn_x_min:
            spawn_x = self._sample_uniform(-half_length + 0.35, -0.20)
        else:
            spawn_x = self._sample_uniform(spawn_x_min, spawn_x_max)
        spawn_y = self._sample_signed_uniform(spawn_jitter)
        spawn_x = min(max(spawn_x, -half_length + 0.35), -0.20)
        spawn_y = min(max(spawn_y, -half_width + 0.30), half_width - 0.30)

        goal_y = spawn_y + self._sample_signed_uniform(goal_jitter)
        goal_y = min(max(goal_y, -half_width + 0.30), half_width - 0.30)
        dy = goal_y - spawn_y
        allowed_abs_dy = max(0.0, target_goal_dist - 1.0e-3)
        dy = min(max(dy, -allowed_abs_dy), allowed_abs_dy)
        goal_y = spawn_y + dy
        goal_dx = math.sqrt(max(target_goal_dist * target_goal_dist - dy * dy, 1.0e-6))
        goal_x = min(max(spawn_x + goal_dx, goal_x_min), goal_x_max)

        pair_shift = self._sample_signed_uniform(float(getattr(self.cfg, "defect_dwb_obstacle_pair_shift_y", 0.0)))
        obstacle_x = self._sample_signed_uniform(float(getattr(self.cfg, "defect_dwb_obstacle_x_jitter", 0.0)))
        obstacle_x = min(max(obstacle_x, -half_length + 0.55), half_length - 0.55)
        radius = float(getattr(self.cfg, "obstacle_radius", 0.2))
        y_min = -half_width + radius + 0.04
        y_max = half_width - radius - 0.04
        obs_y0 = min(max(-0.5 * spacing + pair_shift, y_min), y_max)
        obs_y1 = min(max(0.5 * spacing + pair_shift, y_min), y_max)
        if obs_y1 < obs_y0:
            obs_y0, obs_y1 = obs_y1, obs_y0
        min_sep = max(0.20, 2.0 * radius + 0.05)
        if (obs_y1 - obs_y0) < min_sep:
            center = 0.5 * (obs_y0 + obs_y1)
            half_sep = 0.5 * min_sep
            obs_y0 = max(y_min, center - half_sep)
            obs_y1 = min(y_max, center + half_sep)
            if (obs_y1 - obs_y0) < min_sep:
                obs_y0 = y_min
                obs_y1 = min(y_max, y_min + min_sep)

        obstacles = [(obstacle_x, obs_y0), (obstacle_x, obs_y1)]
        segments = self._build_box_segments_local(
            x_min=-half_length,
            x_max=half_length,
            y_min=-half_width,
            y_max=half_width,
        )
        return (spawn_x, spawn_y), 0.0, (goal_x, goal_y), obstacles, segments

    def _setup_scene_single_obstacle_symmetric_gap(
        self,
    ) -> tuple[tuple[float, float], float, tuple[float, float], list[tuple[float, float]], torch.Tensor]:
        corridor_width = max(0.6, float(getattr(self.cfg, "defect_dwb_corridor_width", 2.0)))
        half_width = 0.5 * corridor_width
        half_length = max(1.2, float(getattr(self.cfg, "defect_dwb_corridor_half_length", 3.0)))
        spawn_jitter = float(getattr(self.cfg, "defect_spawn_jitter_xy", 0.2))
        goal_dist_min = max(0.05, float(getattr(self.cfg, "defect_goal_distance_min", 1.0)))
        goal_dist_max = max(goal_dist_min, float(getattr(self.cfg, "defect_goal_distance_max", goal_dist_min)))
        target_goal_dist = min(max(self._sample_defect_goal_distance(), goal_dist_min), goal_dist_max)
        goal_mode = str(getattr(self.cfg, "defect_single_gap_goal_mode", "rear")).strip().lower()

        half_extent_x = self._get_box_obstacle_half_extent_x() if self._use_box_obstacles() else float(
            getattr(self.cfg, "obstacle_radius", 0.2)
        )
        half_extent_y = self._get_box_obstacle_half_extent_y() if self._use_box_obstacles() else float(
            getattr(self.cfg, "obstacle_radius", 0.2)
        )
        # Keep a conservative hard floor on rear-goal clearance during scene sampling.
        goal_rear_clearance_min = max(
            0.50,
            float(getattr(self.cfg, "defect_single_gap_goal_rear_clearance_min", 0.50)),
        )
        goal_bridge_clearance_min = max(
            0.0,
            float(getattr(self.cfg, "defect_single_gap_goal_bridge_clearance_min", 0.30)),
        )
        goal_lateral_center = float(getattr(self.cfg, "defect_single_gap_goal_rear_lateral_center", 0.0))
        goal_lateral_offset_max = max(
            0.0,
            float(getattr(self.cfg, "defect_single_gap_goal_rear_lateral_offset_max", 0.12)),
        )
        goal_bridge_lateral_center = float(getattr(self.cfg, "defect_single_gap_goal_bridge_lateral_center", 0.20))
        goal_bridge_offset_max = max(
            0.0,
            float(getattr(self.cfg, "defect_single_gap_goal_bridge_lateral_offset_max", 0.02)),
        )
        spawn_obstacle_clearance = max(
            0.10,
            float(getattr(self.cfg, "defect_single_gap_spawn_obstacle_clearance", 0.25)),
        )
        goal_x_min = -half_length + 0.35 if goal_mode == "fixed_local" else 0.20
        goal_x_max = half_length - 0.35
        goal_y_min = -half_width + 0.30
        goal_y_max = half_width - 0.30
        obstacle_x_center = float(getattr(self.cfg, "defect_single_gap_obstacle_x_center", 0.0))
        obstacle_x = obstacle_x_center + self._sample_signed_uniform(
            float(getattr(self.cfg, "defect_dwb_obstacle_x_jitter", 0.0))
        )
        obstacle_x_min = -half_length + half_extent_x + 0.20
        if goal_mode == "fixed_local":
            obstacle_x_max = half_length - half_extent_x - 0.20
        else:
            relative_goal_clearance = goal_rear_clearance_min if goal_mode == "rear" else goal_bridge_clearance_min
            obstacle_x_max = min(
                half_length - half_extent_x - 0.20,
                goal_x_max - half_extent_x - relative_goal_clearance,
            )
        if obstacle_x_max < obstacle_x_min:
            obstacle_x_max = obstacle_x_min
        obstacle_x = min(max(obstacle_x, obstacle_x_min), obstacle_x_max)
        obstacle_y = self._sample_signed_uniform(float(getattr(self.cfg, "defect_single_gap_obstacle_y_jitter", 0.0)))
        obstacle_y = min(max(obstacle_y, -half_width + half_extent_y + 0.04), half_width - half_extent_y - 0.04)

        if goal_mode == "fixed_local":
            goal_x = float(getattr(self.cfg, "defect_single_gap_goal_local_x", -0.05))
            goal_x = goal_x + self._sample_signed_uniform(
                float(getattr(self.cfg, "defect_single_gap_goal_local_jitter_x", 0.0))
            )
            goal_x = min(max(goal_x, goal_x_min), goal_x_max)
            goal_y = float(getattr(self.cfg, "defect_single_gap_goal_local_y", 0.36))
            goal_y = goal_y + self._sample_signed_uniform(
                float(getattr(self.cfg, "defect_single_gap_goal_local_jitter_y", 0.0))
            )
            goal_y = min(max(goal_y, goal_y_min), goal_y_max)
        elif goal_mode == "rear_bridge":
            goal_x = max(goal_x_min, obstacle_x + half_extent_x + goal_bridge_clearance_min)
            goal_x = min(goal_x, goal_x_max)
            goal_y = obstacle_y + goal_bridge_lateral_center + self._sample_signed_uniform(goal_bridge_offset_max)
            goal_y = min(max(goal_y, goal_y_min), goal_y_max)
        else:
            goal_x = max(goal_x_min, obstacle_x + half_extent_x + goal_rear_clearance_min)
            goal_x = min(goal_x, goal_x_max)
            goal_y = obstacle_y + goal_lateral_center + self._sample_signed_uniform(goal_lateral_offset_max)
            goal_y = min(max(goal_y, goal_y_min), goal_y_max)

        spawn_y = self._sample_signed_uniform(spawn_jitter)
        spawn_y = min(max(spawn_y, -half_width + 0.30), half_width - 0.30)
        allowed_abs_dy = max(0.0, target_goal_dist - 1.0e-3)
        dy = goal_y - spawn_y
        if abs(dy) > allowed_abs_dy:
            spawn_y = goal_y - math.copysign(allowed_abs_dy, dy)
            spawn_y = min(max(spawn_y, -half_width + 0.30), half_width - 0.30)
            dy = goal_y - spawn_y
        goal_dx = math.sqrt(max(target_goal_dist * target_goal_dist - dy * dy, 1.0e-6))
        spawn_x_target = goal_x - goal_dx
        spawn_x_min = -half_length + 0.35
        spawn_x_max = min(-0.20, obstacle_x - half_extent_x - spawn_obstacle_clearance)
        if spawn_x_max < spawn_x_min:
            spawn_x_max = spawn_x_min
        spawn_x = min(max(spawn_x_target, spawn_x_min), spawn_x_max)

        goal_x, goal_y = self._project_single_gap_goal_outside_obstacle(
            goal_x=goal_x,
            goal_y=goal_y,
            obstacle_x=obstacle_x,
            obstacle_y=obstacle_y,
            half_extent_x=half_extent_x,
            half_extent_y=half_extent_y,
            goal_x_min=goal_x_min,
            goal_x_max=goal_x_max,
            goal_y_min=goal_y_min,
            goal_y_max=goal_y_max,
        )

        obstacles = [(obstacle_x, obstacle_y)]
        segments = self._build_box_segments_local(
            x_min=-half_length,
            x_max=half_length,
            y_min=-half_width,
            y_max=half_width,
        )
        return (spawn_x, spawn_y), 0.0, (goal_x, goal_y), obstacles, segments

    def _setup_scene_mppi_corner_fail(
        self,
    ) -> tuple[tuple[float, float], float, tuple[float, float], list[tuple[float, float]], torch.Tensor]:
        corridor_width = max(0.5, float(getattr(self.cfg, "defect_corner_corridor_width", 0.8)))
        half_width = 0.5 * corridor_width
        inner_radius_nominal = float(getattr(self.cfg, "defect_corner_inner_radius", 0.5))
        inner_radius_jitter = float(getattr(self.cfg, "defect_corner_inner_radius_jitter", 0.05))
        inner_radius = max(0.2, inner_radius_nominal + self._sample_signed_uniform(inner_radius_jitter))
        outer_radius = inner_radius + corridor_width
        center_radius = inner_radius + half_width
        straight_pre = max(1.0, float(getattr(self.cfg, "defect_corner_straight_pre_length", 2.2)))
        straight_post = max(1.0, float(getattr(self.cfg, "defect_corner_straight_post_length", 2.2)))
        spawn_jitter = float(getattr(self.cfg, "defect_spawn_jitter_xy", 0.2))
        goal_jitter = float(getattr(self.cfg, "defect_goal_jitter_xy", 0.0))
        target_goal_dist = self._sample_defect_goal_distance()
        goal_dist_max = max(target_goal_dist, float(getattr(self.cfg, "defect_goal_distance_max", target_goal_dist)))

        spawn_x = -straight_pre + 0.5 + self._sample_signed_uniform(spawn_jitter)
        spawn_y = self._sample_signed_uniform(spawn_jitter)
        spawn_x = min(max(spawn_x, -straight_pre + 0.3), -0.25)
        spawn_y = min(max(spawn_y, -half_width + 0.08), half_width - 0.08)

        goal_x = center_radius + self._sample_signed_uniform(goal_jitter)
        goal_x = min(max(goal_x, inner_radius + 0.08), outer_radius - 0.08)
        dx = abs(goal_x - spawn_x)
        goal_y_min_margin = max(0.02, float(getattr(self.cfg, "defect_corner_goal_y_min_margin", 0.3)))
        goal_y_max_margin = max(0.02, float(getattr(self.cfg, "defect_corner_goal_y_max_margin", 0.2)))
        goal_y_min = center_radius + goal_y_min_margin
        goal_y_max = center_radius + straight_post - goal_y_max_margin
        if goal_dist_max > dx:
            dy_cap = math.sqrt(max(goal_dist_max * goal_dist_max - dx * dx, 0.0))
            goal_y_max = min(goal_y_max, spawn_y + dy_cap)
        goal_y_max = max(goal_y_min, goal_y_max)
        goal_dist_target = max(target_goal_dist, dx + 1.0e-6)
        dy_target = math.sqrt(max(goal_dist_target * goal_dist_target - dx * dx, 0.0))
        goal_y = spawn_y + dy_target + self._sample_signed_uniform(goal_jitter)
        goal_y = min(max(goal_y, goal_y_min), goal_y_max)

        segments_list: list[torch.Tensor] = []
        obstacles: list[tuple[float, float]] = []
        segments_list.append(
            torch.tensor(
                [[[-straight_pre, -half_width], [0.0, -half_width]]],
                dtype=torch.float32,
                device=self.device,
            )
        )
        segments_list.append(
            torch.tensor(
                [[[-straight_pre, half_width], [0.0, half_width]]],
                dtype=torch.float32,
                device=self.device,
            )
        )
        segments_list.append(
            torch.tensor(
                [[[-straight_pre, -half_width], [-straight_pre, half_width]]],
                dtype=torch.float32,
                device=self.device,
            )
        )
        segments_list.append(
            self._build_arc_segments_local(
                center_x=0.0,
                center_y=center_radius,
                radius=inner_radius,
                theta_start=-0.5 * math.pi,
                theta_end=0.0,
                num_segments=14,
            )
        )
        segments_list.append(
            self._build_arc_segments_local(
                center_x=0.0,
                center_y=center_radius,
                radius=outer_radius,
                theta_start=-0.5 * math.pi,
                theta_end=0.0,
                num_segments=14,
            )
        )
        segments_list.append(
            torch.tensor(
                [[[inner_radius, center_radius], [inner_radius, center_radius + straight_post]]],
                dtype=torch.float32,
                device=self.device,
            )
        )
        segments_list.append(
            torch.tensor(
                [[[outer_radius, center_radius], [outer_radius, center_radius + straight_post]]],
                dtype=torch.float32,
                device=self.device,
            )
        )
        segments_list.append(
            torch.tensor(
                [[[inner_radius, center_radius + straight_post], [outer_radius, center_radius + straight_post]]],
                dtype=torch.float32,
                device=self.device,
            )
        )
        if bool(getattr(self.cfg, "defect_corner_inner_block_enable", True)):
            block_size = max(0.05, float(getattr(self.cfg, "defect_corner_inner_block_size", 0.25)))
            block_half = 0.5 * block_size
            block_offset = max(block_half + 0.01, float(getattr(self.cfg, "defect_corner_inner_block_offset", 0.17)))
            block_cx = inner_radius + block_offset
            block_cy = center_radius - block_offset
            block_cx = min(max(block_cx, inner_radius + block_half + 0.01), outer_radius - block_half - 0.01)
            block_cy = min(max(block_cy, block_half + 0.01), center_radius - block_half - 0.01)
            segments_list.append(
                self._build_box_segments_local(
                    x_min=block_cx - block_half,
                    x_max=block_cx + block_half,
                    y_min=block_cy - block_half,
                    y_max=block_cy + block_half,
                )
            )
            obstacles.append((block_cx, block_cy))
        segments = torch.cat(segments_list, dim=0)
        return (spawn_x, spawn_y), 0.0, (goal_x, goal_y), obstacles, segments

    def _setup_scene_door_deadlock(
        self,
    ) -> tuple[tuple[float, float], float, tuple[float, float], list[tuple[float, float]], torch.Tensor]:
        corridor_width = max(1.0, float(getattr(self.cfg, "defect_door_corridor_width", 2.0)))
        half_width = 0.5 * corridor_width
        half_length = max(2.0, float(getattr(self.cfg, "defect_door_half_length", 3.0)))
        door_width_nominal = float(getattr(self.cfg, "defect_door_width", 0.8))
        door_width_jitter = float(getattr(self.cfg, "defect_door_width_jitter", 0.05))
        door_width = max(0.45, door_width_nominal + self._sample_signed_uniform(door_width_jitter))
        door_half = 0.5 * door_width
        straight_distance = max(0.6, float(getattr(self.cfg, "defect_door_straight_distance", 1.5)))
        line_angle_max_deg = max(0.0, float(getattr(self.cfg, "defect_door_line_angle_max_deg", 15.0)))
        goal_lateral_offset_max = max(0.0, float(getattr(self.cfg, "defect_door_goal_lateral_offset_max", 0.0)))
        spawn_jitter = float(getattr(self.cfg, "defect_spawn_jitter_xy", 0.2))
        goal_jitter = float(getattr(self.cfg, "defect_goal_jitter_xy", 0.0))
        side_margin = 0.35
        side_max = max(side_margin, half_length - side_margin)
        target_goal_dist = min(max(self._sample_defect_goal_distance(), 2.0 * side_margin), 2.0 * side_max)

        robot_dist_min = max(side_margin, target_goal_dist - side_max)
        robot_dist_max = min(side_max, target_goal_dist - side_margin)
        nominal_robot_dist = straight_distance + self._sample_signed_uniform(spawn_jitter)
        robot_dist = min(max(nominal_robot_dist, robot_dist_min), robot_dist_max)
        goal_dist = min(max(target_goal_dist - robot_dist + self._sample_signed_uniform(goal_jitter), side_margin), side_max)

        lateral_margin = max(0.08, 0.5 * float(getattr(self.cfg, "robot_radius", 0.12)))
        lateral_limit = max(0.05, half_width - lateral_margin)
        line_angle = 0.0
        if line_angle_max_deg > 0.0 and lateral_limit > 0.0:
            max_angle_rad = math.radians(line_angle_max_deg)
            robot_angle_bound = math.asin(min(1.0, lateral_limit / max(robot_dist, 1.0e-6)))
            goal_angle_bound = math.asin(min(1.0, lateral_limit / max(goal_dist, 1.0e-6)))
            sampled_angle_bound = min(max_angle_rad, robot_angle_bound, goal_angle_bound)
            if sampled_angle_bound > 1.0e-6:
                line_angle = self._sample_uniform(-sampled_angle_bound, sampled_angle_bound)

        line_dir_x = math.cos(line_angle)
        line_dir_y = math.sin(line_angle)
        line_lat_x = -line_dir_y
        line_lat_y = line_dir_x
        spawn_x = -robot_dist * line_dir_x
        spawn_y = -robot_dist * line_dir_y
        goal_x = goal_dist * line_dir_x
        goal_y = goal_dist * line_dir_y
        if goal_lateral_offset_max > 1.0e-6:
            goal_offset_lo = -goal_lateral_offset_max
            goal_offset_hi = goal_lateral_offset_max

            def _intersect_linear_interval(
                offset_lo: float,
                offset_hi: float,
                base_value: float,
                coeff: float,
                value_min: float,
                value_max: float,
            ) -> tuple[float, float]:
                if abs(coeff) <= 1.0e-6:
                    if value_min <= base_value <= value_max:
                        return offset_lo, offset_hi
                    return 1.0, 0.0
                bound_lo = (value_min - base_value) / coeff
                bound_hi = (value_max - base_value) / coeff
                if bound_lo > bound_hi:
                    bound_lo, bound_hi = bound_hi, bound_lo
                return max(offset_lo, bound_lo), min(offset_hi, bound_hi)

            goal_offset_lo, goal_offset_hi = _intersect_linear_interval(
                goal_offset_lo,
                goal_offset_hi,
                goal_x,
                line_lat_x,
                side_margin,
                side_max,
            )
            goal_offset_lo, goal_offset_hi = _intersect_linear_interval(
                goal_offset_lo,
                goal_offset_hi,
                goal_y,
                line_lat_y,
                -lateral_limit,
                lateral_limit,
            )
            if goal_offset_hi >= goal_offset_lo:
                if goal_offset_hi - goal_offset_lo > 1.0e-6:
                    goal_lateral_offset = self._sample_uniform(goal_offset_lo, goal_offset_hi)
                else:
                    goal_lateral_offset = 0.5 * (goal_offset_lo + goal_offset_hi)
                goal_x = goal_x + goal_lateral_offset * line_lat_x
                goal_y = goal_y + goal_lateral_offset * line_lat_y

        base_segments = self._build_box_segments_local(
            x_min=-half_length,
            x_max=half_length,
            y_min=-half_width,
            y_max=half_width,
        )
        door_segments = torch.tensor(
            [
                [[0.0, -half_width], [0.0, -door_half]],
                [[0.0, door_half], [0.0, half_width]],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        segments = torch.cat((base_segments, door_segments), dim=0)
        return (spawn_x, spawn_y), line_angle, (goal_x, goal_y), [], segments

    def _setup_scene_u_shape_trap(
        self,
    ) -> tuple[tuple[float, float], float, tuple[float, float], list[tuple[float, float]], torch.Tensor]:
        opening_width_nominal = float(getattr(self.cfg, "defect_u_opening_width", 0.6))
        opening_width_jitter = float(getattr(self.cfg, "defect_u_opening_width_jitter", 0.0))
        opening_width = max(0.35, opening_width_nominal + self._sample_signed_uniform(opening_width_jitter))
        half_opening = 0.5 * opening_width
        depth = max(1.2, float(getattr(self.cfg, "defect_u_depth", 2.7)))
        inner_width = max(opening_width + 0.2, float(getattr(self.cfg, "defect_u_inner_width", 2.0)))
        half_inner = 0.5 * inner_width
        arena_half_width = max(half_inner + 0.6, float(getattr(self.cfg, "defect_u_arena_half_width", 2.6)))
        arena_x_min = -depth - 0.8
        arena_x_max = max(2.2, float(getattr(self.cfg, "defect_u_goal_outside_x", 1.8)) + 0.8)
        spawn_jitter = float(getattr(self.cfg, "defect_spawn_jitter_xy", 0.2))
        goal_jitter = float(getattr(self.cfg, "defect_goal_jitter_xy", 0.0))
        goal_dist_min = max(0.05, float(getattr(self.cfg, "defect_goal_distance_min", 1.0)))
        goal_dist_max = max(goal_dist_min, float(getattr(self.cfg, "defect_goal_distance_max", goal_dist_min)))
        target_goal_dist = min(max(self._sample_defect_goal_distance(), goal_dist_min), goal_dist_max)

        spawn_x = -depth + 0.5 + self._sample_signed_uniform(spawn_jitter)
        spawn_y = self._sample_signed_uniform(spawn_jitter)
        spawn_x = min(max(spawn_x, -depth + 0.25), -0.45)
        spawn_y = min(max(spawn_y, -half_inner + 0.08), half_inner - 0.08)
        goal_x_min = 0.45
        goal_x_max = arena_x_max - 0.35

        goal_y = self._sample_signed_uniform(goal_jitter)
        goal_y = min(max(goal_y, -half_opening + 0.05), half_opening - 0.05)
        dy = goal_y - spawn_y
        allowed_abs_dy = max(0.0, target_goal_dist - 1.0e-3)
        dy = min(max(dy, -allowed_abs_dy), allowed_abs_dy)
        goal_y = spawn_y + dy
        goal_y = min(max(goal_y, -half_opening + 0.05), half_opening - 0.05)
        dy = goal_y - spawn_y

        goal_dx = math.sqrt(max(target_goal_dist * target_goal_dist - dy * dy, 1.0e-6))
        goal_x = min(max(spawn_x + goal_dx, goal_x_min), goal_x_max)

        dx = goal_x - spawn_x
        max_abs_dy = math.sqrt(max(goal_dist_max * goal_dist_max - dx * dx, 0.0))
        dy = min(max(goal_y - spawn_y, -max_abs_dy), max_abs_dy)
        goal_y = min(max(spawn_y + dy, -half_opening + 0.05), half_opening - 0.05)

        dy = goal_y - spawn_y
        curr_dist = math.sqrt(dx * dx + dy * dy)
        if curr_dist < goal_dist_min:
            min_dx = math.sqrt(max(goal_dist_min * goal_dist_min - dy * dy, 0.0))
            goal_x = min(max(goal_x, spawn_x + min_dx), goal_x_max)

        segments = torch.tensor(
            [
                [[arena_x_min, -arena_half_width], [arena_x_max, -arena_half_width]],
                [[arena_x_max, -arena_half_width], [arena_x_max, arena_half_width]],
                [[arena_x_max, arena_half_width], [arena_x_min, arena_half_width]],
                [[arena_x_min, arena_half_width], [arena_x_min, -arena_half_width]],
                [[-depth, half_inner], [0.0, half_inner]],
                [[-depth, -half_inner], [0.0, -half_inner]],
                [[-depth, -half_inner], [-depth, half_inner]],
                [[0.0, half_inner], [0.0, half_opening]],
                [[0.0, -half_inner], [0.0, -half_opening]],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return (spawn_x, spawn_y), 0.0, (goal_x, goal_y), [], segments

    def _reset_defect_scene(self, env_ids: torch.Tensor, default_root_state: torch.Tensor) -> None:
        scene_mode = str(getattr(self.cfg, "defect_scene_mode", "dwb_oscillation")).strip().lower()
        emit_single_gap_sample = (
            self._single_gap_goal_debug_enable
            and scene_mode == "single_obstacle_symmetric_gap"
            and (self._single_gap_goal_debug_counter % self._single_gap_goal_debug_every == 0)
        )
        sample_robot_xy: tuple[float, float] | None = None
        sample_goal_xy: tuple[float, float] | None = None
        sample_obstacle_xy: tuple[float, float] | None = None
        scene_index = {
            "dwb_oscillation": self.DEFECT_DWB_OSCILLATION,
            "single_obstacle_symmetric_gap": self.DEFECT_SINGLE_OBSTACLE_SYMMETRIC_GAP,
            "mppi_corner_fail": self.DEFECT_MPPI_CORNER_FAIL,
            "door_deadlock": self.DEFECT_DOOR_DEADLOCK,
            "u_shape_trap": self.DEFECT_U_SHAPE_TRAP,
        }.get(scene_mode, self.DEFECT_DWB_OSCILLATION)

        self.fixed_scene_active_index[env_ids] = scene_index
        self._clear_turn_cmd(env_ids)
        self._set_planner_state(env_ids, "explore")
        self._junction_wall_seg_valid[env_ids] = False
        self._junction_wall_seg_start[env_ids] = 0.0
        self._junction_wall_seg_end[env_ids] = 0.0
        self.layout_open_area[env_ids] = False
        self._defect_dwb_gap_center_local_x[env_ids] = 0.0
        self._defect_dwb_gap_center_local_y[env_ids] = 0.0

        env_ids_cpu = env_ids.detach().cpu().tolist()
        env_origins_xy = self.scene.env_origins[env_ids, :2]
        for local_idx, env_id in enumerate(env_ids_cpu):
            if scene_mode == "mppi_corner_fail":
                robot_xy, robot_yaw, goal_xy, obstacles_xy, segments_local = self._setup_scene_mppi_corner_fail()
            elif scene_mode == "door_deadlock":
                robot_xy, robot_yaw, goal_xy, obstacles_xy, segments_local = self._setup_scene_door_deadlock()
            elif scene_mode == "u_shape_trap":
                robot_xy, robot_yaw, goal_xy, obstacles_xy, segments_local = self._setup_scene_u_shape_trap()
            elif scene_mode == "single_obstacle_symmetric_gap":
                robot_xy, robot_yaw, goal_xy, obstacles_xy, segments_local = self._setup_scene_single_obstacle_symmetric_gap()
            else:
                robot_xy, robot_yaw, goal_xy, obstacles_xy, segments_local = self._setup_scene_dwb_oscillation()

            self.layout_half_width[env_id] = max(0.2, torch.max(torch.abs(segments_local[..., 1])).item())
            self.layout_half_length[env_id] = max(0.2, torch.max(torch.abs(segments_local[..., 0])).item())

            default_root_state[local_idx, 0] = env_origins_xy[local_idx, 0] + float(robot_xy[0])
            default_root_state[local_idx, 1] = env_origins_xy[local_idx, 1] + float(robot_xy[1])
            default_root_state[local_idx, 3:7] = 0.0
            default_root_state[local_idx, 3] = math.cos(0.5 * float(robot_yaw))
            default_root_state[local_idx, 6] = math.sin(0.5 * float(robot_yaw))

            self.goal_pos_w[env_id, 0] = self.scene.env_origins[env_id, 0] + float(goal_xy[0])
            self.goal_pos_w[env_id, 1] = self.scene.env_origins[env_id, 1] + float(goal_xy[1])
            self.goal_pos_w[env_id, 2] = float(self.cfg.goal_radius)

            self._set_obstacles_from_local_xy(env_id=env_id, obstacle_xy=obstacles_xy)
            self._set_custom_wall_segments(env_id=env_id, segments_local=segments_local)
            if self._is_dwb_like_scene_mode(scene_mode) and len(obstacles_xy) >= 1:
                if scene_mode == "dwb_oscillation" and len(obstacles_xy) >= 2:
                    self._defect_dwb_gap_center_local_x[env_id] = 0.5 * (float(obstacles_xy[0][0]) + float(obstacles_xy[1][0]))
                    self._defect_dwb_gap_center_local_y[env_id] = 0.5 * (float(obstacles_xy[0][1]) + float(obstacles_xy[1][1]))
                else:
                    self._defect_dwb_gap_center_local_x[env_id] = float(obstacles_xy[0][0])
                    self._defect_dwb_gap_center_local_y[env_id] = float(obstacles_xy[0][1])
            if emit_single_gap_sample and sample_robot_xy is None:
                sample_robot_xy = (float(robot_xy[0]), float(robot_xy[1]))
                sample_goal_xy = (float(goal_xy[0]), float(goal_xy[1]))
                if len(obstacles_xy) >= 1:
                    sample_obstacle_xy = (float(obstacles_xy[0][0]), float(obstacles_xy[0][1]))

        if scene_mode == "single_obstacle_symmetric_gap":
            self._single_gap_goal_debug_counter += 1
            if emit_single_gap_sample and sample_robot_xy is not None and sample_goal_xy is not None:
                obstacle_str = "none"
                if sample_obstacle_xy is not None:
                    obstacle_str = f"({sample_obstacle_xy[0]:.2f},{sample_obstacle_xy[1]:.2f})"
                print(
                    "[SAMPLE][SingleGap] "
                    f"spawn=({sample_robot_xy[0]:.2f},{sample_robot_xy[1]:.2f}) "
                    f"goal=({sample_goal_xy[0]:.2f},{sample_goal_xy[1]:.2f}) "
                    f"obstacle={obstacle_str} "
                    f"goal_mode={getattr(self.cfg, 'defect_single_gap_goal_mode', 'unknown')} "
                    f"rear_clear={getattr(self.cfg, 'defect_single_gap_goal_rear_clearance_min', float('nan')):.2f} "
                    f"rear_lat={getattr(self.cfg, 'defect_single_gap_goal_rear_lateral_center', float('nan')):.2f}±"
                    f"{getattr(self.cfg, 'defect_single_gap_goal_rear_lateral_offset_max', float('nan')):.2f} "
                    f"bridge_clear={getattr(self.cfg, 'defect_single_gap_goal_bridge_clearance_min', float('nan')):.2f} "
                    f"bridge_lat={getattr(self.cfg, 'defect_single_gap_goal_bridge_lateral_center', float('nan')):.2f}±"
                    f"{getattr(self.cfg, 'defect_single_gap_goal_bridge_lateral_offset_max', float('nan')):.2f}"
                )

    def _parse_fixed_xy(self, attr_name: str, default_xy: tuple[float, float]) -> tuple[float, float]:
        raw = getattr(self.cfg, attr_name, default_xy)
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            try:
                return float(raw[0]), float(raw[1])
            except (TypeError, ValueError):
                return default_xy
        return default_xy

    def _parse_fixed_obstacle_xy(self) -> list[tuple[float, float]]:
        raw = getattr(self.cfg, "fixed_scene_obstacle_xy", ())
        if not isinstance(raw, (list, tuple)):
            return []
        parsed: list[tuple[float, float]] = []
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                parsed.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                continue
        return parsed

    def _parse_fixed_scene_multi_scene_params(self) -> list[tuple[float, float, float, float, float, float]]:
        raw = getattr(self.cfg, "fixed_scene_multi_scene_params", ())
        if not isinstance(raw, (list, tuple)):
            return []
        parsed: list[tuple[float, float, float, float, float, float]] = []
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) < 6:
                continue
            try:
                parsed.append(
                    (
                        float(item[0]),
                        float(item[1]),
                        float(item[2]),
                        float(item[3]),
                        float(item[4]),
                        max(0.2, float(item[5])),
                    )
                )
            except (TypeError, ValueError):
                continue
        return parsed

    def _parse_fixed_scene_multi_obstacle_xy(self) -> list[list[tuple[float, float]]]:
        raw = getattr(self.cfg, "fixed_scene_multi_obstacle_xy", ())
        if not isinstance(raw, (list, tuple)):
            return []
        parsed: list[list[tuple[float, float]]] = []
        for scene in raw:
            if not isinstance(scene, (list, tuple)):
                parsed.append([])
                continue
            scene_obstacles: list[tuple[float, float]] = []
            for item in scene:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                try:
                    scene_obstacles.append((float(item[0]), float(item[1])))
                except (TypeError, ValueError):
                    continue
            parsed.append(scene_obstacles)
        return parsed

    def _parse_fixed_scene_multi_index_weights(self) -> list[float]:
        raw = getattr(self.cfg, "fixed_scene_multi_index_weights", ())
        if not isinstance(raw, (list, tuple)):
            return []
        parsed: list[float] = []
        for item in raw:
            try:
                parsed.append(max(0.0, float(item)))
            except (TypeError, ValueError):
                continue
        return parsed

    def _set_linear_layout_fixed_half_width(self, env_ids: torch.Tensor, half_width: torch.Tensor) -> None:
        if env_ids.numel() == 0 or not bool(getattr(self.cfg, "junction_enable", False)):
            return
        if half_width.numel() != env_ids.numel():
            return

        self._clear_turn_cmd(env_ids)
        self.junction_type[env_ids] = self.TOPO_T
        self.junction_active_arms[env_ids] = False
        self.junction_active_arms[env_ids, self.ARM_W] = True
        self.junction_active_arms[env_ids, self.ARM_E] = True
        self._junction_wall_seg_valid[env_ids] = False
        self._junction_wall_seg_start[env_ids] = 0.0
        self._junction_wall_seg_end[env_ids] = 0.0

        half_length = float(
            self.cfg.junction_half_length if bool(getattr(self.cfg, "junction_enable", False)) else self.cfg.corridor_half_length
        )
        self.layout_half_width[env_ids] = torch.clamp(half_width, min=0.2)
        self.layout_half_length[env_ids] = half_length
        self.layout_open_area[env_ids] = False

        for local_idx, env_id in enumerate(env_ids.detach().cpu().tolist()):
            origin_xy = self.scene.env_origins[env_id, :2]
            segments_local = self._build_junction_boundary_segments_local(
                active_w=True,
                active_e=True,
                active_n=False,
                active_s=False,
                half_w=float(self.layout_half_width[env_id].item()),
                half_l=float(self.layout_half_length[env_id].item()),
            )
            if segments_local.numel() == 0:
                continue
            segments_world = segments_local + origin_xy.view(1, 1, 2)
            seg_count = min(segments_world.shape[0], self._junction_max_segments)
            self._junction_wall_seg_start[env_id, :seg_count] = segments_world[:seg_count, 0, :]
            self._junction_wall_seg_end[env_id, :seg_count] = segments_world[:seg_count, 1, :]
            self._junction_wall_seg_valid[env_id, :seg_count] = True

    def _reset_fixed_scene_multi(self, env_ids: torch.Tensor, default_root_state: torch.Tensor) -> None:
        scene_params = self._parse_fixed_scene_multi_scene_params()
        if len(scene_params) == 0:
            return
        obstacle_scenes = self._parse_fixed_scene_multi_obstacle_xy()
        scene_count = len(scene_params)
        env_origins_xy = self.scene.env_origins[env_ids, :2]
        override_idx = int(getattr(self.cfg, "fixed_scene_multi_index_override", -1))
        if 0 <= override_idx < scene_count:
            scene_idx = torch.full((env_ids.numel(),), override_idx, dtype=torch.long, device=self.device)
        else:
            index_pool_cfg = tuple(getattr(self.cfg, "fixed_scene_multi_index_pool", ()))
            index_pool = [int(idx) for idx in index_pool_cfg if 0 <= int(idx) < scene_count]
            if len(index_pool) > 0:
                index_pool_tensor = torch.tensor(index_pool, dtype=torch.long, device=self.device)
                index_weights = self._parse_fixed_scene_multi_index_weights()
                if len(index_weights) == len(index_pool):
                    weights_tensor = torch.tensor(index_weights, dtype=torch.float32, device=self.device)
                    if float(weights_tensor.sum().item()) > 0.0:
                        sampled = torch.multinomial(
                            weights_tensor / weights_tensor.sum(),
                            env_ids.numel(),
                            replacement=True,
                        )
                    else:
                        sampled = torch.randint(
                            0,
                            index_pool_tensor.numel(),
                            (env_ids.numel(),),
                            dtype=torch.long,
                            device=self.device,
                        )
                else:
                    sampled = torch.randint(
                        0,
                        index_pool_tensor.numel(),
                        (env_ids.numel(),),
                        dtype=torch.long,
                        device=self.device,
                    )
                scene_idx = index_pool_tensor[sampled]
            else:
                scene_idx = torch.randint(0, scene_count, (env_ids.numel(),), dtype=torch.long, device=self.device)

        scene_idx_cpu = scene_idx.detach().cpu().tolist()
        self.fixed_scene_active_index[env_ids] = scene_idx

        if bool(getattr(self.cfg, "fixed_scene_use_linear_layout", True)) and bool(getattr(self.cfg, "junction_enable", False)):
            half_width = torch.tensor(
                [scene_params[idx][5] for idx in scene_idx_cpu],
                dtype=torch.float32,
                device=self.device,
            )
            self._set_linear_layout_fixed_half_width(env_ids, half_width)

        self._clear_turn_cmd(env_ids)
        self._set_planner_state(env_ids, "explore")

        robot_pose = torch.tensor(
            [[scene_params[idx][0], scene_params[idx][1], scene_params[idx][2]] for idx in scene_idx_cpu],
            dtype=torch.float32,
            device=self.device,
        )
        default_root_state[:, 0] = env_origins_xy[:, 0] + robot_pose[:, 0]
        default_root_state[:, 1] = env_origins_xy[:, 1] + robot_pose[:, 1]
        default_root_state[:, 3:7] = 0.0
        default_root_state[:, 3] = torch.cos(0.5 * robot_pose[:, 2])
        default_root_state[:, 6] = torch.sin(0.5 * robot_pose[:, 2])

        goal_xy = torch.tensor(
            [[scene_params[idx][3], scene_params[idx][4]] for idx in scene_idx_cpu],
            dtype=torch.float32,
            device=self.device,
        )
        self.goal_pos_w[env_ids, 0] = env_origins_xy[:, 0] + goal_xy[:, 0]
        self.goal_pos_w[env_ids, 1] = env_origins_xy[:, 1] + goal_xy[:, 1]
        self.goal_pos_w[env_ids, 2] = float(self.cfg.goal_radius)

        inactive_offset_x = float(getattr(self.cfg, "fixed_scene_inactive_obstacle_offset_x", 50.0))
        inactive_offset_y = float(getattr(self.cfg, "fixed_scene_inactive_obstacle_offset_y", 50.0))
        obstacle_z = 0.5 * float(self.cfg.obstacle_height)
        num_obstacles = int(self.cfg.num_obstacles)
        env_ids_cpu = env_ids.detach().cpu().tolist()
        for local_idx, env_id in enumerate(env_ids_cpu):
            scene_obstacles = obstacle_scenes[scene_idx_cpu[local_idx]] if scene_idx_cpu[local_idx] < len(obstacle_scenes) else []
            for obs_idx in range(num_obstacles):
                if obs_idx < len(scene_obstacles):
                    ox, oy = scene_obstacles[obs_idx]
                    self.obstacle_pos_w[env_id, obs_idx, 0] = self.scene.env_origins[env_id, 0] + ox
                    self.obstacle_pos_w[env_id, obs_idx, 1] = self.scene.env_origins[env_id, 1] + oy
                else:
                    self.obstacle_pos_w[env_id, obs_idx, 0] = self.scene.env_origins[env_id, 0] + inactive_offset_x + float(obs_idx)
                    self.obstacle_pos_w[env_id, obs_idx, 1] = self.scene.env_origins[env_id, 1] + inactive_offset_y
                self.obstacle_pos_w[env_id, obs_idx, 2] = obstacle_z

    def _reset_fixed_scene(self, env_ids: torch.Tensor, default_root_state: torch.Tensor) -> None:
        if self._use_fixed_scene_multi_reset():
            self._reset_fixed_scene_multi(env_ids, default_root_state)
            return

        env_origins_xy = self.scene.env_origins[env_ids, :2]
        if bool(getattr(self.cfg, "fixed_scene_use_linear_layout", True)) and bool(getattr(self.cfg, "junction_enable", False)):
            self._set_linear_layout(env_ids)
        self._clear_turn_cmd(env_ids)
        self._set_planner_state(env_ids, "explore")
        self.fixed_scene_active_index[env_ids] = 0

        robot_spawn_xy = self._parse_fixed_xy("fixed_scene_robot_spawn_xy", (-2.0, 0.0))
        robot_spawn_yaw = float(getattr(self.cfg, "fixed_scene_robot_spawn_yaw", 0.0))
        default_root_state[:, 0] = env_origins_xy[:, 0] + robot_spawn_xy[0]
        default_root_state[:, 1] = env_origins_xy[:, 1] + robot_spawn_xy[1]
        default_root_state[:, 3:7] = 0.0
        default_root_state[:, 3] = math.cos(0.5 * robot_spawn_yaw)
        default_root_state[:, 6] = math.sin(0.5 * robot_spawn_yaw)

        goal_xy = self._parse_fixed_xy("fixed_scene_goal_xy", (2.0, 0.0))
        self.goal_pos_w[env_ids, 0] = env_origins_xy[:, 0] + goal_xy[0]
        self.goal_pos_w[env_ids, 1] = env_origins_xy[:, 1] + goal_xy[1]
        self.goal_pos_w[env_ids, 2] = float(self.cfg.goal_radius)

        obstacle_xy = self._parse_fixed_obstacle_xy()
        inactive_offset_x = float(getattr(self.cfg, "fixed_scene_inactive_obstacle_offset_x", 50.0))
        inactive_offset_y = float(getattr(self.cfg, "fixed_scene_inactive_obstacle_offset_y", 50.0))
        obstacle_z = 0.5 * float(self.cfg.obstacle_height)
        for obs_idx in range(int(self.cfg.num_obstacles)):
            if obs_idx < len(obstacle_xy):
                ox, oy = obstacle_xy[obs_idx]
                self.obstacle_pos_w[env_ids, obs_idx, 0] = env_origins_xy[:, 0] + ox
                self.obstacle_pos_w[env_ids, obs_idx, 1] = env_origins_xy[:, 1] + oy
            else:
                self.obstacle_pos_w[env_ids, obs_idx, 0] = env_origins_xy[:, 0] + inactive_offset_x + float(obs_idx)
                self.obstacle_pos_w[env_ids, obs_idx, 1] = env_origins_xy[:, 1] + inactive_offset_y
            self.obstacle_pos_w[env_ids, obs_idx, 2] = obstacle_z

    def _sample_relay_goal_distances(
        self,
        num: int,
        dist_min_attr: str = "relay_goal_distance_min",
        dist_max_attr: str = "relay_goal_distance_max",
        close_prob_attr: str | None = None,
        close_min_attr: str | None = None,
        close_max_attr: str | None = None,
        fallback_min_attr: str | None = None,
        fallback_max_attr: str | None = None,
    ) -> torch.Tensor:
        if fallback_min_attr is None:
            fallback_min_attr = "relay_goal_distance_min"
        if fallback_max_attr is None:
            fallback_max_attr = "relay_goal_distance_max"
        dist_min = float(getattr(self.cfg, dist_min_attr, getattr(self.cfg, fallback_min_attr, 1.5)))
        dist_max = max(dist_min, float(getattr(self.cfg, dist_max_attr, getattr(self.cfg, fallback_max_attr, dist_min))))
        base_samples = torch.rand(num, device=self.device) * max(1e-6, dist_max - dist_min) + dist_min
        if close_prob_attr is None or close_min_attr is None or close_max_attr is None:
            return base_samples
        close_prob = float(getattr(self.cfg, close_prob_attr, 0.0))
        close_prob = min(1.0, max(0.0, close_prob))
        if close_prob <= 0.0:
            return base_samples
        close_min = float(getattr(self.cfg, close_min_attr, dist_min))
        close_max = max(close_min, float(getattr(self.cfg, close_max_attr, close_min)))
        close_samples = torch.rand(num, device=self.device) * max(1e-6, close_max - close_min) + close_min
        use_close = torch.rand(num, device=self.device) < close_prob
        return torch.where(use_close, close_samples, base_samples)

    def _sample_half_width_choices(self, num: int, attr_name: str, default_half_width: float) -> torch.Tensor:
        raw_choices = getattr(self.cfg, attr_name, (default_half_width,))
        if isinstance(raw_choices, (int, float)):
            choices = (float(raw_choices),)
        else:
            choices = tuple(float(choice) for choice in raw_choices)
        choices = tuple(max(0.05, choice) for choice in choices)
        if len(choices) == 0:
            choices = (max(0.05, float(default_half_width)),)
        choice_tensor = torch.tensor(choices, dtype=torch.float32, device=self.device)
        if choice_tensor.numel() == 1:
            return torch.full((num,), float(choice_tensor[0].item()), device=self.device)
        choice_idx = torch.randint(0, choice_tensor.numel(), (num,), device=self.device)
        return choice_tensor[choice_idx]

    def _set_linear_layout(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0 or not bool(getattr(self.cfg, "junction_enable", False)):
            return

        self._clear_turn_cmd(env_ids)
        self.junction_type[env_ids] = self.TOPO_T
        self.junction_active_arms[env_ids] = False
        self.junction_active_arms[env_ids, self.ARM_W] = True
        self.junction_active_arms[env_ids, self.ARM_E] = True
        self._junction_wall_seg_valid[env_ids] = False
        self._junction_wall_seg_start[env_ids] = 0.0
        self._junction_wall_seg_end[env_ids] = 0.0
        num_envs = env_ids.numel()
        default_half_width = float(
            self.cfg.junction_half_width if bool(getattr(self.cfg, "junction_enable", False)) else self.cfg.corridor_half_width
        )
        default_half_length = float(
            self.cfg.junction_half_length if bool(getattr(self.cfg, "junction_enable", False)) else self.cfg.corridor_half_length
        )
        sampled_half_width = self._sample_half_width_choices(
            num_envs,
            "linear_corridor_half_width_choices",
            default_half_width,
        )
        open_prob = min(1.0, max(0.0, float(getattr(self.cfg, "linear_open_area_prob", 0.0))))
        open_mask = torch.rand(num_envs, device=self.device) < open_prob
        open_half_extent = max(
            0.5,
            float(getattr(self.cfg, "linear_open_area_half_extent", default_half_length)),
        )
        self.layout_half_width[env_ids] = torch.where(
            open_mask,
            torch.full_like(sampled_half_width, open_half_extent),
            sampled_half_width,
        )
        self.layout_half_length[env_ids] = torch.where(
            open_mask,
            torch.full_like(sampled_half_width, open_half_extent),
            torch.full_like(sampled_half_width, default_half_length),
        )
        self.layout_open_area[env_ids] = open_mask

        for local_idx, env_id in enumerate(env_ids.detach().cpu().tolist()):
            if bool(open_mask[local_idx].item()):
                continue
            origin_xy = self.scene.env_origins[env_id, :2]
            segments_local = self._build_junction_boundary_segments_local(
                active_w=True,
                active_e=True,
                active_n=False,
                active_s=False,
                half_w=float(self.layout_half_width[env_id].item()),
                half_l=float(self.layout_half_length[env_id].item()),
            )
            if segments_local.numel() == 0:
                continue
            segments_world = segments_local + origin_xy.view(1, 1, 2)
            seg_count = min(segments_world.shape[0], self._junction_max_segments)
            self._junction_wall_seg_start[env_id, :seg_count] = segments_world[:seg_count, 0, :]
            self._junction_wall_seg_end[env_id, :seg_count] = segments_world[:seg_count, 1, :]
            self._junction_wall_seg_valid[env_id, :seg_count] = True

    def _compute_state_base_speed_cap(self, state_idx: torch.Tensor) -> torch.Tensor:
        cap = torch.full(state_idx.shape, float(getattr(self.cfg, "speed_cap_nominal", self.cfg.v_max)), device=self.device)
        if self._use_mixed_planner_state_training():
            explore_cap = float(getattr(self.cfg, "speed_cap_explore", self.cfg.v_max))
            turn_cap = float(getattr(self.cfg, "speed_cap_turn", self.cfg.v_max))
            fallback_cap = float(getattr(self.cfg, "speed_cap_fallback", self.cfg.v_max))
            return_cap = float(getattr(self.cfg, "speed_cap_return", self.cfg.v_max))
            cap = torch.where(state_idx == self.STATE_EXPLORE, torch.minimum(cap, torch.full_like(cap, explore_cap)), cap)
            cap = torch.where(state_idx == self.STATE_JUNCTION, torch.minimum(cap, torch.full_like(cap, turn_cap)), cap)
            cap = torch.where(state_idx == self.STATE_FALLBACK, torch.minimum(cap, torch.full_like(cap, fallback_cap)), cap)
            cap = torch.where(state_idx == self.STATE_RETURN, torch.minimum(cap, torch.full_like(cap, return_cap)), cap)
        elif bool(getattr(self.cfg, "junction_enable", False)):
            cap = torch.minimum(cap, torch.full_like(cap, float(getattr(self.cfg, "speed_cap_turn", self.cfg.v_max))))
        else:
            cap = torch.minimum(cap, torch.full_like(cap, float(getattr(self.cfg, "speed_cap_explore", self.cfg.v_max))))
        return torch.clamp(cap, min=0.0)

    def _compute_speed_cap(self, min_lidar: torch.Tensor) -> torch.Tensor:
        cap = self._compute_state_base_speed_cap(self.planner_state_index).to(dtype=min_lidar.dtype)
        if self._use_mixed_planner_state_training():
            return_mask = self.planner_state_index == self.STATE_RETURN
            if torch.any(return_mask):
                return_cap_high = float(getattr(self.cfg, "return_speed_cap_high_clearance", self.cfg.speed_cap_return))
                return_cap_mid = float(getattr(self.cfg, "return_speed_cap_mid_clearance", self.cfg.speed_cap_return))
                return_cap_low = float(getattr(self.cfg, "return_speed_cap_low_clearance", self.cfg.speed_cap_return))
                return_cap_high_th = float(getattr(self.cfg, "return_speed_cap_high_clearance_threshold", 0.80))
                return_cap_mid_th = float(getattr(self.cfg, "return_speed_cap_mid_clearance_threshold", 0.60))
                return_cap = torch.where(
                    min_lidar > return_cap_high_th,
                    torch.full_like(cap, return_cap_high),
                    torch.where(
                        min_lidar > return_cap_mid_th,
                        torch.full_like(cap, return_cap_mid),
                        torch.full_like(cap, return_cap_low),
                    ),
                )
                cap = torch.where(return_mask, torch.minimum(cap, return_cap), cap)
        narrow_threshold = float(getattr(self.cfg, "speed_cap_narrow_clearance", 0.0))
        if narrow_threshold > 0.0:
            narrow_cap = torch.full_like(cap, float(getattr(self.cfg, "speed_cap_narrow", self.cfg.v_max)))
            narrow_mask = min_lidar <= narrow_threshold
            if self._use_mixed_planner_state_training():
                narrow_mask = torch.logical_and(narrow_mask, self.planner_state_index != self.STATE_RETURN)
            cap = torch.where(narrow_mask, torch.minimum(cap, narrow_cap), cap)
        return torch.clamp(cap, min=0.0)

    def _build_junction_boundary_segments_local(
        self,
        active_w: bool,
        active_e: bool,
        active_n: bool,
        active_s: bool,
        half_w: float | None = None,
        half_l: float | None = None,
    ) -> torch.Tensor:
        """Build local-space boundary segments for a junction road mask."""
        if half_w is None:
            half_w = float(self.cfg.junction_half_width)
        if half_l is None:
            half_l = float(self.cfg.junction_half_length)
        x_edges = [-half_l, -half_w, half_w, half_l]
        y_edges = [-half_l, -half_w, half_w, half_l]

        # 3x3 grid cells: center always open, plus selected arm cells.
        active_cells = [[False, False, False], [False, True, False], [False, False, False]]
        if active_w:
            active_cells[0][1] = True
        if active_e:
            active_cells[2][1] = True
        if active_n:
            active_cells[1][2] = True
        if active_s:
            active_cells[1][0] = True

        segments: list[list[list[float]]] = []
        for i in range(3):
            for j in range(3):
                if not active_cells[i][j]:
                    continue
                x0, x1 = x_edges[i], x_edges[i + 1]
                y0, y1 = y_edges[j], y_edges[j + 1]
                # left edge
                if i == 0 or not active_cells[i - 1][j]:
                    segments.append([[x0, y0], [x0, y1]])
                # right edge
                if i == 2 or not active_cells[i + 1][j]:
                    segments.append([[x1, y0], [x1, y1]])
                # bottom edge
                if j == 0 or not active_cells[i][j - 1]:
                    segments.append([[x0, y0], [x1, y0]])
                # top edge
                if j == 2 or not active_cells[i][j + 1]:
                    segments.append([[x0, y1], [x1, y1]])

        if len(segments) == 0:
            return torch.zeros((0, 2, 2), dtype=torch.float32, device=self.device)
        return torch.tensor(segments, dtype=torch.float32, device=self.device)

    def _sample_junction_layout(self, env_ids: torch.Tensor) -> None:
        """Sample L/T/X topology, command, and wall segments for each environment."""
        if not bool(getattr(self.cfg, "junction_enable", False)):
            return
        if env_ids.numel() == 0:
            return

        dir_lookup = torch.tensor(
            [
                [0.0, 1.0],   # left
                [1.0, 0.0],   # straight
                [0.0, -1.0],  # right
            ],
            device=self.device,
        )
        self._junction_wall_seg_valid[env_ids] = False
        self._junction_wall_seg_start[env_ids] = 0.0
        self._junction_wall_seg_end[env_ids] = 0.0

        env_ids_list = env_ids.detach().cpu().tolist()
        env_origins_xy = self.scene.env_origins[env_ids, :2]
        x_prob = float(getattr(self.cfg, "junction_x_sampling_prob", 0.33))
        x_prob = min(1.0, max(0.0, x_prob))
        sampled_half_width = self._sample_half_width_choices(env_ids.numel(), "junction_half_width_choices", float(self.cfg.junction_half_width))
        self.layout_half_width[env_ids] = sampled_half_width
        self.layout_half_length[env_ids] = float(self.cfg.junction_half_length)
        self.layout_open_area[env_ids] = False
        for local_idx, env_id in enumerate(env_ids_list):
            if float(torch.rand(1, device=self.device).item()) < x_prob:
                topo = self.TOPO_X
            else:
                topo = self.TOPO_L if float(torch.rand(1, device=self.device).item()) < 0.5 else self.TOPO_T
            active_w = True
            active_e = False
            active_n = False
            active_s = False

            if topo == self.TOPO_L:
                # L junction: only one turn option.
                if float(torch.rand(1, device=self.device).item()) < 0.5:
                    active_n = True
                    cmd = self.TURN_LEFT
                else:
                    active_s = True
                    cmd = self.TURN_RIGHT
            elif topo == self.TOPO_T:
                # T junction: remove one of the three outgoing directions.
                missing = int(torch.randint(0, 3, (1,), device=self.device).item())  # left / straight / right
                active_n = missing != 0
                active_e = missing != 1
                active_s = missing != 2
                cmd_options = []
                if active_n:
                    cmd_options.append(self.TURN_LEFT)
                if active_e:
                    cmd_options.append(self.TURN_STRAIGHT)
                if active_s:
                    cmd_options.append(self.TURN_RIGHT)
                cmd = int(cmd_options[int(torch.randint(0, len(cmd_options), (1,), device=self.device).item())])
            else:
                # X junction: all outgoing directions available.
                active_e = True
                active_n = True
                active_s = True
                cmd = int(torch.randint(0, 3, (1,), device=self.device).item())

            self.junction_type[env_id] = topo
            self.turn_cmd_index[env_id] = cmd
            self.junction_active_arms[env_id, self.ARM_W] = active_w
            self.junction_active_arms[env_id, self.ARM_E] = active_e
            self.junction_active_arms[env_id, self.ARM_N] = active_n
            self.junction_active_arms[env_id, self.ARM_S] = active_s

            segments_local = self._build_junction_boundary_segments_local(
                active_w=active_w,
                active_e=active_e,
                active_n=active_n,
                active_s=active_s,
                half_w=float(sampled_half_width[local_idx].item()),
                half_l=float(self.layout_half_length[env_id].item()),
            )
            seg_count = min(int(segments_local.shape[0]), int(self._junction_max_segments))
            if seg_count > 0:
                origin_xy = env_origins_xy[local_idx].view(1, 1, 2)
                segments_world = segments_local[:seg_count] + origin_xy
                self._junction_wall_seg_start[env_id, :seg_count] = segments_world[:, 0, :]
                self._junction_wall_seg_end[env_id, :seg_count] = segments_world[:, 1, :]
                self._junction_wall_seg_valid[env_id, :seg_count] = True

        cmd_idx = self.turn_cmd_index[env_ids]
        self.turn_cmd_onehot[env_ids] = 0.0
        self.turn_cmd_onehot[env_ids, cmd_idx] = 1.0
        self.turn_cmd_dir_b[env_ids] = dir_lookup[cmd_idx]
        self._visualize_junction_walls()

    def _update_terminal_metrics(self, env_ids: torch.Tensor) -> None:
        """Update and print windowed terminal metrics for LiDAR navigation training."""
        valid_mask = self.episode_length_buf[env_ids] > 0
        if not torch.any(valid_mask):
            return
        done_ids = env_ids[valid_mask]
        done_count = int(done_ids.numel())
        if done_count == 0:
            return

        success = self._last_done_success[done_ids]
        collision = self._last_done_collision[done_ids]
        collision_lidar = self._last_done_collision_lidar[done_ids]
        collision_contact = self._last_done_collision_contact[done_ids]
        timeout = self._last_done_timeout[done_ids]

        # Keep terminal categories mutually exclusive for easy interpretation.
        success_only = success
        collision_only = torch.logical_and(~success_only, collision)
        timeout_only = torch.logical_and(~success_only, torch.logical_and(~collision_only, timeout))

        self._metrics_total_episodes += done_count
        self._metrics_window_episodes += done_count
        self._metrics_window_success += int(success_only.sum().item())
        self._metrics_window_collision += int(collision_only.sum().item())
        self._metrics_window_collision_lidar += int(collision_lidar.sum().item())
        self._metrics_window_collision_contact += int(collision_contact.sum().item())
        self._metrics_window_timeout += int(timeout_only.sum().item())
        self._metrics_window_episode_length_sum += float(self.episode_length_buf[done_ids].float().sum().item())
        self._metrics_window_return_sum += float(self.episode_return[done_ids].sum().item())
        self._metrics_window_goal_dist_sum += float(self._last_done_goal_dist[done_ids].sum().item())
        self._metrics_window_min_lidar_sum += float(self._last_done_min_lidar[done_ids].sum().item())

        if self._metrics_window_episodes >= self._metrics_print_interval_episodes:
            total = float(self._metrics_window_episodes)
            iteration = int(self.common_step_counter // self._debug_iteration_rollouts)
            success_rate = self._metrics_window_success / total
            collision_rate = self._metrics_window_collision / total
            collision_lidar_rate = self._metrics_window_collision_lidar / total
            collision_contact_rate = self._metrics_window_collision_contact / total
            timeout_rate = self._metrics_window_timeout / total
            avg_ep_len = self._metrics_window_episode_length_sum / total
            avg_return = self._metrics_window_return_sum / total
            avg_goal_dist = self._metrics_window_goal_dist_sum / total
            avg_min_lidar = self._metrics_window_min_lidar_sum / total
            print(
                (
                    f"[METRIC][LidarNav] iter={iteration} step={self.common_step_counter} "
                    f"ep_total={self._metrics_total_episodes} "
                    f"window={self._metrics_window_episodes} "
                    f"success={success_rate:.1%} collision={collision_rate:.1%} "
                    f"coll_lidar={collision_lidar_rate:.1%} coll_contact={collision_contact_rate:.1%} "
                    f"timeout={timeout_rate:.1%} "
                    f"avg_len={avg_ep_len:.1f} avg_return={avg_return:.2f} "
                    f"avg_goal_end_dist={avg_goal_dist:.2f} avg_min_lidar={avg_min_lidar:.2f} "
                    f"phase={self._debug_phase_tag}"
                ),
                flush=True,
            )
            self._metrics_window_episodes = 0
            self._metrics_window_success = 0
            self._metrics_window_collision = 0
            self._metrics_window_collision_lidar = 0
            self._metrics_window_collision_contact = 0
            self._metrics_window_timeout = 0
            self._metrics_window_episode_length_sum = 0.0
            self._metrics_window_return_sum = 0.0
            self._metrics_window_goal_dist_sum = 0.0
            self._metrics_window_min_lidar_sum = 0.0

    def _update_step_debug(
        self,
        total_reward: torch.Tensor,
        progress_reward: torch.Tensor,
        heading_reward: torch.Tensor,
        forward_goal_reward: torch.Tensor,
        speed_surge_reward: torch.Tensor,
        speed_deficit_penalty: torch.Tensor,
        clearance_reward: torch.Tensor,
        smooth_action_reward: torch.Tensor,
        turn_penalty: torch.Tensor,
        time_penalty: torch.Tensor,
        stall_penalty: torch.Tensor,
        danger_speed_penalty: torch.Tensor,
        defect_dwb_clearance_penalty: torch.Tensor,
        action_saturation_penalty: torch.Tensor,
        defect_single_gap_centerline_penalty: torch.Tensor,
        defect_dwb_post_gap_lateral_align_reward: torch.Tensor,
        success_reward: torch.Tensor,
        collision_penalty: torch.Tensor,
        curr_goal_dist: torch.Tensor,
        min_lidar: torch.Tensor,
        speed_cap: torch.Tensor,
        action_eff_abs_mean: torch.Tensor,
        forward_speed: torch.Tensor,
        yaw_rate_abs_mean: torch.Tensor,
        yaw_rate_abs_p95: torch.Tensor,
        filtered_omega_abs_mean: torch.Tensor,
        yaw_sign_flip_rate: torch.Tensor,
        success: torch.Tensor,
        collision: torch.Tensor,
        collision_lidar: torch.Tensor,
        collision_contact: torch.Tensor,
        stall: torch.Tensor,
        action_saturated: torch.Tensor,
    ) -> None:
        if not self._debug_print_enable:
            return

        self._debug_window_steps += 1
        self._debug_reward_sum += float(total_reward.mean().item())
        self._debug_progress_sum += float(progress_reward.mean().item())
        self._debug_heading_sum += float(heading_reward.mean().item())
        self._debug_forward_goal_sum += float(forward_goal_reward.mean().item())
        self._debug_speed_surge_sum += float(speed_surge_reward.mean().item())
        self._debug_speed_deficit_sum += float(speed_deficit_penalty.mean().item())
        self._debug_clearance_sum += float(clearance_reward.mean().item())
        self._debug_smooth_sum += float(smooth_action_reward.mean().item())
        self._debug_turn_penalty_sum += float(turn_penalty.mean().item())
        self._debug_time_penalty_sum += float(time_penalty.mean().item())
        self._debug_stall_penalty_sum += float(stall_penalty.mean().item())
        self._debug_danger_speed_penalty_sum += float(danger_speed_penalty.mean().item())
        self._debug_dwb_clearance_penalty_sum += float(defect_dwb_clearance_penalty.mean().item())
        self._debug_action_sat_penalty_sum += float(action_saturation_penalty.mean().item())
        self._debug_single_gap_center_penalty_sum += float(defect_single_gap_centerline_penalty.mean().item())
        self._debug_post_gap_lateral_align_sum += float(defect_dwb_post_gap_lateral_align_reward.mean().item())
        self._debug_success_bonus_sum += float(success_reward.mean().item())
        self._debug_collision_penalty_sum += float(collision_penalty.mean().item())
        self._debug_goal_dist_sum += float(curr_goal_dist.mean().item())
        self._debug_min_lidar_sum += float(min_lidar.mean().item())
        self._debug_action_abs_sum += float(torch.mean(torch.abs(self.actions)).item())
        self._debug_action_eff_abs_sum += float(action_eff_abs_mean.mean().item())
        self._debug_forward_speed_sum += float(forward_speed.mean().item())
        self._debug_yaw_rate_abs_mean_sum += float(yaw_rate_abs_mean.item())
        self._debug_yaw_rate_abs_p95_sum += float(yaw_rate_abs_p95.item())
        self._debug_filtered_omega_abs_mean_sum += float(filtered_omega_abs_mean.item())
        self._debug_yaw_sign_flip_frac_sum += float(yaw_sign_flip_rate.item())
        self._debug_success_frac_sum += float(success.float().mean().item())
        self._debug_collision_frac_sum += float(collision.float().mean().item())
        self._debug_collision_lidar_frac_sum += float(collision_lidar.float().mean().item())
        self._debug_collision_contact_frac_sum += float(collision_contact.float().mean().item())
        self._debug_stall_frac_sum += float(stall.float().mean().item())
        self._debug_action_sat_frac_sum += float(action_saturated.float().mean().item())

        if self.common_step_counter <= 0 or (self.common_step_counter % self._debug_print_interval_steps != 0):
            return

        iteration = int(self.common_step_counter // self._debug_iteration_rollouts)
        denom = float(max(1, self._debug_window_steps))
        state_ratios = []
        state_cap_means = []
        for state_idx in range(4):
            state_mask = self.planner_state_index == state_idx
            state_ratios.append(float(state_mask.float().mean().item()))
            if torch.any(state_mask):
                state_cap_means.append(float(speed_cap[state_mask].mean().item()))
            else:
                state_cap_means.append(float("nan"))
        print(
            (
                f"[DEBUG][LidarNav] iter={iteration} step={self.common_step_counter} "
                f"r={self._debug_reward_sum / denom:.3f} "
                f"prog={self._debug_progress_sum / denom:.3f} "
                f"head={self._debug_heading_sum / denom:.3f} "
                f"fwd_goal={self._debug_forward_goal_sum / denom:.3f} "
                f"surge={self._debug_speed_surge_sum / denom:.3f} "
                f"deficit_pen={self._debug_speed_deficit_sum / denom:.3f} "
                f"clear={self._debug_clearance_sum / denom:.3f} "
                f"smooth={self._debug_smooth_sum / denom:.3f} "
                f"turn_pen={self._debug_turn_penalty_sum / denom:.3f} "
                f"time={self._debug_time_penalty_sum / denom:.3f} "
                f"stall_pen={self._debug_stall_penalty_sum / denom:.3f} "
                f"danger_pen={self._debug_danger_speed_penalty_sum / denom:.3f} "
                f"dwb_clear_pen={self._debug_dwb_clearance_penalty_sum / denom:.3f} "
                f"sat_pen={self._debug_action_sat_penalty_sum / denom:.3f} "
                f"front_center_pen={self._debug_single_gap_center_penalty_sum / denom:.3f} "
                f"post_gap_lat={self._debug_post_gap_lateral_align_sum / denom:.3f} "
                f"succ_bonus={self._debug_success_bonus_sum / denom:.3f} "
                f"coll_pen={self._debug_collision_penalty_sum / denom:.3f} "
                f"goal_d={self._debug_goal_dist_sum / denom:.3f} "
                f"min_lidar={self._debug_min_lidar_sum / denom:.3f} "
                f"|a|={self._debug_action_abs_sum / denom:.3f} "
                f"|a_eff|={self._debug_action_eff_abs_sum / denom:.3f} "
                f"v_fwd={self._debug_forward_speed_sum / denom:.3f} "
                f"yaw_abs={self._debug_yaw_rate_abs_mean_sum / denom:.3f} "
                f"yaw_p95={self._debug_yaw_rate_abs_p95_sum / denom:.3f} "
                f"omega_filt={self._debug_filtered_omega_abs_mean_sum / denom:.3f} "
                f"yaw_flip={self._debug_yaw_sign_flip_frac_sum / denom:.1%} "
                f"succ_step={self._debug_success_frac_sum / denom:.1%} "
                f"coll_step={self._debug_collision_frac_sum / denom:.1%} "
                f"coll_lidar_step={self._debug_collision_lidar_frac_sum / denom:.1%} "
                f"coll_contact_step={self._debug_collision_contact_frac_sum / denom:.1%} "
                f"stall_step={self._debug_stall_frac_sum / denom:.1%} "
                f"sat_step={self._debug_action_sat_frac_sum / denom:.1%} "
                f"mix=e:{state_ratios[self.STATE_EXPLORE]:.1%}/f:{state_ratios[self.STATE_FALLBACK]:.1%}/"
                f"j:{state_ratios[self.STATE_JUNCTION]:.1%}/r:{state_ratios[self.STATE_RETURN]:.1%} "
                f"cap=e:{state_cap_means[self.STATE_EXPLORE]:.2f}/f:{state_cap_means[self.STATE_FALLBACK]:.2f}/"
                f"j:{state_cap_means[self.STATE_JUNCTION]:.2f}/r:{state_cap_means[self.STATE_RETURN]:.2f} "
                f"phase={self._debug_phase_tag}"
            ),
            flush=True,
        )

        self._debug_window_steps = 0
        self._debug_reward_sum = 0.0
        self._debug_progress_sum = 0.0
        self._debug_heading_sum = 0.0
        self._debug_forward_goal_sum = 0.0
        self._debug_speed_surge_sum = 0.0
        self._debug_speed_deficit_sum = 0.0
        self._debug_clearance_sum = 0.0
        self._debug_smooth_sum = 0.0
        self._debug_turn_penalty_sum = 0.0
        self._debug_time_penalty_sum = 0.0
        self._debug_stall_penalty_sum = 0.0
        self._debug_danger_speed_penalty_sum = 0.0
        self._debug_dwb_clearance_penalty_sum = 0.0
        self._debug_action_sat_penalty_sum = 0.0
        self._debug_single_gap_center_penalty_sum = 0.0
        self._debug_post_gap_lateral_align_sum = 0.0
        self._debug_success_bonus_sum = 0.0
        self._debug_collision_penalty_sum = 0.0
        self._debug_goal_dist_sum = 0.0
        self._debug_min_lidar_sum = 0.0
        self._debug_action_abs_sum = 0.0
        self._debug_action_eff_abs_sum = 0.0
        self._debug_forward_speed_sum = 0.0
        self._debug_yaw_rate_abs_mean_sum = 0.0
        self._debug_yaw_rate_abs_p95_sum = 0.0
        self._debug_filtered_omega_abs_mean_sum = 0.0
        self._debug_yaw_sign_flip_frac_sum = 0.0
        self._debug_success_frac_sum = 0.0
        self._debug_collision_frac_sum = 0.0
        self._debug_collision_lidar_frac_sum = 0.0
        self._debug_collision_contact_frac_sum = 0.0
        self._debug_stall_frac_sum = 0.0
        self._debug_action_sat_frac_sum = 0.0

    def _update_reset_debug(
        self,
        env_ids: torch.Tensor,
        robot_pos_xy: torch.Tensor,
        robot_quat_wxyz: torch.Tensor,
    ) -> None:
        if not self._debug_print_enable or env_ids.numel() == 0:
            return

        state_idx = self.planner_state_index[env_ids]
        goal_vec_w = torch.zeros((env_ids.numel(), 3), device=self.device)
        goal_vec_w[:, :2] = self.goal_pos_w[env_ids, :2] - robot_pos_xy
        goal_vec_b = math_utils.quat_apply_inverse(robot_quat_wxyz, goal_vec_w)
        goal_xy_b = goal_vec_b[:, :2]
        goal_dist = torch.linalg.norm(goal_xy_b, dim=1)
        yaw_rad = torch.atan2(
            2.0 * (robot_quat_wxyz[:, 0] * robot_quat_wxyz[:, 3] + robot_quat_wxyz[:, 1] * robot_quat_wxyz[:, 2]),
            1.0 - 2.0 * (robot_quat_wxyz[:, 2].square() + robot_quat_wxyz[:, 3].square()),
        )
        yaw_deg = torch.rad2deg(yaw_rad)
        base_speed_cap = self._compute_state_base_speed_cap(state_idx)
        door_mask = self.fixed_scene_active_index[env_ids] == self.DEFECT_DOOR_DEADLOCK
        if bool(torch.any(door_mask).item()):
            door_yaw_deg = yaw_deg[door_mask]
            door_goal_lat = goal_xy_b[door_mask, 1]
            self._reset_debug_door_angle_count += int(door_mask.sum().item())
            self._reset_debug_door_angle_deg_sum += float(door_yaw_deg.sum().item())
            self._reset_debug_door_angle_deg_min = min(
                self._reset_debug_door_angle_deg_min, float(door_yaw_deg.min().item())
            )
            self._reset_debug_door_angle_deg_max = max(
                self._reset_debug_door_angle_deg_max, float(door_yaw_deg.max().item())
            )
            self._reset_debug_door_goal_lat_sum += float(door_goal_lat.sum().item())
            self._reset_debug_door_goal_lat_min = min(
                self._reset_debug_door_goal_lat_min, float(door_goal_lat.min().item())
            )
            self._reset_debug_door_goal_lat_max = max(
                self._reset_debug_door_goal_lat_max, float(door_goal_lat.max().item())
            )

        for planner_state in range(4):
            state_mask = state_idx == planner_state
            state_count = int(state_mask.sum().item())
            if state_count == 0:
                continue
            self._reset_debug_samples += state_count
            self._reset_debug_state_counts[planner_state] += state_count
            state_goal_dist = goal_dist[state_mask]
            state_goal_x_b = goal_xy_b[state_mask, 0]
            state_goal_y_abs = goal_xy_b[state_mask, 1].abs()
            state_yaw_deg = yaw_deg[state_mask]
            state_cap = base_speed_cap[state_mask]
            self._reset_debug_goal_dist_sum[planner_state] += float(state_goal_dist.sum().item())
            self._reset_debug_goal_dist_min[planner_state] = min(
                self._reset_debug_goal_dist_min[planner_state], float(state_goal_dist.min().item())
            )
            self._reset_debug_goal_dist_max[planner_state] = max(
                self._reset_debug_goal_dist_max[planner_state], float(state_goal_dist.max().item())
            )
            self._reset_debug_speed_cap_sum[planner_state] += float(state_cap.sum().item())
            self._reset_debug_goal_x_b_sum[planner_state] += float(state_goal_x_b.sum().item())
            self._reset_debug_goal_y_abs_sum[planner_state] += float(state_goal_y_abs.sum().item())
            self._reset_debug_yaw_deg_sum[planner_state] += float(state_yaw_deg.sum().item())

        if self._reset_debug_samples < self._reset_debug_interval_resets:
            return

        total = float(max(1, self._reset_debug_samples))
        dist_sum_total = sum(self._reset_debug_goal_dist_sum)
        dist_min_total = min(self._reset_debug_goal_dist_min)
        dist_max_total = max(self._reset_debug_goal_dist_max)

        def _mean(values: list[float], counts: list[int], idx: int) -> float:
            return values[idx] / max(1, counts[idx])

        counts = self._reset_debug_state_counts
        ratios = [count / total for count in counts]
        cap_means = [_mean(self._reset_debug_speed_cap_sum, counts, idx) for idx in range(4)]
        goal_x_means = [_mean(self._reset_debug_goal_x_b_sum, counts, idx) for idx in range(4)]
        goal_y_abs_means = [_mean(self._reset_debug_goal_y_abs_sum, counts, idx) for idx in range(4)]
        yaw_deg_means = [_mean(self._reset_debug_yaw_deg_sum, counts, idx) for idx in range(4)]
        door_angle_text = ""
        if self._reset_debug_door_angle_count > 0:
            door_angle_mean = self._reset_debug_door_angle_deg_sum / float(self._reset_debug_door_angle_count)
            door_goal_lat_mean = self._reset_debug_door_goal_lat_sum / float(self._reset_debug_door_angle_count)
            door_angle_text = (
                f"door_line_deg=mean:{door_angle_mean:.1f} "
                f"min:{self._reset_debug_door_angle_deg_min:.1f} "
                f"max:{self._reset_debug_door_angle_deg_max:.1f} "
                f"door_goal_lat_b=mean:{door_goal_lat_mean:.2f} "
                f"min:{self._reset_debug_door_goal_lat_min:.2f} "
                f"max:{self._reset_debug_door_goal_lat_max:.2f} "
            )
        print(
            (
                f"[RESET][LidarNav] samples={self._reset_debug_samples} "
                f"mix=e:{ratios[self.STATE_EXPLORE]:.1%}/f:{ratios[self.STATE_FALLBACK]:.1%}/"
                f"j:{ratios[self.STATE_JUNCTION]:.1%}/r:{ratios[self.STATE_RETURN]:.1%} "
                f"relay_d=mean:{dist_sum_total / total:.2f} min:{dist_min_total:.2f} max:{dist_max_total:.2f} "
                f"cap=e:{cap_means[self.STATE_EXPLORE]:.2f}/f:{cap_means[self.STATE_FALLBACK]:.2f}/"
                f"j:{cap_means[self.STATE_JUNCTION]:.2f}/r:{cap_means[self.STATE_RETURN]:.2f} "
                f"fb(goal_x_b={goal_x_means[self.STATE_FALLBACK]:.2f},|goal_y_b|={goal_y_abs_means[self.STATE_FALLBACK]:.2f},"
                f"yaw_deg={yaw_deg_means[self.STATE_FALLBACK]:.1f}) "
                f"ret(goal_x_b={goal_x_means[self.STATE_RETURN]:.2f},|goal_y_b|={goal_y_abs_means[self.STATE_RETURN]:.2f},"
                f"yaw_deg={yaw_deg_means[self.STATE_RETURN]:.1f}) "
                f"{door_angle_text}"
                f"phase={self._debug_phase_tag}"
            ),
            flush=True,
        )
        self._reset_debug_samples = 0
        self._reset_debug_state_counts = [0, 0, 0, 0]
        self._reset_debug_goal_dist_sum = [0.0, 0.0, 0.0, 0.0]
        self._reset_debug_goal_dist_min = [float("inf"), float("inf"), float("inf"), float("inf")]
        self._reset_debug_goal_dist_max = [0.0, 0.0, 0.0, 0.0]
        self._reset_debug_speed_cap_sum = [0.0, 0.0, 0.0, 0.0]
        self._reset_debug_goal_x_b_sum = [0.0, 0.0, 0.0, 0.0]
        self._reset_debug_goal_y_abs_sum = [0.0, 0.0, 0.0, 0.0]
        self._reset_debug_yaw_deg_sum = [0.0, 0.0, 0.0, 0.0]
        self._reset_debug_door_angle_count = 0
        self._reset_debug_door_angle_deg_sum = 0.0
        self._reset_debug_door_angle_deg_min = float("inf")
        self._reset_debug_door_angle_deg_max = float("-inf")
        self._reset_debug_door_goal_lat_sum = 0.0
        self._reset_debug_door_goal_lat_min = float("inf")
        self._reset_debug_door_goal_lat_max = float("-inf")

    def _spawn_goal_positions(self, env_ids: torch.Tensor, robot_pos_xy: torch.Tensor | None = None) -> None:
        num = len(env_ids)
        if robot_pos_xy is None:
            robot_pos_xy = self.robot.data.root_pos_w[env_ids, :2]
        mixed_relay_mode = self._use_mixed_planner_state_training() and bool(getattr(self.cfg, "relay_goal_enable", False))
        if mixed_relay_mode:
            # Match deployment: train on short relay sub-goals, not far frontier / waypoint targets.
            explore_dists = self._sample_relay_goal_distances(
                num,
                dist_min_attr="explore_relay_goal_distance_min",
                dist_max_attr="explore_relay_goal_distance_max",
                close_prob_attr="explore_relay_goal_close_sampling_prob",
                close_min_attr="explore_relay_goal_close_distance_min",
                close_max_attr="explore_relay_goal_close_distance_max",
            )
            fallback_dists = self._sample_relay_goal_distances(
                num,
                dist_min_attr="fallback_relay_goal_distance_min",
                dist_max_attr="fallback_relay_goal_distance_max",
            )
            junction_straight_dists = self._sample_relay_goal_distances(
                num,
                dist_min_attr="junction_relay_goal_distance_min",
                dist_max_attr="junction_relay_goal_distance_max",
                close_prob_attr="junction_relay_goal_close_sampling_prob",
                close_min_attr="junction_relay_goal_close_distance_min",
                close_max_attr="junction_relay_goal_close_distance_max",
            )
            junction_turn_dists = self._sample_relay_goal_distances(
                num,
                dist_min_attr="junction_turn_relay_goal_distance_min",
                dist_max_attr="junction_turn_relay_goal_distance_max",
                close_prob_attr="junction_turn_relay_goal_close_sampling_prob",
                close_min_attr="junction_turn_relay_goal_close_distance_min",
                close_max_attr="junction_turn_relay_goal_close_distance_max",
                fallback_min_attr="junction_relay_goal_distance_min",
                fallback_max_attr="junction_relay_goal_distance_max",
            )
            lateral = (torch.rand(num, device=self.device) * 2.0 - 1.0) * float(
                getattr(self.cfg, "relay_goal_lateral_range", 0.18)
            )
            goal_xy = robot_pos_xy.clone()
            state_idx = self.planner_state_index[env_ids]

            explore_mask = state_idx == self.STATE_EXPLORE
            if torch.any(explore_mask):
                goal_xy[explore_mask, 0] = robot_pos_xy[explore_mask, 0] + explore_dists[explore_mask]
                goal_xy[explore_mask, 1] = robot_pos_xy[explore_mask, 1] + lateral[explore_mask]

            fallback_mask = state_idx == self.STATE_FALLBACK
            if torch.any(fallback_mask):
                goal_xy[fallback_mask, 0] = robot_pos_xy[fallback_mask, 0] - fallback_dists[fallback_mask]
                goal_xy[fallback_mask, 1] = robot_pos_xy[fallback_mask, 1] + lateral[fallback_mask]

            return_mask = state_idx == self.STATE_RETURN
            if torch.any(return_mask):
                return_dist_min = float(getattr(self.cfg, "return_relay_goal_distance_min", getattr(self.cfg, "relay_goal_distance_min", 1.5)))
                return_dist_max = max(
                    return_dist_min,
                    float(getattr(self.cfg, "return_relay_goal_distance_max", getattr(self.cfg, "relay_goal_distance_max", return_dist_min))),
                )
                return_lat_range = float(
                    getattr(self.cfg, "return_relay_goal_lateral_range", getattr(self.cfg, "relay_goal_lateral_range", 0.18))
                )
                return_lat_min_abs = min(
                    max(0.0, float(getattr(self.cfg, "return_relay_goal_lateral_min_abs", 0.0))),
                    return_lat_range,
                )
                return_dists = (
                    torch.rand(num, device=self.device) * max(1e-6, return_dist_max - return_dist_min) + return_dist_min
                )
                return_lateral = (torch.rand(num, device=self.device) * 2.0 - 1.0) * return_lat_range
                if return_lat_min_abs > 0.0:
                    small_lateral = torch.abs(return_lateral) < return_lat_min_abs
                    if torch.any(small_lateral):
                        lateral_sign = torch.sign(return_lateral[small_lateral])
                        random_sign = torch.where(
                            torch.rand_like(lateral_sign) > 0.5,
                            torch.ones_like(lateral_sign),
                            -torch.ones_like(lateral_sign),
                        )
                        lateral_sign = torch.where(lateral_sign == 0.0, random_sign, lateral_sign)
                        lateral_mag = (
                            torch.rand(small_lateral.sum(), device=self.device)
                            * max(1e-6, return_lat_range - return_lat_min_abs)
                            + return_lat_min_abs
                        )
                        return_lateral[small_lateral] = lateral_sign * lateral_mag
                goal_xy[return_mask, 0] = robot_pos_xy[return_mask, 0] - return_dists[return_mask]
                goal_xy[return_mask, 1] = robot_pos_xy[return_mask, 1] + return_lateral[return_mask]

            junction_mask = state_idx == self.STATE_JUNCTION
            if torch.any(junction_mask):
                junction_env_ids = env_ids[junction_mask]
                junction_goal_xy = robot_pos_xy[junction_mask].clone()
                junction_straight = junction_straight_dists[junction_mask]
                junction_turn = junction_turn_dists[junction_mask]
                cmd_idx = self.turn_cmd_index[junction_env_ids]
                straight_mask = cmd_idx == self.TURN_STRAIGHT
                left_mask = cmd_idx == self.TURN_LEFT
                right_mask = cmd_idx == self.TURN_RIGHT

                if torch.any(straight_mask):
                    junction_goal_xy[straight_mask, 0] = (
                        robot_pos_xy[junction_mask][straight_mask, 0] + junction_straight[straight_mask]
                    )
                    junction_goal_xy[straight_mask, 1] = robot_pos_xy[junction_mask][straight_mask, 1] + lateral[junction_mask][straight_mask]

                turn_relay_min = float(
                    getattr(
                        self.cfg,
                        "junction_turn_relay_goal_distance_min",
                        getattr(self.cfg, "junction_relay_goal_distance_min", getattr(self.cfg, "relay_goal_distance_min", 1.5)),
                    )
                )
                turn_forward = min(
                    float(getattr(self.cfg, "junction_relay_entry_offset", 0.45)),
                    0.95 * turn_relay_min,
                )
                turn_forward = max(0.05, turn_forward)
                turn_forward_tensor = torch.full_like(junction_turn, turn_forward)
                turn_lateral = torch.sqrt(torch.clamp(junction_turn.square() - turn_forward_tensor.square(), min=1e-6))
                if torch.any(left_mask):
                    junction_goal_xy[left_mask, 0] = robot_pos_xy[junction_mask][left_mask, 0] + turn_forward_tensor[left_mask]
                    junction_goal_xy[left_mask, 1] = robot_pos_xy[junction_mask][left_mask, 1] + turn_lateral[left_mask]
                if torch.any(right_mask):
                    junction_goal_xy[right_mask, 0] = robot_pos_xy[junction_mask][right_mask, 0] + turn_forward_tensor[right_mask]
                    junction_goal_xy[right_mask, 1] = robot_pos_xy[junction_mask][right_mask, 1] - turn_lateral[right_mask]

                goal_xy[junction_mask] = junction_goal_xy

            env_origins_xy = self.scene.env_origins[env_ids, :2]
            env_half_l = self.layout_half_length[env_ids]
            env_half_w = self.layout_half_width[env_ids]
            linear_mask = state_idx != self.STATE_JUNCTION
            if torch.any(linear_mask):
                goal_xy[linear_mask, 0] = torch.clamp(
                    goal_xy[linear_mask, 0],
                    min=env_origins_xy[linear_mask, 0] - env_half_l[linear_mask] + 0.45,
                    max=env_origins_xy[linear_mask, 0] + env_half_l[linear_mask] - 0.45,
                )
                goal_xy[linear_mask, 1] = torch.clamp(
                    goal_xy[linear_mask, 1],
                    min=env_origins_xy[linear_mask, 1] - env_half_w[linear_mask] + 0.10,
                    max=env_origins_xy[linear_mask, 1] + env_half_w[linear_mask] - 0.10,
                )
            if torch.any(junction_mask):
                goal_xy[junction_mask, 0] = torch.clamp(
                    goal_xy[junction_mask, 0],
                    min=env_origins_xy[junction_mask, 0] - env_half_l[junction_mask] + 0.45,
                    max=env_origins_xy[junction_mask, 0] + env_half_l[junction_mask] - 0.45,
                )
                goal_xy[junction_mask, 1] = torch.clamp(
                    goal_xy[junction_mask, 1],
                    min=env_origins_xy[junction_mask, 1] - env_half_l[junction_mask] + 0.45,
                    max=env_origins_xy[junction_mask, 1] + env_half_l[junction_mask] - 0.45,
                )

            self.goal_pos_w[env_ids, 0] = goal_xy[:, 0]
            self.goal_pos_w[env_ids, 1] = goal_xy[:, 1]
            self.goal_pos_w[env_ids, 2] = self.cfg.goal_radius
            return

        dists = (
            torch.rand(num, device=self.device)
            * (self.cfg.goal_spawn_range_max - self.cfg.goal_spawn_range_min)
            + self.cfg.goal_spawn_range_min
        )
        if bool(getattr(self.cfg, "junction_enable", False)):
            env_origins_xy = self.scene.env_origins[env_ids, :2]
            cmd_idx = self.turn_cmd_index[env_ids]
            max_fwd = min(float(self.cfg.junction_goal_forward_max), float(self.cfg.junction_half_length) - 0.35)
            min_fwd = min(float(self.cfg.junction_goal_forward_min), max_fwd)
            forward = torch.rand(num, device=self.device) * max(1e-6, max_fwd - min_fwd) + min_fwd
            lateral = (torch.rand(num, device=self.device) * 2.0 - 1.0) * float(self.cfg.junction_goal_lateral_range)
            goal_local = torch.zeros((num, 2), device=self.device)

            left_mask = cmd_idx == self.TURN_LEFT
            straight_mask = cmd_idx == self.TURN_STRAIGHT
            right_mask = cmd_idx == self.TURN_RIGHT
            goal_local[left_mask, 0] = lateral[left_mask]
            goal_local[left_mask, 1] = forward[left_mask]
            goal_local[straight_mask, 0] = forward[straight_mask]
            goal_local[straight_mask, 1] = lateral[straight_mask]
            goal_local[right_mask, 0] = lateral[right_mask]
            goal_local[right_mask, 1] = -forward[right_mask]

            goal_world_xy = env_origins_xy + goal_local
            self.goal_pos_w[env_ids, 0] = goal_world_xy[:, 0]
            self.goal_pos_w[env_ids, 1] = goal_world_xy[:, 1]
        elif bool(getattr(self.cfg, "corridor_enable", False)):
            env_origins_xy = self.scene.env_origins[env_ids, :2]
            center_y = env_origins_xy[:, 1]
            y_low = center_y - float(self.cfg.corridor_goal_lateral_range)
            y_high = center_y + float(self.cfg.corridor_goal_lateral_range)
            goal_y = torch.rand(num, device=self.device) * (y_high - y_low) + y_low
            x_min = env_origins_xy[:, 0] - float(self.cfg.corridor_half_length) + 0.5
            x_max = env_origins_xy[:, 0] + float(self.cfg.corridor_half_length) - 0.5
            goal_x = torch.clamp(robot_pos_xy[:, 0] + dists, min=x_min, max=x_max)
            self.goal_pos_w[env_ids, 0] = goal_x
            self.goal_pos_w[env_ids, 1] = goal_y
        else:
            angles = torch.rand(num, device=self.device) * 2.0 * math.pi
            offset_x = dists * torch.cos(angles)
            offset_y = dists * torch.sin(angles)
            self.goal_pos_w[env_ids, 0] = robot_pos_xy[:, 0] + offset_x
            self.goal_pos_w[env_ids, 1] = robot_pos_xy[:, 1] + offset_y
        self.goal_pos_w[env_ids, 2] = self.cfg.goal_radius

    def _spawn_linear_obstacle_positions(
        self,
        env_ids: torch.Tensor,
        robot_pos_xy: torch.Tensor,
        goal_pos_xy: torch.Tensor,
    ) -> None:
        num = len(env_ids)
        if num == 0:
            return

        env_origins_xy = self.scene.env_origins[env_ids, :2]
        half_width = self.layout_half_width[env_ids]
        half_length = self.layout_half_length[env_ids]
        wall_margin = float(
            getattr(
                self.cfg,
                "junction_obstacle_lateral_margin",
                getattr(self.cfg, "corridor_obstacle_lateral_margin", 0.15),
            )
        )
        y_low = env_origins_xy[:, 1] - half_width + wall_margin
        y_high = env_origins_xy[:, 1] + half_width - wall_margin
        y_mid = env_origins_xy[:, 1]
        too_narrow = y_high <= y_low
        y_low = torch.where(too_narrow, y_mid - 0.05, y_low)
        y_high = torch.where(too_narrow, y_mid + 0.05, y_high)

        forward_min_offset = float(
            getattr(
                self.cfg,
                "junction_obstacle_forward_min_offset",
                getattr(self.cfg, "corridor_obstacle_forward_min_offset", 0.60),
            )
        )
        goal_margin = float(getattr(self.cfg, "corridor_obstacle_goal_margin", 0.70))

        for obs_idx in range(self.cfg.num_obstacles):
            cand = torch.zeros((num, 2), device=self.device)
            forward_min = torch.minimum(robot_pos_xy[:, 0], goal_pos_xy[:, 0]) + forward_min_offset
            forward_max = torch.maximum(robot_pos_xy[:, 0], goal_pos_xy[:, 0]) - goal_margin
            mid_x = 0.5 * (forward_min + forward_max)
            low_ok = forward_min < forward_max
            obs_x = torch.where(
                low_ok,
                torch.rand(num, device=self.device) * (forward_max - forward_min) + forward_min,
                mid_x,
            )
            obs_x = torch.clamp(
                obs_x,
                min=env_origins_xy[:, 0] - half_length + 0.4,
                max=env_origins_xy[:, 0] + half_length - 0.4,
            )
            obs_y = torch.rand(num, device=self.device) * (y_high - y_low) + y_low
            cand[:, 0] = obs_x
            cand[:, 1] = obs_y

            vec_robot = cand - robot_pos_xy
            dist_robot = torch.linalg.norm(vec_robot, dim=1, keepdim=True).clamp(min=1e-6)
            near_robot = dist_robot[:, 0] < self.cfg.obstacle_min_separation
            if torch.any(near_robot):
                safe = robot_pos_xy[near_robot] + vec_robot[near_robot] / dist_robot[near_robot] * self.cfg.obstacle_min_separation
                cand[near_robot] = safe

            vec_goal = cand - goal_pos_xy
            dist_goal = torch.linalg.norm(vec_goal, dim=1, keepdim=True).clamp(min=1e-6)
            near_goal = dist_goal[:, 0] < self.cfg.obstacle_min_separation
            if torch.any(near_goal):
                safe = goal_pos_xy[near_goal] + vec_goal[near_goal] / dist_goal[near_goal] * self.cfg.obstacle_min_separation
                cand[near_goal] = safe

            # Return-mode subgoals are short and nearly axis-aligned. Keep a small centerline
            # corridor free so the policy can learn to go around obstacles instead of always
            # training on head-on blocks that collapse into crash-only behavior.
            return_mask = self.planner_state_index[env_ids] == self.STATE_RETURN
            return_keepout = float(getattr(self.cfg, "return_obstacle_centerline_keepout", 0.0))
            if return_keepout > 0.0 and torch.any(return_mask):
                return_robot_y = robot_pos_xy[return_mask, 1]
                cand_return = cand[return_mask].clone()
                force_opposite_side = bool(getattr(self.cfg, "return_obstacle_force_opposite_goal_side", False))
                if force_opposite_side:
                    goal_delta_y = goal_pos_xy[return_mask, 1] - return_robot_y
                    desired_sign = -torch.sign(goal_delta_y)
                    random_sign = torch.where(
                        torch.rand_like(desired_sign) > 0.5,
                        torch.ones_like(desired_sign),
                        -torch.ones_like(desired_sign),
                    )
                    desired_sign = torch.where(desired_sign == 0.0, random_sign, desired_sign)
                    max_lat_pos = torch.clamp(y_high[return_mask] - return_robot_y, min=0.0)
                    max_lat_neg = torch.clamp(return_robot_y - y_low[return_mask], min=0.0)
                    max_lat = torch.where(desired_sign > 0.0, max_lat_pos, max_lat_neg)
                    min_lat = torch.minimum(torch.full_like(max_lat, return_keepout), max_lat)
                    lat_span = torch.clamp(max_lat - min_lat, min=0.0)
                    sampled_lat = torch.rand_like(max_lat) * lat_span + min_lat
                    cand_return[:, 1] = return_robot_y + desired_sign * sampled_lat
                else:
                    delta_y = cand_return[:, 1] - return_robot_y
                    too_centered = torch.abs(delta_y) < return_keepout
                    if torch.any(too_centered):
                        shift_sign = torch.sign(delta_y[too_centered])
                        random_sign = torch.where(
                            torch.rand_like(shift_sign) > 0.5,
                            torch.ones_like(shift_sign),
                            -torch.ones_like(shift_sign),
                        )
                        shift_sign = torch.where(shift_sign == 0.0, random_sign, shift_sign)
                        adjusted_y = return_robot_y[too_centered] + shift_sign * return_keepout
                        cand_return[too_centered, 1] = adjusted_y
                cand[return_mask] = cand_return

            cand[:, 1] = torch.clamp(cand[:, 1], min=y_low, max=y_high)
            self.obstacle_pos_w[env_ids, obs_idx, 0:2] = cand
            self.obstacle_pos_w[env_ids, obs_idx, 2] = 0.5 * self.cfg.obstacle_height

        fallback_mask = self.planner_state_index[env_ids] == self.STATE_FALLBACK
        if not torch.any(fallback_mask):
            return

        fallback_env_ids = env_ids[fallback_mask]
        fallback_robot_pos_xy = robot_pos_xy[fallback_mask]
        blocker_min = float(getattr(self.cfg, "fallback_blocker_distance_min", 0.55))
        blocker_clearance_min = (
            float(self.cfg.obstacle_radius)
            + float(self.cfg.robot_radius)
            + float(self.cfg.collision_threshold)
            + 0.05
        )
        blocker_min = max(blocker_min, blocker_clearance_min)
        blocker_max = max(blocker_min, float(getattr(self.cfg, "fallback_blocker_distance_max", blocker_min)))
        blocker_lateral_range = float(getattr(self.cfg, "fallback_blocker_lateral_range", 0.16))
        blocker_dist = torch.rand(fallback_env_ids.numel(), device=self.device) * max(1e-6, blocker_max - blocker_min) + blocker_min
        blocker_lat = (torch.rand(fallback_env_ids.numel(), device=self.device) * 2.0 - 1.0) * blocker_lateral_range
        blocker_pos = torch.zeros((fallback_env_ids.numel(), 2), device=self.device)
        blocker_pos[:, 0] = torch.clamp(
            fallback_robot_pos_xy[:, 0] + blocker_dist,
            min=env_origins_xy[fallback_mask, 0] - half_length[fallback_mask] + 0.4,
            max=env_origins_xy[fallback_mask, 0] + half_length[fallback_mask] - 0.4,
        )
        blocker_pos[:, 1] = torch.clamp(
            fallback_robot_pos_xy[:, 1] + blocker_lat,
            min=y_low[fallback_mask],
            max=y_high[fallback_mask],
        )
        self.obstacle_pos_w[fallback_env_ids, 0, 0:2] = blocker_pos
        self.obstacle_pos_w[fallback_env_ids, 0, 2] = 0.5 * self.cfg.obstacle_height

    def _spawn_obstacle_positions(
        self,
        env_ids: torch.Tensor,
        robot_pos_xy: torch.Tensor | None = None,
        goal_pos_xy: torch.Tensor | None = None,
    ) -> None:
        num = len(env_ids)
        if robot_pos_xy is None:
            robot_pos_xy = self.robot.data.root_pos_w[env_ids, :2]
        if goal_pos_xy is None:
            goal_pos_xy = self.goal_pos_w[env_ids, :2]
        junction_mode = bool(getattr(self.cfg, "junction_enable", False))
        corridor_mode = bool(getattr(self.cfg, "corridor_enable", False))
        if self._use_mixed_planner_state_training() and junction_mode:
            linear_mask = self.planner_state_index[env_ids] != self.STATE_JUNCTION
            if torch.any(linear_mask):
                self._spawn_linear_obstacle_positions(env_ids[linear_mask], robot_pos_xy[linear_mask], goal_pos_xy[linear_mask])
            if torch.all(linear_mask):
                return
            env_ids = env_ids[~linear_mask]
            robot_pos_xy = robot_pos_xy[~linear_mask]
            goal_pos_xy = goal_pos_xy[~linear_mask]
            num = len(env_ids)
        if junction_mode:
            half_l = float(self.cfg.junction_half_length)
            lat_limit = max(0.05, float(self.cfg.junction_half_width) - float(self.cfg.junction_obstacle_lateral_margin))
            min_forward = min(half_l - 0.5, float(self.cfg.junction_obstacle_forward_min_offset))
            env_ids_list = env_ids.detach().cpu().tolist()
            for local_idx, env_id in enumerate(env_ids_list):
                origin_xy = self.scene.env_origins[env_id, :2]
                active_arms = [arm for arm in range(4) if bool(self.junction_active_arms[env_id, arm].item())]
                if len(active_arms) == 0:
                    active_arms = [self.ARM_W]
                for obs_idx in range(self.cfg.num_obstacles):
                    cand = torch.zeros((2,), device=self.device)
                    placed = False
                    for _ in range(10):
                        arm = active_arms[int(torch.randint(0, len(active_arms), (1,), device=self.device).item())]
                        if arm == self.ARM_W:
                            x_local = -(torch.rand(1, device=self.device)[0] * max(1e-6, half_l - min_forward) + min_forward)
                            y_local = (torch.rand(1, device=self.device)[0] * 2.0 - 1.0) * lat_limit
                        elif arm == self.ARM_E:
                            x_local = torch.rand(1, device=self.device)[0] * max(1e-6, half_l - min_forward) + min_forward
                            y_local = (torch.rand(1, device=self.device)[0] * 2.0 - 1.0) * lat_limit
                        elif arm == self.ARM_N:
                            x_local = (torch.rand(1, device=self.device)[0] * 2.0 - 1.0) * lat_limit
                            y_local = torch.rand(1, device=self.device)[0] * max(1e-6, half_l - min_forward) + min_forward
                        else:
                            x_local = (torch.rand(1, device=self.device)[0] * 2.0 - 1.0) * lat_limit
                            y_local = -(torch.rand(1, device=self.device)[0] * max(1e-6, half_l - min_forward) + min_forward)
                        cand[0] = origin_xy[0] + x_local
                        cand[1] = origin_xy[1] + y_local
                        if torch.linalg.norm(cand - robot_pos_xy[local_idx]) < self.cfg.obstacle_min_separation:
                            continue
                        if torch.linalg.norm(cand - goal_pos_xy[local_idx]) < self.cfg.obstacle_min_separation:
                            continue
                        placed = True
                        break
                    if not placed:
                        cand[0] = origin_xy[0] + min_forward
                        cand[1] = origin_xy[1]
                    self.obstacle_pos_w[env_id, obs_idx, 0:2] = cand
                    self.obstacle_pos_w[env_id, obs_idx, 2] = 0.5 * self.cfg.obstacle_height
            return
        if corridor_mode:
            env_origins_xy = self.scene.env_origins[env_ids, :2]
            center_y = env_origins_xy[:, 1]
            wall_margin = float(self.cfg.corridor_obstacle_lateral_margin)
            y_low = center_y - float(self.cfg.corridor_half_width) + wall_margin
            y_high = center_y + float(self.cfg.corridor_half_width) - wall_margin
            # keep at least a little usable lateral space
            y_mid = center_y
            too_narrow = y_high <= y_low
            y_low = torch.where(too_narrow, y_mid - 0.05, y_low)
            y_high = torch.where(too_narrow, y_mid + 0.05, y_high)

        for obs_idx in range(self.cfg.num_obstacles):
            cand = torch.zeros((num, 2), device=self.device)
            if corridor_mode:
                forward_min = torch.minimum(robot_pos_xy[:, 0], goal_pos_xy[:, 0]) + float(
                    self.cfg.corridor_obstacle_forward_min_offset
                )
                forward_max = torch.maximum(robot_pos_xy[:, 0], goal_pos_xy[:, 0]) - float(
                    self.cfg.corridor_obstacle_goal_margin
                )
                # fallback when goal is too close
                mid_x = 0.5 * (forward_min + forward_max)
                low_ok = forward_min < forward_max
                obs_x = torch.where(
                    low_ok,
                    torch.rand(num, device=self.device) * (forward_max - forward_min) + forward_min,
                    mid_x,
                )
                obs_x = torch.clamp(
                    obs_x,
                    min=env_origins_xy[:, 0] - float(self.cfg.corridor_half_length) + 0.4,
                    max=env_origins_xy[:, 0] + float(self.cfg.corridor_half_length) - 0.4,
                )
                obs_y = torch.rand(num, device=self.device) * (y_high - y_low) + y_low
                cand[:, 0] = obs_x
                cand[:, 1] = obs_y
            else:
                angles = torch.rand(num, device=self.device) * 2.0 * math.pi
                dists = (
                    torch.rand(num, device=self.device)
                    * (self.cfg.obstacle_spawn_range_max - self.cfg.obstacle_spawn_range_min)
                    + self.cfg.obstacle_spawn_range_min
                )
                cand[:, 0] = robot_pos_xy[:, 0] + dists * torch.cos(angles)
                cand[:, 1] = robot_pos_xy[:, 1] + dists * torch.sin(angles)

            # keep obstacles away from robot center
            vec_robot = cand - robot_pos_xy
            dist_robot = torch.linalg.norm(vec_robot, dim=1, keepdim=True).clamp(min=1e-6)
            near_robot = dist_robot[:, 0] < self.cfg.obstacle_min_separation
            if torch.any(near_robot):
                safe = robot_pos_xy[near_robot] + vec_robot[near_robot] / dist_robot[near_robot] * self.cfg.obstacle_min_separation
                cand[near_robot] = safe

            # keep obstacles away from goal center
            vec_goal = cand - goal_pos_xy
            dist_goal = torch.linalg.norm(vec_goal, dim=1, keepdim=True).clamp(min=1e-6)
            near_goal = dist_goal[:, 0] < self.cfg.obstacle_min_separation
            if torch.any(near_goal):
                safe = goal_pos_xy[near_goal] + vec_goal[near_goal] / dist_goal[near_goal] * self.cfg.obstacle_min_separation
                cand[near_goal] = safe

            if corridor_mode:
                cand[:, 1] = torch.clamp(cand[:, 1], min=y_low, max=y_high)

            self.obstacle_pos_w[env_ids, obs_idx, 0:2] = cand
            self.obstacle_pos_w[env_ids, obs_idx, 2] = 0.5 * self.cfg.obstacle_height

    def _compute_lidar_ranges(self) -> torch.Tensor:
        robot_xy = self.robot.data.root_pos_w[:, :2]  # (N, 2)
        forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)  # (N, 3)
        yaw = torch.atan2(forwards[:, 1], forwards[:, 0])  # (N,)

        beam_angles_world = yaw.unsqueeze(1) + self.lidar_beam_angles.unsqueeze(0)  # (N, B)
        beam_dirs = torch.stack((torch.cos(beam_angles_world), torch.sin(beam_angles_world)), dim=-1)  # (N, B, 2)

        origins = robot_xy.unsqueeze(1).unsqueeze(2)  # (N, 1, 1, 2)
        dirs = beam_dirs.unsqueeze(2)  # (N, B, 1, 2)
        centers = self.obstacle_pos_w[:, : self.cfg.num_obstacles, :2].unsqueeze(1)  # (N, 1, O, 2)

        if self._use_box_obstacles():
            # Ray-AABB intersections against box obstacles expanded by robot radius.
            half_extent_x, half_extent_y = self._get_box_obstacle_half_extents()
            half_extents = torch.tensor(
                (half_extent_x + float(self.cfg.robot_radius), half_extent_y + float(self.cfg.robot_radius)),
                device=self.device,
                dtype=centers.dtype,
            ).view(1, 1, 1, 2)
            box_min = centers - half_extents
            box_max = centers + half_extents
            eps = 1e-6
            inf = torch.full_like(dirs, float("inf"))
            inv_dirs = torch.where(torch.abs(dirs) > eps, 1.0 / dirs, inf)
            t0 = (box_min - origins) * inv_dirs
            t1 = (box_max - origins) * inv_dirs
            t_min = torch.minimum(t0, t1)
            t_max = torch.maximum(t0, t1)
            t_enter = torch.max(t_min, dim=-1).values
            t_exit = torch.min(t_max, dim=-1).values
            valid = t_exit >= torch.maximum(t_enter, torch.full_like(t_enter, 0.0))
            t_enter_pos = torch.where(t_enter > 0.0, t_enter, t_exit)
            t_hit = torch.where(valid & (t_exit > 0.0), t_enter_pos, torch.full_like(t_enter, float("inf")))
        else:
            # Ray-circle intersections for all obstacles.
            # Ray equation: p(t) = o + t*d, t > 0
            # Circle: ||p - c||^2 = r^2
            oc = origins - centers  # (N, B, O, 2)
            b = 2.0 * torch.sum(oc * dirs, dim=-1)  # (N, B, O)
            c = torch.sum(oc * oc, dim=-1) - (self.cfg.obstacle_radius + self.cfg.robot_radius) ** 2  # (N, B, O)
            disc = b * b - 4.0 * c

            valid = disc >= 0.0
            sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
            t1 = (-b - sqrt_disc) / 2.0
            t2 = (-b + sqrt_disc) / 2.0
            inf = torch.full_like(t1, float("inf"))
            t1_pos = torch.where(valid & (t1 > 0.0), t1, inf)
            t2_pos = torch.where(valid & (t2 > 0.0), t2, inf)
            t_hit = torch.minimum(t1_pos, t2_pos)

        ranges = torch.min(t_hit, dim=2).values  # (N, B)

        if bool(getattr(self.cfg, "junction_enable", False)) or self._use_defect_scene_reset():
            junction_ranges = self._compute_junction_wall_ranges(robot_xy=robot_xy, beam_dirs=beam_dirs)
            ranges = torch.minimum(ranges, junction_ranges)

        if bool(getattr(self.cfg, "corridor_enable", False)):
            # Intersections with axis-aligned corridor walls/end-caps in each env.
            env_origins = self.scene.env_origins[:, :2]
            wall_inset = float(self.cfg.robot_radius)
            x_min = (env_origins[:, 0] - float(self.cfg.corridor_half_length) + wall_inset).unsqueeze(1)
            x_max = (env_origins[:, 0] + float(self.cfg.corridor_half_length) - wall_inset).unsqueeze(1)
            y_min = (env_origins[:, 1] - float(self.cfg.corridor_half_width) + wall_inset).unsqueeze(1)
            y_max = (env_origins[:, 1] + float(self.cfg.corridor_half_width) - wall_inset).unsqueeze(1)
            ox = robot_xy[:, 0].unsqueeze(1)
            oy = robot_xy[:, 1].unsqueeze(1)
            dx = beam_dirs[:, :, 0]
            dy = beam_dirs[:, :, 1]
            eps = 1e-6
            inf_wall = torch.full_like(dx, float("inf"))

            t_ymin = torch.where(torch.abs(dy) > eps, (y_min - oy) / dy, inf_wall)
            x_at_ymin = ox + t_ymin * dx
            hit_ymin = (t_ymin > 0.0) & (x_at_ymin >= x_min) & (x_at_ymin <= x_max)
            t_ymin = torch.where(hit_ymin, t_ymin, inf_wall)

            t_ymax = torch.where(torch.abs(dy) > eps, (y_max - oy) / dy, inf_wall)
            x_at_ymax = ox + t_ymax * dx
            hit_ymax = (t_ymax > 0.0) & (x_at_ymax >= x_min) & (x_at_ymax <= x_max)
            t_ymax = torch.where(hit_ymax, t_ymax, inf_wall)

            t_xmin = torch.where(torch.abs(dx) > eps, (x_min - ox) / dx, inf_wall)
            y_at_xmin = oy + t_xmin * dy
            hit_xmin = (t_xmin > 0.0) & (y_at_xmin >= y_min) & (y_at_xmin <= y_max)
            t_xmin = torch.where(hit_xmin, t_xmin, inf_wall)

            t_xmax = torch.where(torch.abs(dx) > eps, (x_max - ox) / dx, inf_wall)
            y_at_xmax = oy + t_xmax * dy
            hit_xmax = (t_xmax > 0.0) & (y_at_xmax >= y_min) & (y_at_xmax <= y_max)
            t_xmax = torch.where(hit_xmax, t_xmax, inf_wall)

            corridor_ranges = torch.minimum(torch.minimum(t_ymin, t_ymax), torch.minimum(t_xmin, t_xmax))
            ranges = torch.minimum(ranges, corridor_ranges)

        ranges = torch.where(torch.isinf(ranges), torch.full_like(ranges, self.cfg.lidar_max_range), ranges)
        ranges = torch.clamp(ranges, min=self.cfg.lidar_min_range, max=self.cfg.lidar_max_range)
        noise_std = max(0.0, float(getattr(self.cfg, "lidar_noise_std", 0.0)))
        if noise_std > 0.0:
            noise = torch.randn_like(ranges) * noise_std
            ranges = torch.clamp(ranges + noise, min=self.cfg.lidar_min_range, max=self.cfg.lidar_max_range)
        return ranges

    def _compute_obstacle_contact_force(self) -> torch.Tensor:
        """Compute max filtered normal contact force between robot body and obstacle rigid bodies."""
        contact_sensor = getattr(self, "_contact_sensor", None)
        if contact_sensor is None:
            return torch.zeros((self.cfg.scene.num_envs,), device=self.device)
        threshold_shape = (self.cfg.scene.num_envs,)
        force_history = contact_sensor.data.force_matrix_w_history
        force_now = contact_sensor.data.force_matrix_w
        if force_history is not None and force_history.numel() > 0 and force_history.shape[-2] > 0:
            # (N, T, B, M, 3) -> max over (T, B, M)
            force_mag = torch.linalg.norm(force_history, dim=-1)
            return torch.amax(force_mag, dim=(1, 2, 3))
        if force_now is not None and force_now.numel() > 0 and force_now.shape[-2] > 0:
            # (N, B, M, 3) -> max over (B, M)
            force_mag = torch.linalg.norm(force_now, dim=-1)
            return torch.amax(force_mag, dim=(1, 2))
        return torch.zeros(threshold_shape, device=self.device)

    def _compute_junction_wall_ranges(self, robot_xy: torch.Tensor, beam_dirs: torch.Tensor) -> torch.Tensor:
        """Compute ray-segment intersections against sampled junction walls."""
        seg_valid = self._junction_wall_seg_valid
        n_envs = int(robot_xy.shape[0])
        n_beams = int(beam_dirs.shape[1])
        inf_out = torch.full((n_envs, n_beams), float("inf"), device=self.device)
        if not torch.any(seg_valid):
            return inf_out

        seg_start = self._junction_wall_seg_start
        seg_end = self._junction_wall_seg_end
        ray_origin = robot_xy.unsqueeze(1).unsqueeze(2)  # (N, B, 1, 2)
        ray_dir = beam_dirs.unsqueeze(2)  # (N, B, 1, 2)
        p = seg_start.unsqueeze(1)  # (N, 1, S, 2)
        s = (seg_end - seg_start).unsqueeze(1)  # (N, 1, S, 2)
        q = p - ray_origin

        den = ray_dir[..., 0] * s[..., 1] - ray_dir[..., 1] * s[..., 0]  # cross(ray_dir, s)
        eps = 1e-6
        den_safe = torch.where(torch.abs(den) > eps, den, torch.ones_like(den))
        t = (q[..., 0] * s[..., 1] - q[..., 1] * s[..., 0]) / den_safe
        u = (q[..., 0] * ray_dir[..., 1] - q[..., 1] * ray_dir[..., 0]) / den_safe

        hit = (
            (torch.abs(den) > eps)
            & (t > 0.0)
            & (u >= 0.0)
            & (u <= 1.0)
            & seg_valid.unsqueeze(1)
        )
        inf = torch.full_like(t, float("inf"))
        t_hit = torch.where(hit, t, inf)
        return torch.min(t_hit, dim=2).values

    def _visualize_goal_obstacles(self):
        goal_orientations = torch.zeros((self.cfg.scene.num_envs, 4), device=self.device)
        goal_orientations[:, 0] = 1.0
        self.goal_markers.visualize(self.goal_pos_w, goal_orientations)

        obstacle_flat = self.obstacle_pos_w.reshape(-1, 3)
        obstacle_orient = torch.zeros((obstacle_flat.shape[0], 4), device=self.device)
        obstacle_orient[:, 0] = 1.0
        self.obstacle_markers.visualize(obstacle_flat, obstacle_orient)

    def _visualize_corridor_walls(self) -> None:
        if self.corridor_wall_markers is None:
            return
        num_envs = int(self.cfg.scene.num_envs)
        if num_envs <= 0:
            return

        half_w = float(self.cfg.corridor_half_width)
        half_l = float(self.cfg.corridor_half_length)
        wall_h = float(getattr(self.cfg, "corridor_wall_height", 0.45))
        wall_t = float(getattr(self.cfg, "corridor_wall_thickness", 0.04))
        wall_z = 0.5 * wall_h
        origins = self.scene.env_origins[:, :2]

        # 4 walls per env: left, right, front, back
        wall_count = num_envs * 4
        translations = torch.zeros((wall_count, 3), device=self.device)
        orientations = torch.zeros((wall_count, 4), device=self.device)
        orientations[:, 0] = 1.0
        scales = torch.ones((wall_count, 3), device=self.device)

        env_idx = torch.arange(num_envs, device=self.device)
        base = env_idx * 4
        ox = origins[:, 0]
        oy = origins[:, 1]

        # left
        li = base
        translations[li, 0] = ox
        translations[li, 1] = oy - half_w
        translations[li, 2] = wall_z
        scales[li, 0] = 2.0 * half_l
        scales[li, 1] = wall_t
        scales[li, 2] = wall_h
        # right
        ri = base + 1
        translations[ri, 0] = ox
        translations[ri, 1] = oy + half_w
        translations[ri, 2] = wall_z
        scales[ri, 0] = 2.0 * half_l
        scales[ri, 1] = wall_t
        scales[ri, 2] = wall_h
        # front cap
        fi = base + 2
        translations[fi, 0] = ox + half_l
        translations[fi, 1] = oy
        translations[fi, 2] = wall_z
        scales[fi, 0] = wall_t
        scales[fi, 1] = 2.0 * half_w
        scales[fi, 2] = wall_h
        # back cap
        bi = base + 3
        translations[bi, 0] = ox - half_l
        translations[bi, 1] = oy
        translations[bi, 2] = wall_z
        scales[bi, 0] = wall_t
        scales[bi, 1] = 2.0 * half_w
        scales[bi, 2] = wall_h

        self.corridor_wall_markers.visualize(
            translations=translations,
            orientations=orientations,
            scales=scales,
        )

    def _visualize_junction_walls(self) -> None:
        if self.junction_wall_markers is None:
            return
        wall_h = float(getattr(self.cfg, "junction_wall_height", 0.45))
        wall_t = float(getattr(self.cfg, "junction_wall_thickness", 0.04))
        wall_z = 0.5 * wall_h

        seg_start = self._junction_wall_seg_start
        seg_end = self._junction_wall_seg_end
        seg_valid = self._junction_wall_seg_valid
        n_envs = seg_start.shape[0]
        n_seg = seg_start.shape[1]
        total = n_envs * n_seg

        start_flat = seg_start.reshape(total, 2)
        end_flat = seg_end.reshape(total, 2)
        valid_flat = seg_valid.reshape(total)
        mid_flat = 0.5 * (start_flat + end_flat)
        delta_flat = end_flat - start_flat
        length_flat = torch.linalg.norm(delta_flat, dim=1).clamp(min=1e-4)
        yaw_flat = torch.atan2(delta_flat[:, 1], delta_flat[:, 0])

        translations = torch.zeros((total, 3), device=self.device)
        orientations = torch.zeros((total, 4), device=self.device)
        scales = torch.ones((total, 3), device=self.device)
        orientations[:, 0] = torch.cos(0.5 * yaw_flat)
        orientations[:, 3] = torch.sin(0.5 * yaw_flat)
        translations[:, 0:2] = mid_flat
        translations[:, 2] = wall_z
        scales[:, 0] = length_flat
        scales[:, 1] = wall_t
        scales[:, 2] = wall_h

        if torch.any(~valid_flat):
            invalid = ~valid_flat
            translations[invalid, 2] = -100.0
            scales[invalid, :] = 1.0e-3
            orientations[invalid, 0] = 1.0
            orientations[invalid, 3] = 0.0

        self.junction_wall_markers.visualize(
            translations=translations,
            orientations=orientations,
            scales=scales,
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions.clone(), -1.0, 1.0)
        self._visualize_goal_obstacles()

    def _apply_action(self) -> None:
        # Policy action -> unicycle command (v, omega).
        a_v = self.actions[:, 0]
        a_omega = self.actions[:, 1]
        if bool(getattr(self.cfg, "forward_only", True)):
            v_target = 0.5 * (a_v + 1.0) * float(self.cfg.v_max)
        else:
            v_target = a_v * float(self.cfg.v_max)
        omega_target = a_omega * float(self.cfg.omega_limit)

        # Slow down near goal/obstacles to reduce overshoot and jitter.
        diff_goal = self.goal_pos_w[:, :2] - self.robot.data.root_pos_w[:, :2]
        curr_goal_dist = torch.linalg.norm(diff_goal, dim=1)
        goal_dir_norm = diff_goal / curr_goal_dist.unsqueeze(-1).clamp(min=1e-6)
        forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)
        heading_cos = torch.sum(forwards[:, :2] * goal_dir_norm, dim=-1)
        near_goal_scale = (
            curr_goal_dist / max(1e-6, float(self.cfg.near_goal_slowdown_distance))
        ).clamp(min=float(self.cfg.near_goal_min_action_scale), max=1.0)
        min_lidar = torch.min(self.latest_lidar_ranges, dim=1).values
        slowdown_lidar = min_lidar
        scene_mode = str(getattr(self.cfg, "defect_scene_mode", "")).strip().lower()
        use_front_slowdown = (
            self._use_defect_scene_reset()
            and scene_mode == "mppi_corner_fail"
            and bool(getattr(self.cfg, "defect_mppi_front_obstacle_slowdown_enable", True))
        )
        if use_front_slowdown:
            half_fov_deg = float(getattr(self.cfg, "defect_mppi_front_obstacle_slowdown_half_fov_deg", 45.0))
            half_fov_deg = min(135.0, max(5.0, half_fov_deg))
            front_mask = torch.abs(self.lidar_beam_angles) <= math.radians(half_fov_deg)
            if bool(torch.any(front_mask).item()):
                slowdown_lidar = torch.min(self.latest_lidar_ranges[:, front_mask], dim=1).values
        obstacle_den = max(1e-6, float(self.cfg.obstacle_slowdown_distance - self.cfg.collision_threshold))
        obstacle_scale = ((slowdown_lidar - self.cfg.collision_threshold) / obstacle_den).clamp(
            min=float(self.cfg.obstacle_min_action_scale), max=1.0
        )
        # Keep angular command available while reducing linear speed when heading is far from goal.
        heading_cos_min = float(getattr(self.cfg, "heading_speed_cos_min", 0.0))
        heading_power = float(getattr(self.cfg, "heading_speed_power", 1.0))
        heading_scale = ((heading_cos - heading_cos_min) / max(1e-6, 1.0 - heading_cos_min)).clamp(min=0.0, max=1.0)
        heading_scale = heading_scale.pow(max(1.0, heading_power))
        # Keep a minimum forward component to avoid spinning in place at large heading error.
        heading_min_scale = float(getattr(self.cfg, "heading_speed_min_scale", 0.0))
        heading_min_scale = min(1.0, max(0.0, heading_min_scale))
        heading_scale = heading_min_scale + (1.0 - heading_min_scale) * heading_scale
        speed_scale = near_goal_scale * obstacle_scale
        v_target = v_target * speed_scale * heading_scale
        omega_target = omega_target * speed_scale
        speed_cap = self._compute_speed_cap(min_lidar)
        if bool(getattr(self.cfg, "forward_only", True)):
            v_target = torch.clamp(v_target, min=0.0)
            v_target = torch.minimum(v_target, speed_cap)
        else:
            v_target = torch.maximum(torch.minimum(v_target, speed_cap), -speed_cap)

        # Optional explicit action smoothing (alpha-blend) for safer control.
        if bool(getattr(self.cfg, "use_action_smoothing", False)):
            alpha = float(getattr(self.cfg, "action_smoothing_alpha", 0.7))
            alpha = max(0.0, min(1.0, alpha))
            alpha_v = alpha
            alpha_omega = alpha
        else:
            # First-order LPF with exact pole matching: alpha = 1 - exp(-dt / tau).
            dt_ctrl = max(1.0e-6, float(self.cfg.decimation) * float(self.cfg.sim.dt))
            tau_v = max(1.0e-6, float(self.cfg.tau_v))
            tau_omega = max(1.0e-6, float(self.cfg.tau_omega))
            alpha_v = 1.0 - math.exp(-dt_ctrl / tau_v)
            alpha_omega = 1.0 - math.exp(-dt_ctrl / tau_omega)
        self.filtered_v_cmd = alpha_v * v_target + (1.0 - alpha_v) * self.filtered_v_cmd
        self.filtered_omega_cmd = alpha_omega * omega_target + (1.0 - alpha_omega) * self.filtered_omega_cmd
        if (
            self._use_defect_scene_reset()
            and str(getattr(self.cfg, "defect_scene_mode", "")).strip().lower() == "door_deadlock"
            and bool(getattr(self.cfg, "defect_door_post_pass_lateral_slowdown_enable", False))
        ):
            local_xy = self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
            goal_vec_w = torch.zeros((self.num_envs, 3), device=self.device)
            goal_vec_w[:, :2] = diff_goal
            goal_vec_b = math_utils.quat_apply_inverse(self.robot.data.root_link_quat_w, goal_vec_w)
            door_mask = self.fixed_scene_active_index == self.DEFECT_DOOR_DEADLOCK
            post_pass_active = torch.logical_and(
                door_mask,
                torch.logical_and(
                    local_xy[:, 0] >= float(getattr(self.cfg, "defect_door_post_pass_x_min", 0.0)),
                    torch.logical_and(
                        curr_goal_dist > self.cfg.goal_reach_threshold,
                        torch.logical_and(
                            goal_vec_b[:, 0] > 0.0,
                            goal_vec_b[:, 1].abs()
                            >= float(getattr(self.cfg, "defect_door_post_pass_goal_lateral_threshold", 0.0)),
                        ),
                    ),
                ),
            )
            door_speed_cap = torch.minimum(
                torch.full_like(self.filtered_v_cmd, float(getattr(self.cfg, "defect_door_post_pass_speed_cap", 0.0))),
                speed_cap,
            )
            self.filtered_v_cmd = torch.where(
                post_pass_active,
                torch.minimum(self.filtered_v_cmd, door_speed_cap),
                self.filtered_v_cmd,
            )
        if self._use_defect_scene_reset() and self._is_dwb_like_scene_mode():
            commit_speed_min = float(getattr(self.cfg, "defect_dwb_commit_forward_speed_min", 0.0))
            if commit_speed_min > 0.0:
                local_xy = self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
                dwb_mask = self._get_dwb_like_scene_mask(self.fixed_scene_active_index)
                single_gap_mask = self.fixed_scene_active_index == self.DEFECT_SINGLE_OBSTACLE_SYMMETRIC_GAP
                gap_rel_y = local_xy[:, 1] - self._defect_dwb_gap_center_local_y
                gap_abs_y = gap_rel_y.abs()
                single_target_abs_y = self._get_single_obstacle_symmetric_gap_target_abs_y()
                obstacle_half_extent_x = self._get_box_obstacle_half_extent_x() if self._use_box_obstacles() else float(
                    getattr(self.cfg, "obstacle_radius", 0.2)
                )
                gap_x_min = self._defect_dwb_gap_center_local_x - float(
                    getattr(self.cfg, "defect_dwb_commit_x_margin_before_gap", 0.0)
                )
                gap_x_max = self._defect_dwb_gap_center_local_x + float(
                    getattr(self.cfg, "defect_dwb_commit_x_margin_after_gap", 0.0)
                )
                gap_x_max = torch.where(
                    single_gap_mask,
                    gap_x_max + obstacle_half_extent_x,
                    gap_x_max,
                )
                commit_lateral_tolerance = float(getattr(self.cfg, "defect_dwb_commit_lateral_tolerance", 0.0))
                commit_lateral_ready = torch.where(
                    single_gap_mask,
                    (gap_abs_y - single_target_abs_y).abs() <= commit_lateral_tolerance,
                    gap_abs_y <= commit_lateral_tolerance,
                )
                commit_ready = torch.logical_and(
                    dwb_mask,
                    torch.logical_and(
                        local_xy[:, 0] >= gap_x_min,
                        torch.logical_and(
                            local_xy[:, 0] <= gap_x_max,
                            torch.logical_and(
                                commit_lateral_ready,
                                torch.logical_and(
                                    heading_cos >= float(getattr(self.cfg, "defect_dwb_commit_heading_cos_min", 0.0)),
                                    min_lidar >= float(getattr(self.cfg, "defect_dwb_commit_min_lidar", 0.0)),
                                ),
                            ),
                        ),
                    ),
                )
                commit_ready = torch.logical_and(commit_ready, curr_goal_dist > self.cfg.goal_reach_threshold)
                commit_release = torch.logical_and(
                    dwb_mask,
                    torch.logical_and(
                        local_xy[:, 0] <= gap_x_max,
                        torch.logical_and(
                            curr_goal_dist > self.cfg.goal_reach_threshold,
                            min_lidar > (self.cfg.collision_threshold + 0.02),
                        ),
                    ),
                )
                self._defect_dwb_gap_commit_latched = torch.logical_or(
                    commit_ready,
                    torch.logical_and(self._defect_dwb_gap_commit_latched, commit_release),
                )
                commit_active = self._defect_dwb_gap_commit_latched
                commit_speed = torch.minimum(
                    torch.full_like(self.filtered_v_cmd, commit_speed_min),
                    speed_cap,
                )
                self.filtered_v_cmd = torch.where(
                    commit_active,
                    torch.maximum(self.filtered_v_cmd, commit_speed),
                    self.filtered_v_cmd,
                )
        v_cmd = self.filtered_v_cmd
        omega_cmd = self.filtered_omega_cmd

        # Joint clamp in (v, omega): |v +/- (L/2) * omega| <= V_max.
        half_wheel_base = 0.5 * float(self.cfg.wheel_base)
        v_max_wheel = float(self.cfg.wheel_linear_speed_max)
        eps = max(1.0e-12, float(self.cfg.diamond_clamp_eps))
        term_r = torch.abs(v_cmd + half_wheel_base * omega_cmd)
        term_l = torch.abs(v_cmd - half_wheel_base * omega_cmd)
        scale_r = v_max_wheel / torch.clamp(term_r, min=eps)
        scale_l = v_max_wheel / torch.clamp(term_l, min=eps)
        scale = torch.minimum(torch.minimum(scale_r, scale_l), torch.ones_like(scale_r))
        v_safe = v_cmd * scale
        omega_safe = omega_cmd * scale

        # Convert to wheel angular velocities.
        wheel_radius = max(1.0e-6, float(self.cfg.wheel_radius))
        left = (v_safe - half_wheel_base * omega_safe) / wheel_radius
        right = (v_safe + half_wheel_base * omega_safe) / wheel_radius
        wheel_targets = torch.stack((left, right), dim=-1)

        delta_limit = max(0.0, float(getattr(self.cfg, "wheel_target_delta_limit", 0.0)))
        if delta_limit > 0.0:
            wheel_delta = (wheel_targets - self.prev_wheel_targets).clamp(min=-delta_limit, max=delta_limit)
            wheel_targets = self.prev_wheel_targets + wheel_delta
        self.prev_wheel_targets = wheel_targets.clone()
        self.robot.set_joint_velocity_target(wheel_targets, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        lidar_ranges = self._compute_lidar_ranges()
        self.latest_lidar_ranges = lidar_ranges
        if bool(getattr(self.cfg, "use_lidar_normalization", False)):
            lidar_ranges_obs = torch.clamp(
                lidar_ranges / max(1.0e-6, float(self.cfg.lidar_max_range)),
                min=0.0,
                max=1.0,
            )
        else:
            lidar_ranges_obs = lidar_ranges

        if self._lidar_frame_stack > 1:
            self.lidar_frame_buffer = torch.roll(self.lidar_frame_buffer, shifts=-1, dims=1)
            self.lidar_frame_buffer[:, -1, :] = lidar_ranges_obs
            # Keep freshly reset envs numerically stable by repeating the current scan across the stack.
            new_episode_mask = self.episode_length_buf <= 1
            if bool(torch.any(new_episode_mask).item()):
                self.lidar_frame_buffer[new_episode_mask] = lidar_ranges_obs[new_episode_mask].unsqueeze(1)
            lidar_policy_obs = self.lidar_frame_buffer.reshape(lidar_ranges_obs.shape[0], -1)
        else:
            lidar_policy_obs = lidar_ranges_obs

        robot_pos = self.robot.data.root_pos_w
        robot_quat = self.robot.data.root_link_quat_w
        goal_vec = self.goal_pos_w - robot_pos
        goal_vec[:, 2] = 0.0
        goal_vec_b = math_utils.quat_apply_inverse(robot_quat, goal_vec)

        base_lin_vel = self.robot.data.root_com_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        projected_gravity = math_utils.quat_apply_inverse(robot_quat, self.gravity_vec_w)
        goal_xy = goal_vec_b[:, :2]
        goal_abs_y_b = goal_xy[:, 1].abs()
        min_lidar = torch.min(lidar_ranges, dim=1).values
        speed_cap = self._compute_speed_cap(min_lidar).unsqueeze(-1)
        goal_age_sec = self.episode_length_buf.unsqueeze(-1).float() * self.control_dt
        goal_changed = (
            goal_age_sec <= float(getattr(self.cfg, "goal_changed_pulse_sec", 0.75))
        ).float()
        goal_age_norm = (
            goal_age_sec / max(1.0e-6, float(getattr(self.cfg, "goal_age_norm_window_sec", 10.0)))
        ).clamp(0.0, 1.0)

        scalar_context = projected_gravity
        if (
            bool(getattr(self.cfg, "defect_scene_enable", False))
            and self._is_dwb_like_scene_mode()
            and bool(getattr(self.cfg, "defect_dwb_observation_enhance_enable", False))
        ):
            env_origins_xy = self.scene.env_origins[:, :2]
            robot_pos_xy_local = robot_pos[:, :2] - env_origins_xy
            gap_center_local = torch.stack(
                (self._defect_dwb_gap_center_local_x, self._defect_dwb_gap_center_local_y),
                dim=-1,
            )
            gap_center_w = gap_center_local + env_origins_xy
            gap_center_vec_w = torch.zeros_like(goal_vec)
            gap_center_vec_w[:, :2] = gap_center_w - robot_pos[:, :2]
            gap_center_vec_b = math_utils.quat_apply_inverse(robot_quat, gap_center_vec_w)
            gap_forward_norm = (
                gap_center_vec_b[:, 0] / max(0.5, float(getattr(self.cfg, "defect_dwb_corridor_half_length", 2.0)))
            ).clamp(-1.0, 1.0)
            gap_lateral_norm = (
                gap_center_vec_b[:, 1] / max(0.3, 0.5 * float(getattr(self.cfg, "defect_dwb_corridor_width", 2.0)))
            ).clamp(-1.0, 1.0)
            gap_clear_margin = float(getattr(self.cfg, "defect_dwb_gap_clear_x_margin", 0.14))
            gap_clear_threshold = self._defect_dwb_gap_center_local_x + gap_clear_margin
            if self._use_box_obstacles():
                gap_clear_threshold = gap_clear_threshold + self._get_box_obstacle_half_extent_x()
            gap_passed = (robot_pos_xy_local[:, 0] > gap_clear_threshold).float()
            scalar_context = torch.stack((gap_forward_norm, gap_lateral_norm, gap_passed), dim=-1)

        use_privileged_obs = bool(getattr(self.cfg, "use_privileged_obs", True))
        use_turn_cmd_obs = bool(getattr(self.cfg, "use_turn_cmd_obs", use_privileged_obs))
        use_planner_state_obs = bool(getattr(self.cfg, "use_planner_state_obs", use_privileged_obs))

        obs_parts = [
            base_lin_vel,
            base_ang_vel,
            scalar_context,
            goal_xy,
            self.prev_actions,
            lidar_policy_obs,
        ]
        if use_turn_cmd_obs:
            obs_parts.append(self.turn_cmd_onehot)
        obs_parts.append(speed_cap)
        if use_planner_state_obs:
            obs_parts.append(self.planner_state_onehot)
        obs_parts.extend([goal_changed, goal_age_norm])
        obs = torch.cat(obs_parts, dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        diff_goal = self.goal_pos_w[:, :2] - self.robot.data.root_pos_w[:, :2]
        curr_goal_dist = torch.linalg.norm(diff_goal, dim=1)
        progress_delta = self.prev_goal_dist - curr_goal_dist
        goal_vec = self.goal_pos_w - self.robot.data.root_pos_w
        goal_vec[:, 2] = 0.0
        goal_vec_b = math_utils.quat_apply_inverse(self.robot.data.root_link_quat_w, goal_vec)
        goal_abs_y_b = goal_vec_b[:, 1].abs()
        state_idx = self.planner_state_index
        fallback_mask = state_idx == self.STATE_FALLBACK
        return_mask = state_idx == self.STATE_RETURN
        recovery_mask = torch.logical_or(fallback_mask, return_mask)
        recovery_enable = bool(getattr(self.cfg, "recovery_state_reward_enable", False)) and self._use_mixed_planner_state_training()

        # Heading reward towards goal.
        forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)
        goal_dir_norm = diff_goal / curr_goal_dist.unsqueeze(-1).clamp(min=1e-6)
        heading_cos = torch.sum(forwards[:, :2] * goal_dir_norm, dim=-1)
        heading_weight = torch.clamp(heading_cos, min=0.0).pow(float(self.cfg.heading_gate_power))
        heading_weight = torch.clamp(heading_weight, min=float(self.cfg.heading_gate_min), max=1.0)

        # Encourage making forward progress when roughly facing the goal.
        forward_speed = self.robot.data.root_com_lin_vel_b[:, 0].clamp(min=0.0)
        recovery_mask_f = recovery_mask.float()
        if recovery_enable:
            forward_goal_mult = torch.ones_like(curr_goal_dist)
            forward_goal_mult = torch.where(
                fallback_mask,
                torch.full_like(forward_goal_mult, float(getattr(self.cfg, "recovery_forward_goal_scale_mult_fallback", 1.0))),
                forward_goal_mult,
            )
            forward_goal_mult = torch.where(
                return_mask,
                torch.full_like(forward_goal_mult, float(getattr(self.cfg, "recovery_forward_goal_scale_mult_return", 1.0))),
                forward_goal_mult,
            )
        else:
            forward_goal_mult = torch.ones_like(curr_goal_dist)
        turn_cmd_reward = torch.zeros_like(curr_goal_dist)
        if bool(getattr(self.cfg, "junction_enable", False)):
            goal_dir_w3 = torch.zeros((diff_goal.shape[0], 3), device=self.device)
            goal_dir_w3[:, :2] = goal_dir_norm
            goal_dir_b3 = math_utils.quat_apply_inverse(self.robot.data.root_link_quat_w, goal_dir_w3)
            turn_align = torch.sum(goal_dir_b3[:, :2] * self.turn_cmd_dir_b, dim=-1)
            turn_cmd_reward = torch.clamp(turn_align, min=0.0) * float(self.cfg.turn_cmd_reward_scale)
            # Additional yaw-consistency shaping for commanded turn behavior.
            yaw_rate_b = self.robot.data.root_ang_vel_b[:, 2]
            yaw_norm = (
                yaw_rate_b / max(1e-6, float(getattr(self.cfg, "turn_cmd_yaw_max_abs", 1.2)))
            ).clamp(-1.0, 1.0)
            yaw_align = torch.zeros_like(curr_goal_dist)
            left_mask = self.turn_cmd_index == self.TURN_LEFT
            right_mask = self.turn_cmd_index == self.TURN_RIGHT
            straight_mask = self.turn_cmd_index == self.TURN_STRAIGHT
            yaw_align[left_mask] = torch.clamp(yaw_norm[left_mask], min=0.0)
            yaw_align[right_mask] = torch.clamp(-yaw_norm[right_mask], min=0.0)
            yaw_align[straight_mask] = 1.0 - torch.clamp(torch.abs(yaw_norm[straight_mask]), 0.0, 1.0)
            turn_cmd_reward = turn_cmd_reward + yaw_align * float(getattr(self.cfg, "turn_cmd_yaw_reward_scale", 0.0))

        lidar_ranges = self._compute_lidar_ranges()
        self.latest_lidar_ranges = lidar_ranges
        min_lidar = torch.min(lidar_ranges, dim=1).values
        left_lidar_ranges = torch.where(
            self.left_lidar_beam_mask.unsqueeze(0),
            lidar_ranges,
            torch.full_like(lidar_ranges, self.cfg.lidar_max_range),
        )
        right_lidar_ranges = torch.where(
            self.right_lidar_beam_mask.unsqueeze(0),
            lidar_ranges,
            torch.full_like(lidar_ranges, self.cfg.lidar_max_range),
        )
        left_min_lidar = torch.min(left_lidar_ranges, dim=1).values
        right_min_lidar = torch.min(right_lidar_ranges, dim=1).values
        if recovery_enable:
            return_heading_gate_min_high = float(getattr(self.cfg, "recovery_heading_gate_min_return", 0.0))
            return_heading_gate_min_mid = float(
                getattr(self.cfg, "recovery_heading_gate_min_return_mid_clearance", return_heading_gate_min_high)
            )
            return_heading_gate_min_low = float(
                getattr(self.cfg, "recovery_heading_gate_min_return_low_clearance", return_heading_gate_min_mid)
            )
            return_heading_gate_high_clearance = float(
                getattr(self.cfg, "recovery_heading_gate_min_return_high_clearance_threshold", 0.80)
            )
            return_heading_gate_mid_clearance = float(
                getattr(self.cfg, "recovery_heading_gate_min_return_clearance_threshold", 0.60)
            )
            if return_heading_gate_min_high > 0.0 or return_heading_gate_min_mid > 0.0 or return_heading_gate_min_low > 0.0:
                return_heading_floor = torch.where(
                    min_lidar > return_heading_gate_high_clearance,
                    torch.full_like(heading_weight, return_heading_gate_min_high),
                    torch.where(
                        min_lidar > return_heading_gate_mid_clearance,
                        torch.full_like(heading_weight, return_heading_gate_min_mid),
                        torch.full_like(heading_weight, return_heading_gate_min_low),
                    ),
                )
                heading_weight_return = torch.minimum(
                    torch.maximum(heading_weight, return_heading_floor),
                    torch.ones_like(heading_weight),
                )
                heading_weight = torch.where(return_mask, heading_weight_return, heading_weight)
        pass_through_heading_fade_distance = max(
            1.0e-6,
            float(getattr(self.cfg, "pass_through_heading_fade_distance", self.cfg.goal_reach_threshold)),
        )
        pass_through_fade = (curr_goal_dist / pass_through_heading_fade_distance).clamp(0.0, 1.0)
        progress_heading_weight = 1.0 - pass_through_fade * (1.0 - heading_weight)
        progress_reward = progress_heading_weight * progress_delta * self.cfg.progress_reward_scale
        if recovery_enable:
            return_progress_gain = float(getattr(self.cfg, "return_progress_age_mult_gain", 0.0))
            return_progress_cap = max(1.0, float(getattr(self.cfg, "return_progress_age_mult_cap", 1.0)))
            if return_progress_gain > 0.0 and return_progress_cap > 1.0:
                goal_age_sec = self.episode_length_buf.float() * self.control_dt
                goal_age_norm = (
                    goal_age_sec / max(1.0e-6, float(getattr(self.cfg, "goal_age_norm_window_sec", 10.0)))
                ).clamp(0.0, 1.0)
                return_progress_mult = torch.clamp(
                    1.0 + return_progress_gain * goal_age_norm,
                    min=1.0,
                    max=return_progress_cap,
                )
                progress_reward = torch.where(return_mask, progress_reward * return_progress_mult, progress_reward)
        heading_reward = heading_weight * self.cfg.heading_reward_scale * pass_through_fade
        forward_goal_reward = heading_weight * forward_speed * self.cfg.forward_goal_reward_scale * forward_goal_mult
        if recovery_enable:
            return_forward_scale_high = float(getattr(self.cfg, "return_forward_goal_scale_high_clearance", 1.0))
            return_forward_scale_mid = float(getattr(self.cfg, "return_forward_goal_scale_mid_clearance", return_forward_scale_high))
            return_forward_scale_low = float(getattr(self.cfg, "return_forward_goal_scale_low_clearance", return_forward_scale_mid))
            return_forward_scale_high_th = float(
                getattr(self.cfg, "return_forward_goal_scale_high_clearance_threshold", 0.80)
            )
            return_forward_scale_mid_th = float(
                getattr(self.cfg, "return_forward_goal_scale_mid_clearance_threshold", 0.60)
            )
            return_forward_scale = torch.where(
                min_lidar > return_forward_scale_high_th,
                torch.full_like(curr_goal_dist, return_forward_scale_high),
                torch.where(
                    min_lidar > return_forward_scale_mid_th,
                    torch.full_like(curr_goal_dist, return_forward_scale_mid),
                    torch.full_like(curr_goal_dist, return_forward_scale_low),
                ),
            )
            forward_goal_reward = torch.where(return_mask, forward_goal_reward * return_forward_scale, forward_goal_reward)

            return_bad_progress_delta = float(
                getattr(
                    self.cfg,
                    "return_no_progress_moving_delta_threshold",
                    getattr(self.cfg, "return_no_progress_delta_threshold", 0.0015),
                )
            )
            return_forward_bad_progress_scale = float(getattr(self.cfg, "return_forward_goal_bad_progress_scale", 1.0))
            if return_forward_bad_progress_scale < 1.0:
                return_bad_progress = torch.logical_and(return_mask, progress_delta < return_bad_progress_delta)
                forward_goal_reward = torch.where(
                    return_bad_progress,
                    forward_goal_reward * return_forward_bad_progress_scale,
                    forward_goal_reward,
                )
        # Near-goal shaping: separate multipliers for progress and forward_goal
        # Safety gate: multiplier only activates when min_lidar > near_goal_clearance_min
        # (prevents "rush into wall" behavior near obstacles)
        near_goal_dist_mid = float(getattr(self.cfg, "near_goal_dist_mid", 1.5))
        near_goal_dist_close = float(getattr(self.cfg, "near_goal_dist_close", 0.8))
        near_goal_clearance_min = float(getattr(self.cfg, "near_goal_clearance_min", 0.6))
        near_goal_safe = min_lidar > near_goal_clearance_min
        near_goal_mult_mid = float(getattr(self.cfg, "near_goal_progress_mult_mid", 1.0))
        near_goal_mult_close = float(getattr(self.cfg, "near_goal_progress_mult_close", 1.0))
        if near_goal_mult_mid > 1.0 or near_goal_mult_close > 1.0:
            near_goal_prog_scale = torch.where(
                near_goal_safe & (curr_goal_dist < near_goal_dist_close),
                torch.full_like(curr_goal_dist, near_goal_mult_close),
                torch.where(
                    near_goal_safe & (curr_goal_dist < near_goal_dist_mid),
                    torch.full_like(curr_goal_dist, near_goal_mult_mid),
                    torch.ones_like(curr_goal_dist),
                ),
            )
            progress_reward = progress_reward * near_goal_prog_scale
        near_goal_fwd_mult_mid = float(getattr(self.cfg, "near_goal_fwd_mult_mid", 1.0))
        near_goal_fwd_mult_close = float(getattr(self.cfg, "near_goal_fwd_mult_close", 1.0))
        if near_goal_fwd_mult_mid > 1.0 or near_goal_fwd_mult_close > 1.0:
            near_goal_fwd_scale = torch.where(
                near_goal_safe & (curr_goal_dist < near_goal_dist_close),
                torch.full_like(curr_goal_dist, near_goal_fwd_mult_close),
                torch.where(
                    near_goal_safe & (curr_goal_dist < near_goal_dist_mid),
                    torch.full_like(curr_goal_dist, near_goal_fwd_mult_mid),
                    torch.ones_like(curr_goal_dist),
                ),
            )
            forward_goal_reward = forward_goal_reward * near_goal_fwd_scale
        defect_dwb_forward_speed_reward = torch.zeros_like(curr_goal_dist)
        if self._use_defect_scene_reset():
            dwb_mask = self._get_dwb_like_scene_mask(self.fixed_scene_active_index)
            if bool(torch.any(dwb_mask).item()):
                dwb_forward_speed_reward_scale = float(getattr(self.cfg, "defect_dwb_forward_speed_reward_scale", 0.0))
                if dwb_forward_speed_reward_scale > 0.0:
                    dwb_forward_speed_min_lidar = float(getattr(self.cfg, "defect_dwb_forward_speed_min_lidar", 0.0))
                    dwb_forward_speed_heading_min = float(getattr(self.cfg, "defect_dwb_forward_speed_heading_min", 0.0))
                    dwb_safe_to_push = torch.logical_and(
                        dwb_mask,
                        torch.logical_and(min_lidar > dwb_forward_speed_min_lidar, heading_weight > dwb_forward_speed_heading_min),
                    )
                    defect_dwb_forward_speed_reward = torch.where(
                        dwb_safe_to_push,
                        torch.clamp(forward_speed, min=0.0) * dwb_forward_speed_reward_scale,
                        torch.zeros_like(curr_goal_dist),
                    )
                    forward_goal_reward = forward_goal_reward + defect_dwb_forward_speed_reward
        contact_force = self._compute_obstacle_contact_force()
        self.latest_contact_force = contact_force
        if recovery_enable:
            clearance_scale_mult = torch.ones_like(curr_goal_dist)
            clearance_scale_mult = torch.where(
                fallback_mask,
                torch.full_like(
                    clearance_scale_mult,
                    float(getattr(self.cfg, "recovery_clearance_scale_mult_fallback", 1.0)),
                ),
                clearance_scale_mult,
            )
            clearance_scale_mult = torch.where(
                return_mask,
                torch.full_like(
                    clearance_scale_mult,
                    float(getattr(self.cfg, "recovery_clearance_scale_mult_return", 1.0)),
                ),
                clearance_scale_mult,
            )
        else:
            clearance_scale_mult = torch.ones_like(curr_goal_dist)
        clearance_reward = (min_lidar / self.cfg.lidar_max_range).clamp(0.0, 1.0) * self.cfg.clearance_reward_scale * clearance_scale_mult
        if recovery_enable:
            clearance_gain = torch.clamp(min_lidar - self.prev_min_lidar, min=0.0) / max(1e-6, float(self.cfg.lidar_max_range))
            clearance_reward = clearance_reward + recovery_mask_f * clearance_gain * float(
                getattr(self.cfg, "recovery_clearance_gain_reward_scale", 0.0)
            )

        smooth_action_reward = (
            torch.sum(torch.square(self.actions - self.prev_actions), dim=1) * self.cfg.smooth_action_reward_scale
        )
        # Penalise sustained angular velocity to suppress spinning and oscillation at turns.
        yaw_rate_b = self.robot.data.root_ang_vel_b[:, 2]
        yaw_rate_abs = self.robot.data.root_ang_vel_b[:, 2].abs()
        yaw_sign_deadzone = max(0.0, float(getattr(self.cfg, "debug_yaw_sign_deadzone", 0.05)))
        raw_yaw_sign = torch.where(
            yaw_rate_b > yaw_sign_deadzone,
            torch.ones_like(yaw_rate_b),
            torch.where(yaw_rate_b < -yaw_sign_deadzone, -torch.ones_like(yaw_rate_b), torch.zeros_like(yaw_rate_b)),
        )
        effective_yaw_sign = torch.where(raw_yaw_sign == 0.0, self._prev_yaw_sign, raw_yaw_sign)
        yaw_sign_flip = torch.logical_and(self._prev_yaw_sign != 0.0, effective_yaw_sign != 0.0)
        yaw_sign_flip = torch.logical_and(yaw_sign_flip, effective_yaw_sign != self._prev_yaw_sign)
        # Stage-11 micro-tune: reward consistent turn direction while moving to reduce dithering.
        persistence_reward = torch.zeros_like(curr_goal_dist)
        flip_penalty = torch.zeros_like(curr_goal_dist)
        direction_bias_reward = torch.zeros_like(curr_goal_dist)
        dwb_like_scene = self._is_dwb_like_scene_mode()
        persistence_stage_indices_cfg = getattr(self.cfg, "yaw_persistence_stage_indices", (11,))
        if isinstance(persistence_stage_indices_cfg, int):
            persistence_stage_indices = {int(persistence_stage_indices_cfg)}
        elif isinstance(persistence_stage_indices_cfg, (list, tuple, set)):
            persistence_stage_indices = {int(stage_idx) for stage_idx in persistence_stage_indices_cfg}
        else:
            persistence_stage_indices = {11}
        current_dwb_stage = int(getattr(self, "_defect_dwb_curriculum_stage", -1))
        if dwb_like_scene and current_dwb_stage in persistence_stage_indices:
            persistence_speed_threshold = float(getattr(self.cfg, "yaw_persistence_speed_threshold", 0.05))
            persistence_reward_scale = float(getattr(self.cfg, "yaw_persistence_reward_scale", 0.03))
            flip_penalty_scale = float(getattr(self.cfg, "yaw_flip_penalty_scale", 0.01))
            same_sign = torch.logical_and(effective_yaw_sign == self._prev_yaw_sign, effective_yaw_sign != 0.0)
            moving_mask = forward_speed > persistence_speed_threshold
            persistence_reward = same_sign.float() * moving_mask.float() * persistence_reward_scale
            flip_penalty = -yaw_sign_flip.float() * moving_mask.float() * flip_penalty_scale
            # Stage-11 symmetry break: apply a tiny left-turn bias while moving.
            if current_dwb_stage == 11:
                direction_bias_scale = float(getattr(self.cfg, "yaw_direction_bias_scale", 0.03))
                direction_bias_reward = (yaw_rate_b > 0.0).float() * moving_mask.float() * direction_bias_scale
        if recovery_enable:
            turn_penalty_scale_mult = torch.ones_like(curr_goal_dist)
            turn_penalty_scale_mult = torch.where(
                fallback_mask,
                torch.full_like(
                    turn_penalty_scale_mult,
                    float(getattr(self.cfg, "recovery_turn_penalty_mult_fallback", 1.0)),
                ),
                turn_penalty_scale_mult,
            )
            turn_penalty_scale_mult = torch.where(
                return_mask,
                torch.full_like(
                    turn_penalty_scale_mult,
                    float(getattr(self.cfg, "recovery_turn_penalty_mult_return", 1.0)),
                ),
                turn_penalty_scale_mult,
            )
        else:
            turn_penalty_scale_mult = torch.ones_like(curr_goal_dist)
        turn_penalty = yaw_rate_abs * self.cfg.turn_penalty_scale * turn_penalty_scale_mult
        turn_penalty = turn_penalty * pass_through_fade
        if recovery_enable:
            yaw_osc = torch.abs(yaw_rate_b - self.prev_yaw_rate)
            turn_penalty = turn_penalty + recovery_mask_f * yaw_osc * float(
                getattr(self.cfg, "recovery_yaw_oscillation_penalty_scale", 0.0)
            )
            return_turn_boost_high = float(getattr(self.cfg, "return_turn_penalty_clearance_boost_high", 1.0))
            return_turn_boost_mid = float(getattr(self.cfg, "return_turn_penalty_clearance_boost_mid", return_turn_boost_high))
            return_turn_boost_low = float(getattr(self.cfg, "return_turn_penalty_clearance_boost_low", return_turn_boost_mid))
            return_turn_boost_high_th = float(getattr(self.cfg, "return_turn_penalty_boost_high_clearance_threshold", 0.80))
            return_turn_boost_mid_th = float(getattr(self.cfg, "return_turn_penalty_boost_mid_clearance_threshold", 0.60))
            return_turn_boost = torch.where(
                min_lidar > return_turn_boost_high_th,
                torch.full_like(curr_goal_dist, return_turn_boost_high),
                torch.where(
                    min_lidar > return_turn_boost_mid_th,
                    torch.full_like(curr_goal_dist, return_turn_boost_mid),
                    torch.full_like(curr_goal_dist, return_turn_boost_low),
                ),
            )
            turn_penalty = torch.where(return_mask, turn_penalty * return_turn_boost, turn_penalty)
        time_penalty = torch.full_like(curr_goal_dist, self.cfg.time_penalty)
        if recovery_enable:
            stall_speed_threshold = torch.full_like(curr_goal_dist, float(self.cfg.stall_speed_threshold))
            stall_speed_threshold = torch.where(
                fallback_mask,
                stall_speed_threshold * float(getattr(self.cfg, "recovery_stall_speed_threshold_mult_fallback", 1.0)),
                stall_speed_threshold,
            )
            stall_speed_threshold = torch.where(
                return_mask,
                stall_speed_threshold * float(getattr(self.cfg, "recovery_stall_speed_threshold_mult_return", 1.0)),
                stall_speed_threshold,
            )
        else:
            stall_speed_threshold = torch.full_like(curr_goal_dist, float(self.cfg.stall_speed_threshold))
        # Stall: low forward speed AND no recent progress toward goal.
        # Use progress_delta instead of distance threshold to catch "fake motion" (spinning in place).
        stall_progress_threshold = float(getattr(self.cfg, "stall_progress_threshold", 0.003))
        stall = torch.logical_and(
            forward_speed < stall_speed_threshold,
            progress_delta < stall_progress_threshold,
        )
        in_success_halo = curr_goal_dist < self.cfg.goal_reach_threshold
        stall = torch.logical_and(stall, ~in_success_halo)
        stall_penalty = stall.float() * self.cfg.stall_penalty
        dwb_no_progress = self._compute_defect_dwb_no_progress_mask(
            curr_goal_dist=curr_goal_dist,
            min_lidar=min_lidar,
            forward_speed=forward_speed,
            progress_delta=progress_delta,
        )
        dwb_no_progress_penalty = float(getattr(self.cfg, "defect_dwb_no_progress_penalty", 0.0))
        if dwb_no_progress_penalty != 0.0:
            stall_penalty = stall_penalty + dwb_no_progress.float() * dwb_no_progress_penalty
        front_lidar_ranges = torch.where(
            self.front_lidar_beam_mask.unsqueeze(0),
            lidar_ranges,
            torch.full_like(lidar_ranges, self.cfg.lidar_max_range),
        )
        front_min_lidar = torch.min(front_lidar_ranges, dim=1).values
        if recovery_enable:
            danger_penalty_mult = torch.ones_like(curr_goal_dist)
            danger_penalty_mult = torch.where(
                fallback_mask,
                torch.full_like(
                    danger_penalty_mult,
                    float(getattr(self.cfg, "recovery_danger_speed_penalty_mult_fallback", 1.0)),
                ),
                danger_penalty_mult,
            )
            danger_penalty_mult = torch.where(
                return_mask,
                torch.full_like(
                    danger_penalty_mult,
                    float(getattr(self.cfg, "recovery_danger_speed_penalty_mult_return", 1.0)),
                ),
                danger_penalty_mult,
            )
        else:
            danger_penalty_mult = torch.ones_like(curr_goal_dist)
        front_danger_distance = float(getattr(self.cfg, "front_danger_distance", self.cfg.danger_speed_distance))
        danger_intrusion = torch.clamp(front_danger_distance - front_min_lidar, min=0.0)
        danger_speed_penalty = (
            forward_speed * danger_intrusion.square() * self.cfg.danger_speed_penalty_scale * danger_penalty_mult
        )
        action_abs_mean = torch.mean(torch.abs(self.actions), dim=1)
        near_goal_scale = (
            curr_goal_dist / max(1e-6, float(self.cfg.near_goal_slowdown_distance))
        ).clamp(min=float(self.cfg.near_goal_min_action_scale), max=1.0)
        obstacle_den = max(1e-6, float(self.cfg.obstacle_slowdown_distance - self.cfg.collision_threshold))
        obstacle_scale = ((min_lidar - self.cfg.collision_threshold) / obstacle_den).clamp(
            min=float(self.cfg.obstacle_min_action_scale), max=1.0
        )
        action_eff_abs_mean = action_abs_mean * near_goal_scale * obstacle_scale
        action_saturated = action_eff_abs_mean > self.cfg.action_saturation_threshold
        sat_ratio = (
            torch.clamp(action_eff_abs_mean - self.cfg.action_saturation_threshold, min=0.0)
            / max(1e-6, 1.0 - float(self.cfg.action_saturation_threshold))
        ).clamp(0.0, 1.0)
        if recovery_enable:
            action_sat_penalty_mult = torch.ones_like(curr_goal_dist)
            action_sat_penalty_mult = torch.where(
                fallback_mask,
                torch.full_like(
                    action_sat_penalty_mult,
                    float(getattr(self.cfg, "recovery_action_saturation_penalty_mult_fallback", 1.0)),
                ),
                action_sat_penalty_mult,
            )
            action_sat_penalty_mult = torch.where(
                return_mask,
                torch.full_like(
                    action_sat_penalty_mult,
                    float(getattr(self.cfg, "recovery_action_saturation_penalty_mult_return", 1.0)),
                ),
                action_sat_penalty_mult,
            )
        else:
            action_sat_penalty_mult = torch.ones_like(curr_goal_dist)
        action_saturation_penalty = (
            sat_ratio.pow(float(self.cfg.action_saturation_penalty_power))
            * self.cfg.action_saturation_penalty_scale
            * action_sat_penalty_mult
        )
        if recovery_enable:
            return_sat_boost_high = float(getattr(self.cfg, "return_action_sat_clearance_boost_high", 1.0))
            return_sat_boost_mid = float(getattr(self.cfg, "return_action_sat_clearance_boost_mid", return_sat_boost_high))
            return_sat_boost_low = float(getattr(self.cfg, "return_action_sat_clearance_boost_low", return_sat_boost_mid))
            return_sat_boost_high_th = float(getattr(self.cfg, "return_action_sat_boost_high_clearance_threshold", 0.80))
            return_sat_boost_mid_th = float(getattr(self.cfg, "return_action_sat_boost_mid_clearance_threshold", 0.60))
            return_sat_boost = torch.where(
                min_lidar > return_sat_boost_high_th,
                torch.full_like(curr_goal_dist, return_sat_boost_high),
                torch.where(
                    min_lidar > return_sat_boost_mid_th,
                    torch.full_like(curr_goal_dist, return_sat_boost_mid),
                    torch.full_like(curr_goal_dist, return_sat_boost_low),
                ),
            )
            action_saturation_penalty = torch.where(
                return_mask,
                action_saturation_penalty * return_sat_boost,
                action_saturation_penalty,
            )
        return_no_progress_penalty = torch.zeros_like(curr_goal_dist)
        if recovery_enable:
            return_no_progress_delta_threshold = float(getattr(self.cfg, "return_no_progress_delta_threshold", 0.0015))
            return_no_progress_speed_threshold = float(getattr(self.cfg, "return_no_progress_speed_threshold", 0.08))
            return_no_progress_clearance_threshold = float(getattr(self.cfg, "return_no_progress_clearance_threshold", 0.55))
            return_no_progress_moving_speed_threshold = float(
                getattr(self.cfg, "return_no_progress_moving_speed_threshold", 0.14)
            )
            return_no_progress_moving_delta_threshold = float(
                getattr(self.cfg, "return_no_progress_moving_delta_threshold", 0.004)
            )

            return_clear_for_unstall = min_lidar > return_no_progress_clearance_threshold
            return_no_progress_stuck = torch.logical_and(
                torch.logical_and(
                    progress_delta < return_no_progress_delta_threshold,
                    torch.logical_and(
                        forward_speed < return_no_progress_speed_threshold,
                        return_clear_for_unstall,
                    ),
                ),
                return_mask,
            )
            # Also treat "moving but not getting closer" as no-progress in return mode.
            return_no_progress_moving = torch.logical_and(
                return_mask,
                torch.logical_and(
                    forward_speed >= return_no_progress_moving_speed_threshold,
                    torch.logical_and(
                        progress_delta < return_no_progress_moving_delta_threshold,
                        return_clear_for_unstall,
                    ),
                ),
            )
            return_no_progress = torch.logical_or(
                return_no_progress_stuck,
                return_no_progress_moving,
            )
            self.return_no_progress_steps = torch.where(
                return_no_progress,
                self.return_no_progress_steps + 1,
                torch.zeros_like(self.return_no_progress_steps),
            )
            return_no_progress_penalty_step = float(getattr(self.cfg, "return_no_progress_penalty_step", 0.0))
            return_no_progress_penalty_cap = max(0.0, float(getattr(self.cfg, "return_no_progress_penalty_cap", 0.0)))
            if return_no_progress_penalty_step > 0.0 and return_no_progress_penalty_cap > 0.0:
                return_no_progress_moving_penalty_mult = float(
                    getattr(self.cfg, "return_no_progress_moving_penalty_mult", 1.0)
                )
                return_no_progress_penalty_mult = torch.where(
                    return_no_progress_moving,
                    torch.full_like(curr_goal_dist, return_no_progress_moving_penalty_mult),
                    torch.ones_like(curr_goal_dist),
                )
                return_no_progress_penalty_mag = torch.clamp(
                    self.return_no_progress_steps.float() * return_no_progress_penalty_step * return_no_progress_penalty_mult,
                    min=0.0,
                    max=return_no_progress_penalty_cap,
                )
                return_no_progress_penalty = torch.where(
                    return_no_progress,
                    -return_no_progress_penalty_mag,
                    torch.zeros_like(curr_goal_dist),
                )
        else:
            self.return_no_progress_steps[:] = 0

        subgoal_progress_reward = torch.zeros_like(curr_goal_dist)
        subgoal_bonus_reward = torch.zeros_like(curr_goal_dist)
        if self._subgoal_reward_enable:
            subgoal_scene_index = int(getattr(self.cfg, "subgoal_scene_index", -1))
            subgoal_scene_mask = self.fixed_scene_active_index == subgoal_scene_index
            if bool(torch.any(subgoal_scene_mask).item()):
                curr_subgoal_dist = torch.linalg.norm(
                    self.subgoal_pos_w[:, :2] - self.robot.data.root_pos_w[:, :2],
                    dim=1,
                )
                active_subgoal_mask = torch.logical_and(subgoal_scene_mask, ~self._subgoal_reached)
                if bool(torch.any(active_subgoal_mask).item()):
                    subgoal_delta = self.prev_subgoal_dist - curr_subgoal_dist
                    subgoal_progress_term = subgoal_delta * float(getattr(self.cfg, "subgoal_progress_reward_scale", 0.0))
                    subgoal_progress_reward = torch.where(
                        active_subgoal_mask,
                        subgoal_progress_term,
                        torch.zeros_like(curr_goal_dist),
                    )
                    entered_subgoal = torch.logical_and(
                        active_subgoal_mask,
                        curr_subgoal_dist < float(getattr(self.cfg, "subgoal_radius", 0.30)),
                    )
                    if bool(torch.any(entered_subgoal).item()):
                        subgoal_bonus_reward = torch.where(
                            entered_subgoal,
                            torch.full_like(curr_goal_dist, float(getattr(self.cfg, "subgoal_bonus", 0.0))),
                            torch.zeros_like(curr_goal_dist),
                        )
                        self._subgoal_reached[entered_subgoal] = True
                self.prev_subgoal_dist = torch.where(subgoal_scene_mask, curr_subgoal_dist, self.prev_subgoal_dist)

        bypass_reward = torch.zeros_like(curr_goal_dist)
        if bool(getattr(self.cfg, "bypass_reward_enable", False)):
            bypass_scene_index = int(getattr(self.cfg, "bypass_scene_index", -1))
            if bypass_scene_index >= 0:
                bypass_scene_mask = self.fixed_scene_active_index == bypass_scene_index
            else:
                bypass_scene_mask = torch.ones_like(curr_goal_dist, dtype=torch.bool)
            bypass_open_side_min = float(getattr(self.cfg, "bypass_open_side_min_lidar", 0.30))
            bypass_blocked_side_max = float(getattr(self.cfg, "bypass_blocked_side_max_lidar", 0.30))
            bypass_near_obs = float(getattr(self.cfg, "bypass_near_obstacle_distance", 0.90))
            bypass_min_speed = float(getattr(self.cfg, "bypass_forward_speed_threshold", 0.05))
            left_open_right_blocked = torch.logical_and(
                left_min_lidar > bypass_open_side_min,
                right_min_lidar < bypass_blocked_side_max,
            )
            right_open_left_blocked = torch.logical_and(
                right_min_lidar > bypass_open_side_min,
                left_min_lidar < bypass_blocked_side_max,
            )
            bypass_side_mask = torch.logical_or(left_open_right_blocked, right_open_left_blocked)
            # In defect scenes we may want bypass shaping without subgoal shaping enabled.
            if self._subgoal_reward_enable:
                bypass_subgoal_gate = ~self._subgoal_reached
            else:
                bypass_subgoal_gate = torch.ones_like(curr_goal_dist, dtype=torch.bool)
            bypass_active = torch.logical_and(
                bypass_scene_mask,
                torch.logical_and(
                    min_lidar < bypass_near_obs,
                    torch.logical_and(forward_speed > bypass_min_speed, bypass_subgoal_gate),
                ),
            )
            bypass_active = torch.logical_and(bypass_active, bypass_side_mask)
            bypass_reward = torch.where(
                bypass_active,
                forward_speed * float(getattr(self.cfg, "bypass_reward_scale", 0.10)),
                torch.zeros_like(curr_goal_dist),
            )

        defect_lateral_penalty = torch.zeros_like(curr_goal_dist)
        defect_narrow_passage_reward = torch.zeros_like(curr_goal_dist)
        defect_dwb_gap_progress_reward = torch.zeros_like(curr_goal_dist)
        defect_dwb_gap_alignment_reward = torch.zeros_like(curr_goal_dist)
        defect_dwb_post_gap_reward = torch.zeros_like(curr_goal_dist)
        defect_dwb_post_gap_lateral_align_reward = torch.zeros_like(curr_goal_dist)
        defect_dwb_side_commit_reward = torch.zeros_like(curr_goal_dist)
        defect_dwb_gap_clear_reward = torch.zeros_like(curr_goal_dist)
        defect_single_gap_centerline_penalty = torch.zeros_like(curr_goal_dist)
        speed_surge_reward = torch.zeros_like(curr_goal_dist)
        speed_deficit_penalty = torch.zeros_like(curr_goal_dist)
        defect_dwb_clearance_penalty = torch.zeros_like(curr_goal_dist)
        defect_turn_completion_reward = torch.zeros_like(curr_goal_dist)
        defect_corner_mid_progress_reward = torch.zeros_like(curr_goal_dist)
        defect_u_escape_reward = torch.zeros_like(curr_goal_dist)
        defect_exploration_reward = torch.zeros_like(curr_goal_dist)
        defect_u_backward_reward = torch.zeros_like(curr_goal_dist)
        if self._use_defect_scene_reset():
            local_xy = self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
            scene_idx = self.fixed_scene_active_index
            dwb_mask = scene_idx == self.DEFECT_DWB_OSCILLATION
            single_gap_mask = scene_idx == self.DEFECT_SINGLE_OBSTACLE_SYMMETRIC_GAP
            dwb_like_mask = self._get_dwb_like_scene_mask(scene_idx)
            mppi_mask = scene_idx == self.DEFECT_MPPI_CORNER_FAIL
            door_mask = scene_idx == self.DEFECT_DOOR_DEADLOCK
            u_mask = scene_idx == self.DEFECT_U_SHAPE_TRAP

            lateral_vel_abs = self.robot.data.root_com_lin_vel_b[:, 1].abs()
            defect_lateral_penalty = lateral_vel_abs * float(getattr(self.cfg, "defect_lateral_velocity_penalty_scale", -0.2))
            speed_surge_scale = float(getattr(self.cfg, "defect_dwb_speed_surge_scale", 0.0))
            if speed_surge_scale > 0.0:
                speed_surge_threshold = float(getattr(self.cfg, "defect_dwb_speed_surge_threshold", 0.15))
                speed_surge_reward = torch.where(
                    dwb_like_mask,
                    torch.clamp(forward_speed - speed_surge_threshold, min=0.0) * speed_surge_scale,
                    torch.zeros_like(curr_goal_dist),
                )
            speed_deficit_scale = float(getattr(self.cfg, "defect_dwb_speed_deficit_scale", 0.0))
            if speed_deficit_scale > 0.0:
                speed_deficit_target = float(getattr(self.cfg, "defect_dwb_speed_deficit_target", 0.15))
                speed_deficit_penalty = torch.where(
                    dwb_like_mask,
                    -torch.clamp(speed_deficit_target - forward_speed, min=0.0) * speed_deficit_scale,
                    torch.zeros_like(curr_goal_dist),
                )
            clearance_penalty_scale = float(getattr(self.cfg, "defect_dwb_clearance_penalty_scale", 0.0))
            front_clearance_penalty_scale = float(
                getattr(self.cfg, "defect_dwb_front_clearance_penalty_scale", 0.0)
            )
            if clearance_penalty_scale != 0.0 or front_clearance_penalty_scale != 0.0:
                clearance_penalty_power = max(1.0, float(getattr(self.cfg, "defect_dwb_clearance_penalty_power", 2.0)))
                dwb_clearance_terms = torch.zeros_like(curr_goal_dist)
                if clearance_penalty_scale != 0.0:
                    clearance_distance = max(
                        0.0,
                        float(getattr(self.cfg, "defect_dwb_clearance_penalty_distance", 0.30)),
                    )
                    clearance_intrusion = torch.clamp(clearance_distance - min_lidar, min=0.0)
                    dwb_clearance_terms = (
                        dwb_clearance_terms
                        + torch.pow(clearance_intrusion, clearance_penalty_power) * clearance_penalty_scale
                    )
                if front_clearance_penalty_scale != 0.0:
                    front_clearance_distance = max(
                        0.0,
                        float(getattr(self.cfg, "defect_dwb_front_clearance_penalty_distance", 0.45)),
                    )
                    front_clearance_intrusion = torch.clamp(front_clearance_distance - front_min_lidar, min=0.0)
                    dwb_clearance_terms = (
                        dwb_clearance_terms
                        + torch.pow(front_clearance_intrusion, clearance_penalty_power) * front_clearance_penalty_scale
                    )
                defect_dwb_clearance_penalty = torch.where(
                    dwb_like_mask,
                    dwb_clearance_terms,
                    defect_dwb_clearance_penalty,
                )

            narrow_lat_tol = max(0.01, float(getattr(self.cfg, "defect_narrow_passage_lateral_tolerance", 0.1)))
            narrow_x_window = max(0.02, float(getattr(self.cfg, "defect_narrow_passage_x_window", 0.25)))
            dwb_gap_dx = (local_xy[:, 0] - self._defect_dwb_gap_center_local_x).abs()
            dwb_gap_rel_y = local_xy[:, 1] - self._defect_dwb_gap_center_local_y
            dwb_gap_dy = (local_xy[:, 1] - self._defect_dwb_gap_center_local_y).abs()
            dwb_gap_dist = torch.sqrt(dwb_gap_dx.square() + dwb_gap_dy.square())
            single_target_abs_y = self._get_single_obstacle_symmetric_gap_target_abs_y()
            single_target_tol = max(
                0.01,
                float(getattr(self.cfg, "defect_single_gap_target_lateral_tolerance", 0.08)),
            )
            single_target_error = (dwb_gap_dy - single_target_abs_y).abs()
            single_forward_dist = torch.clamp(self._defect_dwb_gap_center_local_x - local_xy[:, 0], min=0.0)
            dwb_progress_metric = torch.where(single_gap_mask, single_forward_dist, dwb_gap_dist)
            dwb_alignment_metric = torch.where(single_gap_mask, single_target_error, dwb_gap_dy)
            gap_clear_margin = float(getattr(self.cfg, "defect_dwb_gap_clear_x_margin", 0.18))
            obstacle_half_extent_x = self._get_box_obstacle_half_extent_x() if self._use_box_obstacles() else float(
                getattr(self.cfg, "obstacle_radius", 0.2)
            )
            clear_x_threshold = torch.where(
                single_gap_mask,
                self._defect_dwb_gap_center_local_x + obstacle_half_extent_x + gap_clear_margin,
                self._defect_dwb_gap_center_local_x + gap_clear_margin,
            )
            if bool(torch.any(dwb_like_mask).item()):
                dwb_gap_progress_scale = float(getattr(self.cfg, "defect_dwb_gap_progress_reward_scale", 0.0))
                active_gap_progress = torch.logical_and(dwb_like_mask, ~self._defect_dwb_gap_reached)
                if dwb_gap_progress_scale > 0.0:
                    defect_dwb_gap_progress_reward = torch.where(
                        active_gap_progress,
                        (self._defect_dwb_prev_gap_dist - dwb_progress_metric) * dwb_gap_progress_scale,
                        torch.zeros_like(curr_goal_dist),
                    )
                dwb_gap_align_scale = float(getattr(self.cfg, "defect_dwb_gap_alignment_reward_scale", 0.0))
                if dwb_gap_align_scale > 0.0:
                    gap_align_active = torch.logical_and(
                        active_gap_progress,
                        local_xy[:, 0] <= clear_x_threshold,
                    )
                    defect_dwb_gap_alignment_reward = torch.where(
                        gap_align_active,
                        (self._defect_dwb_prev_gap_abs_y - dwb_alignment_metric) * dwb_gap_align_scale,
                        torch.zeros_like(curr_goal_dist),
                    )
                single_gap_centerline_penalty_scale = float(
                    getattr(self.cfg, "defect_single_gap_centerline_penalty_scale", 0.0)
                )
                if single_gap_centerline_penalty_scale != 0.0:
                    single_gap_centerline_forward_dist = max(
                        0.01,
                        float(getattr(self.cfg, "defect_single_gap_centerline_penalty_forward_dist", 0.90)),
                    )
                    single_gap_centerline_active = torch.logical_and(
                        torch.logical_and(active_gap_progress, single_gap_mask),
                        torch.logical_and(
                            local_xy[:, 0] <= clear_x_threshold,
                            single_forward_dist <= single_gap_centerline_forward_dist,
                        ),
                    )
                    single_gap_centerline_ratio = torch.clamp(single_target_abs_y - dwb_gap_dy, min=0.0) / max(
                        single_target_abs_y,
                        1.0e-6,
                    )
                    single_gap_centerline_proximity = 1.0 - torch.clamp(
                        single_forward_dist / single_gap_centerline_forward_dist,
                        min=0.0,
                        max=1.0,
                    )
                    defect_single_gap_centerline_penalty = torch.where(
                        single_gap_centerline_active,
                        single_gap_centerline_ratio
                        * single_gap_centerline_proximity
                        * single_gap_centerline_penalty_scale,
                        torch.zeros_like(curr_goal_dist),
                    )
                double_gap_reached = torch.logical_and(
                    active_gap_progress,
                    torch.logical_and(dwb_gap_dx <= narrow_x_window, dwb_gap_dy <= narrow_lat_tol),
                )
                single_gap_reached = torch.logical_and(
                    active_gap_progress,
                    torch.logical_and(
                        single_gap_mask,
                        torch.logical_and(
                            dwb_gap_dx <= narrow_x_window,
                            (dwb_gap_dy - single_target_abs_y).abs() <= single_target_tol,
                        ),
                    ),
                )
                gap_reached_now = torch.logical_or(
                    torch.logical_and(double_gap_reached, dwb_mask),
                    single_gap_reached,
                )
                if bool(torch.any(gap_reached_now).item()):
                    self._defect_dwb_gap_reached[gap_reached_now] = True
                gap_cleared_now = torch.logical_and(
                    torch.logical_and(dwb_like_mask, self._defect_dwb_gap_reached),
                    torch.logical_and(
                        ~self._defect_dwb_gap_clear_bonus_given,
                        local_xy[:, 0] > clear_x_threshold,
                    ),
                )
                if bool(torch.any(gap_cleared_now).item()):
                    self._defect_dwb_gap_clear_bonus_given[gap_cleared_now] = True
                defect_dwb_gap_clear_reward = (
                    gap_cleared_now.float() * float(getattr(self.cfg, "defect_dwb_gap_clear_bonus", 0.0))
                )
                post_gap_active = torch.logical_and(
                    torch.logical_and(dwb_like_mask, self._defect_dwb_gap_clear_bonus_given),
                    curr_goal_dist > self.cfg.goal_reach_threshold,
                )
                post_gap_lateral_align_scale = float(
                    getattr(self.cfg, "defect_dwb_post_gap_lateral_align_reward_scale", 0.0)
                )
                if post_gap_lateral_align_scale > 0.0:
                    post_gap_lateral_deadband = max(
                        0.0,
                        float(getattr(self.cfg, "defect_dwb_post_gap_lateral_align_deadband", 0.0)),
                    )
                    post_gap_lateral_x_margin = max(
                        0.0,
                        float(getattr(self.cfg, "defect_dwb_post_gap_lateral_align_x_margin", 0.0)),
                    )
                    post_gap_lateral_active = torch.logical_and(
                        post_gap_active,
                        torch.logical_and(
                            single_gap_mask,
                            local_xy[:, 0] > (clear_x_threshold + post_gap_lateral_x_margin),
                        ),
                    )
                    prev_post_gap_goal_abs_y = torch.clamp(
                        self._defect_dwb_prev_post_gap_goal_abs_y - post_gap_lateral_deadband,
                        min=0.0,
                    )
                    curr_post_gap_goal_abs_y = torch.clamp(goal_abs_y_b - post_gap_lateral_deadband, min=0.0)
                    defect_dwb_post_gap_lateral_align_reward = torch.where(
                        torch.logical_and(post_gap_lateral_active, ~gap_cleared_now),
                        (prev_post_gap_goal_abs_y - curr_post_gap_goal_abs_y) * post_gap_lateral_align_scale,
                        torch.zeros_like(curr_goal_dist),
                    )
                post_gap_forward_scale = float(getattr(self.cfg, "defect_dwb_post_gap_forward_reward_scale", 0.0))
                if post_gap_forward_scale > 0.0:
                    defect_dwb_post_gap_reward = defect_dwb_post_gap_reward + torch.where(
                        post_gap_active,
                        torch.clamp(forward_speed, min=0.0) * post_gap_forward_scale,
                        torch.zeros_like(curr_goal_dist),
                    )
                post_gap_progress_scale = float(getattr(self.cfg, "defect_dwb_post_gap_progress_reward_scale", 0.0))
                if post_gap_progress_scale > 0.0:
                    defect_dwb_post_gap_reward = defect_dwb_post_gap_reward + torch.where(
                        post_gap_active,
                        torch.clamp(progress_delta, min=0.0) * post_gap_progress_scale,
                        torch.zeros_like(curr_goal_dist),
                    )
                side_commit_scale = float(getattr(self.cfg, "defect_dwb_side_commit_reward_scale", 0.0))
                side_switch_penalty_scale = float(getattr(self.cfg, "defect_dwb_side_switch_penalty_scale", 0.0))
                side_commit_deadzone = max(0.0, float(getattr(self.cfg, "defect_dwb_side_commit_deadzone", 0.0)))
                if side_commit_scale > 0.0 or side_switch_penalty_scale > 0.0:
                    commit_active = torch.logical_and(
                        active_gap_progress,
                        local_xy[:, 0] <= clear_x_threshold,
                    )
                    prev_gap_signed_y = self._defect_dwb_prev_gap_signed_y
                    curr_gap_abs_y = dwb_gap_rel_y.abs()
                    prev_gap_abs_y = prev_gap_signed_y.abs()
                    committed_now = torch.logical_and(
                        curr_gap_abs_y > side_commit_deadzone,
                        prev_gap_abs_y > side_commit_deadzone,
                    )
                    same_side_commit = torch.logical_and(committed_now, (dwb_gap_rel_y * prev_gap_signed_y) > 0.0)
                    if side_commit_scale > 0.0:
                        defect_dwb_side_commit_reward = defect_dwb_side_commit_reward + torch.where(
                            torch.logical_and(
                                torch.logical_and(commit_active, same_side_commit),
                                progress_delta > 0.0,
                            ),
                            torch.clamp(curr_gap_abs_y - prev_gap_abs_y, min=0.0) * side_commit_scale,
                            torch.zeros_like(curr_goal_dist),
                        )
                    if side_switch_penalty_scale > 0.0:
                        side_switch = torch.logical_and(commit_active, torch.logical_and(committed_now, (dwb_gap_rel_y * prev_gap_signed_y) < 0.0))
                        defect_dwb_side_commit_reward = defect_dwb_side_commit_reward - (
                            side_switch.float() * side_switch_penalty_scale
                        )
                self._defect_dwb_prev_gap_dist = torch.where(
                    dwb_like_mask,
                    dwb_progress_metric,
                    self._defect_dwb_prev_gap_dist,
                )
                self._defect_dwb_prev_gap_abs_y = torch.where(
                    dwb_like_mask,
                    dwb_alignment_metric,
                    self._defect_dwb_prev_gap_abs_y,
                )
                self._defect_dwb_prev_gap_signed_y = torch.where(
                    dwb_like_mask,
                    dwb_gap_rel_y,
                    self._defect_dwb_prev_gap_signed_y,
                )
                self._defect_dwb_prev_post_gap_goal_abs_y = torch.where(
                    torch.logical_and(dwb_like_mask, self._defect_dwb_gap_clear_bonus_given),
                    goal_abs_y_b,
                    self._defect_dwb_prev_post_gap_goal_abs_y,
                )
            single_narrow_inside = torch.logical_and(
                single_gap_mask,
                torch.logical_and(
                    dwb_gap_dx <= narrow_x_window,
                    (dwb_gap_dy - single_target_abs_y).abs() <= single_target_tol,
                ),
            )
            narrow_inside = torch.logical_or(
                torch.logical_or(
                    torch.logical_and(
                        dwb_mask,
                        torch.logical_and(
                            dwb_gap_dx <= narrow_x_window,
                            dwb_gap_dy <= narrow_lat_tol,
                        ),
                    ),
                    single_narrow_inside,
                ),
                torch.logical_or(
                    torch.logical_and(
                        door_mask,
                        torch.logical_and(
                            local_xy[:, 0].abs() <= narrow_x_window,
                            local_xy[:, 1].abs() <= narrow_lat_tol,
                        ),
                    ),
                    torch.logical_and(
                        u_mask,
                        torch.logical_and(
                            local_xy[:, 0].abs() <= narrow_x_window,
                            local_xy[:, 1].abs() <= narrow_lat_tol,
                        ),
                    ),
                ),
            )
            narrow_enter = torch.logical_and(narrow_inside, ~self._defect_narrow_zone_prev_inside)
            defect_narrow_passage_reward = (
                narrow_enter.float() * float(getattr(self.cfg, "defect_narrow_passage_bonus", 2.0))
            )
            self._defect_narrow_zone_prev_inside[:] = narrow_inside

            corner_inner_radius = max(0.05, float(getattr(self.cfg, "defect_corner_inner_radius", 0.5)))
            corner_corridor_width = max(0.1, float(getattr(self.cfg, "defect_corner_corridor_width", 0.8)))
            corner_center_radius = corner_inner_radius + 0.5 * corner_corridor_width
            corner_post_length = max(0.2, float(getattr(self.cfg, "defect_corner_straight_post_length", 2.2)))
            corner_mid_x = corner_inner_radius + 0.5 * corner_corridor_width
            in_corner_turn = torch.logical_and(
                mppi_mask,
                torch.logical_and(local_xy[:, 0] > -0.05, local_xy[:, 1] > 0.05),
            )
            self._defect_corner_seen_turn = torch.logical_or(self._defect_corner_seen_turn, in_corner_turn)
            robot_quat_wxyz = self.robot.data.root_link_quat_w
            yaw_rad = torch.atan2(
                2.0 * (robot_quat_wxyz[:, 0] * robot_quat_wxyz[:, 3] + robot_quat_wxyz[:, 1] * robot_quat_wxyz[:, 2]),
                1.0 - 2.0 * (robot_quat_wxyz[:, 2].square() + robot_quat_wxyz[:, 3].square()),
            )
            turn_heading_deg_threshold = float(getattr(self.cfg, "defect_corner_turn_heading_deg_threshold", 80.0))
            corner_heading_turned = torch.abs(yaw_rad) > math.radians(turn_heading_deg_threshold)
            corner_lane_gate = torch.logical_and(
                local_xy[:, 1] > (corner_center_radius + 0.45 * corner_post_length),
                torch.logical_and(
                    local_xy[:, 0] > (corner_inner_radius + 0.03),
                    local_xy[:, 0] < (corner_inner_radius + corner_corridor_width - 0.03),
                ),
            )
            corner_mid_passed = local_xy[:, 0] > corner_mid_x
            turn_completion_gate = torch.logical_or(
                corner_lane_gate,
                torch.logical_or(corner_mid_passed, corner_heading_turned),
            )
            turn_completed = torch.logical_and(
                torch.logical_and(mppi_mask, self._defect_corner_seen_turn),
                torch.logical_and(
                    ~self._defect_turn_bonus_given,
                    turn_completion_gate,
                ),
            )
            if bool(torch.any(turn_completed).item()):
                self._defect_turn_bonus_given[turn_completed] = True
            defect_turn_completion_reward = (
                turn_completed.float() * float(getattr(self.cfg, "defect_turn_completion_bonus", 5.0))
            )
            corner_mid_reward_scale = float(getattr(self.cfg, "defect_corner_mid_progress_reward", 0.0))
            if corner_mid_reward_scale > 0.0:
                corner_mid_x_min = float(getattr(self.cfg, "defect_corner_mid_progress_x_min", 1.0))
                corner_mid_x_max = max(
                    corner_mid_x_min,
                    float(getattr(self.cfg, "defect_corner_mid_progress_x_max", corner_mid_x_min)),
                )
                in_corner_mid_band = torch.logical_and(
                    mppi_mask,
                    torch.logical_and(local_xy[:, 0] >= corner_mid_x_min, local_xy[:, 0] <= corner_mid_x_max),
                )
                advancing_corner_mid = torch.logical_and(
                    in_corner_mid_band,
                    torch.logical_and(progress_delta > 0.0, forward_speed > 0.02),
                )
                defect_corner_mid_progress_reward = advancing_corner_mid.float() * corner_mid_reward_scale

            u_depth = max(0.2, float(getattr(self.cfg, "defect_u_depth", 2.7)))
            u_half_inner = 0.5 * max(0.2, float(getattr(self.cfg, "defect_u_inner_width", 2.0)))
            u_half_opening = 0.5 * max(0.2, float(getattr(self.cfg, "defect_u_opening_width", 0.6)))
            inside_u_region = torch.logical_and(
                u_mask,
                torch.logical_and(
                    torch.logical_and(local_xy[:, 0] >= -u_depth, local_xy[:, 0] <= 0.0),
                    local_xy[:, 1].abs() <= u_half_inner,
                ),
            )
            escaped_u_shape = torch.logical_and(
                torch.logical_and(
                    u_mask,
                    torch.logical_and(self._defect_u_was_inside, ~self._defect_escape_bonus_given),
                ),
                torch.logical_and(
                    local_xy[:, 0] > 0.05,
                    torch.logical_and(~inside_u_region, local_xy[:, 1].abs() <= (u_half_opening + 0.15)),
                ),
            )
            if bool(torch.any(escaped_u_shape).item()):
                self._defect_escape_bonus_given[escaped_u_shape] = True
            defect_u_escape_reward = escaped_u_shape.float() * float(getattr(self.cfg, "defect_u_escape_bonus", 20.0))
            self._defect_u_was_inside = torch.where(u_mask, inside_u_region, self._defect_u_was_inside)

            explore_scale = float(getattr(self.cfg, "defect_u_exploration_reward_scale", 0.01))
            if explore_scale > 0.0 and bool(torch.any(u_mask).item()):
                u_env_ids = torch.nonzero(u_mask, as_tuple=False).squeeze(-1)
                new_cells = self._mark_u_explore_cells(u_env_ids, local_xy[u_env_ids])
                defect_exploration_reward[u_env_ids] = new_cells.float() * explore_scale
            backward_bonus = float(getattr(self.cfg, "defect_u_backward_encouragement", 0.0))
            if backward_bonus > 0.0:
                backward_speed_threshold = float(getattr(self.cfg, "defect_u_backward_speed_threshold", -0.1))
                forward_speed_signed = self.robot.data.root_com_lin_vel_b[:, 0]
                backward_active = torch.logical_and(
                    u_mask,
                    torch.logical_and(inside_u_region, forward_speed_signed < backward_speed_threshold),
                )
                defect_u_backward_reward = torch.where(
                    backward_active,
                    torch.full_like(curr_goal_dist, backward_bonus),
                    torch.zeros_like(curr_goal_dist),
                )

        success = curr_goal_dist < self.cfg.goal_reach_threshold
        collision_lidar, _ = self._compute_collision_lidar(lidar_ranges=lidar_ranges, min_lidar=min_lidar)
        collision_contact = contact_force > float(getattr(self.cfg, "contact_collision_force_threshold", 0.0))
        collision = torch.logical_or(collision_lidar, collision_contact)
        contact_force_penalty = (
            torch.clamp(
                contact_force / max(1.0e-6, float(getattr(self.cfg, "contact_collision_force_threshold", 1.0))),
                min=0.0,
                max=1.0,
            )
            * float(getattr(self.cfg, "contact_force_penalty_scale", 0.0))
        )
        success_reward = success.float() * self.cfg.success_bonus
        collision_penalty = collision.float() * self.cfg.collision_penalty
        yaw_rate_abs_mean = yaw_rate_abs.mean()
        yaw_rate_abs_p95 = torch.quantile(yaw_rate_abs, 0.95)
        filtered_omega_abs_mean = self.filtered_omega_cmd.abs().mean()
        yaw_sign_flip_rate = yaw_sign_flip.float().mean()
        yaw_persistence_reward_mean = persistence_reward.mean()
        yaw_flip_penalty_mean = flip_penalty.mean()
        direction_bias_mean = direction_bias_reward.mean()
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        success_rate = success.float().mean()
        collision_rate = collision.float().mean()
        timeout_rate = timeout.float().mean()
        self._yaw_flip_counter = self._yaw_flip_counter + yaw_sign_flip.float()
        self._yaw_flip_step_counter = self._yaw_flip_step_counter + 1.0
        yaw_sign_flip_rate_episode = torch.where(
            self._yaw_flip_step_counter > 0.0,
            self._yaw_flip_counter / self._yaw_flip_step_counter,
            torch.zeros_like(self._yaw_flip_counter),
        ).mean()
        self.extras["log"] = {
            "yaw_rate_abs_mean": yaw_rate_abs_mean.detach(),
            "yaw_rate_abs_p95": yaw_rate_abs_p95.detach(),
            "filtered_omega_abs_mean": filtered_omega_abs_mean.detach(),
            "yaw_sign_flip_rate": yaw_sign_flip_rate_episode.detach(),
            "yaw_sign_flip_rate_step": yaw_sign_flip_rate.detach(),
            "yaw_persistence_reward_mean": yaw_persistence_reward_mean.detach(),
            "yaw_flip_penalty_mean": yaw_flip_penalty_mean.detach(),
            "direction_bias_mean": direction_bias_mean.detach(),
            "success_rate": success_rate.detach(),
            "collision_rate": collision_rate.detach(),
            "timeout_rate": timeout_rate.detach(),
        }

        self.prev_goal_dist = curr_goal_dist.clone()
        self.prev_actions = self.actions.clone()
        self.prev_min_lidar = min_lidar.clone()
        self.prev_yaw_rate = yaw_rate_b.clone()
        self._prev_yaw_sign = effective_yaw_sign.clone()
        self._last_step_turn_cmd_reward[:] = turn_cmd_reward
        self._last_step_turn_penalty[:] = turn_penalty
        self._last_step_stall[:] = stall
        self._last_step_action_saturated[:] = action_saturated
        self._last_step_progress_delta[:] = progress_delta

        total_reward = (
            progress_reward
            + heading_reward
            + forward_goal_reward
            + speed_surge_reward
            + speed_deficit_penalty
            + turn_cmd_reward
            + clearance_reward
            + smooth_action_reward
            + persistence_reward
            + flip_penalty
            + direction_bias_reward
            + turn_penalty
            + time_penalty
            + stall_penalty
            + danger_speed_penalty
            + defect_dwb_clearance_penalty
            + action_saturation_penalty
            + return_no_progress_penalty
            + contact_force_penalty
            + subgoal_progress_reward
            + subgoal_bonus_reward
            + bypass_reward
            + defect_lateral_penalty
            + defect_narrow_passage_reward
            + defect_dwb_gap_progress_reward
            + defect_dwb_gap_alignment_reward
            + defect_dwb_post_gap_reward
            + defect_dwb_post_gap_lateral_align_reward
            + defect_dwb_side_commit_reward
            + defect_dwb_gap_clear_reward
            + defect_single_gap_centerline_penalty
            + defect_turn_completion_reward
            + defect_corner_mid_progress_reward
            + defect_u_escape_reward
            + defect_exploration_reward
            + defect_u_backward_reward
            + success_reward
            + collision_penalty
        )
        self.episode_return += total_reward
        self._update_step_debug(
            total_reward=total_reward,
            progress_reward=progress_reward,
            heading_reward=heading_reward,
            forward_goal_reward=forward_goal_reward,
            speed_surge_reward=speed_surge_reward,
            speed_deficit_penalty=speed_deficit_penalty,
            clearance_reward=clearance_reward,
            smooth_action_reward=smooth_action_reward,
            turn_penalty=turn_penalty,
            time_penalty=time_penalty,
            stall_penalty=stall_penalty,
            danger_speed_penalty=danger_speed_penalty,
            defect_dwb_clearance_penalty=defect_dwb_clearance_penalty,
            action_saturation_penalty=action_saturation_penalty,
            defect_single_gap_centerline_penalty=defect_single_gap_centerline_penalty,
            defect_dwb_post_gap_lateral_align_reward=defect_dwb_post_gap_lateral_align_reward,
            success_reward=success_reward,
            collision_penalty=collision_penalty,
            curr_goal_dist=curr_goal_dist,
            min_lidar=min_lidar,
            speed_cap=self._compute_speed_cap(min_lidar),
            action_eff_abs_mean=action_eff_abs_mean,
            forward_speed=forward_speed,
            yaw_rate_abs_mean=yaw_rate_abs_mean,
            yaw_rate_abs_p95=yaw_rate_abs_p95,
            filtered_omega_abs_mean=filtered_omega_abs_mean,
            yaw_sign_flip_rate=yaw_sign_flip_rate,
            success=success,
            collision=collision,
            collision_lidar=collision_lidar,
            collision_contact=collision_contact,
            stall=stall,
            action_saturated=action_saturated,
        )
        return total_reward.unsqueeze(-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        curr_goal_dist = torch.linalg.norm(self.goal_pos_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        lidar_ranges = self._compute_lidar_ranges()
        self.latest_lidar_ranges = lidar_ranges
        min_lidar = torch.min(lidar_ranges, dim=1).values
        progress_delta = self.prev_goal_dist - curr_goal_dist
        forward_speed = self.robot.data.root_com_lin_vel_b[:, 0]
        dwb_no_progress = self._compute_defect_dwb_no_progress_mask(
            curr_goal_dist=curr_goal_dist,
            min_lidar=min_lidar,
            forward_speed=forward_speed,
            progress_delta=progress_delta,
        )
        dwb_no_progress_terminate_steps = int(getattr(self.cfg, "defect_dwb_no_progress_terminate_steps", 0))
        if dwb_no_progress_terminate_steps > 0:
            self._defect_dwb_no_progress_steps = torch.where(
                dwb_no_progress,
                self._defect_dwb_no_progress_steps + 1,
                torch.zeros_like(self._defect_dwb_no_progress_steps),
            )
            time_out = torch.logical_or(time_out, self._defect_dwb_no_progress_steps >= dwb_no_progress_terminate_steps)
        else:
            self._defect_dwb_no_progress_steps[:] = 0
        contact_force = self._compute_obstacle_contact_force()
        self.latest_contact_force = contact_force
        success = curr_goal_dist < self.cfg.goal_reach_threshold
        collision_lidar, _ = self._compute_collision_lidar(lidar_ranges=lidar_ranges, min_lidar=min_lidar)
        collision_contact = contact_force > float(getattr(self.cfg, "contact_collision_force_threshold", 0.0))
        collision = torch.logical_or(collision_lidar, collision_contact)
        terminated = torch.logical_or(success, collision)
        self._last_done_success[:] = success
        self._last_done_collision[:] = collision
        self._last_done_collision_lidar[:] = collision_lidar
        self._last_done_collision_contact[:] = collision_contact
        self._last_done_timeout[:] = time_out
        self._last_done_goal_dist[:] = curr_goal_dist
        self._last_done_min_lidar[:] = min_lidar
        self._last_done_planner_state[:] = self.planner_state_index
        self._last_done_fixed_scene_index[:] = self.fixed_scene_active_index
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        env_ids_t = self._env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return
        # collect terminal stats before base reset clears buffers
        self._update_terminal_metrics(env_ids_t)
        super()._reset_idx(env_ids_t)

        # reset robot pose
        default_root_state = self.robot.data.default_root_state[env_ids_t]
        default_root_state[:, :3] += self.scene.env_origins[env_ids_t]
        defect_scene_reset = self._use_defect_scene_reset()
        fixed_scene_reset = self._use_fixed_scene_reset()
        if defect_scene_reset:
            self._apply_defect_dwb_curriculum()
            self._apply_defect_ushape_curriculum()
            self._apply_defect_mppi_curriculum()
        if defect_scene_reset:
            self._reset_defect_scene(env_ids_t, default_root_state)
        elif fixed_scene_reset:
            self._reset_fixed_scene(env_ids_t, default_root_state)
        elif self._use_mixed_planner_state_training():
            self.fixed_scene_active_index[env_ids_t] = -1
            # Sample planner context together with spawn geometry so each planner_state sees a consistent regime.
            sampled_states = self._sample_planner_state_indices(env_ids_t)
            env_origins_xy = self.scene.env_origins[env_ids_t, :2]
            default_root_state[:, 3:7] = 0.0
            yaw = torch.zeros(env_ids_t.numel(), device=self.device)

            linear_state_mask = sampled_states != self.STATE_JUNCTION
            if torch.any(linear_state_mask):
                self._set_linear_layout(env_ids_t[linear_state_mask])
                linear_spawn_min = float(getattr(self.cfg, "linear_state_spawn_distance_min", 1.8))
                linear_spawn_max = max(linear_spawn_min, float(getattr(self.cfg, "linear_state_spawn_distance_max", linear_spawn_min)))
                linear_spawn_dist = (
                    torch.rand(env_ids_t.numel(), device=self.device) * max(1e-6, linear_spawn_max - linear_spawn_min)
                    + linear_spawn_min
                )
                linear_lateral = (torch.rand(env_ids_t.numel(), device=self.device) * 2.0 - 1.0) * float(
                    getattr(self.cfg, "linear_state_spawn_lateral_jitter", 0.08)
                )

                explore_mask = sampled_states == self.STATE_EXPLORE
                if torch.any(explore_mask):
                    explore_ids = env_ids_t[explore_mask]
                    self._set_planner_state(explore_ids, "explore")
                    default_root_state[explore_mask, 0] = env_origins_xy[explore_mask, 0] - linear_spawn_dist[explore_mask]
                    default_root_state[explore_mask, 1] = env_origins_xy[explore_mask, 1] + linear_lateral[explore_mask]

                fallback_mask = sampled_states == self.STATE_FALLBACK
                if torch.any(fallback_mask):
                    fallback_ids = env_ids_t[fallback_mask]
                    self._set_planner_state(fallback_ids, "fallback")
                    default_root_state[fallback_mask, 0] = env_origins_xy[fallback_mask, 0] + linear_spawn_dist[fallback_mask]
                    default_root_state[fallback_mask, 1] = env_origins_xy[fallback_mask, 1] + linear_lateral[fallback_mask]

                return_mask = sampled_states == self.STATE_RETURN
                if torch.any(return_mask):
                    return_ids = env_ids_t[return_mask]
                    self._set_planner_state(return_ids, "return")
                    default_root_state[return_mask, 0] = env_origins_xy[return_mask, 0] + linear_spawn_dist[return_mask]
                    default_root_state[return_mask, 1] = env_origins_xy[return_mask, 1] + linear_lateral[return_mask]
                    yaw[return_mask] = math.pi

            junction_mask = sampled_states == self.STATE_JUNCTION
            if torch.any(junction_mask):
                junction_ids = env_ids_t[junction_mask]
                self._sample_junction_layout(junction_ids)
                self._set_planner_state(junction_ids, "junction")
                junction_spawn_min = float(getattr(self.cfg, "junction_state_spawn_distance_min", 0.8))
                junction_spawn_max = max(
                    junction_spawn_min,
                    float(getattr(self.cfg, "junction_state_spawn_distance_max", junction_spawn_min)),
                )
                junction_spawn_dist = (
                    torch.rand(env_ids_t.numel(), device=self.device)
                    * max(1e-6, junction_spawn_max - junction_spawn_min)
                    + junction_spawn_min
                )
                junction_lateral = (torch.rand(env_ids_t.numel(), device=self.device) * 2.0 - 1.0) * float(
                    self.cfg.junction_robot_spawn_lateral_jitter
                )
                default_root_state[junction_mask, 0] = env_origins_xy[junction_mask, 0] - junction_spawn_dist[junction_mask]
                default_root_state[junction_mask, 1] = env_origins_xy[junction_mask, 1] + junction_lateral[junction_mask]

            default_root_state[:, 3] = torch.cos(0.5 * yaw)
            default_root_state[:, 6] = torch.sin(0.5 * yaw)
        elif bool(getattr(self.cfg, "junction_enable", False)):
            self.fixed_scene_active_index[env_ids_t] = -1
            self._sample_junction_layout(env_ids_t)
            self._set_planner_state(env_ids_t, "junction")
            half_l = float(self.cfg.junction_half_length)
            spawn_min = float(self.cfg.junction_robot_spawn_offset_min)
            spawn_max = max(spawn_min, float(self.cfg.junction_robot_spawn_offset_max))
            spawn_offset = torch.rand(env_ids_t.numel(), device=self.device) * max(1e-6, spawn_max - spawn_min) + spawn_min
            lateral = (
                torch.rand(env_ids_t.numel(), device=self.device) * 2.0 - 1.0
            ) * float(self.cfg.junction_robot_spawn_lateral_jitter)
            default_root_state[:, 0] = self.scene.env_origins[env_ids_t, 0] - half_l + spawn_offset
            default_root_state[:, 1] = self.scene.env_origins[env_ids_t, 1] + lateral
        else:
            self.fixed_scene_active_index[env_ids_t] = -1
            self.turn_cmd_index[env_ids_t] = self.TURN_STRAIGHT
            self.turn_cmd_onehot[env_ids_t] = 0.0
            self.turn_cmd_dir_b[env_ids_t, 0] = 1.0
            self.turn_cmd_dir_b[env_ids_t, 1] = 0.0
            self._set_planner_state(env_ids_t, "explore")
        self.robot.write_root_state_to_sim(default_root_state, env_ids_t)

        robot_reset_pos_xy = default_root_state[:, :2]
        if not fixed_scene_reset and not defect_scene_reset:
            self._spawn_goal_positions(env_ids_t, robot_pos_xy=robot_reset_pos_xy)
            self._spawn_obstacle_positions(
                env_ids_t,
                robot_pos_xy=robot_reset_pos_xy,
                goal_pos_xy=self.goal_pos_w[env_ids_t, :2],
            )
        # synchronize sampled obstacle positions to physical rigid bodies
        obstacle_rigid_collection = getattr(self, "obstacle_rigid_collection", None)
        if obstacle_rigid_collection is not None:
            obstacle_state = obstacle_rigid_collection.data.default_object_state[env_ids_t].clone()
            obstacle_state[..., :3] = self.obstacle_pos_w[env_ids_t]
            obstacle_state[..., 3] = 1.0
            obstacle_state[..., 4:7] = 0.0
            obstacle_state[..., 7:] = 0.0
            obstacle_rigid_collection.write_object_state_to_sim(obstacle_state, env_ids=env_ids_t)
            obstacle_rigid_collection.reset(env_ids=env_ids_t)

        diff_goal = self.goal_pos_w[env_ids_t, :2] - robot_reset_pos_xy
        self.prev_goal_dist[env_ids_t] = torch.linalg.norm(diff_goal, dim=1)
        if self._subgoal_reward_enable:
            env_origins_xy = self.scene.env_origins[env_ids_t, :2]
            self.subgoal_pos_w[env_ids_t, 0] = env_origins_xy[:, 0] + self._subgoal_local_xy[0]
            self.subgoal_pos_w[env_ids_t, 1] = env_origins_xy[:, 1] + self._subgoal_local_xy[1]
            self.subgoal_pos_w[env_ids_t, 2] = float(self.cfg.goal_radius)
            subgoal_delta = self.subgoal_pos_w[env_ids_t, :2] - robot_reset_pos_xy
            self.prev_subgoal_dist[env_ids_t] = torch.linalg.norm(subgoal_delta, dim=1)
            subgoal_scene_index = int(getattr(self.cfg, "subgoal_scene_index", -1))
            active_subgoal_scene = self.fixed_scene_active_index[env_ids_t] == subgoal_scene_index
            # Mark non-target scenes as already reached so shaping is truly scene-gated.
            self._subgoal_reached[env_ids_t] = ~active_subgoal_scene
        else:
            self.subgoal_pos_w[env_ids_t] = 0.0
            self.prev_subgoal_dist[env_ids_t] = 0.0
            self._subgoal_reached[env_ids_t] = True
        self.prev_actions[env_ids_t] = 0.0
        self.filtered_v_cmd[env_ids_t] = 0.0
        self.filtered_omega_cmd[env_ids_t] = 0.0
        self.prev_wheel_targets[env_ids_t] = 0.0
        self.prev_min_lidar[env_ids_t] = self.cfg.lidar_max_range
        self._defect_dwb_gap_commit_latched[env_ids_t] = False
        self.prev_yaw_rate[env_ids_t] = 0.0
        self.return_no_progress_steps[env_ids_t] = 0
        self._defect_dwb_no_progress_steps[env_ids_t] = 0
        self._defect_narrow_zone_prev_inside[env_ids_t] = False
        self._defect_dwb_gap_reached[env_ids_t] = False
        self._defect_dwb_gap_clear_bonus_given[env_ids_t] = False
        self._defect_dwb_prev_gap_abs_y[env_ids_t] = 0.0
        self._defect_dwb_prev_gap_signed_y[env_ids_t] = 0.0
        self._defect_dwb_prev_post_gap_goal_abs_y[env_ids_t] = 0.0
        self._defect_turn_bonus_given[env_ids_t] = False
        self._defect_corner_seen_turn[env_ids_t] = False
        self._defect_escape_bonus_given[env_ids_t] = False
        self._defect_u_was_inside[env_ids_t] = False
        self._defect_u_explore_visited[env_ids_t] = False
        if defect_scene_reset:
            u_scene_mask = self.fixed_scene_active_index[env_ids_t] == self.DEFECT_U_SHAPE_TRAP
            if bool(torch.any(u_scene_mask).item()):
                u_env_ids = env_ids_t[u_scene_mask]
                u_local_xy = robot_reset_pos_xy[u_scene_mask] - self.scene.env_origins[u_env_ids, :2]
                u_depth = max(0.2, float(getattr(self.cfg, "defect_u_depth", 2.7)))
                u_half_inner = 0.5 * max(0.2, float(getattr(self.cfg, "defect_u_inner_width", 2.0)))
                inside_u = torch.logical_and(
                    torch.logical_and(u_local_xy[:, 0] >= -u_depth, u_local_xy[:, 0] <= 0.0),
                    u_local_xy[:, 1].abs() <= u_half_inner,
                )
                self._defect_u_was_inside[u_env_ids] = inside_u
                self._mark_u_explore_cells(u_env_ids, u_local_xy)
        if defect_scene_reset:
            dwb_scene_mask = self._get_dwb_like_scene_mask(self.fixed_scene_active_index[env_ids_t])
            if bool(torch.any(dwb_scene_mask).item()):
                dwb_env_ids = env_ids_t[dwb_scene_mask]
                dwb_local_xy = robot_reset_pos_xy[dwb_scene_mask] - self.scene.env_origins[dwb_env_ids, :2]
                single_gap_mask = self.fixed_scene_active_index[dwb_env_ids] == self.DEFECT_SINGLE_OBSTACLE_SYMMETRIC_GAP
                gap_dx = self._defect_dwb_gap_center_local_x[dwb_env_ids] - dwb_local_xy[:, 0]
                gap_dy = self._defect_dwb_gap_center_local_y[dwb_env_ids] - dwb_local_xy[:, 1]
                single_target_abs_y = self._get_single_obstacle_symmetric_gap_target_abs_y()
                progress_metric = torch.where(
                    single_gap_mask,
                    torch.clamp(gap_dx, min=0.0),
                    torch.sqrt(gap_dx.square() + gap_dy.square()),
                )
                align_metric = torch.where(
                    single_gap_mask,
                    (gap_dy.abs() - single_target_abs_y).abs(),
                    gap_dy.abs(),
                )
                self._defect_dwb_prev_gap_dist[dwb_env_ids] = progress_metric
                self._defect_dwb_prev_gap_abs_y[dwb_env_ids] = align_metric
                self._defect_dwb_prev_gap_signed_y[dwb_env_ids] = -gap_dy
        self.latest_lidar_ranges[env_ids_t] = self.cfg.lidar_max_range
        lidar_obs_reset_value = 1.0 if bool(getattr(self.cfg, "use_lidar_normalization", False)) else float(self.cfg.lidar_max_range)
        self.lidar_frame_buffer[env_ids_t] = lidar_obs_reset_value
        if hasattr(self, "latest_contact_force"):
            self.latest_contact_force[env_ids_t] = 0.0
        self._prev_yaw_sign[env_ids_t] = 0.0
        self._yaw_flip_counter[env_ids_t] = 0.0
        self._yaw_flip_step_counter[env_ids_t] = 0.0
        self.episode_return[env_ids_t] = 0.0
        self._update_reset_debug(env_ids_t, robot_reset_pos_xy, default_root_state[:, 3:7])

        self._visualize_goal_obstacles()
        self._visualize_junction_walls()
