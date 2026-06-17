#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run a trained LiDAR-nav policy in ROS2/Gazebo.

Subscribes:
- /scan (sensor_msgs/LaserScan)
- /odom (nav_msgs/Odometry)
- /goal_pose (geometry_msgs/PoseStamped)

Publishes:
- /cmd_vel (geometry_msgs/Twist)

The script reproduces one of two training observation layouts:
  42D: [goal_dir_x, goal_dir_y, goal_dist_norm, forward_speed, yaw_rate, min_lidar_norm, lidar_36_norm]
  45D: [goal_dir_x, goal_dir_y, goal_dist_norm, forward_speed, yaw_rate, min_lidar_norm, turn_cmd_one_hot(3), lidar_36_norm]
and executes deterministic policy actions from a skrl checkpoint (best_agent.pt).
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import rclpy
    from rclpy.duration import Duration
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Float32, String
    from tf2_ros import Buffer, TransformException, TransformListener
except ImportError as exc:
    raise RuntimeError(
        "ROS2 Python packages are not available. Please source ROS2 Humble before running this script."
    ) from exc


class SharedPolicyModel(nn.Module):
    """Policy network matching skrl_lidar_nav_ppo_cfg.yaml shared model."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net_container = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
        )
        self.policy_layer = nn.Linear(128, action_dim)
        # kept for checkpoint compatibility (unused in deterministic inference)
        self.log_std_parameter = nn.Parameter(torch.zeros(action_dim), requires_grad=True)
        # kept for checkpoint compatibility (unused in policy inference)
        self.value_layer = nn.Linear(128, 1)

    def forward_policy(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.net_container(obs)
        return self.policy_layer(features)


def quat_xyzw_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Convert quaternion (xyzw) to yaw."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def resample_lidar(scan: LaserScan, target_angles: np.ndarray, lidar_min: float, lidar_max: float) -> np.ndarray:
    """Resample a LaserScan onto target angles (radians), clipping to [lidar_min, lidar_max]."""
    if scan.angle_increment == 0.0:
        return np.full_like(target_angles, fill_value=lidar_max, dtype=np.float32)

    ranges = np.asarray(scan.ranges, dtype=np.float32)
    # sanitize NaN/inf early
    invalid = ~np.isfinite(ranges)
    if np.any(invalid):
        ranges = ranges.copy()
        ranges[invalid] = lidar_max

    src_angles = scan.angle_min + np.arange(len(ranges), dtype=np.float32) * scan.angle_increment
    sampled = np.interp(target_angles, src_angles, ranges, left=lidar_max, right=lidar_max)
    sampled = np.clip(sampled, lidar_min, lidar_max)
    return sampled.astype(np.float32)


class GazeboPolicyRunner(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("gazebo_policy_runner")

        self.args = args
        self.device = torch.device(args.device)
        self.base_obs_dim = 6 + args.num_beams
        self.obs_dim = self.base_obs_dim
        self.action_dim = 2
        self.turn_cmd_one_hot = (
            self._build_turn_cmd_one_hot(args.turn_cmd)
            if args.turn_cmd not in {"auto", "topic"}
            else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        )
        self.turn_cmd_current = "straight"
        self.speed_cap = float("inf") if args.default_speed_cap <= 0.0 else float(args.default_speed_cap)
        self.use_turn_cmd = False

        self.model, self.running_mean, self.running_variance = self._load_checkpoint(args.checkpoint)
        self.model.to(self.device)
        self.model.eval()

        # Pre-computed beam angles for policy input
        lidar_fov_rad = math.radians(args.lidar_fov_deg)
        self.target_beam_angles = np.linspace(
            -0.5 * lidar_fov_rad,
            0.5 * lidar_fov_rad,
            args.num_beams,
            dtype=np.float32,
        )

        self.scan_msg: LaserScan | None = None
        self.odom_msg: Odometry | None = None
        self.goal_msg: PoseStamped | None = None
        self.last_scan_time = 0.0
        self.last_odom_time = 0.0
        self.last_goal_time = 0.0
        self._frame_warned = False
        self._tf_warned = False
        self._missing_warned = False
        self._stale_warned = False

        self.success_count = 0
        self.collision_count = 0
        self.v_cmd = 0.0
        self.omega_cmd = 0.0
        self.wheel_left_cmd = 0.0
        self.wheel_right_cmd = 0.0
        dt = 1.0 / max(args.control_rate, 1.0e-6)
        self.alpha_v = 1.0 - math.exp(-dt / max(args.tau_v, 1.0e-6))
        self.alpha_omega = 1.0 - math.exp(-dt / max(args.tau_omega, 1.0e-6))
        self.alpha_wheel = self.alpha_v
        self.last_print_time = time.monotonic()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cmd_pub = self.create_publisher(Twist, args.topic_cmd, 10)
        self.create_subscription(LaserScan, args.topic_scan, self._on_scan, 10)
        self.create_subscription(Odometry, args.topic_odom, self._on_odom, 10)
        self.create_subscription(PoseStamped, args.topic_goal, self._on_goal, 10)
        if args.turn_cmd == "topic":
            self.create_subscription(String, args.topic_turn_cmd, self._on_turn_cmd, 10)
        self.create_subscription(Float32, args.topic_speed_cap, self._on_speed_cap, 10)
        self.timer = self.create_timer(1.0 / args.control_rate, self._on_timer)

        self.get_logger().info("Gazebo policy runner started")
        self.get_logger().info(f"checkpoint={args.checkpoint}")
        self.get_logger().info(
            f"topics: scan={args.topic_scan}, odom={args.topic_odom}, goal={args.topic_goal}, cmd={args.topic_cmd}"
        )
        self.get_logger().info(
            (
                f"mode={args.action_mode} action_scale={args.action_scale:.3f} "
                f"v_max={args.v_max:.3f} omega_limit={args.omega_limit:.3f} "
                f"goal_thresh={args.goal_reach_threshold:.3f} coll_thresh={args.collision_threshold:.3f}"
            )
        )
        self.get_logger().info(
            f"obs_dim={self.obs_dim} base_obs_dim={self.base_obs_dim} use_turn_cmd={self.use_turn_cmd} turn_cmd={args.turn_cmd}"
        )
        self.get_logger().info(
            f"speed_cap topic={args.topic_speed_cap} default={self.speed_cap if math.isfinite(self.speed_cap) else 'inf'}"
        )

    @staticmethod
    def _build_turn_cmd_one_hot(turn_cmd: str) -> np.ndarray:
        mapping = {
            "left": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "straight": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "right": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        if turn_cmd not in mapping:
            raise RuntimeError(f"Unsupported turn_cmd: {turn_cmd}")
        return mapping[turn_cmd]

    @staticmethod
    def _infer_obs_dim_from_checkpoint(ckpt: dict) -> int:
        policy_state = ckpt.get("policy")
        if isinstance(policy_state, dict):
            first_weight = policy_state.get("net_container.0.weight")
            if isinstance(first_weight, torch.Tensor) and first_weight.ndim == 2:
                return int(first_weight.shape[1])
        state_pre = ckpt.get("state_preprocessor")
        if isinstance(state_pre, dict) and "running_mean" in state_pre:
            running_mean = state_pre["running_mean"]
            if isinstance(running_mean, torch.Tensor):
                return int(running_mean.numel())
        raise RuntimeError("Cannot infer observation dimension from checkpoint.")

    def _load_checkpoint(
        self, checkpoint_path: str
    ) -> Tuple[SharedPolicyModel, torch.Tensor | None, torch.Tensor | None]:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "policy" not in ckpt:
            raise RuntimeError("Unsupported checkpoint format: missing 'policy' state dict.")

        ckpt_obs_dim = self._infer_obs_dim_from_checkpoint(ckpt)
        if ckpt_obs_dim == self.base_obs_dim:
            self.obs_dim = self.base_obs_dim
            self.use_turn_cmd = False
        elif ckpt_obs_dim == self.base_obs_dim + 3:
            self.obs_dim = ckpt_obs_dim
            self.use_turn_cmd = True
        else:
            raise RuntimeError(
                f"Unsupported checkpoint obs dim: {ckpt_obs_dim}. "
                f"Expected {self.base_obs_dim} (base) or {self.base_obs_dim + 3} (with turn_cmd)."
            )

        model = SharedPolicyModel(obs_dim=self.obs_dim, action_dim=self.action_dim)
        model.load_state_dict(ckpt["policy"], strict=True)

        # skrl RunningStandardScaler buffers
        running_mean = None
        running_variance = None
        state_pre = ckpt.get("state_preprocessor")
        if isinstance(state_pre, dict) and "running_mean" in state_pre and "running_variance" in state_pre:
            running_mean = state_pre["running_mean"].to(torch.float32)
            running_variance = state_pre["running_variance"].to(torch.float32)
            if running_mean.numel() != self.obs_dim:
                raise RuntimeError(
                    f"Checkpoint observation dim mismatch: expected {self.obs_dim}, got {running_mean.numel()}."
                )
        return model, running_mean, running_variance

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.running_mean is None or self.running_variance is None:
            return obs
        mean = self.running_mean.to(device=obs.device, dtype=torch.float32)
        var = self.running_variance.to(device=obs.device, dtype=torch.float32)
        return torch.clamp((obs - mean) / (torch.sqrt(var) + 1e-8), min=-5.0, max=5.0)

    def _on_scan(self, msg: LaserScan) -> None:
        self.scan_msg = msg
        self.last_scan_time = time.monotonic()

    def _on_odom(self, msg: Odometry) -> None:
        self.odom_msg = msg
        self.last_odom_time = time.monotonic()

    def _on_goal(self, msg: PoseStamped) -> None:
        self.goal_msg = msg
        self.last_goal_time = time.monotonic()

    def _on_turn_cmd(self, msg: String) -> None:
        cmd = msg.data.strip().lower()
        if cmd in {"left", "straight", "right"}:
            self.turn_cmd_one_hot = self._build_turn_cmd_one_hot(cmd)
            self.turn_cmd_current = cmd

    def _on_speed_cap(self, msg: Float32) -> None:
        self.speed_cap = max(0.0, float(msg.data))

    def _publish_zero(self) -> None:
        cmd = Twist()
        self.cmd_pub.publish(cmd)

    def _build_observation(self) -> Tuple[np.ndarray, float, float]:
        assert self.scan_msg is not None and self.odom_msg is not None and self.goal_msg is not None

        odom = self.odom_msg
        goal = self.goal_msg
        scan = self.scan_msg

        # Keep runner robust to map/odom frame mismatch by transforming goal into odom frame.
        if goal.header.frame_id and odom.header.frame_id and goal.header.frame_id != odom.header.frame_id:
            try:
                goal = self.tf_buffer.transform(goal, odom.header.frame_id, timeout=Duration(seconds=0.05))
            except TransformException as exc:
                if not self._tf_warned:
                    self.get_logger().warn(
                        f"Failed to transform goal frame '{goal.header.frame_id}' -> '{odom.header.frame_id}': {exc}"
                    )
                    self._tf_warned = True
                raise RuntimeError("goal frame transform failed")
        else:
            self._tf_warned = False

        # robot pose from odom
        rx = float(odom.pose.pose.position.x)
        ry = float(odom.pose.pose.position.y)
        qx = float(odom.pose.pose.orientation.x)
        qy = float(odom.pose.pose.orientation.y)
        qz = float(odom.pose.pose.orientation.z)
        qw = float(odom.pose.pose.orientation.w)
        yaw = quat_xyzw_to_yaw(qx, qy, qz, qw)

        # goal in world frame
        gx = float(goal.pose.position.x)
        gy = float(goal.pose.position.y)
        dx = gx - rx
        dy = gy - ry
        goal_dist = math.sqrt(dx * dx + dy * dy)
        goal_dist_safe = max(goal_dist, 1e-6)
        goal_dir_w_x = dx / goal_dist_safe
        goal_dir_w_y = dy / goal_dist_safe

        # world -> body rotation by -yaw
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        goal_dir_b_x = cos_yaw * goal_dir_w_x + sin_yaw * goal_dir_w_y
        goal_dir_b_y = -sin_yaw * goal_dir_w_x + cos_yaw * goal_dir_w_y
        goal_dist_norm = min(goal_dist / self.args.goal_max_distance, 1.0)

        # velocities - CRITICAL: must be in base_link frame
        # TurtleBot3 official Gazebo publishes odom with child_frame_id="base_footprint"
        # which is equivalent to base_link for velocity (both are body-fixed frames)
        child_frame = odom.child_frame_id
        if child_frame in ["base_link", "base_footprint", ""]:
            # Velocity is already in body frame, use directly
            forward_speed = float(odom.twist.twist.linear.x)
            yaw_rate = float(odom.twist.twist.angular.z)
        else:
            # Unexpected frame - velocity might be in odom frame, needs transformation
            # For safety, we transform assuming it's in odom frame
            if not self._frame_warned:
                self.get_logger().warn(
                    f"Odometry child_frame_id='{child_frame}' is not base_link/base_footprint. "
                    "Assuming velocity is in odom frame and transforming to body frame. "
                    "This may cause incorrect behavior if the assumption is wrong."
                )
                self._frame_warned = True

            # Transform linear velocity from odom frame to body frame
            vx_odom = float(odom.twist.twist.linear.x)
            vy_odom = float(odom.twist.twist.linear.y)
            forward_speed = cos_yaw * vx_odom + sin_yaw * vy_odom
            # Angular velocity is frame-invariant for planar motion, but to be safe:
            yaw_rate = float(odom.twist.twist.angular.z)

        # LiDAR
        lidar_ranges = resample_lidar(
            scan=scan,
            target_angles=self.target_beam_angles,
            lidar_min=self.args.lidar_min_range,
            lidar_max=self.args.lidar_max_range,
        )
        lidar_norm = np.clip(lidar_ranges / self.args.lidar_max_range, 0.0, 1.0)
        min_lidar_norm = float(np.min(lidar_norm))
        min_lidar = float(np.min(lidar_ranges))

        parts = [
            np.array(
                [goal_dir_b_x, goal_dir_b_y, goal_dist_norm, forward_speed, yaw_rate, min_lidar_norm],
                dtype=np.float32,
            )
        ]
        if self.use_turn_cmd:
            if self.args.turn_cmd == "auto":
                if goal_dir_b_y > self.args.turn_cmd_deadzone:
                    turn_one_hot = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                elif goal_dir_b_y < -self.args.turn_cmd_deadzone:
                    turn_one_hot = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                else:
                    turn_one_hot = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                parts.append(turn_one_hot)
            elif self.args.turn_cmd == "topic":
                parts.append(self.turn_cmd_one_hot)
            else:
                parts.append(self.turn_cmd_one_hot)
        parts.append(lidar_norm.astype(np.float32))
        obs = np.concatenate(parts, axis=0)
        return obs, goal_dist, min_lidar

    def _on_timer(self) -> None:
        now = time.monotonic()
        if self.scan_msg is None or self.odom_msg is None or self.goal_msg is None:
            if not self._missing_warned:
                self.get_logger().warn("waiting for /scan, /odom, /goal_pose ...")
                self._missing_warned = True
            self._publish_zero()
            return
        self._missing_warned = False

        stale_scan_odom = (
            (now - self.last_scan_time > self.args.input_timeout_s)
            or (now - self.last_odom_time > self.args.input_timeout_s)
        )
        stale_goal = bool(self.args.goal_timeout_s > 0.0) and ((now - self.last_goal_time) > self.args.goal_timeout_s)
        if stale_scan_odom or stale_goal:
            if not self._stale_warned:
                self.get_logger().warn("stale input detected; publishing zero cmd_vel")
                self._stale_warned = True
            self._publish_zero()
            return
        self._stale_warned = False

        try:
            obs_np, goal_dist, min_lidar = self._build_observation()
        except RuntimeError:
            self._publish_zero()
            return
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(self.device, dtype=torch.float32)
        obs_t = self._normalize_obs(obs_t)

        with torch.inference_mode():
            action = self.model.forward_policy(obs_t)[0]
        # env-side clamp equivalent
        action = torch.clamp(action, min=-1.0, max=1.0)
        action_v = float(action[0].item())
        action_omega = float(action[1].item())

        # Apply action scaling (near-goal and obstacle slowdown)
        # NOTE: If training already applied slowdown in the environment, disable this to avoid double-scaling
        if self.args.disable_action_scaling:
            speed_scale = 1.0
        else:
            near_goal_scale = np.clip(
                goal_dist / max(self.args.near_goal_slowdown_distance, 1.0e-6),
                self.args.near_goal_min_action_scale,
                1.0,
            )
            obstacle_den = max(self.args.obstacle_slowdown_distance - self.args.collision_threshold, 1.0e-6)
            obstacle_scale = np.clip(
                (min_lidar - self.args.collision_threshold) / obstacle_den,
                self.args.obstacle_min_action_scale,
                1.0,
            )
            speed_scale = float(near_goal_scale * obstacle_scale)

        if self.args.action_mode == "wheel":
            # Legacy checkpoints trained with direct wheel actions.
            left_target = action_v * self.args.action_scale * speed_scale
            right_target = action_omega * self.args.action_scale * speed_scale
            self.wheel_left_cmd = self.alpha_wheel * left_target + (1.0 - self.alpha_wheel) * self.wheel_left_cmd
            self.wheel_right_cmd = self.alpha_wheel * right_target + (1.0 - self.alpha_wheel) * self.wheel_right_cmd
            linear = self.args.wheel_radius * 0.5 * (self.wheel_left_cmd + self.wheel_right_cmd)
            angular = self.args.wheel_radius * (self.wheel_right_cmd - self.wheel_left_cmd) / max(
                self.args.wheel_base, 1.0e-6
            )
            # Keep safety limits when publishing cmd_vel.
            linear = max(-self.args.wheel_linear_speed_max, min(self.args.wheel_linear_speed_max, linear))
            angular = max(-self.args.omega_limit, min(self.args.omega_limit, angular))
        else:
            # New checkpoints trained with v-omega actions.
            if self.args.forward_only:
                v_target = 0.5 * (action_v + 1.0) * self.args.v_max
            else:
                v_target = action_v * self.args.v_max
            omega_target = action_omega * self.args.omega_limit
            v_target *= speed_scale
            omega_target *= speed_scale

            self.v_cmd = self.alpha_v * v_target + (1.0 - self.alpha_v) * self.v_cmd
            self.omega_cmd = self.alpha_omega * omega_target + (1.0 - self.alpha_omega) * self.omega_cmd

            half_wheel_base = 0.5 * self.args.wheel_base
            term_r = abs(self.v_cmd + half_wheel_base * self.omega_cmd)
            term_l = abs(self.v_cmd - half_wheel_base * self.omega_cmd)
            s_r = self.args.wheel_linear_speed_max / max(term_r, self.args.diamond_clamp_eps)
            s_l = self.args.wheel_linear_speed_max / max(term_l, self.args.diamond_clamp_eps)
            s = min(s_r, s_l, 1.0)
            linear = self.v_cmd * s
            angular = self.omega_cmd * s

        cmd = Twist()
        if math.isfinite(self.speed_cap):
            linear = max(-self.speed_cap, min(self.speed_cap, linear))
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        self.cmd_pub.publish(cmd)

        # lightweight terminal diagnostics
        success = goal_dist < self.args.goal_reach_threshold
        collision = min_lidar < self.args.collision_threshold
        if success:
            self.success_count += 1
            if self.args.stop_on_success:
                self.get_logger().info(f"goal reached: dist={goal_dist:.3f}, stopping.")
                self._publish_zero()
        if collision:
            self.collision_count += 1

        if now - self.last_print_time >= self.args.print_interval_s:
            self.last_print_time = now
            self.get_logger().info(
                (
                    f"goal_dist={goal_dist:.3f} min_lidar={min_lidar:.3f} "
                    f"action=[{action_v:.3f}, {action_omega:.3f}] "
                    f"cmd_vel=[{linear:.3f} m/s, {angular:.3f} rad/s] "
                    f"succ={self.success_count} coll={self.collision_count}"
                )
            )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run trained LiDAR-nav policy in ROS2/Gazebo.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to skrl checkpoint (best_agent.pt).")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or cuda:0.")

    # topics
    parser.add_argument("--topic_scan", type=str, default="/scan", help="LaserScan topic.")
    parser.add_argument("--topic_odom", type=str, default="/odom", help="Odometry topic.")
    parser.add_argument("--topic_goal", type=str, default="/goal_pose", help="Goal PoseStamped topic.")
    parser.add_argument("--topic_cmd", type=str, default="/cmd_vel", help="Twist command topic.")

    # control and safety
    parser.add_argument("--control_rate", type=float, default=60.0, help="Control loop frequency in Hz.")
    parser.add_argument("--input_timeout_s", type=float, default=0.5, help="Stale input timeout in seconds.")
    parser.add_argument(
        "--goal_timeout_s",
        type=float,
        default=0.0,
        help="Optional stale timeout for /goal_pose (seconds). 0 disables goal staleness checks.",
    )
    parser.add_argument("--print_interval_s", type=float, default=1.0, help="Terminal status print interval.")
    parser.add_argument("--stop_on_success", action="store_true", help="Publish zero velocity when goal is reached.")

    # observation/action parameters (must match training setup unless you know what you're changing)
    parser.add_argument(
        "--action_mode",
        type=str,
        default="vomega",
        choices=["vomega", "wheel"],
        help="Interpret checkpoint action as v-omega (new) or direct wheel commands (legacy).",
    )
    parser.add_argument(
        "--action_scale",
        type=float,
        default=8.0,
        help="Legacy wheel mode scale: wheel_cmd = action * action_scale.",
    )
    parser.add_argument("--num_beams", type=int, default=36, help="Number of LiDAR beams expected by policy.")
    parser.add_argument(
        "--turn_cmd",
        type=str,
        default="straight",
        choices=["left", "straight", "right", "auto", "topic"],
        help="Turn command one-hot used when checkpoint expects 45D observations.",
    )
    parser.add_argument(
        "--topic_turn_cmd",
        type=str,
        default="/turn_cmd",
        help="Turn command topic (std_msgs/String) when --turn_cmd topic is used.",
    )
    parser.add_argument(
        "--topic_speed_cap",
        type=str,
        default="/speed_cap",
        help="Speed cap topic (std_msgs/Float32), used as an upper bound on |cmd.linear.x|.",
    )
    parser.add_argument(
        "--turn_cmd_deadzone",
        type=float,
        default=0.10,
        help="Deadzone for --turn_cmd auto on goal_dir_b_y.",
    )
    parser.add_argument("--lidar_fov_deg", type=float, default=270.0, help="Policy LiDAR FOV in degrees.")
    parser.add_argument("--lidar_min_range", type=float, default=0.05, help="Minimum LiDAR range (m).")
    parser.add_argument("--lidar_max_range", type=float, default=3.0, help="Maximum LiDAR range (m).")
    parser.add_argument("--goal_max_distance", type=float, default=5.0, help="Goal distance normalization factor.")
    parser.add_argument(
        "--default_speed_cap",
        type=float,
        default=-1.0,
        help="Default speed cap before first /speed_cap message. <=0 means disabled.",
    )
    parser.add_argument("--v_max", type=float, default=1.0, help="Maximum forward speed used in training (m/s).")
    parser.add_argument("--omega_limit", type=float, default=6.0, help="Maximum yaw rate used in training (rad/s).")
    parser.add_argument("--forward_only", action="store_true", default=True, help="Use forward-only v mapping.")
    parser.add_argument(
        "--allow_reverse",
        action="store_true",
        help="Allow reverse commands (overrides --forward_only).",
    )
    parser.add_argument(
        "--disable_action_scaling",
        action="store_true",
        help="Disable near-goal and obstacle slowdown (use if training already applied scaling in env).",
    )
    parser.add_argument("--tau_v", type=float, default=0.10, help="LPF time constant tau_v (s).")
    parser.add_argument("--tau_omega", type=float, default=0.05, help="LPF time constant tau_omega (s).")
    parser.add_argument(
        "--near_goal_slowdown_distance",
        type=float,
        default=0.8,
        help="Near-goal slowdown distance used in training (m).",
    )
    parser.add_argument(
        "--near_goal_min_action_scale",
        type=float,
        default=0.30,
        help="Minimum near-goal action scale used in training.",
    )
    parser.add_argument(
        "--obstacle_slowdown_distance",
        type=float,
        default=0.85,
        help="Obstacle slowdown distance used in training (m).",
    )
    parser.add_argument(
        "--obstacle_min_action_scale",
        type=float,
        default=0.25,
        help="Minimum obstacle action scale used in training.",
    )
    parser.add_argument(
        "--wheel_linear_speed_max",
        type=float,
        default=1.0,
        help="Wheel linear speed cap V_max for diamond clamp (m/s).",
    )
    parser.add_argument(
        "--diamond_clamp_eps",
        type=float,
        default=1.0e-9,
        help="Numerical epsilon for diamond clamp denominator.",
    )
    parser.add_argument("--wheel_radius", type=float, default=0.033, help="Wheel radius (m).")
    parser.add_argument("--wheel_base", type=float, default=0.160, help="Wheel base (m).")
    parser.add_argument("--goal_reach_threshold", type=float, default=0.40, help="Goal success threshold (m).")
    parser.add_argument("--collision_threshold", type=float, default=0.20, help="Collision threshold from LiDAR (m).")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    if args.allow_reverse:
        args.forward_only = False

    rclpy.init()
    node = GazeboPolicyRunner(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node._publish_zero()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
