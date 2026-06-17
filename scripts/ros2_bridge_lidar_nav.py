#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""ROS2 bridge loop for TurtleBot3 LiDAR navigation environment.

Publishes:
- LaserScan on /scan
- Odometry on /odom

Subscribes:
- Twist on /cmd_vel
"""

from __future__ import annotations

import argparse
import math
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="ROS2 bridge for TurtleBot3 LiDAR navigation task.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate (must be 1).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--topic_scan", type=str, default="/scan", help="ROS2 LaserScan topic.")
parser.add_argument("--topic_odom", type=str, default="/odom", help="ROS2 Odometry topic.")
parser.add_argument("--topic_cmd", type=str, default="/cmd_vel", help="ROS2 Twist command topic.")
parser.add_argument("--scan_frame", type=str, default="base_scan", help="LaserScan frame id.")
parser.add_argument("--odom_frame", type=str, default="odom", help="Odometry frame id.")
parser.add_argument("--base_frame", type=str, default="base_link", help="Base link frame id.")
parser.add_argument("--cmd_timeout", type=float, default=0.5, help="Timeout for cmd_vel watchdog in seconds.")
parser.add_argument("--spin_timeout", type=float, default=0.0, help="Timeout for rclpy.spin_once in seconds.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import isaac_lab_tutorial.tasks  # noqa: F401

try:
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
except ImportError as exc:
    raise RuntimeError(
        "ROS2 Python packages are not available in this environment. "
        "Please source ROS2 Humble before running this script."
    ) from exc


class LidarNavBridgeNode(Node):
    """ROS2 topic bridge for LiDAR nav environment."""

    def __init__(self, topic_scan: str, topic_odom: str, topic_cmd: str, cmd_timeout: float):
        super().__init__("isaaclab_lidar_nav_bridge")
        self.scan_pub = self.create_publisher(LaserScan, topic_scan, 10)
        self.odom_pub = self.create_publisher(Odometry, topic_odom, 10)
        self.cmd_sub = self.create_subscription(Twist, topic_cmd, self._on_cmd_vel, 10)
        self._cmd_timeout = cmd_timeout
        self._last_cmd_time = time.monotonic()
        self._cmd_linear = 0.0
        self._cmd_angular = 0.0

    def _on_cmd_vel(self, msg: Twist):
        self._cmd_linear = float(msg.linear.x)
        self._cmd_angular = float(msg.angular.z)
        self._last_cmd_time = time.monotonic()

    def get_cmd(self) -> tuple[float, float]:
        if (time.monotonic() - self._last_cmd_time) > self._cmd_timeout:
            return 0.0, 0.0
        return self._cmd_linear, self._cmd_angular

    def publish_scan(
        self,
        ranges: list[float],
        angle_min: float,
        angle_max: float,
        angle_increment: float,
        range_min: float,
        range_max: float,
        frame_id: str,
    ):
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.angle_min = float(angle_min)
        msg.angle_max = float(angle_max)
        msg.angle_increment = float(angle_increment)
        msg.time_increment = 0.0
        msg.scan_time = 0.0
        msg.range_min = float(range_min)
        msg.range_max = float(range_max)
        msg.ranges = ranges
        msg.intensities = [0.0 for _ in ranges]
        self.scan_pub.publish(msg)

    def publish_odom(
        self,
        pos_xyz: list[float],
        quat_wxyz: list[float],
        lin_vel_x: float,
        ang_vel_z: float,
        odom_frame: str,
        base_frame: str,
    ):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = odom_frame
        msg.child_frame_id = base_frame
        msg.pose.pose.position.x = float(pos_xyz[0])
        msg.pose.pose.position.y = float(pos_xyz[1])
        msg.pose.pose.position.z = float(pos_xyz[2])
        # ROS uses xyzw ordering.
        msg.pose.pose.orientation.x = float(quat_wxyz[1])
        msg.pose.pose.orientation.y = float(quat_wxyz[2])
        msg.pose.pose.orientation.z = float(quat_wxyz[3])
        msg.pose.pose.orientation.w = float(quat_wxyz[0])
        msg.twist.twist.linear.x = float(lin_vel_x)
        msg.twist.twist.angular.z = float(ang_vel_z)
        self.odom_pub.publish(msg)


def main():
    if args_cli.num_envs is not None and args_cli.num_envs != 1:
        raise ValueError("ros2_bridge_lidar_nav.py only supports --num_envs 1")

    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # reset environment
    env.reset()
    unwrapped = env.unwrapped

    if not hasattr(unwrapped, "latest_lidar_ranges"):
        raise RuntimeError(
            "Selected task does not expose LiDAR state. Use Isaac-Lab-Tutorial-LidarNav-TurtleBot3-Direct-v0."
        )

    rclpy.init()
    bridge_node = LidarNavBridgeNode(
        topic_scan=args_cli.topic_scan,
        topic_odom=args_cli.topic_odom,
        topic_cmd=args_cli.topic_cmd,
        cmd_timeout=args_cli.cmd_timeout,
    )

    action = torch.zeros((1, env.action_space.shape[-1]), device=unwrapped.device)
    v_max = float(getattr(unwrapped.cfg, "v_max", 0.10))
    omega_limit = float(getattr(unwrapped.cfg, "omega_limit", 1.2))
    forward_only = bool(getattr(unwrapped.cfg, "forward_only", True))

    lidar_fov = math.radians(float(getattr(unwrapped.cfg, "lidar_fov_deg", 270.0)))
    lidar_num_beams = int(getattr(unwrapped.cfg, "lidar_num_beams", 36))
    lidar_min_range = float(getattr(unwrapped.cfg, "lidar_min_range", 0.05))
    lidar_max_range = float(getattr(unwrapped.cfg, "lidar_max_range", 3.0))
    angle_min = -lidar_fov / 2.0
    angle_max = lidar_fov / 2.0
    angle_inc = (angle_max - angle_min) / max(lidar_num_beams - 1, 1)

    print("[ROS2] Bridge started")
    print(f"  Task: {args_cli.task}")
    print(f"  Topics: cmd={args_cli.topic_cmd}, scan={args_cli.topic_scan}, odom={args_cli.topic_odom}")

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                rclpy.spin_once(bridge_node, timeout_sec=args_cli.spin_timeout)
                linear_cmd, angular_cmd = bridge_node.get_cmd()

                # cmd_vel -> normalized policy action [a_v, a_omega].
                if forward_only:
                    linear_cmd = max(0.0, linear_cmd)
                    a_v = 2.0 * linear_cmd / max(v_max, 1.0e-6) - 1.0
                else:
                    a_v = linear_cmd / max(v_max, 1.0e-6)
                a_omega = angular_cmd / max(omega_limit, 1.0e-6)
                action[0, 0] = max(-1.0, min(1.0, a_v))
                action[0, 1] = max(-1.0, min(1.0, a_omega))

                env.step(action)

                lidar_ranges = unwrapped.latest_lidar_ranges[0].detach().cpu().tolist()
                bridge_node.publish_scan(
                    ranges=lidar_ranges,
                    angle_min=angle_min,
                    angle_max=angle_max,
                    angle_increment=angle_inc,
                    range_min=lidar_min_range,
                    range_max=lidar_max_range,
                    frame_id=args_cli.scan_frame,
                )

                pos = unwrapped.robot.data.root_pos_w[0, :3].detach().cpu().tolist()
                quat_wxyz = unwrapped.robot.data.root_link_quat_w[0, :4].detach().cpu().tolist()
                lin_vel_x = float(unwrapped.robot.data.root_com_lin_vel_b[0, 0].item())
                ang_vel_z = float(unwrapped.robot.data.root_ang_vel_b[0, 2].item())
                bridge_node.publish_odom(
                    pos_xyz=pos,
                    quat_wxyz=quat_wxyz,
                    lin_vel_x=lin_vel_x,
                    ang_vel_z=ang_vel_z,
                    odom_frame=args_cli.odom_frame,
                    base_frame=args_cli.base_frame,
                )
    finally:
        env.close()
        bridge_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
