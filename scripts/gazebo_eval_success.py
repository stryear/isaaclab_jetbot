#!/usr/bin/env python3
"""Evaluate navigation success rate in ROS2/Gazebo.

This script assumes a policy node (e.g. scripts/gazebo_policy_runner.py) is already running
and publishing /cmd_vel from /scan, /odom, /goal_pose.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Pose, Twist
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import Empty

try:
    from gazebo_msgs.srv import DeleteEntity, SetEntityState, SpawnEntity
except ImportError:
    DeleteEntity = None  # type: ignore[assignment]
    SetEntityState = None  # type: ignore[assignment]
    SpawnEntity = None  # type: ignore[assignment]


def quat_xyzw_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class EpisodeResult:
    outcome: str  # success / collision / timeout
    elapsed_s: float
    final_dist: float
    min_scan: float
    goal_x: float
    goal_y: float
    goal_dx: float
    goal_dy: float


@dataclass
class FixedCase:
    name: str
    spawn_x: float
    spawn_y: float
    spawn_yaw: float
    goal_x: float
    goal_y: float
    turn_cmd: str


class GazeboSuccessEvaluator(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("gazebo_success_evaluator")
        self.args = args
        self.odom_msg: Odometry | None = None
        self.scan_msg: LaserScan | None = None
        self.last_odom_t = 0.0
        self.last_scan_t = 0.0

        self.goal_pub = self.create_publisher(PoseStamped, args.topic_goal, 10)
        self.turn_cmd_pub = self.create_publisher(String, args.topic_turn_cmd, 10)
        self.create_subscription(Odometry, args.topic_odom, self._on_odom, 10)
        self.create_subscription(LaserScan, args.topic_scan, self._on_scan, 10)
        self.reset_client = None
        if args.reset_each_episode:
            self.reset_client = self.create_client(Empty, args.reset_service)

        self.fixed_cases: list[FixedCase] = []
        self.set_state_client = None
        self.set_state_service_name = ""
        self._set_state_clients: dict[str, SetEntityState.Client] = {}
        self._set_state_candidates: list[str] = []
        self.delete_client = None
        self.spawn_client = None
        self.delete_service_name = ""
        self.spawn_service_name = ""
        self._delete_clients: dict[str, DeleteEntity.Client] = {}
        self._spawn_clients: dict[str, SpawnEntity.Client] = {}
        self._delete_candidates: list[str] = []
        self._spawn_candidates: list[str] = []
        self.spawn_entity_xml = ""
        if args.fixed_cases_file:
            self.fixed_cases = self._load_fixed_cases(args.fixed_cases_file)
            if SetEntityState is None:
                raise RuntimeError("gazebo_msgs.srv.SetEntityState is required for --fixed_cases_file mode.")
            if args.respawn_fallback:
                if DeleteEntity is None or SpawnEntity is None:
                    raise RuntimeError("gazebo_msgs DeleteEntity/SpawnEntity are required for --respawn_fallback.")
                if args.spawn_entity_sdf:
                    with open(args.spawn_entity_sdf, "r", encoding="utf-8") as f:
                        self.spawn_entity_xml = f.read()

        random.seed(args.seed)

    def _on_odom(self, msg: Odometry) -> None:
        self.odom_msg = msg
        self.last_odom_t = time.monotonic()

    def _on_scan(self, msg: LaserScan) -> None:
        self.scan_msg = msg
        self.last_scan_t = time.monotonic()

    def _wait_inputs(self, timeout_s: float) -> bool:
        t0 = time.monotonic()
        while rclpy.ok() and time.monotonic() - t0 < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.odom_msg is not None and self.scan_msg is not None:
                return True
        return False

    def _current_pose_yaw(self) -> tuple[float, float, float]:
        assert self.odom_msg is not None
        p = self.odom_msg.pose.pose.position
        q = self.odom_msg.pose.pose.orientation
        yaw = quat_xyzw_to_yaw(q.x, q.y, q.z, q.w)
        return float(p.x), float(p.y), yaw

    def _scan_min(self) -> float:
        assert self.scan_msg is not None
        vals = []
        range_min = float(self.scan_msg.range_min) if math.isfinite(self.scan_msg.range_min) else 0.0
        for r in self.scan_msg.ranges:
            if math.isfinite(r) and r >= range_min:
                vals.append(float(r))
        if not vals:
            return float("inf")
        return min(vals)

    def _scan_range_at_angle(self, angle_rad: float) -> float:
        """Interpolate LiDAR range at a body-frame angle."""
        assert self.scan_msg is not None
        scan = self.scan_msg
        n = len(scan.ranges)
        if n == 0 or not math.isfinite(scan.angle_increment) or abs(scan.angle_increment) < 1.0e-9:
            return float("inf")
        if angle_rad < scan.angle_min or angle_rad > scan.angle_max:
            return 0.0

        idx_f = (angle_rad - scan.angle_min) / scan.angle_increment
        i0 = int(math.floor(idx_f))
        i1 = min(i0 + 1, n - 1)
        w = idx_f - i0

        def _sanitize(v: float) -> float:
            if not math.isfinite(v):
                return float(scan.range_max)
            return max(float(scan.range_min), min(float(scan.range_max), float(v)))

        r0 = _sanitize(float(scan.ranges[max(0, min(i0, n - 1))]))
        r1 = _sanitize(float(scan.ranges[max(0, min(i1, n - 1))]))
        return (1.0 - w) * r0 + w * r1

    def _reset_world(self) -> bool:
        if self.reset_client is None:
            return True
        if not self.reset_client.wait_for_service(timeout_sec=self.args.reset_service_wait_s):
            self.get_logger().warn(f"reset service not available: {self.args.reset_service}")
            return False
        req = Empty.Request()
        fut = self.reset_client.call_async(req)
        t0 = time.monotonic()
        while rclpy.ok() and not fut.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.monotonic() - t0 > self.args.reset_service_wait_s:
                self.get_logger().warn("reset service call timed out")
                return False
        return fut.done() and fut.exception() is None

    def _publish_goal(self, gx: float, gy: float, frame_id: str) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.pose.position.x = gx
        msg.pose.position.y = gy
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

    def _publish_turn_cmd(self, cmd: str) -> None:
        if cmd not in {"left", "straight", "right"}:
            return
        msg = String()
        msg.data = cmd
        self.turn_cmd_pub.publish(msg)

    def _set_robot_pose(self, x: float, y: float, yaw: float) -> bool:
        if self.args.pose_reset_mode == "respawn":
            return self._respawn_robot(x, y, yaw)
        if not self._ensure_set_state_client():
            if self.args.respawn_fallback:
                self.get_logger().warn("set_state service unavailable; trying respawn fallback.")
                return self._respawn_robot(x, y, yaw)
            return False

        qz = math.sin(0.5 * yaw)
        qw = math.cos(0.5 * yaw)

        req = SetEntityState.Request()
        req.state.name = self.args.spawn_entity_name
        req.state.pose = Pose()
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = 0.01
        req.state.pose.orientation.z = float(qz)
        req.state.pose.orientation.w = float(qw)
        req.state.twist = Twist()
        req.state.reference_frame = "world"

        # Try primary service first, then fallback to alternatives if timeout/failure.
        service_order = []
        if self.set_state_service_name:
            service_order.append(self.set_state_service_name)
        for name in self._set_state_candidates:
            if name not in service_order:
                service_order.append(name)

        for idx, svc_name in enumerate(service_order):
            client = self._set_state_clients.get(svc_name)
            if client is None:
                continue
            ok = self._call_set_state(client, svc_name, req)
            if ok:
                if self.set_state_service_name != svc_name:
                    self.set_state_service_name = svc_name
                    self.set_state_client = client
                    self.get_logger().info(f"Switched set_state service to: {svc_name}")
                return True
            if idx + 1 < len(service_order):
                self.get_logger().warn(f"set_state via {svc_name} failed, trying fallback service...")
        if self.args.pose_reset_mode == "set_state":
            return False
        if self.args.respawn_fallback:
            self.get_logger().warn("set_state failed; trying respawn fallback (/delete_entity + /spawn_entity).")
            return self._respawn_robot(x, y, yaw)
        return False

    def _respawn_robot(self, x: float, y: float, yaw: float) -> bool:
        if not self._ensure_delete_spawn_clients():
            return False
        if self.delete_client is None or self.spawn_client is None or not self.spawn_entity_xml:
            self.get_logger().warn("respawn fallback unavailable (missing services/clients or spawn SDF).")
            return False

        # 1) delete (best-effort)
        dreq = DeleteEntity.Request()
        dreq.name = self.args.spawn_entity_name
        delete_ok = self._call_delete_entity(self.delete_client, self.delete_service_name, dreq)
        if not delete_ok:
            self.get_logger().warn("delete failed or timed out; will still try spawn to continue.")
        time.sleep(0.15)

        # 2) spawn at requested pose
        qz = math.sin(0.5 * yaw)
        qw = math.cos(0.5 * yaw)
        sreq = SpawnEntity.Request()
        sreq.name = self.args.spawn_entity_name
        sreq.xml = self.spawn_entity_xml
        sreq.robot_namespace = ""
        sreq.reference_frame = "world"
        sreq.initial_pose = Pose()
        sreq.initial_pose.position.x = float(x)
        sreq.initial_pose.position.y = float(y)
        sreq.initial_pose.position.z = 0.01
        sreq.initial_pose.orientation.z = float(qz)
        sreq.initial_pose.orientation.w = float(qw)

        return self._call_spawn_entity(self.spawn_client, self.spawn_service_name, sreq)

    def _call_set_state(self, client: SetEntityState.Client, svc_name: str, req: SetEntityState.Request) -> bool:
        fut = client.call_async(req)
        t0 = time.monotonic()
        while rclpy.ok() and not fut.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.monotonic() - t0 > self.args.set_state_wait_s:
                self.get_logger().warn(f"set_state service call timed out: {svc_name}")
                return False
        if fut.exception() is not None:
            self.get_logger().warn(f"set_state service failed ({svc_name}): {fut.exception()}")
            return False
        resp = fut.result()
        if resp is None or not bool(resp.success):
            self.get_logger().warn(f"set_state response unsuccessful: {svc_name}")
            return False
        return True

    def _ensure_set_state_client(self) -> bool:
        if self.set_state_client is not None:
            return True
        if SetEntityState is None:
            return False

        candidates: list[str] = []
        requested = (self.args.set_state_service or "").strip()
        if requested and requested != "auto":
            candidates.append(requested)
        # Common Gazebo Classic ROS2 service names
        candidates.extend(["/gazebo/set_entity_state", "/set_entity_state"])
        # Deduplicate while keeping order
        candidates = list(dict.fromkeys(candidates))
        self._set_state_candidates = candidates

        # Build clients once
        for name in candidates:
            if name not in self._set_state_clients:
                self._set_state_clients[name] = self.create_client(SetEntityState, name)

        # Robust wait loop: service graph can lag behind process startup.
        deadline = time.monotonic() + max(float(self.args.set_state_wait_s), 6.0)
        while rclpy.ok() and time.monotonic() < deadline:
            for name in candidates:
                client = self._set_state_clients[name]
                if client.wait_for_service(timeout_sec=0.2):
                    self.set_state_client = client
                    self.set_state_service_name = name
                    self.get_logger().info(f"Using set_state service: {name}")
                    return True
            rclpy.spin_once(self, timeout_sec=0.05)

        available_with_types = [(n, t) for n, t in self.get_service_names_and_types() if "set_entity_state" in n]
        available_names = [n for n, _ in available_with_types]
        # Fallback: if service is visible in graph, try calling anyway.
        for name in candidates:
            if name in available_names:
                self.set_state_client = self._set_state_clients[name]
                self.set_state_service_name = name
                self.get_logger().warn(
                    f"set_state wait timed out, but service is visible in graph. Proceeding with: {name}"
                )
                return True

        self.get_logger().warn(
            "set_state service not available. "
            f"tried={candidates}, discovered={available_with_types if available_with_types else 'none'}"
        )
        return False

    def _ensure_delete_spawn_clients(self) -> bool:
        if DeleteEntity is None or SpawnEntity is None:
            self.get_logger().warn("DeleteEntity/SpawnEntity service types are unavailable.")
            return False
        if not self.spawn_entity_xml:
            self.get_logger().warn("spawn_entity_sdf is empty; cannot use respawn fallback.")
            return False
        if self.delete_client is not None and self.spawn_client is not None:
            return True

        delete_requested = (self.args.delete_entity_service or "").strip()
        spawn_requested = (self.args.spawn_entity_service or "").strip()

        delete_candidates: list[str] = []
        spawn_candidates: list[str] = []
        if delete_requested and delete_requested != "auto":
            delete_candidates.append(delete_requested)
        if spawn_requested and spawn_requested != "auto":
            spawn_candidates.append(spawn_requested)

        delete_candidates.extend(["/gazebo/delete_entity", "/delete_entity"])
        spawn_candidates.extend(["/gazebo/spawn_entity", "/spawn_entity"])
        delete_candidates = list(dict.fromkeys(delete_candidates))
        spawn_candidates = list(dict.fromkeys(spawn_candidates))
        self._delete_candidates = delete_candidates
        self._spawn_candidates = spawn_candidates

        for name in delete_candidates:
            if name not in self._delete_clients:
                self._delete_clients[name] = self.create_client(DeleteEntity, name)
        for name in spawn_candidates:
            if name not in self._spawn_clients:
                self._spawn_clients[name] = self.create_client(SpawnEntity, name)

        deadline = time.monotonic() + max(float(self.args.set_state_wait_s), 6.0)
        delete_ready_name = ""
        spawn_ready_name = ""
        while rclpy.ok() and time.monotonic() < deadline:
            if not delete_ready_name:
                for name in delete_candidates:
                    if self._delete_clients[name].wait_for_service(timeout_sec=0.2):
                        delete_ready_name = name
                        break
            if not spawn_ready_name:
                for name in spawn_candidates:
                    if self._spawn_clients[name].wait_for_service(timeout_sec=0.2):
                        spawn_ready_name = name
                        break
            if delete_ready_name and spawn_ready_name:
                self.delete_service_name = delete_ready_name
                self.spawn_service_name = spawn_ready_name
                self.delete_client = self._delete_clients[delete_ready_name]
                self.spawn_client = self._spawn_clients[spawn_ready_name]
                self.get_logger().info(
                    f"Using respawn services: delete={delete_ready_name}, spawn={spawn_ready_name}"
                )
                return True
            rclpy.spin_once(self, timeout_sec=0.05)

        discovered = self.get_service_names_and_types()
        found_delete = [n for n, _ in discovered if "delete_entity" in n]
        found_spawn = [n for n, _ in discovered if "spawn_entity" in n]
        # Fallback: service graph visible but wait failed. Pick graph-visible names.
        if not delete_ready_name:
            for name in delete_candidates:
                if name in found_delete:
                    delete_ready_name = name
                    break
        if not spawn_ready_name:
            for name in spawn_candidates:
                if name in found_spawn:
                    spawn_ready_name = name
                    break
        if delete_ready_name and spawn_ready_name:
            self.delete_service_name = delete_ready_name
            self.spawn_service_name = spawn_ready_name
            self.delete_client = self._delete_clients[delete_ready_name]
            self.spawn_client = self._spawn_clients[spawn_ready_name]
            self.get_logger().warn(
                "respawn service wait timed out, but services are visible in graph. "
                f"Proceeding with delete={delete_ready_name}, spawn={spawn_ready_name}"
            )
            return True

        self.get_logger().warn(
            "respawn services not available. "
            f"delete_tried={delete_candidates}, delete_found={found_delete or 'none'}, "
            f"spawn_tried={spawn_candidates}, spawn_found={found_spawn or 'none'}"
        )
        return False

    def _call_delete_entity(self, client: DeleteEntity.Client, svc_name: str, req: DeleteEntity.Request) -> bool:
        fut = client.call_async(req)
        t0 = time.monotonic()
        while rclpy.ok() and not fut.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.monotonic() - t0 > self.args.set_state_wait_s:
                self.get_logger().warn(f"delete service call timed out: {svc_name}")
                return False
        if fut.exception() is not None:
            self.get_logger().warn(f"delete service failed ({svc_name}): {fut.exception()}")
            return False
        resp = fut.result()
        if resp is None:
            self.get_logger().warn(f"delete response missing: {svc_name}")
            return False
        if not bool(resp.success):
            # Deleting a non-existing entity is acceptable in this workflow.
            msg = getattr(resp, "status_message", "") or "unknown"
            if "does not exist" in msg.lower():
                return True
            self.get_logger().warn(f"delete response unsuccessful: {svc_name}: {msg}")
            return False
        return True

    def _call_spawn_entity(self, client: SpawnEntity.Client, svc_name: str, req: SpawnEntity.Request) -> bool:
        fut = client.call_async(req)
        t0 = time.monotonic()
        while rclpy.ok() and not fut.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.monotonic() - t0 > self.args.set_state_wait_s:
                self.get_logger().warn(f"spawn service call timed out: {svc_name}")
                return False
        if fut.exception() is not None:
            self.get_logger().warn(f"spawn service failed ({svc_name}): {fut.exception()}")
            return False
        resp = fut.result()
        if resp is None or not bool(resp.success):
            self.get_logger().warn(
                f"spawn response unsuccessful: {svc_name}: {getattr(resp, 'status_message', 'unknown')}"
            )
            return False
        return True

    @staticmethod
    def _load_fixed_cases(path: str) -> list[FixedCase]:
        cases: list[FixedCase] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"spawn_x", "spawn_y", "goal_x", "goal_y"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                raise RuntimeError(
                    "fixed_cases_file must contain headers: spawn_x,spawn_y,goal_x,goal_y "
                    "(optional: name,spawn_yaw)"
                )
            for i, row in enumerate(reader, start=1):
                name = (row.get("name") or f"case_{i:03d}").strip()
                spawn_x = float(row["spawn_x"])
                spawn_y = float(row["spawn_y"])
                spawn_yaw = float(row.get("spawn_yaw") or 0.0)
                goal_x = float(row["goal_x"])
                goal_y = float(row["goal_y"])
                turn_cmd = (row.get("turn_cmd") or "").strip().lower()
                if turn_cmd not in {"left", "straight", "right"}:
                    turn_cmd = ""
                cases.append(
                    FixedCase(
                        name=name,
                        spawn_x=spawn_x,
                        spawn_y=spawn_y,
                        spawn_yaw=spawn_yaw,
                        goal_x=goal_x,
                        goal_y=goal_y,
                        turn_cmd=turn_cmd,
                    )
                )
        if not cases:
            raise RuntimeError("fixed_cases_file is empty.")
        return cases

    def _distance_to_goal(self, gx: float, gy: float) -> float:
        x, y, _ = self._current_pose_yaw()
        return math.hypot(gx - x, gy - y)

    def _sample_relative_goal(self) -> tuple[float, float, str, float, float]:
        x, y, yaw = self._current_pose_yaw()
        dx = random.uniform(self.args.goal_forward_min, self.args.goal_forward_max)
        dy = random.uniform(-self.args.goal_lateral_max, self.args.goal_lateral_max)
        if bool(self.args.scan_aware_goal_sampling) and self.scan_msg is not None:
            sampled = False
            for _ in range(max(1, int(self.args.goal_sample_max_tries))):
                dx_try = random.uniform(self.args.goal_forward_min, self.args.goal_forward_max)
                dy_try = random.uniform(-self.args.goal_lateral_max, self.args.goal_lateral_max)
                dist_try = math.hypot(dx_try, dy_try)
                bearing_try = math.atan2(dy_try, dx_try)
                ray_range = self._scan_range_at_angle(bearing_try)
                if ray_range >= dist_try + float(self.args.goal_obstacle_margin_m):
                    dx, dy = dx_try, dy_try
                    sampled = True
                    break
            if not sampled:
                self.get_logger().warn(
                    "scan-aware sampling failed to find a fully free goal; using last sampled candidate."
                )
        gx = x + math.cos(yaw) * dx - math.sin(yaw) * dy
        gy = y + math.sin(yaw) * dx + math.cos(yaw) * dy
        frame_id = self.odom_msg.header.frame_id if self.odom_msg and self.odom_msg.header.frame_id else "odom"
        return gx, gy, frame_id, dx, dy

    def run(self) -> list[EpisodeResult]:
        if not self._wait_inputs(self.args.startup_timeout_s):
            raise RuntimeError("Did not receive /odom and /scan within startup timeout.")

        results: list[EpisodeResult] = []
        if self.fixed_cases:
            ep_cases = []
            for rep in range(max(1, self.args.fixed_cases_repeat)):
                for case in self.fixed_cases:
                    ep_cases.append((rep + 1, case))
            total_episodes = len(ep_cases)
            self.get_logger().info(
                f"Starting Gazebo eval (fixed-cases): episodes={total_episodes}, timeout={self.args.episode_timeout_s}s, "
                f"cases={len(self.fixed_cases)}, repeat={max(1, self.args.fixed_cases_repeat)}"
            )
        else:
            ep_cases = []
            total_episodes = self.args.episodes
            self.get_logger().info(
                f"Starting Gazebo eval: episodes={self.args.episodes}, timeout={self.args.episode_timeout_s}s"
            )

        for ep in range(1, total_episodes + 1):
            if self.args.reset_each_episode:
                self._reset_world()
                # Allow sensor/state topics to settle after reset.
                settle_t0 = time.monotonic()
                while rclpy.ok() and time.monotonic() - settle_t0 < self.args.post_reset_settle_s:
                    rclpy.spin_once(self, timeout_sec=0.02)

            if self.fixed_cases:
                _, case = ep_cases[ep - 1]
                if not self._set_robot_pose(case.spawn_x, case.spawn_y, case.spawn_yaw):
                    raise RuntimeError(
                        f"Failed to set robot pose for case '{case.name}' at "
                        f"({case.spawn_x}, {case.spawn_y}, yaw={case.spawn_yaw})."
                    )
                settle_t0 = time.monotonic()
                while rclpy.ok() and time.monotonic() - settle_t0 < self.args.spawn_settle_s:
                    rclpy.spin_once(self, timeout_sec=0.02)
                gx, gy = case.goal_x, case.goal_y
                frame_id = self.odom_msg.header.frame_id if self.odom_msg and self.odom_msg.header.frame_id else "odom"
                x, y, yaw = self._current_pose_yaw()
                dxw, dyw = gx - x, gy - y
                gdx = math.cos(yaw) * dxw + math.sin(yaw) * dyw
                gdy = -math.sin(yaw) * dxw + math.cos(yaw) * dyw
                if self.args.publish_turn_cmd and case.turn_cmd:
                    # publish a few times to survive occasional first-frame drops
                    for _ in range(3):
                        self._publish_turn_cmd(case.turn_cmd)
                        rclpy.spin_once(self, timeout_sec=0.01)
            else:
                gx, gy, frame_id, gdx, gdy = self._sample_relative_goal()
            self._publish_goal(gx, gy, frame_id)
            t0 = time.monotonic()
            min_scan = float("inf")
            outcome = "timeout"
            coll_count = 0

            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.02)
                elapsed = time.monotonic() - t0
                if elapsed > self.args.episode_timeout_s:
                    outcome = "timeout"
                    break

                # stale input guard
                now = time.monotonic()
                if (now - self.last_odom_t > self.args.input_timeout_s) or (now - self.last_scan_t > self.args.input_timeout_s):
                    continue

                d = self._distance_to_goal(gx, gy)
                smin = self._scan_min()
                min_scan = min(min_scan, smin)

                if elapsed >= self.args.collision_grace_s and smin < self.args.collision_threshold:
                    coll_count += 1
                else:
                    coll_count = 0

                if coll_count >= self.args.collision_consecutive:
                    outcome = "collision"
                    break
                if d < self.args.goal_reach_threshold:
                    outcome = "success"
                    break

            final_dist = self._distance_to_goal(gx, gy)
            res = EpisodeResult(
                outcome=outcome,
                elapsed_s=time.monotonic() - t0,
                final_dist=final_dist,
                min_scan=min_scan,
                goal_x=gx,
                goal_y=gy,
                goal_dx=gdx,
                goal_dy=gdy,
            )
            results.append(res)
            self.get_logger().info(
                (
                    f"[EP {ep:03d}] {outcome:<9} t={res.elapsed_s:5.1f}s "
                    f"dist={res.final_dist:.3f} min_scan={res.min_scan:.3f} "
                    f"goal=({res.goal_x:.2f},{res.goal_y:.2f}) rel=({res.goal_dx:.2f},{res.goal_dy:.2f})"
                )
            )
            if self.args.inter_episode_sleep_s > 0.0:
                time.sleep(self.args.inter_episode_sleep_s)

        return results


def summarize(results: list[EpisodeResult]) -> None:
    total = len(results)
    succ = sum(1 for r in results if r.outcome == "success")
    coll = sum(1 for r in results if r.outcome == "collision")
    tout = sum(1 for r in results if r.outcome == "timeout")
    mean_t = sum(r.elapsed_s for r in results) / max(total, 1)
    mean_dist = sum(r.final_dist for r in results) / max(total, 1)
    print(f"[GAZEBO][RESULT] episodes={total}")
    print(f"[GAZEBO][RESULT] success_rate={succ/total:.4f} ({succ}/{total})")
    print(f"[GAZEBO][RESULT] collision_rate={coll/total:.4f} ({coll}/{total})")
    print(f"[GAZEBO][RESULT] timeout_rate={tout/total:.4f} ({tout}/{total})")
    print(f"[GAZEBO][RESULT] ep_time_mean={mean_t:.2f}s")
    print(f"[GAZEBO][RESULT] final_dist_mean={mean_dist:.3f}m")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate success rate in ROS2/Gazebo by sending relative goals.")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--episode_timeout_s", type=float, default=30.0)
    p.add_argument("--startup_timeout_s", type=float, default=10.0)
    p.add_argument("--input_timeout_s", type=float, default=0.5)
    p.add_argument("--inter_episode_sleep_s", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fixed_cases_file", type=str, default="")
    p.add_argument("--fixed_cases_repeat", type=int, default=1)

    p.add_argument("--goal_forward_min", type=float, default=1.0)
    p.add_argument("--goal_forward_max", type=float, default=2.4)
    p.add_argument("--goal_lateral_max", type=float, default=0.8)
    p.add_argument(
        "--scan_aware_goal_sampling",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable scan-aware goal feasibility check before publishing goals.",
    )
    p.add_argument(
        "--goal_obstacle_margin_m",
        type=float,
        default=0.25,
        help="Required free margin beyond sampled goal distance along its LiDAR ray.",
    )
    p.add_argument(
        "--goal_sample_max_tries",
        type=int,
        default=40,
        help="Maximum rejection-sampling attempts for scan-aware goal placement.",
    )

    p.add_argument("--goal_reach_threshold", type=float, default=0.40)
    p.add_argument("--collision_threshold", type=float, default=0.20)
    p.add_argument("--collision_grace_s", type=float, default=1.0)
    p.add_argument("--collision_consecutive", type=int, default=3)
    p.add_argument("--reset_each_episode", action="store_true")
    p.add_argument("--reset_service", type=str, default="/reset_simulation")
    p.add_argument("--reset_service_wait_s", type=float, default=2.0)
    p.add_argument("--post_reset_settle_s", type=float, default=0.5)
    p.add_argument("--spawn_entity_name", type=str, default="burger")
    p.add_argument(
        "--set_state_service",
        type=str,
        default="auto",
        help='SetEntityState service name (default: "auto", tries common names).',
    )
    p.add_argument("--delete_entity_service", type=str, default="/delete_entity")
    p.add_argument("--spawn_entity_service", type=str, default="/spawn_entity")
    p.add_argument(
        "--spawn_entity_sdf",
        type=str,
        default="/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf",
    )
    p.add_argument("--respawn_fallback", type=int, choices=[0, 1], default=1)
    p.add_argument("--set_state_wait_s", type=float, default=2.0)
    p.add_argument("--spawn_settle_s", type=float, default=0.5)
    p.add_argument(
        "--pose_reset_mode",
        type=str,
        default="auto",
        choices=["auto", "set_state", "respawn"],
        help="Robot pose reset mode in fixed-cases: auto=try set_state then fallback, set_state=only set_state, respawn=only delete+spawn.",
    )

    p.add_argument("--topic_scan", type=str, default="/scan")
    p.add_argument("--topic_odom", type=str, default="/odom")
    p.add_argument("--topic_goal", type=str, default="/goal_pose")
    p.add_argument("--topic_turn_cmd", type=str, default="/turn_cmd")
    p.add_argument(
        "--publish_turn_cmd",
        type=int,
        choices=[0, 1],
        default=0,
        help="Publish per-case turn_cmd (requires turn_cmd column in fixed_cases_file).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    rclpy.init()
    node = GazeboSuccessEvaluator(args)
    try:
        results = node.run()
        summarize(results)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
