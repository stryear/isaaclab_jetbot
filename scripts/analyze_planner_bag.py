#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


STATE_NAMES = ("explore", "fallback", "junction", "return")
STATE_TO_IDX = {name: idx for idx, name in enumerate(STATE_NAMES)}


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    arr = sorted(values)
    if len(arr) == 1:
        return float(arr[0])
    p = max(0.0, min(1.0, q)) * (len(arr) - 1)
    i0 = int(math.floor(p))
    i1 = min(i0 + 1, len(arr) - 1)
    w = p - i0
    return float(arr[i0] * (1.0 - w) + arr[i1] * w)


def _safe_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "min": None, "max": None, "p10": None, "p50": None, "p90": None}
    n = len(values)
    mean = sum(values) / max(1, n)
    return {
        "n": n,
        "mean": float(mean),
        "min": float(min(values)),
        "max": float(max(values)),
        "p10": _quantile(values, 0.10),
        "p50": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
    }


def _state_name(idx: int | None) -> str:
    if idx is None or idx < 0 or idx >= len(STATE_NAMES):
        return "unknown"
    return STATE_NAMES[idx]


def _yaw_from_quat_xyzw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _to_base_frame(dx_w: float, dy_w: float, yaw_wb: float) -> tuple[float, float]:
    c = math.cos(yaw_wb)
    s = math.sin(yaw_wb)
    x_b = c * dx_w + s * dy_w
    y_b = -s * dx_w + c * dy_w
    return x_b, y_b


def _is_finite(x: float) -> bool:
    return math.isfinite(x)


def _try_get_scalar(msg: Any) -> float | None:
    for name in ("data", "value", "id", "goal_id"):
        if hasattr(msg, name):
            v = getattr(msg, name)
            if isinstance(v, (int, float, bool)):
                return float(v)
    return None


def _try_get_string(msg: Any) -> str | None:
    if hasattr(msg, "data"):
        v = getattr(msg, "data")
        if isinstance(v, str):
            return v
    for name in ("source", "state", "planner_state"):
        if hasattr(msg, name):
            v = getattr(msg, name)
            if isinstance(v, str):
                return v
    return None


def _try_get_goal_xy_frame(msg: Any) -> tuple[float, float, str | None] | None:
    frame_id = None
    if hasattr(msg, "header") and hasattr(msg.header, "frame_id"):
        frame_id = str(msg.header.frame_id)

    if hasattr(msg, "point"):
        x = float(getattr(msg.point, "x", float("nan")))
        y = float(getattr(msg.point, "y", float("nan")))
        return (x, y, frame_id)

    if hasattr(msg, "pose") and hasattr(msg.pose, "position"):
        x = float(getattr(msg.pose.position, "x", float("nan")))
        y = float(getattr(msg.pose.position, "y", float("nan")))
        return (x, y, frame_id)

    if hasattr(msg, "x") and hasattr(msg, "y"):
        x = float(getattr(msg, "x", float("nan")))
        y = float(getattr(msg, "y", float("nan")))
        return (x, y, frame_id)

    return None


def _coerce_planner_state(msg: Any) -> int | None:
    val = None
    if hasattr(msg, "data"):
        val = getattr(msg, "data")
    elif hasattr(msg, "state"):
        val = getattr(msg, "state")
    elif hasattr(msg, "planner_state"):
        val = getattr(msg, "planner_state")
    if val is None:
        return None
    if isinstance(val, str):
        key = val.strip().lower()
        if key in STATE_TO_IDX:
            return STATE_TO_IDX[key]
        return None
    if isinstance(val, (int, float)):
        idx = int(val)
        if 0 <= idx < len(STATE_NAMES):
            return idx
    return None


def _field_names(msg: Any) -> list[str]:
    if hasattr(msg, "get_fields_and_field_types"):
        return list(msg.get_fields_and_field_types().keys())
    return []


def _find_numeric_field(msg: Any, candidates: Iterable[str], depth: int = 0) -> float | None:
    if depth > 3:
        return None
    cset = set(candidates)
    names = _field_names(msg)
    for n in names:
        if n in cset:
            v = getattr(msg, n, None)
            if isinstance(v, (int, float, bool)):
                return float(v)
    for n in names:
        v = getattr(msg, n, None)
        if hasattr(v, "get_fields_and_field_types"):
            out = _find_numeric_field(v, candidates, depth + 1)
            if out is not None:
                return out
    return None


@dataclass
class OdomState:
    x: float = float("nan")
    y: float = float("nan")
    yaw: float = float("nan")
    frame_id: str | None = None
    t: float = 0.0


class PlannerBagAnalyzer:
    def __init__(self):
        self.goal_dists: list[float] = []
        self.goal_bearings: list[float] = []
        self.goal_dists_by_state: dict[str, list[float]] = defaultdict(list)
        self.goal_bearings_by_state: dict[str, list[float]] = defaultdict(list)
        self.goal_unresolved_frame_count = 0

        self.goal_nan_segments: list[float] = []
        self._goal_nan_active = False
        self._goal_nan_start = 0.0
        self.goal_nan_count = 0

        self.goal_id_durations: list[float] = []
        self.goal_id_durations_by_id: dict[str, list[float]] = defaultdict(list)
        self._goal_id_curr: str | None = None
        self._goal_id_start_t: float | None = None

        self.state_time_sec = [0.0, 0.0, 0.0, 0.0]
        self.state_dwell_sec: dict[str, list[float]] = defaultdict(list)
        self.state_transition = [[0 for _ in range(4)] for _ in range(4)]
        self._state_last_idx: int | None = None
        self._state_last_change_t: float | None = None

        self.speed_cap_all: list[float] = []
        self.speed_cap_by_state: dict[str, list[float]] = defaultdict(list)
        self._last_speed_cap: float | None = None

        self._odom = OdomState()
        self._last_goal_dist: float | None = None
        self._last_progress_delta: float | None = None
        self._last_min_lidar: float | None = None

        self.exec_counter = {
            "local_timeout": 0.0,
            "no_progress": 0.0,
            "unsafe_clearance": 0.0,
            "fallback_count": 0.0,
            "replan_count": 0.0,
        }
        self._exec_prev_values: dict[str, float] = {}
        self.timeout_context_rows: list[dict[str, Any]] = []
        self.unsafe_context_rows: list[dict[str, Any]] = []

        self.topic_msg_count: Counter[str] = Counter()
        self.exec_source_counts: Counter[str] = Counter()
        self.exec_source_switches = 0
        self._last_exec_source: str | None = None

        self.action_status_counts: Counter[str] = Counter()
        self.action_status_code_counts: Counter[str] = Counter()

        self.topic_type_warnings: list[str] = []
        self.bags_processed: list[str] = []

    def _record_goal_nan(self, t_sec: float, is_nan: bool):
        if is_nan and not self._goal_nan_active:
            self._goal_nan_active = True
            self._goal_nan_start = t_sec
            self.goal_nan_count += 1
        elif (not is_nan) and self._goal_nan_active:
            self.goal_nan_segments.append(max(0.0, t_sec - self._goal_nan_start))
            self._goal_nan_active = False

    def _close_open_segments(self, t_end: float):
        if self._goal_nan_active:
            self.goal_nan_segments.append(max(0.0, t_end - self._goal_nan_start))
            self._goal_nan_active = False

        if self._goal_id_curr is not None and self._goal_id_start_t is not None:
            dur = max(0.0, t_end - self._goal_id_start_t)
            self.goal_id_durations.append(dur)
            self.goal_id_durations_by_id[self._goal_id_curr].append(dur)
            self._goal_id_curr = None
            self._goal_id_start_t = None

        if self._state_last_idx is not None and self._state_last_change_t is not None:
            dur = max(0.0, t_end - self._state_last_change_t)
            self.state_time_sec[self._state_last_idx] += dur
            self.state_dwell_sec[_state_name(self._state_last_idx)].append(dur)
            self._state_last_change_t = None

    def _on_goal_id(self, t_sec: float, msg: Any):
        scalar = _try_get_scalar(msg)
        if scalar is None:
            if hasattr(msg, "data"):
                scalar = str(getattr(msg, "data"))
            else:
                return
        gid = str(int(scalar)) if isinstance(scalar, float) and scalar.is_integer() else str(scalar)
        if self._goal_id_curr is None:
            self._goal_id_curr = gid
            self._goal_id_start_t = t_sec
            return
        if gid != self._goal_id_curr:
            if self._goal_id_start_t is not None:
                dur = max(0.0, t_sec - self._goal_id_start_t)
                self.goal_id_durations.append(dur)
                self.goal_id_durations_by_id[self._goal_id_curr].append(dur)
            self._goal_id_curr = gid
            self._goal_id_start_t = t_sec

    def _on_planner_state(self, t_sec: float, msg: Any):
        idx = _coerce_planner_state(msg)
        if idx is None:
            return
        if self._state_last_idx is None:
            self._state_last_idx = idx
            self._state_last_change_t = t_sec
            return
        if idx == self._state_last_idx:
            return
        if self._state_last_change_t is not None:
            dur = max(0.0, t_sec - self._state_last_change_t)
            self.state_time_sec[self._state_last_idx] += dur
            self.state_dwell_sec[_state_name(self._state_last_idx)].append(dur)
        self.state_transition[self._state_last_idx][idx] += 1
        self._state_last_idx = idx
        self._state_last_change_t = t_sec

    def _on_speed_cap(self, msg: Any):
        v = _try_get_scalar(msg)
        if v is None or not _is_finite(v):
            return
        self._last_speed_cap = float(v)
        self.speed_cap_all.append(float(v))
        key = _state_name(self._state_last_idx)
        self.speed_cap_by_state[key].append(float(v))

    def _on_odom(self, t_sec: float, msg: Any):
        if not hasattr(msg, "pose") or not hasattr(msg.pose, "pose"):
            return
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._odom.x = float(getattr(p, "x", float("nan")))
        self._odom.y = float(getattr(p, "y", float("nan")))
        self._odom.yaw = _yaw_from_quat_xyzw(
            float(getattr(q, "x", 0.0)),
            float(getattr(q, "y", 0.0)),
            float(getattr(q, "z", 0.0)),
            float(getattr(q, "w", 1.0)),
        )
        if hasattr(msg, "header") and hasattr(msg.header, "frame_id"):
            self._odom.frame_id = str(msg.header.frame_id)
        self._odom.t = t_sec

    def _on_scan(self, msg: Any):
        if not hasattr(msg, "ranges"):
            return
        mins = [float(r) for r in msg.ranges if _is_finite(float(r)) and float(r) > 0.0]
        if mins:
            self._last_min_lidar = min(mins)

    def _goal_age(self, t_sec: float) -> float | None:
        if self._goal_id_start_t is None:
            return None
        return max(0.0, t_sec - self._goal_id_start_t)

    def _snapshot_context(self, t_sec: float) -> dict[str, Any]:
        return {
            "t_sec": float(t_sec),
            "planner_state": _state_name(self._state_last_idx),
            "goal_dist": self._last_goal_dist,
            "progress_delta": self._last_progress_delta,
            "min_lidar": self._last_min_lidar,
            "speed_cap": self._last_speed_cap,
            "goal_id_age": self._goal_age(t_sec),
        }

    def _on_planner_exec_stats(self, t_sec: float, msg: Any):
        field_map = {
            "local_timeout": {"local_timeout", "local_timeouts", "timeout", "timeouts"},
            "no_progress": {"no_progress", "no_progress_count", "stuck_count"},
            "unsafe_clearance": {"unsafe_clearance", "unsafe_clearance_count", "collision_count"},
            "fallback_count": {"fallback_count", "fallbacks"},
            "replan_count": {"replan_count", "replan", "replans"},
        }
        for key, candidates in field_map.items():
            cur = _find_numeric_field(msg, candidates)
            if cur is None:
                continue
            prev = self._exec_prev_values.get(key)
            if prev is None:
                self._exec_prev_values[key] = cur
                continue
            delta = cur - prev
            self._exec_prev_values[key] = cur
            if delta <= 0:
                continue
            self.exec_counter[key] += float(delta)
            if key == "local_timeout":
                for _ in range(max(1, int(round(delta)))):
                    self.timeout_context_rows.append(self._snapshot_context(t_sec))
            if key == "unsafe_clearance":
                for _ in range(max(1, int(round(delta)))):
                    self.unsafe_context_rows.append(self._snapshot_context(t_sec))

    def _on_goal(self, t_sec: float, msg: Any):
        parsed = _try_get_goal_xy_frame(msg)
        if parsed is None:
            return
        gx, gy, frame = parsed
        if not _is_finite(gx) or not _is_finite(gy):
            self._record_goal_nan(t_sec, True)
            return

        rel_x = float("nan")
        rel_y = float("nan")
        frame_l = (frame or "").strip().lower()
        if frame_l in {"base_footprint", "base_link"}:
            rel_x, rel_y = gx, gy
        elif _is_finite(self._odom.x) and _is_finite(self._odom.y) and _is_finite(self._odom.yaw):
            if self._odom.frame_id is None or frame is None or str(frame) == str(self._odom.frame_id):
                dx = gx - self._odom.x
                dy = gy - self._odom.y
                rel_x, rel_y = _to_base_frame(dx, dy, self._odom.yaw)
            else:
                self.goal_unresolved_frame_count += 1
                return
        else:
            self.goal_unresolved_frame_count += 1
            return

        dist = math.hypot(rel_x, rel_y)
        bearing = math.atan2(rel_y, rel_x)
        if not _is_finite(dist) or not _is_finite(bearing):
            self._record_goal_nan(t_sec, True)
            return

        self._record_goal_nan(t_sec, False)
        self.goal_dists.append(dist)
        self.goal_bearings.append(bearing)
        state_key = _state_name(self._state_last_idx)
        self.goal_dists_by_state[state_key].append(dist)
        self.goal_bearings_by_state[state_key].append(bearing)

        if self._last_goal_dist is not None:
            self._last_progress_delta = self._last_goal_dist - dist
        self._last_goal_dist = dist

    def _on_exec_source(self, msg: Any):
        v = _try_get_string(msg)
        if not v:
            scalar = _try_get_scalar(msg)
            if scalar is not None:
                v = str(scalar)
        if not v:
            return
        key = v.strip().lower()
        if not key:
            return
        self.exec_source_counts[key] += 1
        if self._last_exec_source is not None and key != self._last_exec_source:
            self.exec_source_switches += 1
        self._last_exec_source = key

    def _on_action_status(self, msg: Any):
        status_map = {
            0: "unknown",
            1: "accepted",
            2: "executing",
            3: "canceling",
            4: "succeeded",
            5: "canceled",
            6: "aborted",
        }
        if not hasattr(msg, "status_list"):
            return
        for st in getattr(msg, "status_list", []):
            code = int(getattr(st, "status", 0))
            name = status_map.get(code, f"code_{code}")
            self.action_status_counts[name] += 1
            self.action_status_code_counts[str(code)] += 1

    def process_bag(self, bag_path: Path):
        try:
            import rosbag2_py
            from rclpy.serialization import deserialize_message
            from rosidl_runtime_py.utilities import get_message
        except Exception as exc:
            raise RuntimeError("ROS2 Python runtime unavailable. Please source ROS2 before running this script.") from exc

        storage_candidates = ("sqlite3", "mcap")
        reader = rosbag2_py.SequentialReader()
        opened = False
        last_err = None
        for sid in storage_candidates:
            try:
                reader.open(rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=sid), rosbag2_py.ConverterOptions("", ""))
                opened = True
                break
            except Exception as exc:  # pragma: no cover - runtime dependent
                last_err = exc
        if not opened:
            raise RuntimeError(f"Failed to open bag {bag_path}: {last_err}")

        topic_type = {t.name: t.type for t in reader.get_all_topics_and_types()}
        msg_cls_cache: dict[str, Any] = {}
        for tp, type_name in topic_type.items():
            try:
                msg_cls_cache[tp] = get_message(type_name)
            except Exception:
                self.topic_type_warnings.append(f"{bag_path}: cannot import message type '{type_name}' for topic '{tp}'")

        t_end = 0.0
        while reader.has_next():
            topic, raw, t_ns = reader.read_next()
            t_sec = float(t_ns) * 1.0e-9
            t_end = max(t_end, t_sec)
            self.topic_msg_count[topic] += 1
            msg_cls = msg_cls_cache.get(topic)
            if msg_cls is None:
                continue
            try:
                msg = deserialize_message(raw, msg_cls)
            except Exception:
                continue

            if topic == "/planner_state":
                self._on_planner_state(t_sec, msg)
            elif topic == "/speed_cap":
                self._on_speed_cap(msg)
            elif topic == "/goal_id":
                self._on_goal_id(t_sec, msg)
            elif topic == "/odom":
                self._on_odom(t_sec, msg)
            elif topic == "/scan":
                self._on_scan(msg)
            elif topic == "/planner_exec_stats":
                self._on_planner_exec_stats(t_sec, msg)
            elif topic == "/active_waypoint_goal":
                self._on_goal(t_sec, msg)
            elif topic == "/exec_source":
                self._on_exec_source(msg)
            elif topic == "/navigate_to_pose/_action/status":
                self._on_action_status(msg)

        self._close_open_segments(t_end=t_end)
        self.bags_processed.append(str(bag_path))

    def build_summary(self) -> dict[str, Any]:
        total_state_time = sum(self.state_time_sec)
        state_ratio = {}
        for idx, t in enumerate(self.state_time_sec):
            key = _state_name(idx)
            state_ratio[key] = float(t / total_state_time) if total_state_time > 0 else None

        state_dwell_mean = {}
        for name in STATE_NAMES:
            vals = self.state_dwell_sec.get(name, [])
            state_dwell_mean[name] = float(sum(vals) / len(vals)) if vals else None

        speed_by_state = {}
        for name in list(STATE_NAMES) + ["unknown"]:
            speed_by_state[name] = _safe_stats(self.speed_cap_by_state.get(name, []))

        goal_by_state = {}
        for name in list(STATE_NAMES) + ["unknown"]:
            goal_by_state[name] = {
                "distance": _safe_stats(self.goal_dists_by_state.get(name, [])),
                "bearing_rad": _safe_stats(self.goal_bearings_by_state.get(name, [])),
            }

        top_goal_ids = Counter({k: len(v) for k, v in self.goal_id_durations_by_id.items()})
        top_goal_rows = []
        for gid, seg_count in top_goal_ids.most_common(10):
            durs = self.goal_id_durations_by_id[gid]
            top_goal_rows.append({"goal_id": gid, "segments": int(seg_count), "duration_sec": _safe_stats(durs)})

        key_topics = [
            "/cmd_vel_drl",
            "/cmd_vel",
            "/cmd_vel_nav",
            "/cmd_vel_nav2",
            "/exec_source",
            "/exec_source_hz",
            "/navigate_to_pose/_action/status",
            "/navigate_to_pose/_action/feedback",
        ]
        key_topic_counts = {tp: int(self.topic_msg_count.get(tp, 0)) for tp in key_topics}

        return {
            "bags_processed": self.bags_processed,
            "topic_message_count": {k: int(v) for k, v in sorted(self.topic_msg_count.items())},
            "nav2_exec_chain": {
                "key_topic_counts": key_topic_counts,
                "exec_source_counts": {k: int(v) for k, v in self.exec_source_counts.items()},
                "exec_source_switches": int(self.exec_source_switches),
                "action_status_counts": {k: int(v) for k, v in self.action_status_counts.items()},
                "action_status_code_counts": {k: int(v) for k, v in self.action_status_code_counts.items()},
            },
            "goal_distribution": {
                "distance": _safe_stats(self.goal_dists),
                "bearing_rad": _safe_stats(self.goal_bearings),
                "by_planner_state": goal_by_state,
                "unresolved_frame_count": int(self.goal_unresolved_frame_count),
            },
            "goal_id": {
                "duration_sec": _safe_stats(self.goal_id_durations),
                "top_goal_ids": top_goal_rows,
            },
            "goal_nan": {
                "segment_count": int(self.goal_nan_count),
                "duration_sec": _safe_stats(self.goal_nan_segments),
            },
            "planner_state": {
                "time_ratio": state_ratio,
                "dwell_mean_sec": state_dwell_mean,
                "dwell_sec": {name: _safe_stats(self.state_dwell_sec.get(name, [])) for name in STATE_NAMES},
                "transition_matrix": {
                    STATE_NAMES[r]: {STATE_NAMES[c]: int(self.state_transition[r][c]) for c in range(4)} for r in range(4)
                },
            },
            "speed_cap": {
                "all": _safe_stats(self.speed_cap_all),
                "by_planner_state": speed_by_state,
            },
            "planner_exec_stats": {
                "local_timeout": float(self.exec_counter["local_timeout"]),
                "no_progress": float(self.exec_counter["no_progress"]),
                "unsafe_clearance": float(self.exec_counter["unsafe_clearance"]),
                "fallback_count": float(self.exec_counter["fallback_count"]),
                "replan_count": float(self.exec_counter["replan_count"]),
                "timeout_context": {
                    "count": len(self.timeout_context_rows),
                    "goal_dist": _safe_stats([r["goal_dist"] for r in self.timeout_context_rows if r["goal_dist"] is not None]),
                    "progress_delta": _safe_stats(
                        [r["progress_delta"] for r in self.timeout_context_rows if r["progress_delta"] is not None]
                    ),
                    "goal_id_age": _safe_stats([r["goal_id_age"] for r in self.timeout_context_rows if r["goal_id_age"] is not None]),
                },
                "unsafe_clearance_context": {
                    "count": len(self.unsafe_context_rows),
                    "goal_dist": _safe_stats([r["goal_dist"] for r in self.unsafe_context_rows if r["goal_dist"] is not None]),
                    "min_lidar": _safe_stats([r["min_lidar"] for r in self.unsafe_context_rows if r["min_lidar"] is not None]),
                    "speed_cap": _safe_stats([r["speed_cap"] for r in self.unsafe_context_rows if r["speed_cap"] is not None]),
                },
            },
            "warnings": self.topic_type_warnings,
        }

    def dump_csvs(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)

        trans_csv = out_dir / "state_transition_matrix.csv"
        with trans_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["from_state", "to_state", "count"])
            for i, from_name in enumerate(STATE_NAMES):
                for j, to_name in enumerate(STATE_NAMES):
                    w.writerow([from_name, to_name, int(self.state_transition[i][j])])

        speed_csv = out_dir / "speed_cap_by_state.csv"
        with speed_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["state", "n", "mean", "p10", "p50", "p90", "min", "max"])
            for state in list(STATE_NAMES) + ["unknown"]:
                stats = _safe_stats(self.speed_cap_by_state.get(state, []))
                w.writerow([state, stats["n"], stats["mean"], stats["p10"], stats["p50"], stats["p90"], stats["min"], stats["max"]])

        gid_csv = out_dir / "goal_id_dwell.csv"
        with gid_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["goal_id", "duration_sec"])
            for gid, durs in sorted(self.goal_id_durations_by_id.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                for d in durs:
                    w.writerow([gid, float(d)])

        timeout_csv = out_dir / "timeout_context.csv"
        with timeout_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_sec", "planner_state", "goal_dist", "progress_delta", "min_lidar", "speed_cap", "goal_id_age"])
            for row in self.timeout_context_rows:
                w.writerow(
                    [
                        row.get("t_sec"),
                        row.get("planner_state"),
                        row.get("goal_dist"),
                        row.get("progress_delta"),
                        row.get("min_lidar"),
                        row.get("speed_cap"),
                        row.get("goal_id_age"),
                    ]
                )

        unsafe_csv = out_dir / "unsafe_clearance_context.csv"
        with unsafe_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_sec", "planner_state", "goal_dist", "progress_delta", "min_lidar", "speed_cap", "goal_id_age"])
            for row in self.unsafe_context_rows:
                w.writerow(
                    [
                        row.get("t_sec"),
                        row.get("planner_state"),
                        row.get("goal_dist"),
                        row.get("progress_delta"),
                        row.get("min_lidar"),
                        row.get("speed_cap"),
                        row.get("goal_id_age"),
                    ]
                )

        topic_csv = out_dir / "topic_message_count.csv"
        with topic_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["topic", "count"])
            for topic, count in sorted(self.topic_msg_count.items()):
                w.writerow([topic, int(count)])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze planner-related ROS2 bags for relay-goal training calibration.")
    p.add_argument("--bag", action="append", required=True, help="Path to a rosbag directory. Repeatable.")
    p.add_argument("--out-dir", required=True, help="Output directory for summary JSON and CSV tables.")
    p.add_argument("--label", type=str, default="", help="Optional dataset label (e.g., topo_nav2_ref_seed42).")
    return p.parse_args()


def main():
    args = _parse_args()
    bags = [Path(b).expanduser().resolve() for b in args.bag]
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    analyzer = PlannerBagAnalyzer()
    for bag in bags:
        if not bag.exists() or not bag.is_dir():
            raise FileNotFoundError(f"Bag path not found or not a directory: {bag}")
        analyzer.process_bag(bag)

    summary = analyzer.build_summary()
    if args.label:
        summary["label"] = args.label

    summary_path = out_dir / "planner_bag_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.flush()

    analyzer.dump_csvs(out_dir)

    print(f"[ANALYZE] bags={len(bags)} label={args.label or '-'}")
    print(f"[ANALYZE] summary: {summary_path}")
    print(f"[ANALYZE] csv: {out_dir / 'state_transition_matrix.csv'}")
    print(f"[ANALYZE] csv: {out_dir / 'speed_cap_by_state.csv'}")
    print(f"[ANALYZE] csv: {out_dir / 'goal_id_dwell.csv'}")
    print(f"[ANALYZE] csv: {out_dir / 'timeout_context.csv'}")
    print(f"[ANALYZE] csv: {out_dir / 'unsafe_clearance_context.csv'}")
    print(f"[ANALYZE] csv: {out_dir / 'topic_message_count.csv'}")


if __name__ == "__main__":
    main()
