#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import rclpy
from action_msgs.msg import GoalStatusArray
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from rclpy.node import Node
from std_msgs.msg import String


STATUS_NAMES = {
    0: "UNKNOWN",
    1: "ACCEPTED",
    2: "EXECUTING",
    3: "CANCELING",
    4: "SUCCEEDED",
    5: "CANCELED",
    6: "ABORTED",
}
RECOVERY_STATES = {"fallback", "return"}


@dataclass
class RecoveryWindow:
    state: str
    start_time: float
    goal_status_msgs: int = 0
    feedback_msgs: int = 0
    nav2_cmd_msgs: int = 0
    nav2_cmd_nonzero_msgs: int = 0
    nav2_max_lin: float = 0.0
    nav2_max_ang: float = 0.0
    exec_nav2_seen: bool = False
    nav2_age_positive_seen: bool = False


class RecoveryMonitor(Node):
    def __init__(self, log_path: Optional[Path], summary_period: float):
        super().__init__("recovery_monitor")
        self.log_path = log_path
        self.summary_period = float(summary_period)
        self._log_fp = None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fp = self.log_path.open("a", encoding="utf-8")

        self.current_state = "unknown"
        self.current_exec_source = "unknown"
        self.last_exec_hz_raw = ""
        self.last_nav2_age = -1.0
        self.last_drl_age = -1.0
        self.last_goal_status = "NONE"
        self.last_goal_status_count = 0
        self.last_nav2_cmd_nonzero = False
        self.window: Optional[RecoveryWindow] = None

        self.create_subscription(String, "/planner_state", self._on_planner_state, 10)
        self.create_subscription(String, "/exec_source_hz", self._on_exec_source_hz, 10)
        self.create_subscription(GoalStatusArray, "/navigate_to_pose/_action/status", self._on_nav_status, 10)
        self.create_subscription(NavigateToPose.Impl.FeedbackMessage, "/navigate_to_pose/_action/feedback", self._on_nav_feedback, 10)
        self.create_subscription(Twist, "/cmd_vel_nav2", self._on_nav2_cmd, 20)

        if self.summary_period > 0.0:
            self.create_timer(self.summary_period, self._on_summary_timer)

        self._log("monitor_start", f"watching recovery states {sorted(RECOVERY_STATES)}")

    def destroy_node(self):
        if self.window is not None:
            self._close_window("shutdown")
        if self._log_fp is not None:
            self._log_fp.flush()
            self._log_fp.close()
        super().destroy_node()

    def _stamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def _log(self, tag: str, message: str):
        line = f"[{self._stamp()}] {tag}: {message}"
        print(line, flush=True)
        if self._log_fp is not None:
            self._log_fp.write(line + "\n")
            self._log_fp.flush()

    def _start_window(self, state: str):
        now = self.get_clock().now().nanoseconds / 1e9
        self.window = RecoveryWindow(state=state, start_time=now)
        self._log("recovery_start", f"state={state}")

    def _close_window(self, reason: str):
        if self.window is None:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        dur = max(0.0, now - self.window.start_time)
        self._log(
            "recovery_end",
            (
                f"state={self.window.state} dur={dur:.2f}s reason={reason} "
                f"exec_nav2={self.window.exec_nav2_seen} nav2_age_pos={self.window.nav2_age_positive_seen} "
                f"status_msgs={self.window.goal_status_msgs} feedback_msgs={self.window.feedback_msgs} "
                f"nav2_cmd_msgs={self.window.nav2_cmd_msgs} nav2_cmd_nonzero={self.window.nav2_cmd_nonzero_msgs} "
                f"nav2_max_lin={self.window.nav2_max_lin:.3f} nav2_max_ang={self.window.nav2_max_ang:.3f}"
            ),
        )
        self.window = None

    def _on_planner_state(self, msg: String):
        state = (msg.data or "").strip().lower()
        if state != self.current_state:
            prev = self.current_state
            self.current_state = state
            self._log("planner_state", f"{prev} -> {state}")

            if state in RECOVERY_STATES and self.window is None:
                self._start_window(state)
            elif state not in RECOVERY_STATES and self.window is not None:
                self._close_window(f"state_exit->{state}")

    def _on_exec_source_hz(self, msg: String):
        raw = msg.data or ""
        src = self.current_exec_source
        nav2_age = self.last_nav2_age
        drl_age = self.last_drl_age

        parts = {}
        for token in raw.split():
            if "=" in token:
                key, value = token.split("=", 1)
                parts[key.strip()] = value.strip()
        src = parts.get("src", src)

        try:
            nav2_age = float(parts.get("nav2_age", str(nav2_age)).rstrip("s"))
        except ValueError:
            pass
        try:
            drl_age = float(parts.get("drl_age", str(drl_age)).rstrip("s"))
        except ValueError:
            pass

        changed = (src != self.current_exec_source) or (raw != self.last_exec_hz_raw and self.current_state in RECOVERY_STATES)
        self.current_exec_source = src
        self.last_exec_hz_raw = raw
        self.last_nav2_age = nav2_age
        self.last_drl_age = drl_age

        if self.window is not None:
            if src == "nav2":
                self.window.exec_nav2_seen = True
            if nav2_age >= 0.0:
                self.window.nav2_age_positive_seen = True

        if changed:
            self._log("exec_source", raw)

    def _on_nav_status(self, msg: GoalStatusArray):
        count = len(msg.status_list)
        if count == 0:
            status_name = "EMPTY"
        else:
            status = msg.status_list[-1].status
            status_name = STATUS_NAMES.get(status, str(status))
        if self.window is not None:
            self.window.goal_status_msgs += 1
        if status_name != self.last_goal_status or count != self.last_goal_status_count:
            self.last_goal_status = status_name
            self.last_goal_status_count = count
            self._log("nav2_status", f"status={status_name} count={count}")

    def _on_nav_feedback(self, msg: NavigateToPose.Impl.FeedbackMessage):
        if self.window is not None:
            self.window.feedback_msgs += 1
            feedback = msg.feedback
            try:
                dist = feedback.distance_remaining
            except AttributeError:
                dist = float("nan")
            if math.isfinite(dist):
                self._log("nav2_feedback", f"distance_remaining={dist:.3f}")
            else:
                self._log("nav2_feedback", "feedback")

    def _on_nav2_cmd(self, msg: Twist):
        lin = float(msg.linear.x)
        ang = float(msg.angular.z)
        nonzero = abs(lin) > 1e-4 or abs(ang) > 1e-4
        if self.window is not None:
            self.window.nav2_cmd_msgs += 1
            self.window.nav2_max_lin = max(self.window.nav2_max_lin, abs(lin))
            self.window.nav2_max_ang = max(self.window.nav2_max_ang, abs(ang))
            if nonzero:
                self.window.nav2_cmd_nonzero_msgs += 1
        if nonzero and not self.last_nav2_cmd_nonzero:
            self._log("nav2_cmd", f"ACTIVE lin={lin:.3f} ang={ang:.3f}")
        elif (not nonzero) and self.last_nav2_cmd_nonzero:
            self._log("nav2_cmd", "IDLE")
        self.last_nav2_cmd_nonzero = nonzero

    def _on_summary_timer(self):
        summary = (
            f"state={self.current_state} src={self.current_exec_source} "
            f"drl_age={self.last_drl_age:.2f}s nav2_age={self.last_nav2_age:.2f}s "
            f"nav_status={self.last_goal_status}"
        )
        if self.window is not None:
            summary += (
                f" | recovery_window={self.window.state} "
                f"nav2_cmd_nonzero={self.window.nav2_cmd_nonzero_msgs} "
                f"status_msgs={self.window.goal_status_msgs} feedback_msgs={self.window.feedback_msgs}"
            )
        self._log("summary", summary)


def main():
    parser = argparse.ArgumentParser(description="Monitor fallback/return -> Nav2 recovery chain")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional file to append monitor output")
    parser.add_argument("--summary-period", type=float, default=2.0, help="Seconds between periodic summaries")
    args = parser.parse_args()

    rclpy.init()
    node = RecoveryMonitor(log_path=args.log_file, summary_period=args.summary_period)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
