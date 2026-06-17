#!/usr/bin/env bash

set -euo pipefail

GOAL_X="-4.5"
GOAL_Y="-4.0"
FRAME_ID="map"
MONITOR_SEC=25
OUT_DIR="/home/cs/isaaclab_jetbot/logs/nav2_online_probe"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_nav2_online_probe.sh [options]

Options:
  --goal-x X          goal x in map frame (default: -4.5)
  --goal-y Y          goal y in map frame (default: -4.0)
  --frame-id FRAME    goal frame id (default: map)
  --monitor-sec N     monitor seconds (default: 25)
  --out-dir DIR       output directory root (default: /home/cs/isaaclab_jetbot/logs/nav2_online_probe)
  -h, --help          show help

This script verifies the pure Nav2 execution chain online:
1) send a nearby NavigateToPose goal
2) monitor /cmd_vel_nav and /cmd_vel_nav2 rates
3) monitor /exec_source_hz and hidden action status topic
4) print pass/fail gates
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --goal-x)
      GOAL_X="$2"
      shift 2
      ;;
    --goal-y)
      GOAL_Y="$2"
      shift 2
      ;;
    --frame-id)
      FRAME_ID="$2"
      shift 2
      ;;
    --monitor-sec)
      MONITOR_SEC="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown arg: $1"
      usage
      exit 2
      ;;
  esac
done

if ! [[ "$MONITOR_SEC" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] --monitor-sec must be integer"
  exit 2
fi

stamp="$(date +%Y%m%d_%H%M%S)"
run_dir="${OUT_DIR}/probe_${stamp}"
mkdir -p "$run_dir"

for cmd in ros2 timeout awk grep; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[ERROR] Required command not found: $cmd"
    exit 2
  fi
done

echo "[INFO] run_dir: $run_dir"
echo "[INFO] goal: frame=${FRAME_ID}, x=${GOAL_X}, y=${GOAL_Y}"
echo "[INFO] monitor_sec: ${MONITOR_SEC}"

echo "[INFO] Action server check:"
ros2 action list 2>/dev/null | grep -E '^/navigate_to_pose$' || echo "  (not listed yet)"

HZ_NAV_LOG="${run_dir}/hz_cmd_vel_nav.log"
HZ_NAV2_LOG="${run_dir}/hz_cmd_vel_nav2.log"
EXEC_HZ_LOG="${run_dir}/exec_source_hz.log"
STATUS_LOG="${run_dir}/nav2_action_status.log"
GOAL_LOG="${run_dir}/navigate_to_pose_goal.log"

timeout "${MONITOR_SEC}" ros2 topic hz /cmd_vel_nav --window 20 >"${HZ_NAV_LOG}" 2>&1 &
PID_HZ_NAV=$!
timeout "${MONITOR_SEC}" ros2 topic hz /cmd_vel_nav2 --window 20 >"${HZ_NAV2_LOG}" 2>&1 &
PID_HZ_NAV2=$!
timeout "${MONITOR_SEC}" ros2 topic echo /exec_source_hz >"${EXEC_HZ_LOG}" 2>&1 &
PID_EXEC_HZ=$!
timeout "${MONITOR_SEC}" ros2 topic echo --include-hidden-topics /navigate_to_pose/_action/status >"${STATUS_LOG}" 2>&1 &
PID_STATUS=$!

GOAL_PAYLOAD="{pose: {header: {frame_id: '${FRAME_ID}'}, pose: {position: {x: ${GOAL_X}, y: ${GOAL_Y}, z: 0.0}, orientation: {w: 1.0}}}}"
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "${GOAL_PAYLOAD}" --feedback >"${GOAL_LOG}" 2>&1 &
PID_GOAL=$!

sleep "${MONITOR_SEC}"

for pid in "$PID_HZ_NAV" "$PID_HZ_NAV2" "$PID_EXEC_HZ" "$PID_STATUS"; do
  kill "$pid" 2>/dev/null || true
done
kill "$PID_GOAL" 2>/dev/null || true
wait "$PID_HZ_NAV" "$PID_HZ_NAV2" "$PID_EXEC_HZ" "$PID_STATUS" 2>/dev/null || true
wait "$PID_GOAL" 2>/dev/null || true

parse_rate() {
  local file="$1"
  local rate
  rate="$(awk '/average rate:/{r=$3} END{if (r=="") print "0"; else print r}' "$file" 2>/dev/null || echo "0")"
  echo "$rate"
}

RATE_NAV="$(parse_rate "$HZ_NAV_LOG")"
RATE_NAV2="$(parse_rate "$HZ_NAV2_LOG")"

GOAL_ACCEPTED=0
if grep -qi "Goal accepted" "$GOAL_LOG"; then
  GOAL_ACCEPTED=1
fi

STATUS_EXECUTING=0
if grep -Eqi 'status:[[:space:]]*2\b|executing' "$STATUS_LOG"; then
  STATUS_EXECUTING=1
fi

echo
echo "=== Nav2 Online Probe Result ==="
echo "goal_accepted=${GOAL_ACCEPTED}"
echo "status_executing_seen=${STATUS_EXECUTING}"
echo "cmd_vel_nav_hz=${RATE_NAV}"
echo "cmd_vel_nav2_hz=${RATE_NAV2}"
echo "logs=${run_dir}"

PASS=1
if [[ "$GOAL_ACCEPTED" -ne 1 ]]; then
  PASS=0
  echo "[FAIL] Goal not accepted by /navigate_to_pose"
fi
if [[ "$STATUS_EXECUTING" -ne 1 ]]; then
  PASS=0
  echo "[FAIL] action status did not show executing"
fi
awk -v r="$RATE_NAV" 'BEGIN{exit !(r+0>0.01)}' || { PASS=0; echo "[FAIL] /cmd_vel_nav has no sustained output"; }
awk -v r="$RATE_NAV2" 'BEGIN{exit !(r+0>0.01)}' || { PASS=0; echo "[FAIL] /cmd_vel_nav2 has no sustained output"; }

if [[ "$PASS" -eq 1 ]]; then
  echo "[PASS] Nav2 velocity chain is active"
else
  echo "[NEXT] If /cmd_vel_nav2==0 while /cmd_vel_nav>0, check velocity_smoother remap and node wiring."
fi
