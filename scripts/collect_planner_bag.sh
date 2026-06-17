#!/usr/bin/env bash

set -euo pipefail

MODE="reference"                # reference | failure
DURATION_MIN=20                 # recommended: 15~20
SEED_TAG="seed0"
POSE_TAG=""
OUT_ROOT="/home/cs/isaaclab_jetbot/logs/rosbags"
NO_TIMEOUT=0
INCLUDE_HIDDEN=1
INCLUDE_UNPUBLISHED=1

EXTRA_TOPICS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/collect_planner_bag.sh [options]

Options:
  --mode MODE               reference | failure (default: reference)
  --duration-min N          recording duration in minutes (default: 20)
  --seed-tag TAG            run seed/initial-pose tag (default: seed0)
  --pose-tag TAG            optional initial pose tag
  --out-root DIR            bag output root dir (default: /home/cs/isaaclab_jetbot/logs/rosbags)
  --extra-topic TOPIC       append extra topic to rosbag record (repeatable)
  --no-timeout              do not auto-stop (manual Ctrl+C)
  --no-hidden-topics        do not pass --include-hidden-topics
  --no-unpublished-topics   do not pass --include-unpublished-topics
  -h, --help                show this help

Examples:
  scripts/collect_planner_bag.sh --mode reference --seed-tag seed42 --duration-min 18
  scripts/collect_planner_bag.sh --mode failure --seed-tag seed17 --pose-tag spawnB --duration-min 15
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --duration-min)
      DURATION_MIN="$2"
      shift 2
      ;;
    --seed-tag)
      SEED_TAG="$2"
      shift 2
      ;;
    --pose-tag)
      POSE_TAG="$2"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --extra-topic)
      EXTRA_TOPICS+=("$2")
      shift 2
      ;;
    --no-timeout)
      NO_TIMEOUT=1
      shift 1
      ;;
    --no-hidden-topics)
      INCLUDE_HIDDEN=0
      shift 1
      ;;
    --no-unpublished-topics)
      INCLUDE_UNPUBLISHED=0
      shift 1
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

if [[ "$MODE" != "reference" && "$MODE" != "failure" ]]; then
  echo "[ERROR] --mode must be reference or failure"
  exit 2
fi

if ! [[ "$DURATION_MIN" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] --duration-min must be integer minutes"
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
run_tag="${MODE}_${SEED_TAG}"
if [[ -n "$POSE_TAG" ]]; then
  run_tag="${run_tag}_${POSE_TAG}"
fi
run_tag="${run_tag}_${timestamp}"

bag_dir="${OUT_ROOT}/${run_tag}"
mkdir -p "$OUT_ROOT"

TOPICS=(
  /active_waypoint_goal
  /planner_state
  /speed_cap
  /goal_id
  /turn_cmd
  /planner_exec_stats
  /cmd_vel_drl
  /cmd_vel_nav
  /cmd_vel_nav2
  /cmd_vel
  /exec_source
  /exec_source_hz
  /navigate_to_pose/_action/status
  /navigate_to_pose/_action/feedback
  /odom
  /imu
  /scan
  /tf
  /tf_static
)

echo "[INFO] Collection mode: $MODE"
echo "[INFO] Output bag: $bag_dir"
echo "[INFO] Duration: ${DURATION_MIN} min"
echo "[INFO] Seed tag: $SEED_TAG"
if [[ -n "$POSE_TAG" ]]; then
  echo "[INFO] Pose tag: $POSE_TAG"
fi
echo "[INFO] include_hidden_topics: $INCLUDE_HIDDEN"
echo "[INFO] include_unpublished_topics: $INCLUDE_UNPUBLISHED"
echo "[INFO] Topics (${#TOPICS[@]} + ${#EXTRA_TOPICS[@]} extra):"
printf '  %s\n' "${TOPICS[@]}"
if [[ ${#EXTRA_TOPICS[@]} -gt 0 ]]; then
  printf '  %s\n' "${EXTRA_TOPICS[@]}"
fi
echo
echo "[INFO] Recommended launch before recording:"
if [[ "$MODE" == "reference" ]]; then
  echo "  ros2 launch wpr_simulation2 bringup_sim.launch.py execution_mode:=nav2 robot_sensor:=plain_lidar"
else
  echo "  ros2 launch wpr_simulation2 bringup_sim.launch.py execution_mode:=drl robot_sensor:=plain_lidar rl_checkpoint:=<policy.onnx>"
fi
echo

RECORD_CMD=(ros2 bag record -o "$bag_dir")
if [[ "$INCLUDE_HIDDEN" -eq 1 ]]; then
  RECORD_CMD+=(--include-hidden-topics)
fi
if [[ "$INCLUDE_UNPUBLISHED" -eq 1 ]]; then
  RECORD_CMD+=(--include-unpublished-topics)
fi
RECORD_CMD+=("${TOPICS[@]}" "${EXTRA_TOPICS[@]}")

if [[ "$NO_TIMEOUT" -eq 1 ]]; then
  echo "[INFO] Starting ros2 bag record (manual stop: Ctrl+C)..."
  "${RECORD_CMD[@]}"
else
  duration_sec=$((DURATION_MIN * 60))
  echo "[INFO] Starting ros2 bag record with auto-stop after ${duration_sec}s..."
  timeout --signal=INT "${duration_sec}" "${RECORD_CMD[@]}"
fi

echo "[INFO] Done. Bag saved at: $bag_dir"
