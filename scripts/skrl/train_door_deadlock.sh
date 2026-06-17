#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

NUM_ENVS=100
MAX_ITERATIONS=5000
HEADLESS=1
WARM_START=1
CHECKPOINT=""
DOOR_WIDTH=""
DOOR_WIDTH_JITTER=""
CORRIDOR_WIDTH=""
HALF_LENGTH=""
STRAIGHT_DISTANCE=""
LINE_ANGLE_DEG=""
GOAL_LATERAL_OFFSET_MAX=""
GOAL_DISTANCE_MIN=""
GOAL_DISTANCE_MAX=""
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/skrl/train_door_deadlock.sh [options] [extra launch/train args]

Options:
  --num-envs N             Number of parallel environments (default: 100)
  --max-iterations N       skrl max_iterations value (default: 5000)
  --checkpoint PATH        Resume or warm-start from this checkpoint
  --no-warm-start          Start from scratch instead of auto-loading latest v18a checkpoint
  --gui                    Disable --headless
  --door-width M           Override defect_door_width
  --door-width-jitter M    Override defect_door_width_jitter
  --corridor-width M       Override defect_door_corridor_width
  --half-length M          Override defect_door_half_length
  --straight-distance M    Override defect_door_straight_distance
  --line-angle-deg M       Override defect_door_line_angle_max_deg
  --goal-lateral-offset-max M
                          Override defect_door_goal_lateral_offset_max
  --goal-distance-min M    Override defect_goal_distance_min
  --goal-distance-max M    Override defect_goal_distance_max
  -h, --help               Show this help

Examples:
  scripts/skrl/train_door_deadlock.sh
  scripts/skrl/train_door_deadlock.sh --door-width 0.72 --corridor-width 1.8
  scripts/skrl/train_door_deadlock.sh --no-warm-start --seed 7
EOF
}

find_latest_v18a_checkpoint() {
  local ckpt_root="$ROOT/logs/skrl/lidar_short_nav_v18a_direct"
  local ckpt=""

  ckpt="$(ls -1dt "$ckpt_root"/*/checkpoints/best_agent.pt 2>/dev/null | head -n 1 || true)"
  if [[ -n "$ckpt" && -f "$ckpt" ]]; then
    printf '%s\n' "$ckpt"
    return 0
  fi

  ckpt="$(ls -1dt "$ckpt_root"/*/checkpoints/agent_*.pt 2>/dev/null | head -n 1 || true)"
  if [[ -n "$ckpt" && -f "$ckpt" ]]; then
    printf '%s\n' "$ckpt"
    return 0
  fi

  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-envs)
      NUM_ENVS="$2"
      shift 2
      ;;
    --max-iterations)
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --no-warm-start)
      WARM_START=0
      shift
      ;;
    --gui)
      HEADLESS=0
      shift
      ;;
    --door-width)
      DOOR_WIDTH="$2"
      shift 2
      ;;
    --door-width-jitter)
      DOOR_WIDTH_JITTER="$2"
      shift 2
      ;;
    --corridor-width)
      CORRIDOR_WIDTH="$2"
      shift 2
      ;;
    --half-length)
      HALF_LENGTH="$2"
      shift 2
      ;;
    --straight-distance)
      STRAIGHT_DISTANCE="$2"
      shift 2
      ;;
    --line-angle-deg)
      LINE_ANGLE_DEG="$2"
      shift 2
      ;;
    --goal-lateral-offset-max)
      GOAL_LATERAL_OFFSET_MAX="$2"
      shift 2
      ;;
    --goal-distance-min)
      GOAL_DISTANCE_MIN="$2"
      shift 2
      ;;
    --goal-distance-max)
      GOAL_DISTANCE_MAX="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$CHECKPOINT" && "$WARM_START" -eq 1 ]]; then
  CHECKPOINT="$(find_latest_v18a_checkpoint || true)"
fi

CMD=(
  "$ROOT/launch.sh"
  train
  --robot turtlebot3_burger
  --mode door_deadlock
  --algorithm PPO
  --num_envs "$NUM_ENVS"
  --max_iterations "$MAX_ITERATIONS"
)

if [[ "$HEADLESS" -eq 1 ]]; then
  CMD+=(--headless)
fi
if [[ -n "$CHECKPOINT" ]]; then
  echo "[door-deadlock] checkpoint=$CHECKPOINT"
  CMD+=(--checkpoint "$CHECKPOINT")
else
  echo "[door-deadlock] checkpoint=<scratch>"
fi
if [[ -n "$DOOR_WIDTH" ]]; then
  CMD+=(--defect_door_width "$DOOR_WIDTH")
fi
if [[ -n "$DOOR_WIDTH_JITTER" ]]; then
  CMD+=(--defect_door_width_jitter "$DOOR_WIDTH_JITTER")
fi
if [[ -n "$CORRIDOR_WIDTH" ]]; then
  CMD+=(--defect_door_corridor_width "$CORRIDOR_WIDTH")
fi
if [[ -n "$HALF_LENGTH" ]]; then
  CMD+=(--defect_door_half_length "$HALF_LENGTH")
fi
if [[ -n "$STRAIGHT_DISTANCE" ]]; then
  CMD+=(--defect_door_straight_distance "$STRAIGHT_DISTANCE")
fi
if [[ -n "$LINE_ANGLE_DEG" ]]; then
  CMD+=(--defect_door_line_angle_max_deg "$LINE_ANGLE_DEG")
fi
if [[ -n "$GOAL_LATERAL_OFFSET_MAX" ]]; then
  CMD+=(--defect_door_goal_lateral_offset_max "$GOAL_LATERAL_OFFSET_MAX")
fi
if [[ -n "$GOAL_DISTANCE_MIN" ]]; then
  CMD+=(--defect_goal_distance_min "$GOAL_DISTANCE_MIN")
fi
if [[ -n "$GOAL_DISTANCE_MAX" ]]; then
  CMD+=(--defect_goal_distance_max "$GOAL_DISTANCE_MAX")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[door-deadlock] num_envs=$NUM_ENVS max_iterations=$MAX_ITERATIONS headless=$HEADLESS"
printf '[door-deadlock] cmd:'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
