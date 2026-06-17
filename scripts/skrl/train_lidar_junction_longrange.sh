#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_SCRIPT="${ROOT_DIR}/scripts/skrl/train.py"

ISAACLAB_SH="${ISAACLAB_SH:-$HOME/IsaacLab/isaaclab.sh}"
TASK="${TASK:-Isaac-Lab-Tutorial-LidarJunction-TurtleBot3-Direct-v0}"
ALGORITHM="PPO"

NUM_ENVS=100
MAX_ITER=12000
HEADLESS=1
DEBUG_PRINT_ENABLE=1
DEBUG_PRINT_INTERVAL_STEPS=200
DEBUG_PHASE_TAG=40

# Long-range single-shot defaults
EPISODE_LENGTH_S=20
GOAL_MAX_DISTANCE=10.0
GOAL_SPAWN_RANGE_MIN=3.0
GOAL_SPAWN_RANGE_MAX=8.0
JUNCTION_HALF_WIDTH=1.20
JUNCTION_HALF_LENGTH=18.0
JUNCTION_GOAL_FORWARD_MIN=3.0
JUNCTION_GOAL_FORWARD_MAX=8.0
JUNCTION_ROBOT_SPAWN_OFFSET_MIN=4.6
JUNCTION_ROBOT_SPAWN_OFFSET_MAX=5.4
JUNCTION_ROBOT_SPAWN_LATERAL_JITTER=0.03
JUNCTION_X_SAMPLING_PROB=0.05
NUM_OBSTACLES=2
GOAL_REACH_THRESHOLD=0.50

# Keep deployment-consistent action limits unless user overrides.
V_MAX=1.0
OMEGA_LIMIT=6.0
TAU_V=0.10
TAU_OMEGA=0.05
WHEEL_LINEAR_SPEED_MAX=1.0
FORWARD_ONLY=1

CKPT_ROOT="${ROOT_DIR}/logs/skrl/lidar_junction_direct"
CHECKPOINT="${CHECKPOINT:-}"
USE_LATEST_CHECKPOINT=1

print_usage() {
  cat <<'EOF'
Usage:
  scripts/skrl/train_lidar_junction_longrange.sh [options]

Options:
  --num_envs N
  --max_iterations N
  --headless | --no-headless
  --checkpoint PATH              Resume from this checkpoint
  --no-latest-checkpoint         Start from scratch unless --checkpoint is provided

  --episode_length_s S
  --goal_max_distance M
  --goal_spawn_range_min M
  --goal_spawn_range_max M
  --junction_half_width M
  --junction_half_length M
  --junction_goal_forward_min M
  --junction_goal_forward_max M
  --junction_x_sampling_prob P
  --num_obstacles N
  --v_max MPS
  --omega_limit RADPS
  --goal_reach_threshold M
  --tau_v S
  --tau_omega S
  --wheel_linear_speed_max MPS

  --help

Defaults are tuned for long-range single-shot navigation retraining.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num_envs) NUM_ENVS="$2"; shift 2 ;;
    --max_iterations) MAX_ITER="$2"; shift 2 ;;
    --headless) HEADLESS=1; shift ;;
    --no-headless) HEADLESS=0; shift ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --no-latest-checkpoint) USE_LATEST_CHECKPOINT=0; shift ;;
    --episode_length_s) EPISODE_LENGTH_S="$2"; shift 2 ;;
    --goal_max_distance) GOAL_MAX_DISTANCE="$2"; shift 2 ;;
    --goal_spawn_range_min) GOAL_SPAWN_RANGE_MIN="$2"; shift 2 ;;
    --goal_spawn_range_max) GOAL_SPAWN_RANGE_MAX="$2"; shift 2 ;;
    --junction_half_width) JUNCTION_HALF_WIDTH="$2"; shift 2 ;;
    --junction_half_length) JUNCTION_HALF_LENGTH="$2"; shift 2 ;;
    --junction_goal_forward_min) JUNCTION_GOAL_FORWARD_MIN="$2"; shift 2 ;;
    --junction_goal_forward_max) JUNCTION_GOAL_FORWARD_MAX="$2"; shift 2 ;;
    --junction_x_sampling_prob) JUNCTION_X_SAMPLING_PROB="$2"; shift 2 ;;
    --num_obstacles) NUM_OBSTACLES="$2"; shift 2 ;;
    --v_max) V_MAX="$2"; shift 2 ;;
    --omega_limit) OMEGA_LIMIT="$2"; shift 2 ;;
    --goal_reach_threshold) GOAL_REACH_THRESHOLD="$2"; shift 2 ;;
    --tau_v) TAU_V="$2"; shift 2 ;;
    --tau_omega) TAU_OMEGA="$2"; shift 2 ;;
    --wheel_linear_speed_max) WHEEL_LINEAR_SPEED_MAX="$2"; shift 2 ;;
    --help|-h) print_usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; print_usage; exit 1 ;;
  esac
done

if [[ ! -x "${ISAACLAB_SH}" ]]; then
  echo "[ERROR] isaaclab.sh not executable: ${ISAACLAB_SH}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "[ERROR] train script not found: ${TRAIN_SCRIPT}" >&2
  exit 1
fi

latest_ckpt() {
  local run_dir ckpt
  while IFS= read -r run_dir; do
    ckpt="${run_dir}/checkpoints/best_agent.pt"
    if [[ -f "${ckpt}" ]]; then
      echo "${ckpt}"
      return 0
    fi
  done < <(ls -1dt "${CKPT_ROOT}"/*_ppo_torch 2>/dev/null || true)
  return 1
}

if [[ -z "${CHECKPOINT}" && "${USE_LATEST_CHECKPOINT}" == "1" ]]; then
  if CHECKPOINT="$(latest_ckpt)"; then
    echo "[INFO] Resuming from latest checkpoint: ${CHECKPOINT}"
  else
    echo "[WARN] No latest checkpoint found under ${CKPT_ROOT}; training from scratch."
    CHECKPOINT=""
  fi
fi

CMD=(
  env PYTHONNOUSERSITE=1 PYTHONPATH=
  "${ISAACLAB_SH}" -p "${TRAIN_SCRIPT}"
  --task "${TASK}"
  --algorithm "${ALGORITHM}"
  --num_envs "${NUM_ENVS}"
  --max_iterations "${MAX_ITER}"
  --debug_print_enable "${DEBUG_PRINT_ENABLE}"
  --debug_print_interval_steps "${DEBUG_PRINT_INTERVAL_STEPS}"
  --debug_phase_tag "${DEBUG_PHASE_TAG}"
  --episode_length_s "${EPISODE_LENGTH_S}"
  --goal_max_distance "${GOAL_MAX_DISTANCE}"
  --goal_spawn_range_min "${GOAL_SPAWN_RANGE_MIN}"
  --goal_spawn_range_max "${GOAL_SPAWN_RANGE_MAX}"
  --junction_half_width "${JUNCTION_HALF_WIDTH}"
  --junction_half_length "${JUNCTION_HALF_LENGTH}"
  --junction_goal_forward_min "${JUNCTION_GOAL_FORWARD_MIN}"
  --junction_goal_forward_max "${JUNCTION_GOAL_FORWARD_MAX}"
  --junction_robot_spawn_offset_min "${JUNCTION_ROBOT_SPAWN_OFFSET_MIN}"
  --junction_robot_spawn_offset_max "${JUNCTION_ROBOT_SPAWN_OFFSET_MAX}"
  --junction_robot_spawn_lateral_jitter "${JUNCTION_ROBOT_SPAWN_LATERAL_JITTER}"
  --junction_x_sampling_prob "${JUNCTION_X_SAMPLING_PROB}"
  --num_obstacles "${NUM_OBSTACLES}"
  --goal_reach_threshold "${GOAL_REACH_THRESHOLD}"
  --v_max "${V_MAX}"
  --omega_limit "${OMEGA_LIMIT}"
  --tau_v "${TAU_V}"
  --tau_omega "${TAU_OMEGA}"
  --wheel_linear_speed_max "${WHEEL_LINEAR_SPEED_MAX}"
  --forward_only "${FORWARD_ONLY}"
)

if [[ "${HEADLESS}" == "1" ]]; then
  CMD+=(--headless)
fi
if [[ -n "${CHECKPOINT}" ]]; then
  CMD+=(--checkpoint "${CHECKPOINT}")
fi

echo "[RUN] long-range retraining"
echo "[RUN] max_iter=${MAX_ITER} num_envs=${NUM_ENVS} episode_length_s=${EPISODE_LENGTH_S}"
echo "[RUN] goal_max_distance=${GOAL_MAX_DISTANCE} junction_goal_forward=[${JUNCTION_GOAL_FORWARD_MIN}, ${JUNCTION_GOAL_FORWARD_MAX}]"
echo "[RUN] junction_size=(${JUNCTION_HALF_WIDTH}, ${JUNCTION_HALF_LENGTH}) x_prob=${JUNCTION_X_SAMPLING_PROB} obstacles=${NUM_OBSTACLES}"
echo "[RUN] action_limits v_max=${V_MAX} omega_limit=${OMEGA_LIMIT} goal_reach_threshold=${GOAL_REACH_THRESHOLD}"

(cd "${ROOT_DIR}" && "${CMD[@]}")
