#!/usr/bin/env bash
set -euo pipefail

# Train suite for TurtleBot3 high-speed + long-range:
#   Stage 1: LidarNav (far distance, high speed)
#   Stage 2: LidarCorridor (resume from stage 1)
#   Stage 3: LidarJunction (resume from latest junction ckpt if available, else scratch)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ISAACLAB="${ISAACLAB:-/home/cs/IsaacLab}"

TASK_NAV="Isaac-Lab-Tutorial-LidarNav-TurtleBot3-Direct-v0"
TASK_COR="Isaac-Lab-Tutorial-LidarCorridor-TurtleBot3-Direct-v0"
TASK_JUN="Isaac-Lab-Tutorial-LidarJunction-TurtleBot3-Direct-v0"

NUM_ENVS=100
SEED=42
START_STAGE="nav"   # nav|corridor|junction
HEADLESS=1          # 1=headless, 0=GUI
ENV_SPACING=30.0

NAV_ITERS=4000
COR_ITERS=3000
JUN_ITERS=6000

# Stage-wise speed/turn limits
# NAV: high-speed straight/open navigation
NAV_V_MAX=1.0
NAV_OMEGA_LIMIT=3.2
NAV_TAU_V=0.10
NAV_TAU_OMEGA=0.05
NAV_WHEEL_LINEAR_SPEED_MAX=1.0
# CORRIDOR: moderate speed
COR_V_MAX=0.8
COR_OMEGA_LIMIT=2.6
COR_TAU_V=0.12
COR_TAU_OMEGA=0.06
COR_WHEEL_LINEAR_SPEED_MAX=0.8
# JUNCTION: conservative and stable
JUN_V_MAX=0.6
JUN_OMEGA_LIMIT=2.0
JUN_TAU_V=0.12
JUN_TAU_OMEGA=0.08
JUN_WHEEL_LINEAR_SPEED_MAX=0.6

# Long-range settings
GOAL_MIN=3.0
GOAL_MAX=12.0
GOAL_NORM=20.0
EP_LEN=40.0

# Optional explicit checkpoints
NAV_CKPT=""
JUN_CKPT=""

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --num_envs N            Parallel envs (default: ${NUM_ENVS})
  --seed N                Seed (default: ${SEED})
  --env_spacing M         Environment spacing in meters (default: ${ENV_SPACING})
  --start_stage S         nav|corridor|junction (default: ${START_STAGE})
  --gui                   Run with GUI (disable --headless)
  --headless              Force headless mode (default)
  --nav_checkpoint PATH   Use this checkpoint as nav stage input (optional)
  --junction_checkpoint PATH
                          Use this checkpoint as junction stage input (optional)
  --nav_iters N           Stage1 iterations (default: ${NAV_ITERS})
  --corridor_iters N      Stage2 iterations (default: ${COR_ITERS})
  --junction_iters N      Stage3 iterations (default: ${JUN_ITERS})
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num_envs) NUM_ENVS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --env_spacing) ENV_SPACING="$2"; shift 2 ;;
    --start_stage) START_STAGE="$2"; shift 2 ;;
    --gui) HEADLESS=0; shift ;;
    --headless) HEADLESS=1; shift ;;
    --nav_checkpoint) NAV_CKPT="$2"; shift 2 ;;
    --junction_checkpoint) JUN_CKPT="$2"; shift 2 ;;
    --nav_iters) NAV_ITERS="$2"; shift 2 ;;
    --corridor_iters) COR_ITERS="$2"; shift 2 ;;
    --junction_iters) JUN_ITERS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

latest_best() {
  local exp_dir="$1"
  ls -1t "${PROJ}/logs/skrl/${exp_dir}"/*/checkpoints/best_agent.pt 2>/dev/null | head -n 1 || true
}

run_train() {
  echo "[RUN] $*"
  env PYTHONNOUSERSITE=1 PYTHONPATH= \
    "${ISAACLAB}/isaaclab.sh" -p "${PROJ}/scripts/skrl/train.py" "$@"
}

# Common args
COMMON_ARGS=(
  --algorithm PPO
  --num_envs "${NUM_ENVS}"
  --env_spacing "${ENV_SPACING}"
  --seed "${SEED}"
  --goal_spawn_range_min "${GOAL_MIN}"
  --goal_spawn_range_max "${GOAL_MAX}"
  --goal_max_distance "${GOAL_NORM}"
  --episode_length_s "${EP_LEN}"
)
if [[ "${HEADLESS}" == "1" ]]; then
  COMMON_ARGS+=(--headless)
else
  echo "[INFO] GUI mode enabled (--headless disabled)."
fi

if [[ "${START_STAGE}" == "nav" ]]; then
  echo "========== Stage 1: NAV =========="
  ARGS=(
    --task "${TASK_NAV}"
    --max_iterations "${NAV_ITERS}"
    "${COMMON_ARGS[@]}"
    --v_max "${NAV_V_MAX}"
    --omega_limit "${NAV_OMEGA_LIMIT}"
    --tau_v "${NAV_TAU_V}"
    --tau_omega "${NAV_TAU_OMEGA}"
    --wheel_linear_speed_max "${NAV_WHEEL_LINEAR_SPEED_MAX}"
    --num_obstacles 4
  )
  if [[ -n "${NAV_CKPT}" ]]; then
    ARGS+=( --checkpoint "${NAV_CKPT}" )
  fi
  run_train "${ARGS[@]}"
  NAV_CKPT="$(latest_best lidar_nav_direct)"
  [[ -n "${NAV_CKPT}" ]] || { echo "[ERROR] no nav checkpoint found"; exit 1; }
  echo "[CKPT][NAV] ${NAV_CKPT}"
else
  NAV_CKPT="${NAV_CKPT:-$(latest_best lidar_nav_direct)}"
  echo "[INFO] skip nav stage, using NAV_CKPT=${NAV_CKPT:-<empty>}"
fi

if [[ "${START_STAGE}" == "nav" || "${START_STAGE}" == "corridor" ]]; then
  echo "========== Stage 2: CORRIDOR =========="
  [[ -n "${NAV_CKPT}" ]] || { echo "[ERROR] corridor stage needs NAV_CKPT"; exit 1; }
  run_train \
    --task "${TASK_COR}" \
    --max_iterations "${COR_ITERS}" \
    --checkpoint "${NAV_CKPT}" \
    "${COMMON_ARGS[@]}" \
    --v_max "${COR_V_MAX}" \
    --omega_limit "${COR_OMEGA_LIMIT}" \
    --tau_v "${COR_TAU_V}" \
    --tau_omega "${COR_TAU_OMEGA}" \
    --wheel_linear_speed_max "${COR_WHEEL_LINEAR_SPEED_MAX}" \
    --num_obstacles 5 \
    --corridor_half_length 12.0
  COR_CKPT="$(latest_best lidar_corridor_direct)"
  [[ -n "${COR_CKPT}" ]] || { echo "[ERROR] no corridor checkpoint found"; exit 1; }
  echo "[CKPT][CORRIDOR] ${COR_CKPT}"
fi

if [[ "${START_STAGE}" == "nav" || "${START_STAGE}" == "corridor" || "${START_STAGE}" == "junction" ]]; then
  echo "========== Stage 3: JUNCTION =========="
  # Junction observation dim differs from nav/corridor, so do NOT resume from corridor ckpt.
  if [[ -z "${JUN_CKPT}" ]]; then
    JUN_CKPT="$(latest_best lidar_junction_direct)"
  fi
  J_ARGS=(
    --task "${TASK_JUN}"
    --max_iterations "${JUN_ITERS}"
    "${COMMON_ARGS[@]}"
    --v_max "${JUN_V_MAX}"
    --omega_limit "${JUN_OMEGA_LIMIT}"
    --tau_v "${JUN_TAU_V}"
    --tau_omega "${JUN_TAU_OMEGA}"
    --wheel_linear_speed_max "${JUN_WHEEL_LINEAR_SPEED_MAX}"
    --num_obstacles 4
    --junction_half_length 10.0
    --junction_goal_forward_min 2.0
    --junction_goal_forward_max 6.0
    --junction_x_sampling_prob 0.02
  )
  if [[ -n "${JUN_CKPT}" ]]; then
    J_ARGS+=( --checkpoint "${JUN_CKPT}" )
    echo "[INFO] junction resume from ${JUN_CKPT}"
  else
    echo "[INFO] no junction checkpoint found, start junction from scratch"
  fi
  run_train "${J_ARGS[@]}"
  JUN_NEW="$(latest_best lidar_junction_direct)"
  [[ -n "${JUN_NEW}" ]] || { echo "[ERROR] no junction checkpoint found after training"; exit 1; }
  echo "[CKPT][JUNCTION] ${JUN_NEW}"
fi

echo "[DONE] high-speed suite finished"
