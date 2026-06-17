#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/skrl/run_state_bucket_eval.sh \
    [--run-dir /abs/path/to/run_dir] \
    [--log-root /abs/path/to/logs/skrl/lidar_junction_direct] \
    [--best-ckpt /abs/path/to/best_agent.pt] \
    [--final-ckpt /abs/path/to/final_or_agent_xxx.pt] \
    [--task Isaac-Lab-Tutorial-LidarJunction-TurtleBot3-Direct-v0] \
    [--algorithm PPO] \
    [--episodes 500] [--num-envs 64] [--seed 42] \
    [--output-dir /abs/path/to/output]

Description:
  1) Resolve best/final checkpoints (same run by default).
  2) Ensure final checkpoint is preserved as checkpoints/final_agent.pt.
  3) Run the same eval script twice (best + final), each with state-bucket JSON output.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

ISAACLAB="${ISAACLAB:-/home/cs/IsaacLab}"
TASK=""
ALGO="PPO"
EPISODES=500
NUM_ENVS=64
SEED=42

RUN_DIR=""
LOG_ROOT="${PROJ_ROOT}/logs/skrl/lidar_junction_direct"
BEST_CKPT=""
FINAL_CKPT=""
OUTPUT_DIR=""
NO_PROGRESS_MIN_DELTA_M=""
NO_PROGRESS_MIN_RATIO=""
NO_PROGRESS_STALL_RATIO=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --best-ckpt) BEST_CKPT="$2"; shift 2 ;;
    --final-ckpt) FINAL_CKPT="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --algorithm) ALGO="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --no-progress-min-delta-m) NO_PROGRESS_MIN_DELTA_M="$2"; shift 2 ;;
    --no-progress-min-ratio) NO_PROGRESS_MIN_RATIO="$2"; shift 2 ;;
    --no-progress-stall-ratio) NO_PROGRESS_STALL_RATIO="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

ALGO_LC="$(echo "${ALGO}" | tr '[:upper:]' '[:lower:]')"

if [[ -z "${RUN_DIR}" ]]; then
  if [[ ! -d "${LOG_ROOT}" ]]; then
    echo "[ERROR] log root not found: ${LOG_ROOT}" >&2
    exit 1
  fi
  RUN_DIR="$(ls -dt "${LOG_ROOT}"/*_"${ALGO_LC}"_torch 2>/dev/null | head -n 1 || true)"
  if [[ -z "${RUN_DIR}" ]]; then
    echo "[ERROR] no run dir found under ${LOG_ROOT}" >&2
    exit 1
  fi
fi

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "[ERROR] run dir not found: ${RUN_DIR}" >&2
  exit 1
fi

if [[ -z "${TASK}" ]]; then
  case "${RUN_DIR}" in
    *"/lidar_short_nav_curriculum_p1_direct/"*|*"/lidar_short_nav_curriculum_p1_direct")
      TASK="Isaac-Lab-Tutorial-LidarShortNavCurriculumP1-TurtleBot3-Direct-v0"
      ;;
    *"/lidar_short_nav_v14_direct/"*|*"/lidar_short_nav_v14_direct")
      TASK="Isaac-Lab-Tutorial-LidarShortNavV14-TurtleBot3-Direct-v0"
      ;;
    *"/lidar_short_nav_v15_direct/"*|*"/lidar_short_nav_v15_direct")
      TASK="Isaac-Lab-Tutorial-LidarShortNavV15-TurtleBot3-Direct-v0"
      ;;
    *"/lidar_short_nav_v15a_direct/"*|*"/lidar_short_nav_v15a_direct")
      TASK="Isaac-Lab-Tutorial-LidarShortNavV15A-TurtleBot3-Direct-v0"
      ;;
    *"/lidar_short_nav_v15b_direct/"*|*"/lidar_short_nav_v15b_direct")
      TASK="Isaac-Lab-Tutorial-LidarShortNavV15B-TurtleBot3-Direct-v0"
      ;;
    *"/lidar_short_nav_v16_direct/"*|*"/lidar_short_nav_v16_direct")
      TASK="Isaac-Lab-Tutorial-LidarShortNavV16-TurtleBot3-Direct-v0"
      ;;
    *"/lidar_short_nav_unified_direct/"*|*"/lidar_short_nav_unified_direct")
      TASK="Isaac-Lab-Tutorial-LidarShortNavUnified-TurtleBot3-Direct-v0"
      ;;
    *"/lidar_short_nav_junction_direct/"*|*"/lidar_short_nav_junction_direct")
      TASK="Isaac-Lab-Tutorial-LidarShortNavJunction-TurtleBot3-Direct-v0"
      ;;
    *"/lidar_junction_direct/"*|*"/lidar_junction_direct")
      TASK="Isaac-Lab-Tutorial-LidarJunction-TurtleBot3-Direct-v0"
      ;;
    *"/lidar_corridor_direct/"*|*"/lidar_corridor_direct")
      TASK="Isaac-Lab-Tutorial-LidarCorridor-TurtleBot3-Direct-v0"
      ;;
    *"/lidar_nav_direct/"*|*"/lidar_nav_direct")
      TASK="Isaac-Lab-Tutorial-LidarNav-TurtleBot3-Direct-v0"
      ;;
    *)
      TASK="Isaac-Lab-Tutorial-LidarJunction-TurtleBot3-Direct-v0"
      ;;
  esac
fi

CKPT_DIR="${RUN_DIR}/checkpoints"
if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "[ERROR] checkpoints dir not found: ${CKPT_DIR}" >&2
  exit 1
fi

if [[ -z "${BEST_CKPT}" ]]; then
  if [[ -f "${CKPT_DIR}/best_agent.pt" ]]; then
    BEST_CKPT="${CKPT_DIR}/best_agent.pt"
  fi
fi
if [[ -z "${BEST_CKPT}" || ! -f "${BEST_CKPT}" ]]; then
  echo "[ERROR] best checkpoint not found. Pass --best-ckpt explicitly." >&2
  exit 1
fi

if [[ -z "${FINAL_CKPT}" ]]; then
  if [[ -f "${CKPT_DIR}/final_agent.pt" ]]; then
    FINAL_CKPT="${CKPT_DIR}/final_agent.pt"
  else
    latest_step=-1
    latest_agent=""
    for ck in "${CKPT_DIR}"/agent_*.pt; do
      [[ -e "${ck}" ]] || continue
      name="$(basename "${ck}")"
      step="${name#agent_}"
      step="${step%.pt}"
      if [[ "${step}" =~ ^[0-9]+$ ]] && (( step > latest_step )); then
        latest_step="${step}"
        latest_agent="${ck}"
      fi
    done
    if [[ -z "${latest_agent}" ]]; then
      echo "[ERROR] final checkpoint not found: neither final_agent.pt nor agent_*.pt exists." >&2
      exit 1
    fi
    FINAL_CKPT="${CKPT_DIR}/final_agent.pt"
    cp -f "${latest_agent}" "${FINAL_CKPT}"
    echo "[INFO] Preserved final checkpoint: ${FINAL_CKPT} (from ${latest_agent})"
  fi
fi

if [[ ! -f "${FINAL_CKPT}" ]]; then
  echo "[ERROR] final checkpoint not found: ${FINAL_CKPT}" >&2
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${PROJ_ROOT}/logs/eval_state_bucket_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${OUTPUT_DIR}"

run_eval() {
  local label="$1"
  local ckpt="$2"
  local log_file="${OUTPUT_DIR}/${label}_eval.log"
  local json_file="${OUTPUT_DIR}/${label}_state_report.json"

  echo "[RUN] ${label}: ${ckpt}"
  local extra=()
  if [[ -n "${NO_PROGRESS_MIN_DELTA_M}" ]]; then extra+=(--no_progress_min_delta_m "${NO_PROGRESS_MIN_DELTA_M}"); fi
  if [[ -n "${NO_PROGRESS_MIN_RATIO}" ]]; then extra+=(--no_progress_min_ratio "${NO_PROGRESS_MIN_RATIO}"); fi
  if [[ -n "${NO_PROGRESS_STALL_RATIO}" ]]; then extra+=(--no_progress_stall_ratio "${NO_PROGRESS_STALL_RATIO}"); fi

  env PYTHONNOUSERSITE=1 PYTHONPATH= \
    "${ISAACLAB}/isaaclab.sh" -p "${PROJ_ROOT}/scripts/skrl/eval_lidar_nav.py" \
    --task "${TASK}" \
    --algorithm "${ALGO}" \
    --checkpoint "${ckpt}" \
    --num_envs "${NUM_ENVS}" \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    --headless \
    --state_report_json "${json_file}" \
    "${extra[@]}" 2>&1 | tee "${log_file}"

  if [[ ! -s "${json_file}" ]]; then
    echo "[ERROR] state report not generated: ${json_file}" >&2
    echo "[ERROR] see eval log: ${log_file}" >&2
    return 2
  fi
  if ! grep -q "\[EVAL\]\[RESULT\] episodes=" "${log_file}"; then
    echo "[ERROR] eval summary marker missing in log: ${log_file}" >&2
    return 2
  fi
}

run_eval "best" "${BEST_CKPT}"
run_eval "final" "${FINAL_CKPT}"

cat > "${OUTPUT_DIR}/checkpoint_manifest.txt" <<EOF
run_dir=${RUN_DIR}
best_ckpt=${BEST_CKPT}
final_ckpt=${FINAL_CKPT}
best_report=${OUTPUT_DIR}/best_state_report.json
final_report=${OUTPUT_DIR}/final_state_report.json
EOF

echo "[DONE] outputs:"
echo "  - ${OUTPUT_DIR}/best_eval.log"
echo "  - ${OUTPUT_DIR}/final_eval.log"
echo "  - ${OUTPUT_DIR}/best_state_report.json"
echo "  - ${OUTPUT_DIR}/final_state_report.json"
echo "  - ${OUTPUT_DIR}/checkpoint_manifest.txt"
