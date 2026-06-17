#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/skrl/run_eval_3stage_summary.sh \
    --ckpt-stage0 /path/to/stage0_best_agent.pt \
    --ckpt-stage1 /path/to/stage1_best_agent.pt \
    --ckpt-stage2 /path/to/stage2_best_agent.pt \
    [--episodes 500] [--num_envs 64] [--seed 42] \
    [--task Isaac-Lab-Tutorial-LidarNav-TurtleBot3-Direct-v0] \
    [--algorithm PPO] \
    [--stage2-goal-min 3.0 --stage2-goal-max 12.0 --stage2-goal-norm 20.0] \
    [--output-dir /abs/path/output]

Description:
  1) 顺序运行三次 eval（stage0 -> stage1 -> stage2）
  2) 自动解析每次输出中的 success/collision/timeout
  3) 生成汇总文件:
     - summary.csv
     - summary.md
     - stage0_eval.log / stage1_eval.log / stage2_eval.log
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

ISAACLAB="${ISAACLAB:-/home/cs/IsaacLab}"
TASK="Isaac-Lab-Tutorial-LidarNav-TurtleBot3-Direct-v0"
ALGO="PPO"
EPISODES=500
NUM_ENVS=64
SEED=42

CKPT_STAGE0=""
CKPT_STAGE1=""
CKPT_STAGE2=""

STAGE2_GOAL_MIN=""
STAGE2_GOAL_MAX=""
STAGE2_GOAL_NORM=""

OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt-stage0) CKPT_STAGE0="$2"; shift 2 ;;
    --ckpt-stage1) CKPT_STAGE1="$2"; shift 2 ;;
    --ckpt-stage2) CKPT_STAGE2="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --num_envs) NUM_ENVS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --algorithm) ALGO="$2"; shift 2 ;;
    --stage2-goal-min) STAGE2_GOAL_MIN="$2"; shift 2 ;;
    --stage2-goal-max) STAGE2_GOAL_MAX="$2"; shift 2 ;;
    --stage2-goal-norm) STAGE2_GOAL_NORM="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$CKPT_STAGE0" || -z "$CKPT_STAGE1" || -z "$CKPT_STAGE2" ]]; then
  echo "[ERROR] --ckpt-stage0/1/2 are required." >&2
  usage
  exit 1
fi

for ck in "$CKPT_STAGE0" "$CKPT_STAGE1" "$CKPT_STAGE2"; do
  if [[ ! -f "$ck" ]]; then
    echo "[ERROR] checkpoint not found: $ck" >&2
    exit 1
  fi
done

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${PROJ_ROOT}/logs/eval_3stage_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT_DIR"

run_eval() {
  local stage_name="$1"
  local checkpoint="$2"
  local log_file="$3"
  shift 3
  local extra_args=("$@")

  echo "[RUN] ${stage_name}"
  echo "      ckpt=${checkpoint}"
  echo "      log=${log_file}"

  env PYTHONNOUSERSITE=1 PYTHONPATH= \
    "${ISAACLAB}/isaaclab.sh" -p "${PROJ_ROOT}/scripts/skrl/eval_lidar_nav.py" \
    --task "${TASK}" \
    --algorithm "${ALGO}" \
    --checkpoint "${checkpoint}" \
    --num_envs "${NUM_ENVS}" \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    --headless \
    "${extra_args[@]}" | tee "${log_file}"
}

extract_rate() {
  local key="$1"
  local log_file="$2"
  grep -F "[EVAL][RESULT] ${key}=" "${log_file}" | tail -n 1 | sed -E "s/.*${key}=([0-9.]+).*/\\1/"
}

extract_count() {
  local key="$1"
  local log_file="$2"
  grep -F "[EVAL][RESULT] ${key}=" "${log_file}" | tail -n 1 | sed -E 's/.*\(([0-9]+\/[0-9]+)\).*/\1/'
}

to_pct() {
  local x="$1"
  awk -v v="$x" 'BEGIN { printf("%.1f%%", v * 100.0) }'
}

LOG0="${OUTPUT_DIR}/stage0_eval.log"
LOG1="${OUTPUT_DIR}/stage1_eval.log"
LOG2="${OUTPUT_DIR}/stage2_eval.log"

run_eval "stage0" "$CKPT_STAGE0" "$LOG0"
run_eval "stage1" "$CKPT_STAGE1" "$LOG1"

STAGE2_EXTRA_ARGS=()
if [[ -n "$STAGE2_GOAL_MIN" ]]; then STAGE2_EXTRA_ARGS+=(--goal_spawn_range_min "$STAGE2_GOAL_MIN"); fi
if [[ -n "$STAGE2_GOAL_MAX" ]]; then STAGE2_EXTRA_ARGS+=(--goal_spawn_range_max "$STAGE2_GOAL_MAX"); fi
if [[ -n "$STAGE2_GOAL_NORM" ]]; then STAGE2_EXTRA_ARGS+=(--goal_max_distance "$STAGE2_GOAL_NORM"); fi
run_eval "stage2" "$CKPT_STAGE2" "$LOG2" "${STAGE2_EXTRA_ARGS[@]}"

S0="$(extract_rate success_rate "$LOG0")"
C0="$(extract_rate collision_rate "$LOG0")"
T0="$(extract_rate timeout_rate "$LOG0")"
S0C="$(extract_count success_rate "$LOG0")"
C0C="$(extract_count collision_rate "$LOG0")"
T0C="$(extract_count timeout_rate "$LOG0")"

S1="$(extract_rate success_rate "$LOG1")"
C1="$(extract_rate collision_rate "$LOG1")"
T1="$(extract_rate timeout_rate "$LOG1")"
S1C="$(extract_count success_rate "$LOG1")"
C1C="$(extract_count collision_rate "$LOG1")"
T1C="$(extract_count timeout_rate "$LOG1")"

S2="$(extract_rate success_rate "$LOG2")"
C2="$(extract_rate collision_rate "$LOG2")"
T2="$(extract_rate timeout_rate "$LOG2")"
S2C="$(extract_count success_rate "$LOG2")"
C2C="$(extract_count collision_rate "$LOG2")"
T2C="$(extract_count timeout_rate "$LOG2")"

CSV_OUT="${OUTPUT_DIR}/summary.csv"
cat > "${CSV_OUT}" <<EOF
stage,checkpoint,success_rate,success_count,collision_rate,collision_count,timeout_rate,timeout_count
stage0,${CKPT_STAGE0},${S0},${S0C},${C0},${C0C},${T0},${T0C}
stage1,${CKPT_STAGE1},${S1},${S1C},${C1},${C1C},${T1},${T1C}
stage2,${CKPT_STAGE2},${S2},${S2C},${C2},${C2C},${T2},${T2C}
EOF

MD_OUT="${OUTPUT_DIR}/summary.md"
cat > "${MD_OUT}" <<EOF
# Three-Stage Eval Summary

| Stage  | Success | Collision | Timeout |
|-------:|--------:|----------:|--------:|
| stage0 | $(to_pct "$S0") (${S0C}) | $(to_pct "$C0") (${C0C}) | $(to_pct "$T0") (${T0C}) |
| stage1 | $(to_pct "$S1") (${S1C}) | $(to_pct "$C1") (${C1C}) | $(to_pct "$T1") (${T1C}) |
| stage2 | $(to_pct "$S2") (${S2C}) | $(to_pct "$C2") (${C2C}) | $(to_pct "$T2") (${T2C}) |

## Run Config
- task: \`${TASK}\`
- algorithm: \`${ALGO}\`
- episodes: \`${EPISODES}\`
- num_envs: \`${NUM_ENVS}\`
- seed: \`${SEED}\`
- stage2 goal overrides: min=\`${STAGE2_GOAL_MIN:-N/A}\`, max=\`${STAGE2_GOAL_MAX:-N/A}\`, norm=\`${STAGE2_GOAL_NORM:-N/A}\`
EOF

echo
echo "[DONE] Summary generated:"
echo "  - ${CSV_OUT}"
echo "  - ${MD_OUT}"
echo "  - ${LOG0}"
echo "  - ${LOG1}"
echo "  - ${LOG2}"
