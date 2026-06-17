#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Usage:
#   ./scripts/run_eval_testworld_fixed.sh [repeat] [seeds] [cases_csv] [episode_timeout_s] [goal_max_distance]
REPEAT="${1:-1}"
SEEDS="${2:-42}"
CASES_ARG="${3:-}"
EP_TIMEOUT="${4:-180}"
GOAL_MAX_DISTANCE="${5:-16.0}"

CHECKPOINT="${CHECKPOINT:-/home/cs/isaaclab_jetbot/logs/skrl/lidar_junction_direct/2026-03-08_08-11-22_ppo_torch/checkpoints/best_agent.pt}"
WORLD="${WORLD:-/home/cs/drl_planner/src/wpr_simulation2/worlds/test.world}"
CASES_DEFAULT="/home/cs/isaaclab_jetbot/scripts/testworld_fixed_cases.csv"
CASES="${CASES_ARG:-${CASES:-${CASES_DEFAULT}}}"

echo "[INFO] run_eval_testworld_fixed.sh using cases: ${CASES}"
echo "[INFO] run_eval_testworld_fixed.sh episode_timeout_s=${EP_TIMEOUT} goal_max_distance=${GOAL_MAX_DISTANCE}"

"${SCRIPT_DIR}/run_gazebo_eval_multiscene.sh" \
  --checkpoint "${CHECKPOINT}" \
  --worlds "${WORLD}" \
  --episodes 1 \
  --seeds "${SEEDS}" \
  --policy-runtime isaaclab \
  --turn-cmd topic \
  --fixed-cases-file "${CASES}" \
  --fixed-cases-repeat "${REPEAT}" \
  --reset-each-episode 0 \
  --set-state-wait-s 15 \
  --episode-timeout-s "${EP_TIMEOUT}" \
  --goal-max-distance "${GOAL_MAX_DISTANCE}" \
  --goal-reach-threshold 0.45 \
  --obstacle-min-scale 0.55 \
  --obstacle-slowdown-distance 0.55 \
  --near-goal-min-scale 0.45
