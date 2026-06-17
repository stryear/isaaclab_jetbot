#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Short wrapper for the long multi-scene command.
# Usage:
#   ./scripts/run_eval_testworld.sh [spawn_x] [spawn_y] [episodes] [seeds]
SPAWN_X="${1:-16.0}"
SPAWN_Y="${2:--0.2}"
EPISODES="${3:-10}"
SEEDS="${4:-42}"

CHECKPOINT="${CHECKPOINT:-/home/cs/isaaclab_jetbot/logs/skrl/lidar_junction_direct/2026-03-08_08-11-22_ppo_torch/checkpoints/best_agent.pt}"
WORLD="${WORLD:-/home/cs/drl_planner/src/wpr_simulation2/worlds/test.world}"

"${SCRIPT_DIR}/run_gazebo_eval_multiscene.sh" \
  --checkpoint "${CHECKPOINT}" \
  --worlds "${WORLD}" \
  --spawn-x "${SPAWN_X}" \
  --spawn-y "${SPAWN_Y}" \
  --episodes "${EPISODES}" \
  --seeds "${SEEDS}" \
  --policy-runtime isaaclab \
  --turn-cmd straight \
  --episode-timeout-s 40 \
  --goal-reach-threshold 0.45 \
  --obstacle-min-scale 0.55 \
  --obstacle-slowdown-distance 0.55 \
  --near-goal-min-scale 0.45
