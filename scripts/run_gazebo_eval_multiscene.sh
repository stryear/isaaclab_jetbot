#!/usr/bin/env bash
set -euo pipefail

# Avoid "unbound variable" failures inside ROS setup scripts.
set +u
source /opt/ros/humble/setup.bash
if [ -f /home/cs/drl_planner/install/setup.bash ]; then
  source /home/cs/drl_planner/install/setup.bash
fi
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT=""
DEVICE="cpu"
ACTION_MODE="vomega"
ACTION_SCALE="8.0"
TURN_CMD="auto"
TB3_MODEL="${TURTLEBOT3_MODEL:-burger}"
EPISODES=100
SEEDS_CSV="42,43,44"
WORLDS_CSV="turtlebot3_world,turtlebot3_house,turtlebot3_dqn_stage4"
SLEEP_AFTER_WORLD=6
SLEEP_AFTER_POLICY=3
OUT_DIR="${REPO_ROOT}/logs/gazebo_eval_multiscene/$(date +%Y%m%d_%H%M%S)"
POLICY_RUNTIME="auto" # auto | system | isaaclab
ISAACLAB_SH="/home/cs/IsaacLab/isaaclab.sh"
DISABLE_ACTION_SCALING=0
ALLOW_REVERSE=0
SPAWN_X=""
SPAWN_Y=""
FIXED_CASES_FILE=""
FIXED_CASES_REPEAT=1
RESET_EACH_EPISODE="auto"  # auto | 0 | 1
SET_STATE_WAIT_S=10.0

# Eval defaults (aligned with your current setup)
EPISODE_TIMEOUT=30
GOAL_FORWARD_MIN=1.0
GOAL_FORWARD_MAX=2.4
GOAL_LATERAL_MAX=0.8
SCAN_AWARE=1
GOAL_OBSTACLE_MARGIN=0.25
GOAL_SAMPLE_MAX_TRIES=40
GOAL_REACH_THRESHOLD=0.45
COLLISION_THRESHOLD=0.20
COLLISION_GRACE_S=1.0
COLLISION_CONSEC=3
RESET_SERVICE="/reset_simulation"
RESET_WAIT_S=2.0
POST_RESET_SETTLE_S=0.5

# Policy defaults
CONTROL_RATE=60
GOAL_TIMEOUT_S=0.0
V_MAX="${V_MAX:-0.55}"
OMEGA_LIMIT="${OMEGA_LIMIT:-3.2}"
GOAL_MAX_DISTANCE=5.0
TAU_V=0.10
TAU_OMEGA=0.05
NEAR_GOAL_SLOWDOWN_DISTANCE=0.8
NEAR_GOAL_MIN_ACTION_SCALE=0.40
OBSTACLE_SLOWDOWN_DISTANCE=0.75
OBSTACLE_MIN_ACTION_SCALE=0.35
PRINT_INTERVAL_S=1.0

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --checkpoint <best_agent.pt> [options]

Required:
  --checkpoint PATH                 Policy checkpoint path

Options:
  --episodes N                      Episodes per seed (default: ${EPISODES})
  --seeds CSV                       Seed list, e.g. 42,43,44
  --worlds CSV                      World keys or absolute .world paths (comma-separated)
  --out-dir DIR                     Output directory
  --model NAME                      TURTLEBOT3_MODEL (default: ${TB3_MODEL})
  --device DEV                      Policy torch device (default: ${DEVICE})
  --action-mode MODE                vomega | wheel (default: ${ACTION_MODE})
  --action-scale S                  Wheel mode action scale (default: ${ACTION_SCALE})
  --turn-cmd CMD                    left | straight | right | auto | topic (default: ${TURN_CMD})
  --spawn-x X                       Override robot spawn x for all worlds
  --spawn-y Y                       Override robot spawn y for all worlds
  --fixed-cases-file PATH           Use fixed spawn/goal cases CSV for evaluator
  --fixed-cases-repeat N            Repeat fixed case list N times (default: ${FIXED_CASES_REPEAT})
  --reset-each-episode {0|1|auto}   Evaluator reset mode (default: ${RESET_EACH_EPISODE})
  --set-state-wait-s SEC            Wait timeout for set_entity_state service (default: ${SET_STATE_WAIT_S})
  --disable-action-scaling          Disable runner-side speed scaling
  --allow-reverse                  Allow reverse linear commands in policy runner
  --episode-timeout-s SEC           Episode timeout for evaluator (default: ${EPISODE_TIMEOUT})
  --goal-reach-threshold M          Success threshold for runner/evaluator (default: ${GOAL_REACH_THRESHOLD})
  --v-max M                         Policy v_max (default: ${V_MAX})
  --omega-limit R                   Policy omega_limit (default: ${OMEGA_LIMIT})
  --goal-max-distance M             Goal distance normalization denominator for policy runner (default: ${GOAL_MAX_DISTANCE})
  --near-goal-min-scale S           Near-goal min action scale (default: ${NEAR_GOAL_MIN_ACTION_SCALE})
  --obstacle-min-scale S            Obstacle min action scale (default: ${OBSTACLE_MIN_ACTION_SCALE})
  --obstacle-slowdown-distance M    Obstacle slowdown distance (default: ${OBSTACLE_SLOWDOWN_DISTANCE})
  --sleep-after-world SEC           Wait after world launch (default: ${SLEEP_AFTER_WORLD})
  --sleep-after-policy SEC          Wait after policy launch (default: ${SLEEP_AFTER_POLICY})
  --policy-runtime MODE             auto | system | isaaclab (default: ${POLICY_RUNTIME})
  --isaaclab-sh PATH                isaaclab.sh path (default: ${ISAACLAB_SH})

  --help                            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --seeds) SEEDS_CSV="$2"; shift 2 ;;
    --worlds) WORLDS_CSV="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --model) TB3_MODEL="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --action-mode) ACTION_MODE="$2"; shift 2 ;;
    --action-scale) ACTION_SCALE="$2"; shift 2 ;;
    --turn-cmd) TURN_CMD="$2"; shift 2 ;;
    --spawn-x) SPAWN_X="$2"; shift 2 ;;
    --spawn-y) SPAWN_Y="$2"; shift 2 ;;
    --fixed-cases-file) FIXED_CASES_FILE="$2"; shift 2 ;;
    --fixed-cases-repeat) FIXED_CASES_REPEAT="$2"; shift 2 ;;
    --reset-each-episode) RESET_EACH_EPISODE="$2"; shift 2 ;;
    --set-state-wait-s) SET_STATE_WAIT_S="$2"; shift 2 ;;
    --disable-action-scaling) DISABLE_ACTION_SCALING=1; shift ;;
    --allow-reverse) ALLOW_REVERSE=1; shift ;;
    --episode-timeout-s) EPISODE_TIMEOUT="$2"; shift 2 ;;
    --goal-reach-threshold) GOAL_REACH_THRESHOLD="$2"; shift 2 ;;
    --v-max) V_MAX="$2"; shift 2 ;;
    --omega-limit) OMEGA_LIMIT="$2"; shift 2 ;;
    --goal-max-distance) GOAL_MAX_DISTANCE="$2"; shift 2 ;;
    --near-goal-min-scale) NEAR_GOAL_MIN_ACTION_SCALE="$2"; shift 2 ;;
    --obstacle-min-scale) OBSTACLE_MIN_ACTION_SCALE="$2"; shift 2 ;;
    --obstacle-slowdown-distance) OBSTACLE_SLOWDOWN_DISTANCE="$2"; shift 2 ;;
    --sleep-after-world) SLEEP_AFTER_WORLD="$2"; shift 2 ;;
    --sleep-after-policy) SLEEP_AFTER_POLICY="$2"; shift 2 ;;
    --policy-runtime) POLICY_RUNTIME="$2"; shift 2 ;;
    --isaaclab-sh) ISAACLAB_SH="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${CHECKPOINT}" ]]; then
  echo "[ERROR] --checkpoint is required"
  usage
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] checkpoint not found: ${CHECKPOINT}"
  exit 1
fi
if [[ "${POLICY_RUNTIME}" != "auto" && "${POLICY_RUNTIME}" != "system" && "${POLICY_RUNTIME}" != "isaaclab" ]]; then
  echo "[ERROR] --policy-runtime must be one of: auto, system, isaaclab"
  exit 1
fi
if [[ "${TURN_CMD}" != "left" && "${TURN_CMD}" != "straight" && "${TURN_CMD}" != "right" && "${TURN_CMD}" != "auto" && "${TURN_CMD}" != "topic" ]]; then
  echo "[ERROR] --turn-cmd must be one of: left, straight, right, auto, topic"
  exit 1
fi
if [[ -n "${FIXED_CASES_FILE}" && ! -f "${FIXED_CASES_FILE}" ]]; then
  echo "[ERROR] fixed cases file not found: ${FIXED_CASES_FILE}"
  exit 1
fi
if [[ "${RESET_EACH_EPISODE}" != "auto" && "${RESET_EACH_EPISODE}" != "0" && "${RESET_EACH_EPISODE}" != "1" ]]; then
  echo "[ERROR] --reset-each-episode must be one of: auto, 0, 1"
  exit 1
fi

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
IFS=',' read -r -a WORLDS <<< "${WORLDS_CSV}"

mkdir -p "${OUT_DIR}"
export TURTLEBOT3_MODEL="${TB3_MODEL}"

# If user has local turtlebot3_simulations source, include it in Gazebo search paths.
if [[ -d "/home/cs/drl_planner/src/turtlebot3_simulations/turtlebot3_gazebo/models" ]]; then
  export GAZEBO_MODEL_PATH="/home/cs/drl_planner/src/turtlebot3_simulations/turtlebot3_gazebo/models:${GAZEBO_MODEL_PATH:-}"
fi
if [[ -d "/home/cs/drl_planner/src/turtlebot3_simulations/turtlebot3_gazebo/worlds" ]]; then
  export GAZEBO_RESOURCE_PATH="/home/cs/drl_planner/src/turtlebot3_simulations/turtlebot3_gazebo/worlds:${GAZEBO_RESOURCE_PATH:-}"
fi
if [[ -d "/home/cs/drl_planner/src/wpr_simulation2/models" ]]; then
  export GAZEBO_MODEL_PATH="/home/cs/drl_planner/src/wpr_simulation2/models:${GAZEBO_MODEL_PATH:-}"
fi
if [[ -d "/home/cs/drl_planner/src/wpr_simulation2/worlds" ]]; then
  export GAZEBO_RESOURCE_PATH="/home/cs/drl_planner/src/wpr_simulation2/worlds:${GAZEBO_RESOURCE_PATH:-}"
fi

declare -A LAUNCH_FILE=(
  ["turtlebot3_world"]="turtlebot3_world.launch.py"
  ["turtlebot3_house"]="turtlebot3_house.launch.py"
  ["turtlebot3_dqn_stage4"]="turtlebot3_dqn_stage4.launch.py"
)

# Default spawn points per world.
# turtlebot3_house default launch uses (-2.0, -0.5), which can start outside the house layout.
declare -A DEFAULT_SPAWN_X=(
  ["turtlebot3_world"]="-2.0"
  ["turtlebot3_house"]="-7.0"
  ["turtlebot3_dqn_stage4"]="0.0"
)
declare -A DEFAULT_SPAWN_Y=(
  ["turtlebot3_world"]="-0.5"
  ["turtlebot3_house"]="-2.0"
  ["turtlebot3_dqn_stage4"]="0.0"
)

cleanup_all() {
  pkill -f "gazebo_policy_runner.py" || true
  pkill -f "gazebo_eval_success.py" || true
  pkill -f "spawn_entity.py" || true
  pkill -f "robot_state_publisher" || true
  pkill -f "gzserver" || true
  pkill -f "gzclient" || true
  sleep 2
}

trap cleanup_all EXIT

wait_topic() {
  local topic="$1"
  local timeout_s="$2"
  local t0
  t0="$(date +%s)"
  while true; do
    if ros2 topic list 2>/dev/null | grep -qx "${topic}"; then
      return 0
    fi
    if (( "$(date +%s)" - t0 >= timeout_s )); then
      return 1
    fi
    sleep 1
  done
}

echo "[INFO] Output dir: ${OUT_DIR}"
echo "[INFO] Worlds: ${WORLDS_CSV}"
echo "[INFO] Seeds: ${SEEDS_CSV}"
echo "[INFO] Model: ${TB3_MODEL}"
echo "[INFO] Action mode: ${ACTION_MODE}"
echo "[INFO] Policy limits: v_max=${V_MAX}, omega_limit=${OMEGA_LIMIT}"
echo "[INFO] Turn cmd: ${TURN_CMD}"
if [[ -n "${FIXED_CASES_FILE}" ]]; then
  echo "[INFO] Fixed cases: ${FIXED_CASES_FILE} (repeat=${FIXED_CASES_REPEAT})"
fi
echo "[INFO] Reset each episode: ${RESET_EACH_EPISODE}"
echo "[INFO] Disable action scaling: ${DISABLE_ACTION_SCALING}"
echo "[INFO] Policy runtime: ${POLICY_RUNTIME}"
echo

for W in "${WORLDS[@]}"; do
  world_mode=""
  world_key_or_name="${W}"
  world_path=""
  if [[ -n "${LAUNCH_FILE[$W]+x}" ]]; then
    world_mode="tb3_builtin"
  elif [[ "${W}" == /*.world && -f "${W}" ]]; then
    world_mode="custom_file"
    world_path="${W}"
    world_key_or_name="$(basename "${W}" .world)"
  else
    echo "[ERROR] Unsupported world entry: ${W}"
    echo "Supported keys: ${!LAUNCH_FILE[*]}"
    echo "Or pass absolute .world file path"
    exit 1
  fi

  log_world="${world_key_or_name//[^a-zA-Z0-9_.-]/_}"
  echo "========== WORLD: ${world_key_or_name} ==========" | tee -a "${OUT_DIR}/summary.log"
  cleanup_all

  if [[ "${world_mode}" == "tb3_builtin" ]]; then
    world_spawn_x="${SPAWN_X:-${DEFAULT_SPAWN_X[$W]}}"
    world_spawn_y="${SPAWN_Y:-${DEFAULT_SPAWN_Y[$W]}}"
  else
    world_spawn_x="${SPAWN_X:-0.0}"
    world_spawn_y="${SPAWN_Y:-0.0}"
  fi
  echo "[INFO] world=${world_key_or_name} mode=${world_mode} spawn=(${world_spawn_x}, ${world_spawn_y})" | tee -a "${OUT_DIR}/summary.log"

  WORLD_PIDS=()
  if [[ "${world_mode}" == "tb3_builtin" ]]; then
    ros2 launch turtlebot3_gazebo "${LAUNCH_FILE[$W]}" \
      x_pose:="${world_spawn_x}" y_pose:="${world_spawn_y}" \
      > "${OUT_DIR}/${log_world}_gazebo.log" 2>&1 &
    WORLD_PIDS+=("$!")
  else
    ros2 launch gazebo_ros gzserver.launch.py world:="${world_path}" \
      > "${OUT_DIR}/${log_world}_gzserver.log" 2>&1 &
    WORLD_PIDS+=("$!")
    ros2 launch turtlebot3_gazebo robot_state_publisher.launch.py use_sim_time:=true \
      > "${OUT_DIR}/${log_world}_rsp.log" 2>&1 &
    WORLD_PIDS+=("$!")
    # One-shot spawn launch (foreground): ensures robot entity exists before evaluation.
    ros2 launch turtlebot3_gazebo spawn_turtlebot3.launch.py \
      x_pose:="${world_spawn_x}" y_pose:="${world_spawn_y}" \
      > "${OUT_DIR}/${log_world}_spawn.log" 2>&1
  fi

  sleep "${SLEEP_AFTER_WORLD}"

  if ! wait_topic "/scan" 30; then
    echo "[ERROR] /scan not ready in world ${world_key_or_name}" | tee -a "${OUT_DIR}/summary.log"
    for pid in "${WORLD_PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
    exit 1
  fi
  if ! wait_topic "/odom" 30; then
    echo "[ERROR] /odom not ready in world ${world_key_or_name}" | tee -a "${OUT_DIR}/summary.log"
    for pid in "${WORLD_PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
    exit 1
  fi

  POLICY_LOG="${OUT_DIR}/${log_world}_policy.log"
  POLICY_ARGS=(
    --checkpoint "${CHECKPOINT}"
    --device "${DEVICE}"
    --action_mode "${ACTION_MODE}"
    --action_scale "${ACTION_SCALE}"
    --turn_cmd "${TURN_CMD}"
    --control_rate "${CONTROL_RATE}"
    --goal_timeout_s "${GOAL_TIMEOUT_S}"
    --v_max "${V_MAX}"
    --omega_limit "${OMEGA_LIMIT}"
    --goal_max_distance "${GOAL_MAX_DISTANCE}"
    --tau_v "${TAU_V}"
    --tau_omega "${TAU_OMEGA}"
    --near_goal_slowdown_distance "${NEAR_GOAL_SLOWDOWN_DISTANCE}"
    --near_goal_min_action_scale "${NEAR_GOAL_MIN_ACTION_SCALE}"
    --obstacle_slowdown_distance "${OBSTACLE_SLOWDOWN_DISTANCE}"
    --obstacle_min_action_scale "${OBSTACLE_MIN_ACTION_SCALE}"
    --goal_reach_threshold "${GOAL_REACH_THRESHOLD}"
    --collision_threshold "${COLLISION_THRESHOLD}"
    --print_interval_s "${PRINT_INTERVAL_S}"
  )
  if [[ "${DISABLE_ACTION_SCALING}" == "1" ]]; then
    POLICY_ARGS+=(--disable_action_scaling)
  fi
  if [[ "${ALLOW_REVERSE}" == "1" ]]; then
    POLICY_ARGS+=(--allow_reverse)
  fi

  if [[ "${POLICY_RUNTIME}" == "system" ]]; then
    python3 "${SCRIPT_DIR}/gazebo_policy_runner.py" "${POLICY_ARGS[@]}" > "${POLICY_LOG}" 2>&1 &
  elif [[ "${POLICY_RUNTIME}" == "isaaclab" ]]; then
    "${ISAACLAB_SH}" -p "${SCRIPT_DIR}/gazebo_policy_runner.py" "${POLICY_ARGS[@]}" > "${POLICY_LOG}" 2>&1 &
  else
    if python3 -c "import torch" >/dev/null 2>&1; then
      python3 "${SCRIPT_DIR}/gazebo_policy_runner.py" "${POLICY_ARGS[@]}" > "${POLICY_LOG}" 2>&1 &
    else
      "${ISAACLAB_SH}" -p "${SCRIPT_DIR}/gazebo_policy_runner.py" "${POLICY_ARGS[@]}" > "${POLICY_LOG}" 2>&1 &
    fi
  fi
  POLICY_PID=$!

  sleep "${SLEEP_AFTER_POLICY}"
  if ! kill -0 "${POLICY_PID}" 2>/dev/null; then
    echo "[ERROR] policy runner exited early in world ${world_key_or_name}. Last log lines:" | tee -a "${OUT_DIR}/summary.log"
    tail -n 40 "${POLICY_LOG}" | tee -a "${OUT_DIR}/summary.log"
    for pid in "${WORLD_PIDS[@]}"; do
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    done
    exit 1
  fi

  for S in "${SEEDS[@]}"; do
    echo "[RUN] world=${world_key_or_name} seed=${S}" | tee -a "${OUT_DIR}/summary.log"

    EVAL_ARGS=(
      --episodes "${EPISODES}"
      --seed "${S}"
      --startup_timeout_s 20
      --episode_timeout_s "${EPISODE_TIMEOUT}"
      --inter_episode_sleep_s 0.3
      --goal_forward_min "${GOAL_FORWARD_MIN}"
      --goal_forward_max "${GOAL_FORWARD_MAX}"
      --goal_lateral_max "${GOAL_LATERAL_MAX}"
      --scan_aware_goal_sampling "${SCAN_AWARE}"
      --goal_obstacle_margin_m "${GOAL_OBSTACLE_MARGIN}"
      --goal_sample_max_tries "${GOAL_SAMPLE_MAX_TRIES}"
      --goal_reach_threshold "${GOAL_REACH_THRESHOLD}"
      --collision_threshold "${COLLISION_THRESHOLD}"
      --collision_grace_s "${COLLISION_GRACE_S}"
      --collision_consecutive "${COLLISION_CONSEC}"
      --reset_service "${RESET_SERVICE}"
      --reset_service_wait_s "${RESET_WAIT_S}"
      --post_reset_settle_s "${POST_RESET_SETTLE_S}"
      --set_state_wait_s "${SET_STATE_WAIT_S}"
    )
    do_reset_each_episode=1
    if [[ "${RESET_EACH_EPISODE}" == "0" ]]; then
      do_reset_each_episode=0
    elif [[ "${RESET_EACH_EPISODE}" == "1" ]]; then
      do_reset_each_episode=1
    else
      # auto: fixed-cases mode prefers teleport-based reset without world reset.
      if [[ -n "${FIXED_CASES_FILE}" ]]; then
        do_reset_each_episode=0
      fi
    fi
    if [[ "${do_reset_each_episode}" == "1" ]]; then
      EVAL_ARGS+=(--reset_each_episode)
    fi
    if [[ -n "${FIXED_CASES_FILE}" ]]; then
      EVAL_ARGS+=(
        --fixed_cases_file "${FIXED_CASES_FILE}"
        --fixed_cases_repeat "${FIXED_CASES_REPEAT}"
        --pose_reset_mode respawn
      )
      if [[ "${TURN_CMD}" == "topic" ]]; then
        EVAL_ARGS+=(--publish_turn_cmd 1)
      fi
    fi

    python3 "${SCRIPT_DIR}/gazebo_eval_success.py" \
      "${EVAL_ARGS[@]}" \
      2>&1 | tee "${OUT_DIR}/eval_${log_world}_s${S}.log"
  done

  kill "${POLICY_PID}" || true
  wait "${POLICY_PID}" 2>/dev/null || true
  for pid in "${WORLD_PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  done
done

python3 - "${OUT_DIR}" <<'PY'
import glob
import os
import re
import statistics
import sys

out = sys.argv[1]
pat = re.compile(r"\[GAZEBO\]\[RESULT\] (success_rate|collision_rate|timeout_rate)=([0-9.]+)")
by_world = {}

for p in sorted(glob.glob(os.path.join(out, "eval_*_s*.log"))):
    m = re.match(r"eval_(.+)_s\d+\.log", os.path.basename(p))
    if not m:
        continue
    world = m.group(1)
    vals = {}
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            mm = pat.search(line)
            if mm:
                vals[mm.group(1)] = float(mm.group(2))
    if len(vals) == 3:
        by_world.setdefault(world, []).append(vals)

print("\n===== MULTI-SCENE SUMMARY =====")
all_s, all_c, all_t = [], [], []
for world, runs in sorted(by_world.items()):
    s = [r["success_rate"] for r in runs]
    c = [r["collision_rate"] for r in runs]
    t = [r["timeout_rate"] for r in runs]
    all_s += s
    all_c += c
    all_t += t
    print(
        f"{world:24s} "
        f"success={statistics.mean(s):.4f}±{statistics.pstdev(s):.4f} "
        f"collision={statistics.mean(c):.4f}±{statistics.pstdev(c):.4f} "
        f"timeout={statistics.mean(t):.4f}±{statistics.pstdev(t):.4f} (n={len(runs)})"
    )

if all_s:
    print(
        f"OVERALL                  "
        f"success={statistics.mean(all_s):.4f}±{statistics.pstdev(all_s):.4f} "
        f"collision={statistics.mean(all_c):.4f}±{statistics.pstdev(all_c):.4f} "
        f"timeout={statistics.mean(all_t):.4f}±{statistics.pstdev(all_t):.4f} (n={len(all_s)})"
    )
print(f"\nlogs: {out}")
PY

echo "[DONE] logs: ${OUT_DIR}"
