#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cs/isaaclab_jetbot"
CFG="$ROOT/source/isaac_lab_tutorial/isaac_lab_tutorial/tasks/direct/isaac_lab_tutorial/isaac_lab_tutorial_env_cfg.py"
PROFILE="mild"
CKPT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --checkpoint)
      CKPT="$2"
      shift 2
      ;;
    *)
      CKPT="$1"
      shift
      ;;
  esac
done

set_tuple_line() {
  local key="$1"
  local value="$2"
  sed -i -E "s|^([[:space:]]*${key}: tuple\[[^]]+\] = ).*$|\\1${value}|" "$CFG"
}

find_latest_dwb_checkpoint() {
  local latest_log=""
  local exp_name=""
  local run_dir=""
  local loaded_ckpt=""

  latest_log="$(ls -1dt "$ROOT"/logs/train_jobs/dwb_oscillation_curriculum*.log 2>/dev/null | head -n 1 || true)"
  if [[ -n "$latest_log" ]]; then
    exp_name="$(rg -o '20[0-9]{2}-[0-9]{2}-[0-9]{2}_[0-9-]{8}_ppo_torch' "$latest_log" | head -n 1 || true)"
    if [[ -n "$exp_name" ]]; then
      run_dir="$ROOT/logs/skrl/lidar_short_nav_v18a_direct/$exp_name/checkpoints"
      CKPT="$(ls -1dt "$run_dir"/agent_*.pt 2>/dev/null | head -n 1 || true)"
      if [[ -n "$CKPT" ]]; then
        return 0
      fi
    fi

    loaded_ckpt="$(sed -n 's|.*Loading model checkpoint from: ||p' "$latest_log" | tail -n 1 || true)"
    if [[ -n "$loaded_ckpt" && -f "$loaded_ckpt" ]]; then
      CKPT="$loaded_ckpt"
      return 0
    fi
  fi

  CKPT="$(ls -1dt "$ROOT"/logs/skrl/lidar_short_nav_v18a_direct/*_ppo_torch/checkpoints/agent_*.pt 2>/dev/null | head -n 1 || true)"
}

if [[ "$PROFILE" == "extreme" ]]; then
  echo "[recover] applying extreme stage0 dwb curriculum values"
  set_tuple_line "defect_dwb_curriculum_collision_thresholds" "(0.08, 0.09, 0.10)"
  set_tuple_line "defect_dwb_curriculum_contact_force_thresholds" "(10.0, 9.0, 8.0)"
  set_tuple_line "defect_dwb_curriculum_goal_distance_mins" "(0.8, 0.9, 1.0)"
  set_tuple_line "defect_dwb_curriculum_goal_distance_maxs" "(1.3, 1.8, 2.2)"
  set_tuple_line "defect_dwb_curriculum_goal_reach_thresholds" "(0.50, 0.45, 0.45)"
  set_tuple_line "defect_dwb_curriculum_corridor_widths" "(3.5, 3.0, 2.6)"
  set_tuple_line "defect_dwb_curriculum_obstacle_spacings" "(1.2, 1.0, 0.9)"
  set_tuple_line "defect_dwb_curriculum_v_maxes" "(0.23, 0.25, 0.35)"
  set_tuple_line "defect_dwb_curriculum_progress_reward_scales" "(12.0, 10.0, 10.0)"
  set_tuple_line "defect_dwb_curriculum_forward_goal_reward_scales" "(0.84, 0.70, 0.70)"
  set_tuple_line "defect_dwb_curriculum_stall_penalties" "(-0.05, -0.06, -0.06)"
  set_tuple_line "defect_dwb_curriculum_lateral_penalty_weights" "(-0.4, -0.45, -0.5)"
  set_tuple_line "defect_dwb_curriculum_passage_bonuses" "(8.0, 6.0, 4.0)"
  set_tuple_line "defect_dwb_curriculum_spawn_jitters" "(0.05, 0.06, 0.08)"
  set_tuple_line "defect_dwb_curriculum_goal_jitters" "(0.05, 0.06, 0.08)"
  set_tuple_line "defect_dwb_curriculum_obstacle_pair_shift_ys" "(0.00, 0.01, 0.03)"
  set_tuple_line "defect_dwb_curriculum_obstacle_x_jitters" "(0.00, 0.01, 0.02)"
  set_tuple_line "defect_dwb_curriculum_obstacle_spacing_jitters" "(0.00, 0.00, 0.01)"
elif [[ "$PROFILE" == "aggressive" ]]; then
  echo "[recover] applying aggressive stage0 dwb curriculum values"
  set_tuple_line "defect_dwb_curriculum_collision_thresholds" "(0.10, 0.16, 0.22)"
  set_tuple_line "defect_dwb_curriculum_contact_force_thresholds" "(8.0, 4.0, 2.0)"
  set_tuple_line "defect_dwb_curriculum_goal_distance_mins" "(0.8, 1.0, 1.0)"
  set_tuple_line "defect_dwb_curriculum_goal_distance_maxs" "(1.6, 2.0, 2.4)"
  set_tuple_line "defect_dwb_curriculum_corridor_widths" "(3.0, 2.0, 2.0)"
  set_tuple_line "defect_dwb_curriculum_obstacle_spacings" "(1.0, 0.6, 0.6)"
  set_tuple_line "defect_dwb_curriculum_v_maxes" "(0.20, 0.50, 0.50)"
  set_tuple_line "defect_dwb_curriculum_lateral_penalty_weights" "(-0.8, -0.6, -0.5)"
  set_tuple_line "defect_dwb_curriculum_passage_bonuses" "(3.0, 2.0, 2.0)"
  set_tuple_line "defect_dwb_curriculum_spawn_jitters" "(0.08, 0.12, 0.20)"
  set_tuple_line "defect_dwb_curriculum_goal_jitters" "(0.08, 0.12, 0.20)"
  set_tuple_line "defect_dwb_curriculum_obstacle_pair_shift_ys" "(0.02, 0.08, 0.12)"
  set_tuple_line "defect_dwb_curriculum_obstacle_x_jitters" "(0.02, 0.05, 0.08)"
  set_tuple_line "defect_dwb_curriculum_obstacle_spacing_jitters" "(0.01, 0.02, 0.03)"
else
  echo "[recover] applying mild stage0 dwb curriculum values"
  set_tuple_line "defect_dwb_curriculum_collision_thresholds" "(0.12, 0.18, 0.24)"
  set_tuple_line "defect_dwb_curriculum_contact_force_thresholds" "(6.0, 3.0, 2.0)"
  set_tuple_line "defect_dwb_curriculum_goal_distance_mins" "(0.9, 1.0, 1.0)"
  set_tuple_line "defect_dwb_curriculum_goal_distance_maxs" "(1.8, 2.2, 2.6)"
  set_tuple_line "defect_dwb_curriculum_corridor_widths" "(2.5, 2.0, 2.0)"
  set_tuple_line "defect_dwb_curriculum_obstacle_spacings" "(0.8, 0.6, 0.6)"
  set_tuple_line "defect_dwb_curriculum_v_maxes" "(0.25, 0.50, 0.50)"
  set_tuple_line "defect_dwb_curriculum_lateral_penalty_weights" "(-0.3, -0.4, -0.5)"
  set_tuple_line "defect_dwb_curriculum_passage_bonuses" "(2.0, 2.0, 2.0)"
  set_tuple_line "defect_dwb_curriculum_spawn_jitters" "(0.10, 0.15, 0.20)"
  set_tuple_line "defect_dwb_curriculum_goal_jitters" "(0.10, 0.15, 0.20)"
  set_tuple_line "defect_dwb_curriculum_obstacle_pair_shift_ys" "(0.04, 0.08, 0.12)"
  set_tuple_line "defect_dwb_curriculum_obstacle_x_jitters" "(0.04, 0.06, 0.08)"
  set_tuple_line "defect_dwb_curriculum_obstacle_spacing_jitters" "(0.01, 0.02, 0.03)"
fi

python3 -m py_compile "$CFG"

echo "[recover] stopping existing dwb process"
pkill -f "Isaac-Lab-Tutorial-LidarShortNavDefectDwbOscillation" || true
sleep 1

if [[ -z "$CKPT" ]]; then
  find_latest_dwb_checkpoint
fi

if [[ ! -f "$CKPT" ]]; then
  echo "[recover][error] checkpoint not found: $CKPT" >&2
  exit 1
fi

ts="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/logs/train_jobs/dwb_oscillation_curriculum_recover_${ts}.log"

echo "[recover] restart from checkpoint: $CKPT"
nohup "$ROOT"/launch.sh train \
  --robot turtlebot3_burger \
  --mode dwb_oscillation \
  --algorithm PPO \
  --num_envs 100 \
  --headless \
  --checkpoint "$CKPT" \
  > "$LOG" 2>&1 &

PID=$!
echo "[recover] started pid=$PID"
echo "[recover] log=$LOG"
