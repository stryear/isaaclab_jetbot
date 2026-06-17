#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cs/isaaclab_jetbot"
CONFIG_FILE="$ROOT/source/isaac_lab_tutorial/isaac_lab_tutorial/tasks/direct/isaac_lab_tutorial/isaac_lab_tutorial_env_cfg.py"
MPPI_LOG_DEFAULT="$ROOT/logs/train_jobs/mppi_corner_fail_curriculum_v3_fixed_20260327_142646.log"

APPLY=0
RESTART=0
CHECKPOINT=""
MPPI_LOG="$MPPI_LOG_DEFAULT"

usage() {
  cat <<'EOF'
Usage:
  scripts/mppi_stage2_prepare.sh --apply [--restart] [--checkpoint /abs/path/agent_x.pt] [--log /abs/path/train.log]

Options:
  --apply                 Relax stage-2 curriculum values in the config file.
  --restart               Restart the running mppi training from the latest checkpoint after applying changes.
  --checkpoint PATH       Explicit checkpoint to use on restart.
  --log PATH              mppi train-job log used to locate the active run directory and latest step.
  -h, --help              Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      APPLY=1
      shift
      ;;
    --restart)
      RESTART=1
      shift
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --log)
      MPPI_LOG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[mppi-stage2][error] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$RESTART" -eq 1 ]]; then
  APPLY=1
fi

if [[ "$APPLY" -ne 1 ]]; then
  echo "[mppi-stage2][error] nothing to do; pass --apply and optionally --restart" >&2
  usage >&2
  exit 2
fi

set_tuple_line() {
  local key="$1"
  local value="$2"
  sed -i -E "s|^([[:space:]]*${key}: tuple\\[[^]]+\\] = ).*$|\\1${value}|" "$CONFIG_FILE"
}

latest_mppi_step() {
  local log_path="$1"
  if [[ ! -f "$log_path" ]]; then
    return 1
  fi
  tr '\r' '\n' < "$log_path" \
    | rg '\[METRIC\]\[LidarNav\]' \
    | tail -n 1 \
    | sed -n 's/.*step=\([0-9]\+\).*/\1/p'
}

active_run_dir_from_log() {
  local log_path="$1"
  local run_root
  local exp_name
  if [[ ! -f "$log_path" ]]; then
    return 1
  fi
  run_root="$(
    sed -n 's|^\[INFO\] Logging experiment in directory: \(.*\)|\1|p' "$log_path" | tail -n 1
  )"
  exp_name="$(
    sed -n 's/^Exact experiment name requested from command line \(.*\)$/\1/p' "$log_path" | tail -n 1
  )"
  if [[ -n "$run_root" && -n "$exp_name" && -d "$run_root/$exp_name" ]]; then
    printf '%s\n' "$run_root/$exp_name"
    return 0
  fi
  return 1
}

select_checkpoint_from_run_dir() {
  local run_dir="$1"
  local step_limit="${2:-}"
  local best_ckpt=""
  local best_step=-1
  local path
  local step

  if [[ ! -d "$run_dir/checkpoints" ]]; then
    return 1
  fi

  while IFS= read -r path; do
    step="$(basename "$path" | sed -n 's/^agent_\([0-9]\+\)\.pt$/\1/p')"
    if [[ -z "$step" ]]; then
      continue
    fi
    if [[ -n "$step_limit" && "$step" -gt "$step_limit" ]]; then
      continue
    fi
    if [[ "$step" -gt "$best_step" ]]; then
      best_step="$step"
      best_ckpt="$path"
    fi
  done < <(find "$run_dir/checkpoints" -maxdepth 1 -type f -name 'agent_*.pt' | sort)

  if [[ -n "$best_ckpt" ]]; then
    printf '%s\n' "$best_ckpt"
    return 0
  fi
  return 1
}

running_mppi_checkpoint() {
  ps -eo cmd \
    | rg 'Isaac-Lab-Tutorial-LidarShortNavDefectMppiCornerFail' \
    | sed -n 's/.*--checkpoint \([^ ]*agent_[0-9]\+\.pt\).*/\1/p' \
    | tail -n 1
}

echo "[mppi-stage2] backing up config"
backup_path="${CONFIG_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
cp "$CONFIG_FILE" "$backup_path"
echo "[mppi-stage2] backup=$backup_path"

echo "[mppi-stage2] applying relaxed stage-2 values"
set_tuple_line "defect_mppi_curriculum_stage_steps" "(0, 180_000, 3_000_000)"
set_tuple_line "defect_mppi_curriculum_collision_thresholds" "(0.18, 0.22, 0.24)"
set_tuple_line "defect_mppi_curriculum_corridor_widths" "(1.00, 0.90, 0.85)"
set_tuple_line "defect_mppi_curriculum_inner_radii" "(0.65, 0.60, 0.55)"
set_tuple_line "defect_mppi_curriculum_inner_radius_jitters" "(0.02, 0.04, 0.04)"
set_tuple_line "defect_mppi_curriculum_spawn_jitters" "(0.05, 0.15, 0.15)"
set_tuple_line "defect_mppi_curriculum_goal_jitters" "(0.05, 0.15, 0.15)"
set_tuple_line "defect_mppi_curriculum_straight_pre_lengths" "(1.2, 1.2, 1.1)"
set_tuple_line "defect_mppi_curriculum_straight_post_lengths" "(1.8, 1.7, 1.6)"
set_tuple_line "defect_mppi_curriculum_turn_completion_bonuses" "(15.0, 10.0, 12.0)"
set_tuple_line "defect_mppi_curriculum_time_penalties" "(-0.02, -0.03, -0.03)"
set_tuple_line "defect_mppi_curriculum_goal_distance_maxs" "(2.4, 2.7, 2.5)"
set_tuple_line "defect_mppi_curriculum_goal_y_min_margins" "(0.10, 0.20, 0.20)"
set_tuple_line "defect_mppi_curriculum_goal_y_max_margins" "(0.05, 0.10, 0.15)"
set_tuple_line "defect_mppi_curriculum_v_maxes" "(0.35, 0.40, 0.45)"
set_tuple_line "defect_mppi_curriculum_speed_cap_explores" "(0.30, 0.22, 0.22)"
set_tuple_line "defect_mppi_curriculum_speed_cap_turns" "(0.28, 0.20, 0.19)"
set_tuple_line "defect_mppi_curriculum_speed_cap_narrows" "(0.24, 0.19, 0.19)"
set_tuple_line "defect_mppi_curriculum_speed_cap_narrow_clearances" "(0.30, 0.38, 0.42)"
set_tuple_line "defect_mppi_curriculum_front_danger_distances" "(0.55, 0.65, 0.70)"
set_tuple_line "defect_mppi_curriculum_danger_speed_penalty_scales" "(-0.8, -1.2, -1.5)"
set_tuple_line "defect_mppi_curriculum_progress_reward_scales" "(12.0, 10.0, 9.0)"

python3 -m py_compile "$CONFIG_FILE"
echo "[mppi-stage2] config updated and syntax-checked"

if [[ "$RESTART" -ne 1 ]]; then
  echo "[mppi-stage2] apply-only complete; no restart requested"
  exit 0
fi

if [[ -z "$CHECKPOINT" ]]; then
  run_dir="$(active_run_dir_from_log "$MPPI_LOG" || true)"
  latest_step="$(latest_mppi_step "$MPPI_LOG" || true)"
  if [[ -n "${run_dir:-}" ]]; then
    CHECKPOINT="$(select_checkpoint_from_run_dir "$run_dir" "${latest_step:-}" || true)"
  fi
fi

if [[ -z "$CHECKPOINT" ]]; then
  CHECKPOINT="$(running_mppi_checkpoint || true)"
fi

if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
  echo "[mppi-stage2][error] failed to locate a restart checkpoint" >&2
  exit 1
fi

echo "[mppi-stage2] stopping current mppi training"
pkill -f 'Isaac-Lab-Tutorial-LidarShortNavDefectMppiCornerFail' || true
pkill -f 'launch.sh train --robot turtlebot3_burger --mode mppi_corner_fail' || true
sleep 2

restart_log="$ROOT/logs/train_jobs/mppi_stage2_prepare_restart_$(date +%Y%m%d_%H%M%S).log"
echo "[mppi-stage2] restart_checkpoint=$CHECKPOINT"
echo "[mppi-stage2] restart_log=$restart_log"

nohup "$ROOT/launch.sh" train \
  --robot turtlebot3_burger \
  --mode mppi_corner_fail \
  --algorithm PPO \
  --num_envs 100 \
  --headless \
  --checkpoint "$CHECKPOINT" \
  > "$restart_log" 2>&1 &

echo "[mppi-stage2] started pid=$!"
