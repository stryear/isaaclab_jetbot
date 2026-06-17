#!/usr/bin/env bash
set -euo pipefail

INTERVAL_SEC="${INTERVAL_SEC:-7200}"
MPPI_LOG="${MPPI_LOG:-/home/cs/isaaclab_jetbot/logs/train_jobs/mppi_corner_fail_curriculum_v3_fixed_20260327_142646.log}"
DWB_LOG="${DWB_LOG:-/home/cs/isaaclab_jetbot/logs/train_jobs/dwb_oscillation_curriculum_relaxed_20260327_150948.log}"
OUT_LOG="${OUT_LOG:-/home/cs/isaaclab_jetbot/logs/train_jobs/specialized_monitor_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$(dirname "$OUT_LOG")"

echo "[monitor] start: interval=${INTERVAL_SEC}s" | tee -a "$OUT_LOG"
echo "[monitor] mppi_log=$MPPI_LOG" | tee -a "$OUT_LOG"
echo "[monitor] dwb_log=$DWB_LOG" | tee -a "$OUT_LOG"

while true; do
  now="$(date '+%F %T')"
  {
    echo "===== ${now} ====="
    echo "[gpu]"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

    echo "[mppi latest METRIC]"
    if [[ -f "$MPPI_LOG" ]]; then
      tr '\r' '\n' < "$MPPI_LOG" | rg "\[METRIC\]\[LidarNav\]" | tail -n 1
    else
      echo "mppi log not found: $MPPI_LOG"
    fi

    echo "[dwb latest METRIC]"
    if [[ -f "$DWB_LOG" ]]; then
      tr '\r' '\n' < "$DWB_LOG" | rg "\[METRIC\]\[LidarNav\]" | tail -n 1
    else
      echo "dwb log not found: $DWB_LOG"
    fi
  } | tee -a "$OUT_LOG"

  sleep "$INTERVAL_SEC"
done
