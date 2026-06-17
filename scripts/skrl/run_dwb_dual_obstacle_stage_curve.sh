#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/skrl/run_dwb_dual_obstacle_stage_curve.sh \
    [--episodes 100] [--num-envs 32] [--seed 42] \
    [--task Isaac-Lab-Tutorial-LidarShortNavDefectDwbOscillation-TurtleBot3-Direct-v0] \
    [--algorithm PPO] \
    [--output-dir /abs/path/output]

Description:
  Re-run independent evaluations for typical staged checkpoints on the
  dwb_oscillation dual-obstacle task and generate:
    - per-checkpoint eval logs
    - per-checkpoint state reports
    - dual_obstacle_stage_eval_summary.csv
    - dual_obstacle_stage_eval_summary.md
    - dual_obstacle_stage_eval_summary.png
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

ISAACLAB="${ISAACLAB:-/home/cs/IsaacLab}"
TASK="Isaac-Lab-Tutorial-LidarShortNavDefectDwbOscillation-TurtleBot3-Direct-v0"
ALGO="PPO"
EPISODES=100
NUM_ENVS=32
SEED=42
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episodes) EPISODES="$2"; shift 2 ;;
    --num-envs|--num_envs) NUM_ENVS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --algorithm) ALGO="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -x "${ISAACLAB}/isaaclab.sh" ]]; then
  echo "[ERROR] isaaclab launcher not found: ${ISAACLAB}/isaaclab.sh" >&2
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${PROJ_ROOT}/logs/dwb_stage_eval/$(date +%Y%m%d_%H%M%S)_dual_obstacle_stage_curve"
fi
mkdir -p "${OUTPUT_DIR}"

STAGE_LABELS=(
  "2458.4k"
  "2514.4k"
  "2524.5k"
  "2529.6k"
  "2534.4k"
  "2544.6k"
)

CHECKPOINTS=(
  "${PROJ_ROOT}/logs/resume_aliases/agent_2458400.pt"
  "${PROJ_ROOT}/logs/resume_aliases/agent_2514396.pt"
  "${PROJ_ROOT}/logs/resume_aliases/agent_2524518.pt"
  "${PROJ_ROOT}/logs/resume_aliases/agent_2529638.pt"
  "${PROJ_ROOT}/logs/resume_aliases/agent_2534361.pt"
  "${PROJ_ROOT}/logs/resume_aliases/agent_2544569.pt"
)

for ckpt in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "${ckpt}" ]]; then
    echo "[ERROR] checkpoint not found: ${ckpt}" >&2
    exit 1
  fi
done

run_eval() {
  local stage_label="$1"
  local checkpoint="$2"
  local stem
  stem="$(basename "${checkpoint}" .pt)"
  local log_file="${OUTPUT_DIR}/eval_${stem}.log"
  local report_file="${OUTPUT_DIR}/state_report_${stem}.json"

  echo "[RUN] stage=${stage_label} checkpoint=${checkpoint}"

  env PYTHONNOUSERSITE=1 PYTHONPATH= \
    "${ISAACLAB}/isaaclab.sh" -p "${PROJ_ROOT}/scripts/skrl/eval_lidar_nav.py" \
    --task "${TASK}" \
    --algorithm "${ALGO}" \
    --ml_framework torch \
    --checkpoint "${checkpoint}" \
    --num_envs "${NUM_ENVS}" \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    --headless \
    --state_report_json "${report_file}" \
    > "${log_file}" 2>&1

  echo "[DONE] log=${log_file}"
  echo "[DONE] report=${report_file}"
}

for idx in "${!STAGE_LABELS[@]}"; do
  run_eval "${STAGE_LABELS[$idx]}" "${CHECKPOINTS[$idx]}"
done

export OUTPUT_DIR
export TASK
export ALGO
export EPISODES
export STAGE_LABELS_JSON
STAGE_LABELS_JSON="$(printf '%s\n' "${STAGE_LABELS[@]}" | python3 -c 'import json,sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))')"
export CHECKPOINTS_JSON
CHECKPOINTS_JSON="$(printf '%s\n' "${CHECKPOINTS[@]}" | python3 -c 'import json,sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))')"

python3 <<'PY'
import csv
import json
import os
from pathlib import Path

output_dir = Path(os.environ["OUTPUT_DIR"])
task = os.environ["TASK"]
algo = os.environ["ALGO"]
episodes = int(os.environ["EPISODES"])
stage_labels = json.loads(os.environ["STAGE_LABELS_JSON"])
checkpoints = json.loads(os.environ["CHECKPOINTS_JSON"])


def rate_pct(value: float) -> float:
    return round(float(value) * 100.0, 1)

rows = []
for stage_label, checkpoint in zip(stage_labels, checkpoints):
    stem = Path(checkpoint).stem
    report_path = output_dir / f"state_report_{stem}.json"
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    overall = report["overall"]
    scene0 = report.get("by_scene", {}).get("scene_0", {})
    rows.append(
        {
            "stage": stage_label,
            "checkpoint": checkpoint,
            "episodes": int(report["episodes"]),
            "success_rate_pct": rate_pct(overall["success_rate"]),
            "collision_rate_pct": rate_pct(overall["collision_rate"]),
            "timeout_rate_pct": rate_pct(overall["timeout_rate"]),
            "scene0_success_rate_pct": rate_pct(scene0.get("success_rate", overall["success_rate"])),
            "scene0_collision_rate_pct": rate_pct(scene0.get("collision_rate", overall["collision_rate"])),
            "scene0_timeout_rate_pct": rate_pct(scene0.get("timeout_rate", overall["timeout_rate"])),
        }
    )

summary_csv = output_dir / "dual_obstacle_stage_eval_summary.csv"
with summary_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "stage",
            "checkpoint",
            "episodes",
            "success_rate_pct",
            "collision_rate_pct",
            "timeout_rate_pct",
            "scene0_success_rate_pct",
            "scene0_collision_rate_pct",
            "scene0_timeout_rate_pct",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

summary_md = output_dir / "dual_obstacle_stage_eval_summary.md"
with summary_md.open("w", encoding="utf-8") as handle:
    handle.write("# DWB Dual-Obstacle Stage Eval Summary\n\n")
    handle.write("| Stage | Success | Collision | Timeout |\n")
    handle.write("| --- | ---: | ---: | ---: |\n")
    for row in rows:
        handle.write(
            f"| {row['stage']} | {row['success_rate_pct']:.1f}% | "
            f"{row['collision_rate_pct']:.1f}% | {row['timeout_rate_pct']:.1f}% |\n"
        )
    handle.write("\n")
    handle.write("## Run Config\n\n")
    handle.write(f"- episodes: `{episodes}`\n")
    handle.write(f"- task: `{task}`\n")
    handle.write(f"- algorithm: `{algo}`\n")

try:
    import matplotlib.pyplot as plt
except ImportError:
    print(f"[WARN] matplotlib is unavailable. Wrote {summary_csv} and {summary_md}")
else:
    x = list(range(len(rows)))
    labels = [row["stage"] for row in rows]
    success = [row["success_rate_pct"] for row in rows]
    collision = [row["collision_rate_pct"] for row in rows]
    timeout = [row["timeout_rate_pct"] for row in rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, success, marker="o", linewidth=2.2, label="Success")
    ax.plot(x, collision, marker="s", linewidth=2.2, label="Collision")
    ax.plot(x, timeout, marker="^", linewidth=2.2, label="Timeout")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Stage Checkpoint")
    ax.set_ylabel("Rate (%)")
    ax.set_title("DWB Dual-Obstacle Independent Eval Curve")
    ax.set_ylim(0.0, max(success + collision + timeout) * 1.1 + 1.0)
    ax.legend(loc="best")
    fig.tight_layout()
    summary_png = output_dir / "dual_obstacle_stage_eval_summary.png"
    fig.savefig(summary_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] wrote {summary_png}")

print(f"[DONE] wrote {summary_csv}")
print(f"[DONE] wrote {summary_md}")
PY

echo "[DONE] output_dir=${OUTPUT_DIR}"
