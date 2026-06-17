#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/skrl/run_dwb_candidate_family_curve.sh \
    [--episodes 200] [--num-envs 32] [--seed 42] \
    [--task Isaac-Lab-Tutorial-LidarShortNavDefectDwbOscillation-TurtleBot3-Direct-v0] \
    [--algorithm PPO] \
    [--output-dir /abs/path/output]

Description:
  Re-run the independent dual-obstacle evaluation curve for the DWB finetune
  checkpoint family that leads to:
    logs/skrl/lidar_short_nav_defect_dwb_finetune_direct/dwb_candidate_best_20260405_145329/best_agent.pt

  Outputs:
    - per-checkpoint eval logs
    - per-checkpoint state reports
    - dwb_candidate_family_eval_summary.csv
    - dwb_candidate_family_eval_summary.md
    - dwb_candidate_family_eval_summary.png
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

ISAACLAB="${ISAACLAB:-/home/cs/IsaacLab}"
TASK="Isaac-Lab-Tutorial-LidarShortNavDefectDwbOscillation-TurtleBot3-Direct-v0"
ALGO="PPO"
EPISODES=200
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
  OUTPUT_DIR="${PROJ_ROOT}/logs/dwb_stage_eval/$(date +%Y%m%d_%H%M%S)_candidate_family_curve"
fi
mkdir -p "${OUTPUT_DIR}"

MILESTONES=(
  "0403_1946_best"
  "0404_1335_best"
  "0404_1551_best"
  "0404_1722_best"
  "0405_0001_best"
  "0405_official_best"
)

CHECKPOINTS=(
  "${PROJ_ROOT}/logs/skrl/lidar_short_nav_defect_dwb_finetune_direct/2026-04-03_19-46-12_ppo_torch/checkpoints/best_agent.pt"
  "${PROJ_ROOT}/logs/skrl/lidar_short_nav_defect_dwb_finetune_direct/2026-04-04_13-35-33_ppo_torch/checkpoints/best_agent.pt"
  "${PROJ_ROOT}/logs/skrl/lidar_short_nav_defect_dwb_finetune_direct/2026-04-04_15-51-45_ppo_torch/checkpoints/best_agent.pt"
  "${PROJ_ROOT}/logs/skrl/lidar_short_nav_defect_dwb_finetune_direct/2026-04-04_17-22-12_ppo_torch/checkpoints/best_agent.pt"
  "${PROJ_ROOT}/logs/skrl/lidar_short_nav_defect_dwb_finetune_direct/2026-04-05_00-01-26_ppo_torch/checkpoints/best_agent.pt"
  "${PROJ_ROOT}/logs/skrl/lidar_short_nav_defect_dwb_finetune_direct/dwb_candidate_best_20260405_145329/best_agent.pt"
)

for ckpt in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "${ckpt}" ]]; then
    echo "[ERROR] checkpoint not found: ${ckpt}" >&2
    exit 1
  fi
done

run_eval() {
  local milestone="$1"
  local checkpoint="$2"
  local log_file="${OUTPUT_DIR}/eval_${milestone}.log"
  local report_file="${OUTPUT_DIR}/state_report_${milestone}.json"

  echo "[RUN] milestone=${milestone} checkpoint=${checkpoint}"

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

for idx in "${!MILESTONES[@]}"; do
  run_eval "${MILESTONES[$idx]}" "${CHECKPOINTS[$idx]}"
done

export OUTPUT_DIR
export TASK
export ALGO
export EPISODES
export MILESTONES_JSON
MILESTONES_JSON="$(printf '%s\n' "${MILESTONES[@]}" | python3 -c 'import json,sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))')"
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
milestones = json.loads(os.environ["MILESTONES_JSON"])
checkpoints = json.loads(os.environ["CHECKPOINTS_JSON"])


def rate_pct(value: float) -> float:
    return round(float(value) * 100.0, 1)


rows = []
for milestone, checkpoint in zip(milestones, checkpoints):
    report_path = output_dir / f"state_report_{milestone}.json"
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    overall = report["overall"]
    scene0 = report.get("by_scene", {}).get("scene_0", {})
    rows.append(
        {
            "milestone": milestone,
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

summary_csv = output_dir / "dwb_candidate_family_eval_summary.csv"
with summary_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "milestone",
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

summary_md = output_dir / "dwb_candidate_family_eval_summary.md"
with summary_md.open("w", encoding="utf-8") as handle:
    handle.write("# DWB Candidate Family Independent Eval Summary\n\n")
    handle.write("| Milestone | Success | Collision | Timeout |\n")
    handle.write("| --- | ---: | ---: | ---: |\n")
    for row in rows:
        handle.write(
            f"| {row['milestone']} | {row['success_rate_pct']:.1f}% | "
            f"{row['collision_rate_pct']:.1f}% | {row['timeout_rate_pct']:.1f}% |\n"
        )
    handle.write("\n")
    handle.write("## Run Config\n\n")
    handle.write(f"- episodes: `{episodes}`\n")
    handle.write(f"- task: `{task}`\n")
    handle.write(f"- algorithm: `{algo}`\n")
    handle.write("\n")
    handle.write("## Checkpoints\n\n")
    for row in rows:
        handle.write(f"- `{row['milestone']}`: `{row['checkpoint']}`\n")

try:
    import matplotlib.pyplot as plt
except ImportError:
    print(f"[WARN] matplotlib is unavailable. Wrote {summary_csv} and {summary_md}")
else:
    x = list(range(len(rows)))
    display_labels = {
        "0403_1946_best": "0403",
        "0404_1335_best": "0404-1",
        "0404_1551_best": "0404-2",
        "0404_1722_best": "0404-3",
        "0405_0001_best": "0405",
        "0405_official_best": "best",
    }
    labels = [display_labels.get(row["milestone"], row["milestone"]) for row in rows]
    success = [row["success_rate_pct"] for row in rows]
    collision = [row["collision_rate_pct"] for row in rows]
    timeout = [row["timeout_rate_pct"] for row in rows]

    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    ax.plot(x, success, marker="o", linewidth=2.2, markersize=8.5, label="成功率")
    ax.plot(x, collision, marker="s", linewidth=2.2, markersize=8.5, label="碰撞率")
    ax.plot(x, timeout, marker="^", linewidth=2.2, markersize=8.5, label="超时率")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("阶段模型")
    ax.set_ylabel("比例 / %")
    ax.set_title("双障碍辅助场景下候选 best 模型族的独立评估结果")
    ax.set_ylim(0.0, 100.0)
    ax.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.35)
    ax.legend(loc="center right", frameon=False)

    for xs, ys, color, dy in (
        (x, success, "#2a77c7", 2.0),
        (x, collision, "#c83c2c", 2.0),
        (x, timeout, "#2e8b57", 2.0),
    ):
        for xi, yi in zip(xs, ys):
            ax.text(xi, yi + dy, f"{yi:.1f}", color=color, ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    summary_png = output_dir / "dwb_candidate_family_eval_summary.png"
    fig.savefig(summary_png, dpi=180, bbox_inches="tight")
    summary_pdf = output_dir / "dwb_candidate_family_eval_summary.pdf"
    fig.savefig(summary_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] wrote {summary_png}")
    print(f"[DONE] wrote {summary_pdf}")

print(f"[DONE] wrote {summary_csv}")
print(f"[DONE] wrote {summary_md}")
PY

echo "[DONE] output_dir=${OUTPUT_DIR}"
