#!/usr/bin/env python3
"""
Interactive demo controller - Manual control and policy comparison
Usage: python scripts/demo_controller.py --checkpoint best_agent.pt
"""

import argparse
from pathlib import Path
import subprocess
import time

DEFAULT_TASK = "Isaac-Lab-Tutorial-LidarShortNavDemo-TurtleBot3-Direct-v0"


def launch_interactive_demo(checkpoint: Path = None, enable_gui: bool = True, task: str = DEFAULT_TASK):
    """Launch interactive demo with optional policy."""
    print(f"🎮 Launching interactive demo...")

    cmd = [
        "python3", "scripts/skrl/play.py",
        "--task", task,
        "--num_envs", "1",
    ]

    if checkpoint:
        cmd.extend(["--checkpoint", str(checkpoint)])
        print(f"   Policy: {checkpoint}")
    else:
        print(f"   Mode: Random policy")

    if not enable_gui:
        cmd.append("--headless")

    print(f"\n{'='*60}")
    print(f"Controls:")
    print(f"  - Watch the robot navigate")
    print(f"  - Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    subprocess.run(cmd)


def benchmark_demo_policy(checkpoint: Path, num_runs: int = 10, task: str = DEFAULT_TASK):
    """Benchmark policy on demo environment."""
    print(f"⏱️  Benchmarking policy ({num_runs} runs)...")
    print(f"   Checkpoint: {checkpoint}")

    results = []

    for i in range(num_runs):
        print(f"\n   Run {i+1}/{num_runs}...")

        cmd = [
            "python3", "scripts/skrl/play.py",
            "--task", task,
            "--num_envs", "1",
            "--checkpoint", str(checkpoint),
            "--headless",
        ]

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time

        # Parse result
        success = "success" in result.stdout.lower()
        results.append({
            "run": i+1,
            "success": success,
            "time": elapsed,
        })

    # Print summary
    success_count = sum(1 for r in results if r["success"])
    avg_time = sum(r["time"] for r in results) / len(results)

    print(f"\n{'='*60}")
    print(f"Benchmark Results")
    print(f"{'='*60}")
    print(f"Success rate: {success_count}/{num_runs} ({success_count/num_runs*100:.1f}%)")
    print(f"Average time: {avg_time:.2f}s per run")
    print(f"{'='*60}")


def create_demo_script(checkpoint: Path, output_path: Path = None, task: str = DEFAULT_TASK):
    """Create a standalone demo script."""
    if output_path is None:
        output_path = Path("demos/run_demo.sh")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    script_content = f"""#!/bin/bash
# Standalone demo script
# Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REPO_ROOT="$(cd "${{SCRIPT_DIR}}/.." && pwd)"
ISAACLAB_SH="${{ISAACLAB_SH:-$HOME/IsaacLab/isaaclab.sh}}"
CHECKPOINT="{checkpoint.absolute()}"

echo "=========================================="
echo "Isaac Lab Navigation Demo"
echo "=========================================="
echo ""
echo "Checkpoint: $CHECKPOINT"
echo ""

if [ ! -x "$ISAACLAB_SH" ]; then
    echo "[ERROR] isaaclab.sh not executable: $ISAACLAB_SH"
    exit 1
fi

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Run demo
env PYTHONNOUSERSITE=1 PYTHONPATH= \\
    "$ISAACLAB_SH" -p "$REPO_ROOT/scripts/skrl/play.py" \\
    --task {task} \\
    --num_envs 1 \\
    --checkpoint "$CHECKPOINT" \\
    --video_folder "$SCRIPT_DIR/videos/play" \\
    --video_full_episode \\
    "$@"
"""

    with open(output_path, 'w') as f:
        f.write(script_content)

    output_path.chmod(0o755)

    print(f"✓ Demo script created: {output_path}")
    print(f"\nRun with:")
    print(f"  {output_path}")
    print(f"  {output_path} --video  # Record video")


def main():
    parser = argparse.ArgumentParser(description="Interactive demo controller")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Launch command
    launch_parser = subparsers.add_parser("launch", help="Launch interactive demo")
    launch_parser.add_argument("--checkpoint", help="Policy checkpoint (optional)")
    launch_parser.add_argument("--headless", action="store_true", help="Run without GUI")
    launch_parser.add_argument("--task", default=DEFAULT_TASK, help="Gym task ID")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark policy")
    bench_parser.add_argument("--checkpoint", required=True, help="Policy checkpoint")
    bench_parser.add_argument("--runs", type=int, default=10, help="Number of runs")
    bench_parser.add_argument("--task", default=DEFAULT_TASK, help="Gym task ID")

    # Create script command
    script_parser = subparsers.add_parser("create-script", help="Create standalone demo script")
    script_parser.add_argument("--checkpoint", required=True, help="Policy checkpoint")
    script_parser.add_argument("--output", help="Output script path")
    script_parser.add_argument("--task", default=DEFAULT_TASK, help="Gym task ID")

    args = parser.parse_args()

    if args.command == "launch":
        ckpt = Path(args.checkpoint) if args.checkpoint else None
        launch_interactive_demo(ckpt, not args.headless, args.task)

    elif args.command == "benchmark":
        ckpt = Path(args.checkpoint)
        benchmark_demo_policy(ckpt, args.runs, args.task)

    elif args.command == "create-script":
        ckpt = Path(args.checkpoint)
        output = Path(args.output) if args.output else None
        create_demo_script(ckpt, output, args.task)


if __name__ == "__main__":
    main()
