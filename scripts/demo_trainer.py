#!/usr/bin/env python3
"""
Demo training helper - Quick overfitting on deterministic scene
Usage: python scripts/demo_trainer.py --base-checkpoint v18a_best.pt --iterations 500
"""

import argparse
from pathlib import Path
import subprocess
from datetime import datetime
import shutil

DEFAULT_TASK = "Isaac-Lab-Tutorial-LidarShortNavDemo-TurtleBot3-Direct-v0"


def train_demo_policy(
    base_checkpoint: Path = None,
    max_iterations: int = 500,
    num_envs: int = 1,
    task: str = DEFAULT_TASK,
):
    """Train demo policy with optional warm start."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(f"logs/skrl/demo_train_{timestamp}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Training demo policy...")
    print(f"   Max iterations: {max_iterations}")
    print(f"   Environments: {num_envs}")

    if base_checkpoint:
        print(f"   Warm start from: {base_checkpoint}")

    cmd = [
        "python3", "scripts/skrl/train.py",
        "--task", task,
        "--num_envs", str(num_envs),
        "--max_iterations", str(max_iterations),
        "--headless",
    ]

    if base_checkpoint:
        cmd.extend(["--checkpoint", str(base_checkpoint)])

    print(f"\n   Log: {log_file}")
    print(f"\n   Starting training...")

    with open(log_file, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    print(f"\n✓ Training complete")
    print(f"\nFind checkpoint:")
    print(f"  python scripts/checkpoint_manager.py best --version demo")


def quick_demo_test(checkpoint: Path, num_episodes: int = 10, task: str = DEFAULT_TASK):
    """Quick test of demo policy."""
    print(f"🧪 Quick test ({num_episodes} episodes)...")

    cmd = [
        "python3", "scripts/skrl/play.py",
        "--task", task,
        "--num_envs", "1",
        "--checkpoint", str(checkpoint),
        "--headless",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse output for success rate
    print(f"\n{'='*60}")
    print(f"Quick Test Results")
    print(f"{'='*60}")

    # Look for metrics in output
    for line in result.stdout.split('\n'):
        if 'success' in line.lower() or 'collision' in line.lower():
            print(line)


def create_demo_package(checkpoint: Path, output_dir: Path = None, task: str = DEFAULT_TASK):
    """Package demo for distribution."""
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"demos/package_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Creating demo package...")
    print(f"   Output: {output_dir}")

    # Copy checkpoint
    ckpt_dest = output_dir / "policy.pt"
    shutil.copy(checkpoint, ckpt_dest)
    print(f"   ✓ Copied checkpoint")

    # Create README
    readme_content = f"""# Isaac Lab Navigation Demo

## Quick Start

1. Run the demo:
   ```bash
   ./run_demo.sh
   ```

2. Record video:
   ```bash
   ./run_demo.sh --video
   ```

## Files

- `policy.pt` - Trained policy checkpoint
- `run_demo.sh` - Demo launcher script

## Requirements

- Isaac Lab 4.5+
- Python 3.10+
- CUDA-capable GPU

## Generated

- Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Checkpoint: {checkpoint.name}
- Task: {task}
"""

    with open(output_dir / "README.md", 'w') as f:
        f.write(readme_content)
    print(f"   ✓ Created README")

    # Create run script
    run_script = f"""#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REPO_ROOT="$(cd "${{SCRIPT_DIR}}/../.." && pwd)"
ISAACLAB_SH="${{ISAACLAB_SH:-$HOME/IsaacLab/isaaclab.sh}}"

if [ ! -x "${{ISAACLAB_SH}}" ]; then
    echo "[ERROR] isaaclab.sh not executable: $ISAACLAB_SH"
    exit 1
fi

env PYTHONNOUSERSITE=1 PYTHONPATH= \\
    "${{ISAACLAB_SH}}" -p "${{REPO_ROOT}}/scripts/skrl/play.py" \\
    --task {task} \\
    --num_envs 1 \\
    --checkpoint "${{SCRIPT_DIR}}/policy.pt" \\
    --video_folder "${{SCRIPT_DIR}}/videos/play" \\
    --video_full_episode \\
    "$@"
"""

    script_path = output_dir / "run_demo.sh"
    with open(script_path, 'w') as f:
        f.write(run_script)
    script_path.chmod(0o755)
    print(f"   ✓ Created run script")

    print(f"\n✓ Demo package ready: {output_dir}")
    print(f"\nTest with:")
    print(f"  cd {output_dir} && ./run_demo.sh")


def main():
    parser = argparse.ArgumentParser(description="Demo training helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train demo policy")
    train_parser.add_argument("--base-checkpoint", help="Base checkpoint for warm start")
    train_parser.add_argument("--iterations", type=int, default=500, help="Max iterations")
    train_parser.add_argument("--num-envs", type=int, default=1, help="Parallel environments")
    train_parser.add_argument("--task", default=DEFAULT_TASK, help="Gym task ID")

    # Test command
    test_parser = subparsers.add_parser("test", help="Quick test demo policy")
    test_parser.add_argument("--checkpoint", required=True, help="Policy checkpoint")
    test_parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    test_parser.add_argument("--task", default=DEFAULT_TASK, help="Gym task ID")

    # Package command
    package_parser = subparsers.add_parser("package", help="Create demo package")
    package_parser.add_argument("--checkpoint", required=True, help="Policy checkpoint")
    package_parser.add_argument("--output", help="Output directory")
    package_parser.add_argument("--task", default=DEFAULT_TASK, help="Gym task ID")

    args = parser.parse_args()

    if args.command == "train":
        base_ckpt = Path(args.base_checkpoint) if args.base_checkpoint else None
        train_demo_policy(base_ckpt, args.iterations, args.num_envs, args.task)

    elif args.command == "test":
        ckpt = Path(args.checkpoint)
        quick_demo_test(ckpt, args.episodes, args.task)

    elif args.command == "package":
        ckpt = Path(args.checkpoint)
        output = Path(args.output) if args.output else None
        create_demo_package(ckpt, output, args.task)


if __name__ == "__main__":
    main()
