#!/usr/bin/env python3
"""
Demo mode visualization and recording tool
Usage: python scripts/demo_visualizer.py --checkpoint best_agent.pt --episodes 5
"""

import argparse
import subprocess
from pathlib import Path
import json
from datetime import datetime


def record_demo_episode(checkpoint: Path, num_episodes: int = 5, output_dir: Path = None):
    """Record demo episodes with visualization."""
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"demos/demo_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎬 Recording {num_episodes} demo episodes...")
    print(f"   Checkpoint: {checkpoint}")
    print(f"   Output: {output_dir}")

    # Run demo with video recording
    cmd = [
        "python3", "scripts/skrl/play.py",
        "--task", "Isaac-Lab-Tutorial-LidarShortNavDemo-TurtleBot3-Direct-v0",
        "--num_envs", "1",
        "--checkpoint", str(checkpoint),
        "--video",
        "--video_length", str(num_episodes * 1800),  # Assume max 1800 steps per episode
    ]

    log_file = output_dir / "recording.log"
    with open(log_file, 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    if result.returncode == 0:
        print(f"✓ Recording complete")
        print(f"   Log: {log_file}")

        # Find generated video
        video_files = list(Path("videos").glob("*.mp4"))
        if video_files:
            latest_video = max(video_files, key=lambda p: p.stat().st_mtime)
            print(f"   Video: {latest_video}")
    else:
        print(f"❌ Recording failed (see {log_file})")


def analyze_demo_performance(checkpoint: Path, num_episodes: int = 50):
    """Analyze demo performance statistics."""
    print(f"📊 Analyzing demo performance ({num_episodes} episodes)...")

    cmd = [
        "python3", "scripts/skrl/play.py",
        "--task", "Isaac-Lab-Tutorial-LidarShortNavDemo-TurtleBot3-Direct-v0",
        "--num_envs", "1",
        "--checkpoint", str(checkpoint),
    ]

    # Run and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse metrics from output
    metrics = {
        "success_count": 0,
        "collision_count": 0,
        "timeout_count": 0,
        "total_episodes": num_episodes,
    }

    for line in result.stdout.split('\n'):
        if 'success' in line.lower():
            # Parse success rate
            pass

    print(f"\n{'='*60}")
    print(f"Demo Performance Summary")
    print(f"{'='*60}")
    print(f"Success rate: {metrics['success_count']/num_episodes*100:.1f}%")
    print(f"Collision rate: {metrics['collision_count']/num_episodes*100:.1f}%")
    print(f"Timeout rate: {metrics['timeout_count']/num_episodes*100:.1f}%")


def create_demo_comparison(checkpoints: list, output_path: Path = None):
    """Create side-by-side comparison of multiple checkpoints."""
    if output_path is None:
        output_path = Path("demos/comparison.html")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"🎥 Creating demo comparison...")

    # Record each checkpoint
    videos = []
    for ckpt in checkpoints:
        ckpt_path = Path(ckpt)
        print(f"   Recording {ckpt_path.name}...")

        # Record single episode
        output_dir = Path(f"demos/temp_{ckpt_path.stem}")
        record_demo_episode(ckpt_path, num_episodes=1, output_dir=output_dir)

        # Find video
        video_files = list(output_dir.glob("*.mp4"))
        if video_files:
            videos.append(video_files[0])

    # Create HTML comparison page
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Demo Comparison</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .video-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
        .video-item {{ border: 1px solid #ccc; padding: 10px; }}
        video {{ width: 100%; }}
    </style>
</head>
<body>
    <h1>Demo Comparison</h1>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <div class="video-grid">
"""

    for i, (ckpt, video) in enumerate(zip(checkpoints, videos)):
        html_content += f"""
        <div class="video-item">
            <h3>Checkpoint {i+1}: {Path(ckpt).name}</h3>
            <video controls>
                <source src="{video.relative_to(output_path.parent)}" type="video/mp4">
            </video>
        </div>
"""

    html_content += """
    </div>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"✓ Comparison page created: {output_path}")
    print(f"   Open in browser: file://{output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="Demo visualization tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Record command
    record_parser = subparsers.add_parser("record", help="Record demo episodes")
    record_parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    record_parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    record_parser.add_argument("--output", help="Output directory")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze demo performance")
    analyze_parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    analyze_parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple checkpoints")
    compare_parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths")
    compare_parser.add_argument("--output", help="Output HTML path")

    args = parser.parse_args()

    if args.command == "record":
        ckpt = Path(args.checkpoint)
        output = Path(args.output) if args.output else None
        record_demo_episode(ckpt, args.episodes, output)

    elif args.command == "analyze":
        ckpt = Path(args.checkpoint)
        analyze_demo_performance(ckpt, args.episodes)

    elif args.command == "compare":
        output = Path(args.output) if args.output else None
        create_demo_comparison(args.checkpoints, output)


if __name__ == "__main__":
    main()
