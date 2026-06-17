#!/usr/bin/env python3
"""
Checkpoint manager - Find, compare, and manage training checkpoints
Usage:
  python scripts/checkpoint_manager.py list --version v17
  python scripts/checkpoint_manager.py best --version v17
  python scripts/checkpoint_manager.py compare --checkpoints ckpt1.pt ckpt2.pt
"""

import argparse
from pathlib import Path
import torch
from datetime import datetime
from typing import List, Dict
import json


def find_checkpoints(version: str, log_root: Path = Path("logs/skrl")) -> List[Path]:
    """Find all checkpoints for a version."""
    pattern = f"lidar_short_nav_{version}_direct/*/checkpoints/*.pt"
    checkpoints = list(log_root.glob(pattern))
    return sorted(checkpoints, key=lambda p: p.stat().st_mtime)


def load_checkpoint_info(ckpt_path: Path) -> Dict:
    """Extract metadata from checkpoint."""
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        info = {
            "path": str(ckpt_path),
            "name": ckpt_path.name,
            "size_mb": ckpt_path.stat().st_size / (1024 * 1024),
            "modified": datetime.fromtimestamp(ckpt_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Extract training info if available
        if isinstance(ckpt, dict):
            info["timestep"] = ckpt.get("timestep", "unknown")
            info["iteration"] = ckpt.get("iteration", "unknown")

            # Try to get network size
            if "policy" in ckpt:
                policy_params = sum(p.numel() for p in ckpt["policy"].values() if isinstance(p, torch.Tensor))
                info["policy_params"] = f"{policy_params:,}"

        return info
    except Exception as e:
        return {
            "path": str(ckpt_path),
            "name": ckpt_path.name,
            "error": str(e)
        }


def list_checkpoints(version: str):
    """List all checkpoints with metadata."""
    checkpoints = find_checkpoints(version)

    if not checkpoints:
        print(f"❌ No checkpoints found for {version}")
        return

    print(f"\n{'='*100}")
    print(f"Checkpoints for {version} ({len(checkpoints)} found)")
    print(f"{'='*100}\n")

    print(f"{'Name':<30} {'Size (MB)':<12} {'Modified':<20} {'Timestep':<12}")
    print("-"*100)

    for ckpt in checkpoints:
        info = load_checkpoint_info(ckpt)
        if "error" not in info:
            print(f"{info['name']:<30} {info['size_mb']:<12.2f} {info['modified']:<20} {info.get('timestep', 'N/A'):<12}")
        else:
            print(f"{info['name']:<30} {'ERROR':<12} {info.get('modified', 'N/A'):<20} {'N/A':<12}")


def find_best_checkpoint(version: str):
    """Find the best checkpoint (by name convention)."""
    checkpoints = find_checkpoints(version)

    # Look for best_agent.pt
    best_ckpts = [c for c in checkpoints if "best" in c.name.lower()]

    if best_ckpts:
        best = best_ckpts[0]
        info = load_checkpoint_info(best)
        print(f"\n🏆 Best checkpoint for {version}:")
        print(f"   Path: {info['path']}")
        print(f"   Size: {info['size_mb']:.2f} MB")
        print(f"   Modified: {info['modified']}")
        if "policy_params" in info:
            print(f"   Policy params: {info['policy_params']}")
        print(f"\nTo evaluate:")
        print(f"  make play CHECKPOINT={info['path']}")
    else:
        print(f"❌ No 'best' checkpoint found for {version}")
        if checkpoints:
            latest = checkpoints[-1]
            print(f"\n💡 Latest checkpoint: {latest}")


def compare_checkpoints(ckpt_paths: List[str]):
    """Compare multiple checkpoints."""
    print(f"\n{'='*100}")
    print(f"Checkpoint Comparison")
    print(f"{'='*100}\n")

    infos = [load_checkpoint_info(Path(p)) for p in ckpt_paths]

    print(f"{'Checkpoint':<40} {'Size (MB)':<12} {'Params':<15} {'Timestep':<12}")
    print("-"*100)

    for info in infos:
        if "error" not in info:
            print(f"{info['name']:<40} {info['size_mb']:<12.2f} "
                  f"{info.get('policy_params', 'N/A'):<15} {info.get('timestep', 'N/A'):<12}")
        else:
            print(f"{info['name']:<40} {'ERROR':<12} {'N/A':<15} {'N/A':<12}")


def cleanup_old_checkpoints(version: str, keep_last: int = 5, dry_run: bool = True):
    """Remove old checkpoints, keeping only the most recent."""
    checkpoints = find_checkpoints(version)

    # Separate best checkpoints from regular ones
    best_ckpts = [c for c in checkpoints if "best" in c.name.lower()]
    regular_ckpts = [c for c in checkpoints if "best" not in c.name.lower()]

    if len(regular_ckpts) <= keep_last:
        print(f"✓ Only {len(regular_ckpts)} regular checkpoints, nothing to clean")
        return

    to_delete = regular_ckpts[:-keep_last]
    total_size = sum(c.stat().st_size for c in to_delete) / (1024 * 1024)

    print(f"\n{'='*80}")
    print(f"Cleanup Plan for {version}")
    print(f"{'='*80}")
    print(f"Total checkpoints: {len(checkpoints)}")
    print(f"Best checkpoints: {len(best_ckpts)} (will keep)")
    print(f"Regular checkpoints: {len(regular_ckpts)}")
    print(f"To delete: {len(to_delete)} (freeing {total_size:.2f} MB)")
    print(f"To keep: {keep_last} most recent")

    if dry_run:
        print(f"\n⚠️  DRY RUN - No files will be deleted")
        print(f"Add --no-dry-run to actually delete files")
    else:
        print(f"\n⚠️  DELETING {len(to_delete)} checkpoints...")
        for ckpt in to_delete:
            ckpt.unlink()
            print(f"   Deleted: {ckpt.name}")
        print(f"✓ Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description="Checkpoint manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # List command
    list_parser = subparsers.add_parser("list", help="List all checkpoints")
    list_parser.add_argument("--version", required=True, help="Version (e.g., v17)")

    # Best command
    best_parser = subparsers.add_parser("best", help="Find best checkpoint")
    best_parser.add_argument("--version", required=True, help="Version (e.g., v17)")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare checkpoints")
    compare_parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths")

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Remove old checkpoints")
    cleanup_parser.add_argument("--version", required=True, help="Version (e.g., v17)")
    cleanup_parser.add_argument("--keep-last", type=int, default=5, help="Keep N most recent")
    cleanup_parser.add_argument("--no-dry-run", action="store_true", help="Actually delete files")

    args = parser.parse_args()

    if args.command == "list":
        list_checkpoints(args.version)
    elif args.command == "best":
        find_best_checkpoint(args.version)
    elif args.command == "compare":
        compare_checkpoints(args.checkpoints)
    elif args.command == "cleanup":
        cleanup_old_checkpoints(args.version, args.keep_last, not args.no_dry_run)


if __name__ == "__main__":
    main()
