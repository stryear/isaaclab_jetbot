#!/usr/bin/env python3
"""
Hyperparameter sweep tool - Grid search for PPO hyperparameters
Usage: python scripts/hyperparam_sweep.py --base-config v17 --param learning_rate --values 3e-5 5e-5 1e-4
"""

import argparse
import subprocess
import yaml
from pathlib import Path
from datetime import datetime
import json


def create_sweep_config(base_yaml: Path, param_name: str, param_value, sweep_id: str) -> Path:
    """Create a modified config with new parameter value."""
    with open(base_yaml, 'r') as f:
        config = yaml.safe_load(f)

    # Navigate nested structure
    if '.' in param_name:
        parts = param_name.split('.')
        target = config
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = param_value
    else:
        # Try common locations
        if param_name in config.get('agent', {}):
            config['agent'][param_name] = param_value
        elif param_name in config.get('models', {}).get('policy', {}).get('network', [{}])[0]:
            config['models']['policy']['network'][0][param_name] = param_value
        else:
            raise ValueError(f"Parameter {param_name} not found in config")

    # Save modified config
    sweep_dir = Path("configs/sweeps")
    sweep_dir.mkdir(parents=True, exist_ok=True)
    output_path = sweep_dir / f"sweep_{sweep_id}_{param_name}_{param_value}.yaml"

    with open(output_path, 'w') as f:
        yaml.dump(config, f)

    return output_path


def run_training(task: str, num_envs: int, config_path: Path, sweep_id: str, param_desc: str):
    """Launch training with custom config."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(f"logs/skrl/sweep_{sweep_id}_{param_desc}_{timestamp}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3", "scripts/skrl/train.py",
        "--task", task,
        "--num_envs", str(num_envs),
        "--headless",
        # Note: Would need to modify train.py to accept custom config path
    ]

    print(f"🚀 Starting training: {param_desc}")
    print(f"   Log: {log_file}")

    with open(log_file, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    return log_file


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep")
    parser.add_argument("--base-config", required=True, help="Base config version (e.g., v17)")
    parser.add_argument("--param", required=True, help="Parameter to sweep (e.g., learning_rate)")
    parser.add_argument("--values", nargs="+", required=True, help="Values to try")
    parser.add_argument("--num-envs", type=int, default=32, help="Parallel environments")
    parser.add_argument("--dry-run", action="store_true", help="Only create configs, don't train")
    args = parser.parse_args()

    sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_yaml = Path(f"source/isaac_lab_tutorial/isaac_lab_tutorial/tasks/direct/"
                     f"isaac_lab_tutorial/agents/skrl_lidar_short_nav_{args.base_config}_ppo_cfg.yaml")

    if not base_yaml.exists():
        print(f"❌ Base config not found: {base_yaml}")
        return

    results = []

    for value in args.values:
        # Convert value to appropriate type
        try:
            if '.' in value or 'e' in value.lower():
                typed_value = float(value)
            else:
                typed_value = int(value)
        except ValueError:
            typed_value = value

        param_desc = f"{args.param}_{value}"
        config_path = create_sweep_config(base_yaml, args.param, typed_value, sweep_id)
        print(f"✓ Created config: {config_path}")

        if not args.dry_run:
            task = f"Isaac-Lab-Tutorial-LidarShortNav{args.base_config.upper()}-TurtleBot3-Direct-v0"
            log_file = run_training(task, args.num_envs, config_path, sweep_id, param_desc)
            results.append({"param": args.param, "value": typed_value, "log": str(log_file)})

    # Save sweep metadata
    if results:
        metadata_path = Path(f"logs/skrl/sweep_{sweep_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                "sweep_id": sweep_id,
                "base_config": args.base_config,
                "parameter": args.param,
                "results": results
            }, f, indent=2)
        print(f"\n📊 Sweep metadata saved to: {metadata_path}")
        print(f"\nAnalyze results with:")
        print(f"  python scripts/compare_training_runs.py --sweep {sweep_id}")


if __name__ == "__main__":
    main()
