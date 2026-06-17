# V17 Training Guide - Quick Start

## ✓ Configuration Complete

All v17 files are ready:
- Environment config: `LidarShortNavV17TurtleBot3EnvCfg`
- PPO config: `skrl_lidar_short_nav_v17_ppo_cfg.yaml`
- Makefile integration: `MODE=v17`
- Training scripts: `scripts/train_v17.sh`

## Training Commands

### Option 1: Using Makefile (Recommended)

```bash
# Smoke test (2 minutes, 4 envs, 10 iterations)
make train MODE=v17 NUM_ENVS=4 ALGORITHM=PPO

# Quick test (15 minutes, 32 envs, 100 iterations)
make train MODE=v17 NUM_ENVS=32 ALGORITHM=PPO

# Full training (2-3 hours, 64 envs, ~1200 iterations)
make train MODE=v17 NUM_ENVS=64 ALGORITHM=PPO
```

### Option 2: Using launch.sh

```bash
# Full training
./launch.sh train --robot turtlebot3_burger --mode v17 --num_envs 64 --algorithm PPO

# With custom max iterations
./launch.sh train --robot turtlebot3_burger --mode v17 --num_envs 64 --max_iterations 500
```

### Option 3: Using helper script

```bash
# Smoke test
./scripts/train_v17.sh smoke

# Quick test
./scripts/train_v17.sh quick

# Full training
./scripts/train_v17.sh
```

## Monitoring Training

### Real-time metrics
```bash
# Watch training progress
tail -f logs/skrl/short_nav_v17_*.log | grep METRIC

# Example output:
# [METRIC][LidarNav] iter=100 step=16384 success=45.0% collision=48.0% timeout=7.0%
```

### Key metrics to watch
- **success**: Target >55% (v16 was 43%)
- **collision**: Target <40% (v16 was 53%)
- **avg_return**: Should increase over time
- **avg_len**: Should increase (policy becomes more cautious)

### Automated metric monitoring (threshold alerts)
```bash
# One-time check (prints latest values + alerts)
make monitor-metrics EXTRA_ARGS="--once"

# Continuous monitoring every 10 minutes
make monitor-metrics MONITOR_POLL_SECONDS=600
```

Default alert thresholds in `monitor_training_metrics.py`:
- `Policy / Entropy` < `0.10`
- `Learning / KL divergence` outside `[0.005, 0.020]`
- `Value / Explained variance` < `0.80`
- `Optimization / Grad norm` > `5.0`

Monitor outputs:
- `logs/auto_monitor_v17/metrics_summary.csv`
- `logs/auto_monitor_v17/metrics_summary.jsonl`

### Automated checkpoint evaluation
```bash
# Dry-run: only detect pending checkpoints
make auto-eval-checkpoints EXTRA_ARGS="--once --dry-run"

# Continuous evaluation: check every 10 minutes, eval 50 episodes on 16 envs
make auto-eval-checkpoints MONITOR_POLL_SECONDS=600 \
  EXTRA_ARGS="--episodes 50 --num-envs 16"

# If you want to backfill all existing checkpoints, add:
# EXTRA_ARGS="--episodes 50 --num-envs 16 --catch-up-existing"
```

Evaluation outputs:
- `logs/auto_eval_v17/summary.csv`
- `logs/auto_eval_v17/eval_logs/*.log`
- `logs/auto_eval_v17/eval_reports/*_state_report.json`

The auto-eval worker skips existing checkpoints on first startup and then only evaluates newly generated `agent_*.pt`.
Use `--include-best` in `EXTRA_ARGS` if you also want to track updates to `best_agent.pt`.

### Find checkpoints
```bash
# List all checkpoints
ls -lht logs/skrl/lidar_short_nav_v17_direct/*/checkpoints/

# Find best checkpoint
find logs/skrl/lidar_short_nav_v17_direct -name "best_agent.pt"
```

## Configuration Summary

### Network Architecture
```yaml
Policy & Value: [512, 256, 128]  # First layer 2x larger than v16
Activation: ELU
Shared: Yes (separate=False)
```

### Training Hyperparameters
```yaml
Rollout: 256 steps (4.27 seconds @ 60Hz)
Learning rate: 5e-5 (KL-adaptive)
Ratio clip: 0.15
Mini-batches: 8
Epochs: 8
Training steps: 200,000
```

### Observation Space
```
286 dimensions:
  - 16 base features (velocity, goal, actions, etc.)
  - 270 LiDAR features (90 beams × 3 frames)
```

## Expected Timeline

| Mode | Envs | Iterations | Time | Purpose |
|------|------|------------|------|---------|
| Smoke | 4 | 10 | 2 min | Verify config works |
| Quick | 32 | 100 | 15 min | Check learning trend |
| Full | 64 | ~1200 | 2-3 hrs | Complete training |

## Troubleshooting

### Out of memory
```bash
# Reduce parallel environments
make train MODE=v17 NUM_ENVS=32 ALGORITHM=PPO
```

### Training too slow
```bash
# Check GPU utilization
nvidia-smi

# Reduce environments if GPU is maxed out
make train MODE=v17 NUM_ENVS=48 ALGORITHM=PPO
```

### Want to resume training
```bash
# Find latest checkpoint
CKPT=$(find logs/skrl/lidar_short_nav_v17_direct -name "agent_*.pt" | sort | tail -1)

# Resume from checkpoint
./launch.sh train --robot turtlebot3_burger --mode v17 --num_envs 64 --checkpoint $CKPT
```

## Next Steps After Training

### Evaluate the policy
```bash
# Play the best checkpoint
make play CHECKPOINT=logs/skrl/lidar_short_nav_v17_direct/*/checkpoints/best_agent.pt
```

### Compare with v16
```bash
# Extract final metrics
echo "=== V16 Final ==="
grep METRIC logs/skrl/short_nav_v16_train400_20260317_182052.log | tail -5

echo "=== V17 Final ==="
grep METRIC logs/skrl/short_nav_v17_*.log | tail -5
```

### If v17 succeeds (>55% success)
1. Try ablation: single-frame LiDAR (remove stacking)
2. Experiment with CNN or LSTM architectures
3. Fine-tune hyperparameters

### If v17 struggles (<50% success)
1. Check training curves for instability
2. Simplify observation (remove frame stacking)
3. Add 1D CNN for spatial features
4. Try LSTM for temporal modeling

## Files Reference

```
Configuration:
  source/isaac_lab_tutorial/isaac_lab_tutorial/tasks/direct/isaac_lab_tutorial/
    ├── isaac_lab_tutorial_env_cfg.py (LidarShortNavV17TurtleBot3EnvCfg)
    └── agents/skrl_lidar_short_nav_v17_ppo_cfg.yaml

Scripts:
  scripts/
    ├── train_v17.sh (helper script)
    └── skrl/train.py (main training script)

Logs:
  logs/skrl/
    ├── short_nav_v17_*.log (training logs)
    └── lidar_short_nav_v17_direct/ (checkpoints & tensorboard)

Documentation:
  v17_training_plan.md (detailed analysis)
  v17_quickstart.md (this file)
```

## Support

If you encounter issues:
1. Check logs: `tail -100 logs/skrl/short_nav_v17_*.log`
2. Verify GPU: `nvidia-smi`
3. Check disk space: `df -h`
4. Review training plan: `cat v17_training_plan.md`

Good luck with training! 🚀
