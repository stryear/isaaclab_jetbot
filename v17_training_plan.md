# V17 Training Plan: Network Capacity Fix

## Changes from V16

### Network Architecture
- **First layer**: 256 → **512 neurons** (fixes information bottleneck)
- **Architecture**: `[512, 256, 128]` (was `[256, 256, 128]`)
- Shared Actor-Critic (separate=False)
- Activation: ELU

### Training Hyperparameters
- **Rollout**: 64 → **256 steps** (1.07s → 4.27s, captures longer-term dependencies)
- **Learning rate**: 2e-5 → **5e-5** (2.5x faster convergence)
- **Ratio clip**: 0.08 → **0.15** (less conservative updates)
- **Training steps**: 60k → **200k** (3.3x more samples)

### Unchanged from V16
- Observation: 286 dims (16 base + 90 beams × 3 frames)
- Perception: 360° LiDAR, 90 beams, 3-frame stacking
- Reward function: v15b shaping (progress=12.0, forward_goal=0.80, danger=-12.0, clearance=0.24)
- Environment: 0.5-1.0m goals, 80% explore / 20% junction

## Expected Outcomes

### Success Criteria
- **Target success rate**: >55% (vs v16's 43%)
- **Collision rate**: <40% (vs v16's 53%)
- **Training stability**: success rate variance <5% over last 50 iterations

### Diagnostic Metrics to Monitor
1. **Policy entropy**: Should decay gradually (not collapse early)
2. **Value loss**: Should converge smoothly
3. **KL divergence**: Should stay near 0.01 (adaptive LR working)
4. **Gradient norm**: Should be stable (no explosion/vanishing)
5. **Episode length**: Should increase as policy improves

## Training Commands

### Quick Smoke Test (5 min)
```bash
make train MODE=v17 NUM_ENVS=64 ALGORITHM=PPO
# Stop after iter 10, check metrics look reasonable
```

### Full Training Run (estimated 2-3 hours on RTX 4060 Ti)
```bash
make train MODE=v17 NUM_ENVS=64 ALGORITHM=PPO
# Will run 200k steps = ~1200 iterations
# Checkpoints saved every 3200 steps to logs/skrl/lidar_short_nav_v17_direct/
```

### Monitor Training
```bash
# Watch live metrics
tail -f logs/skrl/short_nav_v17_train_*.log | grep METRIC

# Check tensorboard (if enabled)
tensorboard --logdir logs/skrl/lidar_short_nav_v17_direct/
```

## Next Steps Based on Results

### If v17 succeeds (>55% success):
1. **Ablation study**: Test without 3-frame stacking (single frame 106D obs)
2. **Further improvements**: Try 1D CNN or LSTM for time-series modeling
3. **Hyperparameter tuning**: Fine-tune learning rate, rollout length

### If v17 still struggles (<50% success):
1. **Check training curves**: Is it converging? Oscillating? Collapsing?
2. **Reduce observation complexity**: Try single-frame 106D (remove stacking)
3. **Introduce CNN**: 1D convolution over LiDAR beams
4. **Add LSTM**: Replace frame stacking with recurrent memory

### If training is unstable (high variance):
1. **Increase mini-batches**: 8 → 16 (reduce gradient variance)
2. **Reduce learning rate**: 5e-5 → 3e-5
3. **Tighten ratio clip**: 0.15 → 0.12
4. **Add gradient clipping**: Already at 1.0, could try 0.5

## Comparison Table

| Metric | v15b (baseline) | v16 (failed) | v17 (target) |
|--------|----------------|--------------|--------------|
| Observation | 51D (36 beams) | 286D (90×3) | 286D (90×3) |
| Network | [256,256,128] | [256,256,128] | [512,256,128] |
| Rollout | 64 steps | 64 steps | 256 steps |
| Learning rate | 2e-5 | 2e-5 | 5e-5 |
| Ratio clip | 0.08 | 0.08 | 0.15 |
| Training steps | 60k | 60k | 200k |
| Success rate | ~50% | 43% | >55% (goal) |
| Collision rate | ~45% | 53% | <40% (goal) |

## Implementation Notes

### Files Modified
- `isaac_lab_tutorial_env_cfg.py`: Added `LidarShortNavV17TurtleBot3EnvCfg`
- `agents/skrl_lidar_short_nav_v17_ppo_cfg.yaml`: New PPO config
- `__init__.py`: Registered v17 environment
- `Makefile`: Added v17 mode shortcuts

### Network Capacity Calculation
- **v16 bottleneck**: 286 inputs → 256 neurons = 0.89x compression (information loss)
- **v17 expansion**: 286 inputs → 512 neurons = 1.79x expansion (room for features)
- **Parameter increase**: ~147k → ~294k params (2x, still manageable for RTX 4060 Ti)

### Memory Requirements
- **v16**: 64 envs × 64 rollout × 286 obs = 1.17M floats ≈ 4.7 MB
- **v17**: 64 envs × 256 rollout × 286 obs = 4.68M floats ≈ 18.7 MB
- **Network**: ~294k params × 4 bytes ≈ 1.2 MB
- **Total estimated**: <100 MB (well within 16GB VRAM)

## Troubleshooting

### If training crashes with OOM:
```bash
# Reduce parallel environments
make train MODE=v17 NUM_ENVS=32 ALGORITHM=PPO
```

### If convergence is too slow:
```bash
# Check if learning rate is being adapted
grep "learning_rate" logs/skrl/lidar_short_nav_v17_direct/*/agent.txt
```

### If policy collapses (entropy → 0 early):
- Increase `entropy_loss_scale` from 0.001 to 0.01 in yaml
- Reduce `ratio_clip` to prevent large updates

## References
- v16 training log: `logs/skrl/short_nav_v16_train400_20260317_182052.log`
- v16 final metrics: iter 346, success=43%, collision=53%
- Memory notes: `/home/cs/.claude/projects/-home-cs-isaaclab-jetbot/memory/MEMORY.md`
