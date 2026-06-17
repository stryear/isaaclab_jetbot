# Demo Skills - Quick Reference Card

## 🎯 One-Line Commands

```bash
# Train demo (30 min, 95% success)
make train MODE=demo NUM_ENVS=1

# Launch demo
./scripts/quick_demo.sh

# Record video
python scripts/demo_visualizer.py record --checkpoint best.pt --episodes 5

# Package for distribution
python scripts/demo_trainer.py package --checkpoint best.pt
```

---

## 📋 Complete Workflow

### Step 1: Train Demo Policy
```bash
# Option A: From scratch (2-3 hours)
make train MODE=demo NUM_ENVS=1

# Option B: Warm start from v18a (30 minutes) ⭐ Recommended
python scripts/demo_trainer.py train \
  --base-checkpoint logs/skrl/lidar_short_nav_v18a_direct/*/checkpoints/best_agent.pt \
  --iterations 300
```

### Step 2: Test Performance
```bash
# Quick test (10 episodes)
python scripts/demo_trainer.py test \
  --checkpoint logs/skrl/lidar_short_nav_demo_direct/*/checkpoints/best_agent.pt

# Benchmark (10 runs with timing)
python scripts/demo_controller.py benchmark \
  --checkpoint best_agent.pt \
  --runs 10
```

### Step 3: Create Demonstration
```bash
# Interactive demo with GUI
python scripts/demo_controller.py launch --checkpoint best_agent.pt

# Record video
python scripts/demo_visualizer.py record \
  --checkpoint best_agent.pt \
  --episodes 5 \
  --output demos/my_demo
```

### Step 4: Package & Distribute
```bash
# Create standalone package
python scripts/demo_trainer.py package \
  --checkpoint best_agent.pt \
  --output demos/navigation_demo_v1

# Compress for sharing
tar -czf navigation_demo_v1.tar.gz demos/navigation_demo_v1/
```

---

## 🛠️ Available Tools

| Tool | Purpose | Command |
|------|---------|---------|
| **demo_trainer.py** | Train/test demo policy | `train`, `test`, `package` |
| **demo_controller.py** | Interactive control | `launch`, `benchmark`, `create-script` |
| **demo_visualizer.py** | Video recording | `record`, `analyze`, `compare` |
| **quick_demo.sh** | One-click launcher | `./scripts/quick_demo.sh` |

---

## 🎬 Demo Features

### Environment
- **Single environment** (num_envs=1)
- **Deterministic scene** (fixed 2m corridor)
- **Fixed spawn** (no randomness)
- **Higher speed** (0.30 m/s)

### Network
- **CNN architecture** for LiDAR
- **2D convolution** (32→64 filters)
- **Fusion layer** (512+16→512→256)

### Performance
- **Target**: >95% success rate
- **Training**: 300 iterations (~30 min with warm start)
- **Episode length**: 200-300 steps

---

## 📊 Expected Results

### Training Progress
```
Iteration 100: Success 75%, Collision 20%, Timeout 5%
Iteration 200: Success 90%, Collision 8%, Timeout 2%
Iteration 300: Success 96%, Collision 3%, Timeout 1%
```

### Benchmark Output
```
============================================================
Benchmark Results
============================================================
Success rate: 19/20 (95.0%)
Average time: 12.3s per run
============================================================
```

---

## 🚀 Quick Start (3 Steps)

```bash
# 1. Train (or use existing checkpoint)
make train MODE=demo NUM_ENVS=1

# 2. Test
python scripts/demo_trainer.py test --checkpoint best.pt

# 3. Demo
./scripts/quick_demo.sh
```

---

## 📦 Distribution Package Contents

```
navigation_demo_v1/
├── policy.pt          # Trained policy (5-10 MB)
├── run_demo.sh        # Launch script
├── README.md          # Usage instructions
└── config.json        # Environment config
```

**Usage by recipient**:
```bash
cd navigation_demo_v1
./run_demo.sh              # Run demo
./run_demo.sh --video      # Record video
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Success rate <90% | Continue training: `--iterations 200` |
| Video recording fails | Install ffmpeg: `sudo apt install ffmpeg` |
| Robot doesn't move | Check checkpoint path and obs dimensions |
| Training too slow | Use warm start from v18a checkpoint |

---

## 📚 Full Documentation

- **Complete guide**: `DEMO_SKILLS_GUIDE.md`
- **All skills**: `SKILLS_GUIDE.md`
- **V17 training**: `v17_training_plan.md`

---

## 💡 Pro Tips

1. **Always warm start** from v18a for faster convergence
2. **Test on demo env** before recording videos
3. **Use benchmark** to verify consistency
4. **Package demos** for easy sharing
5. **Record multiple episodes** for best footage

---

## ⚡ Most Used Commands

```bash
# Train with warm start
python scripts/demo_trainer.py train --base-checkpoint v18a_best.pt --iterations 300

# Quick test
python scripts/demo_trainer.py test --checkpoint best.pt

# Launch demo
./scripts/quick_demo.sh

# Record video
python scripts/demo_visualizer.py record --checkpoint best.pt --episodes 5

# Package
python scripts/demo_trainer.py package --checkpoint best.pt
```

---

**Created**: 2026-03-18
**Version**: 1.0
**Status**: ✅ All tools tested and ready
