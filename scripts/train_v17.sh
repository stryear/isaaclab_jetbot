#!/bin/bash
# Quick start script for v17 training

set -e

echo "=========================================="
echo "V17 Training: Network Capacity Fix"
echo "=========================================="
echo ""
echo "Changes from v16:"
echo "  - Network: [256,256,128] → [512,256,128]"
echo "  - Rollout: 64 → 256 steps"
echo "  - Learning rate: 2e-5 → 5e-5"
echo "  - Ratio clip: 0.08 → 0.15"
echo "  - Training steps: 60k → 200k"
echo ""
echo "Expected: >55% success rate (vs v16's 43%)"
echo ""

# Parse arguments
MODE="full"
if [ "$1" = "smoke" ]; then
    MODE="smoke"
    echo "Running SMOKE TEST (10 iterations, ~2 minutes)"
elif [ "$1" = "quick" ]; then
    MODE="quick"
    echo "Running QUICK TEST (100 iterations, ~15 minutes)"
else
    echo "Running FULL TRAINING (1200 iterations, ~2-3 hours)"
fi
echo ""

# Set parameters based on mode
if [ "$MODE" = "smoke" ]; then
    MAX_ITER=10
    NUM_ENVS=32
elif [ "$MODE" = "quick" ]; then
    MAX_ITER=100
    NUM_ENVS=64
else
    MAX_ITER=1200
    NUM_ENVS=64
fi

# Create log directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/skrl/short_nav_v17_${MODE}_${TIMESTAMP}.log"
mkdir -p logs/skrl

echo "Configuration:"
echo "  Environments: $NUM_ENVS"
echo "  Max iterations: $MAX_ITER"
echo "  Log file: $LOG_FILE"
echo ""
echo "Starting training in 3 seconds..."
sleep 3

# Run training
make train MODE=v17 NUM_ENVS=$NUM_ENVS ALGORITHM=PPO 2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo ""
echo "Log saved to: $LOG_FILE"
echo ""
echo "Check final metrics:"
echo "  grep METRIC $LOG_FILE | tail -10"
echo ""
echo "Find best checkpoint:"
echo "  ls -lht logs/skrl/lidar_short_nav_v17_direct/*/checkpoints/"
echo ""
