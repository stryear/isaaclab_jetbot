#!/bin/bash
# Quick demo launcher

echo "=========================================="
echo "Isaac Lab Navigation - Quick Demo"
echo "=========================================="
echo ""

# Find latest demo checkpoint
DEMO_CKPT=$(find logs/skrl/lidar_short_nav_demo_direct -name "best_agent.pt" 2>/dev/null | head -1)

if [ -z "$DEMO_CKPT" ]; then
    echo "❌ No demo checkpoint found"
    echo ""
    echo "Train a demo policy first:"
    echo "  make train MODE=demo NUM_ENVS=1"
    echo ""
    echo "Or use a checkpoint from another version:"
    echo "  python scripts/demo_controller.py launch --checkpoint <path>"
    exit 1
fi

echo "Found checkpoint: $DEMO_CKPT"
echo ""
echo "Launching demo..."
echo ""

python3 scripts/demo_controller.py launch --checkpoint "$DEMO_CKPT"
