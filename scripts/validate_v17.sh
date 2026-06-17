#!/bin/bash
# Quick validation script for v17 configuration

echo "=========================================="
echo "V17 Configuration Validation"
echo "=========================================="
echo ""

# Test 1: Check environment registration
echo "[1/4] Checking environment registration..."
if python3 scripts/list_envs.py 2>&1 | grep -q "LidarShortNavV17"; then
    echo "  ✓ V17 environment registered"
else
    echo "  ✗ V17 environment NOT found"
    exit 1
fi

# Test 2: Check PPO config file
echo "[2/4] Checking PPO config file..."
if [ -f "source/isaac_lab_tutorial/isaac_lab_tutorial/tasks/direct/isaac_lab_tutorial/agents/skrl_lidar_short_nav_v17_ppo_cfg.yaml" ]; then
    echo "  ✓ PPO config exists"
    echo "    Network layers: $(grep -A 1 'layers:' source/isaac_lab_tutorial/isaac_lab_tutorial/tasks/direct/isaac_lab_tutorial/agents/skrl_lidar_short_nav_v17_ppo_cfg.yaml | tail -1 | tr -d ' ')"
    echo "    Rollouts: $(grep 'rollouts:' source/isaac_lab_tutorial/isaac_lab_tutorial/tasks/direct/isaac_lab_tutorial/agents/skrl_lidar_short_nav_v17_ppo_cfg.yaml | awk '{print $2}')"
    echo "    Learning rate: $(grep 'learning_rate:' source/isaac_lab_tutorial/isaac_lab_tutorial/tasks/direct/isaac_lab_tutorial/agents/skrl_lidar_short_nav_v17_ppo_cfg.yaml | head -1 | awk '{print $2}')"
else
    echo "  ✗ PPO config NOT found"
    exit 1
fi

# Test 3: Check Makefile integration
echo "[3/4] Checking Makefile integration..."
if grep -q "TASK_TURTLEBOT3_SHORT_NAV_V17" Makefile; then
    echo "  ✓ Makefile configured"
else
    echo "  ✗ Makefile NOT configured"
    exit 1
fi

# Test 4: Dry-run smoke test (just parse args, don't launch sim)
echo "[4/4] Testing command-line parsing..."
if python3 scripts/skrl/train.py --help 2>&1 | grep -q "task"; then
    echo "  ✓ Training script accessible"
else
    echo "  ✗ Training script has issues"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ All validation checks passed!"
echo "=========================================="
echo ""
echo "Ready to train. Run one of:"
echo "  ./scripts/train_v17.sh smoke    # 2-min smoke test"
echo "  ./scripts/train_v17.sh quick    # 15-min quick test"
echo "  ./scripts/train_v17.sh          # Full training (2-3 hours)"
echo ""
echo "Or use Makefile directly:"
echo "  make train MODE=v17 NUM_ENVS=64"
echo ""
