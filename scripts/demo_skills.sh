#!/bin/bash
# Quick demo of available skills

echo "=========================================="
echo "DRL Navigation Project - Skills Demo"
echo "=========================================="
echo ""

# Check if training logs exist
if [ ! -d "logs/skrl" ]; then
    echo "⚠️  No training logs found. Run training first:"
    echo "   make train MODE=v17 NUM_ENVS=4"
    exit 1
fi

echo "Available Skills:"
echo ""

echo "1. Training Comparison"
echo "   Command: python scripts/compare_training_runs.py v16 v17 v18"
if ls logs/skrl/short_nav_v1*_*.log 2>/dev/null | head -1 > /dev/null; then
    echo "   Status: ✓ Training logs found"
    VERSIONS=$(ls logs/skrl/short_nav_v1*_*.log 2>/dev/null | sed 's/.*short_nav_\(v[0-9]*\).*/\1/' | sort -u | head -3 | tr '\n' ' ')
    echo "   Available versions: $VERSIONS"
else
    echo "   Status: ⚠️  No training logs yet"
fi
echo ""

echo "2. Checkpoint Manager"
echo "   Command: python scripts/checkpoint_manager.py list --version v17"
if ls logs/skrl/lidar_short_nav_v*/*/checkpoints/*.pt 2>/dev/null | head -1 > /dev/null; then
    CKPT_COUNT=$(ls logs/skrl/lidar_short_nav_v*/*/checkpoints/*.pt 2>/dev/null | wc -l)
    echo "   Status: ✓ Found $CKPT_COUNT checkpoints"
else
    echo "   Status: ⚠️  No checkpoints yet"
fi
echo ""

echo "3. Training Visualization"
echo "   Command: python scripts/plot_training_curves.py v17 v18"
if command -v python3 &> /dev/null; then
    if python3 -c "import matplotlib" 2>/dev/null; then
        echo "   Status: ✓ matplotlib installed"
    else
        echo "   Status: ⚠️  matplotlib not installed (pip install matplotlib)"
    fi
else
    echo "   Status: ⚠️  python3 not found"
fi
echo ""

echo "4. Hyperparameter Sweep"
echo "   Command: python scripts/hyperparam_sweep.py --base-config v17 --param learning_rate --values 3e-5 5e-5"
echo "   Status: ✓ Ready to use"
echo ""

echo "5. Automated Monitoring (Built-in)"
echo "   Command: make monitor-metrics"
if [ -f "scripts/skrl/monitor_training_metrics.py" ]; then
    echo "   Status: ✓ Monitor script available"
else
    echo "   Status: ⚠️  Monitor script not found"
fi
echo ""

echo "6. Automated Checkpoint Evaluation (Built-in)"
echo "   Command: make auto-eval-checkpoints"
if [ -f "scripts/skrl/auto_eval_checkpoints.py" ]; then
    echo "   Status: ✓ Auto-eval script available"
else
    echo "   Status: ⚠️  Auto-eval script not found"
fi
echo ""

echo "=========================================="
echo "Quick Start Examples:"
echo "=========================================="
echo ""

# If we have training logs, show a real example
if ls logs/skrl/short_nav_v1*_*.log 2>/dev/null | head -1 > /dev/null; then
    VERSIONS=$(ls logs/skrl/short_nav_v1*_*.log 2>/dev/null | sed 's/.*short_nav_\(v[0-9]*\).*/\1/' | sort -u | head -3 | tr '\n' ' ')
    echo "Try comparing your training runs:"
    echo "  python scripts/compare_training_runs.py $VERSIONS"
    echo ""
fi

if ls logs/skrl/lidar_short_nav_v*/*/checkpoints/*.pt 2>/dev/null | head -1 > /dev/null; then
    LATEST_VERSION=$(ls -td logs/skrl/lidar_short_nav_v*/ 2>/dev/null | head -1 | sed 's/.*lidar_short_nav_\(v[0-9]*\).*/\1/')
    echo "Find your best checkpoint:"
    echo "  python scripts/checkpoint_manager.py best --version $LATEST_VERSION"
    echo ""
fi

echo "Start automated monitoring:"
echo "  make monitor-metrics MONITOR_POLL_SECONDS=600 &"
echo ""

echo "For full documentation, see:"
echo "  cat SKILLS_GUIDE.md"
echo ""
