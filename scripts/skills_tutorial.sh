#!/bin/bash
# Interactive Skills Tutorial
# 逐步演示如何使用所有创建的 skills

clear
echo "=========================================="
echo "Isaac Lab Navigation - Skills Tutorial"
echo "=========================================="
echo ""
echo "本教程将演示如何使用所有创建的工具"
echo ""
read -p "按 Enter 继续..."

# ============================================================================
# Part 1: 训练对比工具
# ============================================================================
clear
echo "=========================================="
echo "Part 1: 训练对比工具"
echo "=========================================="
echo ""
echo "用途：对比多个训练版本的性能"
echo ""
echo "示例命令："
echo "  python scripts/compare_training_runs.py v16 v17 v18"
echo ""
echo "输出："
echo "  - 最终成功率"
echo "  - 峰值成功率"
echo "  - 碰撞率"
echo "  - 改进幅度"
echo ""
read -p "是否运行示例？(y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # 查找可用的版本
    VERSIONS=$(ls logs/skrl/short_nav_v*_*.log 2>/dev/null | sed 's/.*short_nav_\(v[0-9]*\).*/\1/' | sort -u | head -3 | tr '\n' ' ')
    if [ -n "$VERSIONS" ]; then
        echo ""
        echo "运行: python scripts/compare_training_runs.py $VERSIONS"
        echo ""
        python scripts/compare_training_runs.py $VERSIONS
    else
        echo "⚠️  未找到训练日志，跳过演示"
    fi
fi
echo ""
read -p "按 Enter 继续..."

# ============================================================================
# Part 2: Checkpoint 管理工具
# ============================================================================
clear
echo "=========================================="
echo "Part 2: Checkpoint 管理工具"
echo "=========================================="
echo ""
echo "用途：查找、对比和管理训练 checkpoint"
echo ""
echo "常用命令："
echo "  1. 列出所有 checkpoint"
echo "     python scripts/checkpoint_manager.py list --version v17"
echo ""
echo "  2. 找到最佳 checkpoint"
echo "     python scripts/checkpoint_manager.py best --version v17"
echo ""
echo "  3. 清理旧 checkpoint（保留最新 5 个）"
echo "     python scripts/checkpoint_manager.py cleanup --version v17 --keep-last 5"
echo ""
read -p "选择操作 (1=列出, 2=找最佳, 3=清理, 0=跳过): " choice
echo ""

case $choice in
    1)
        # 找一个有 checkpoint 的版本
        VERSION=$(ls -d logs/skrl/lidar_short_nav_v*_direct 2>/dev/null | head -1 | sed 's/.*lidar_short_nav_\(v[0-9a-z]*\)_direct/\1/')
        if [ -n "$VERSION" ]; then
            echo "运行: python scripts/checkpoint_manager.py list --version $VERSION"
            echo ""
            python scripts/checkpoint_manager.py list --version $VERSION | head -20
        else
            echo "⚠️  未找到 checkpoint 目录"
        fi
        ;;
    2)
        VERSION=$(ls -d logs/skrl/lidar_short_nav_v*_direct 2>/dev/null | head -1 | sed 's/.*lidar_short_nav_\(v[0-9a-z]*\)_direct/\1/')
        if [ -n "$VERSION" ]; then
            echo "运行: python scripts/checkpoint_manager.py best --version $VERSION"
            echo ""
            python scripts/checkpoint_manager.py best --version $VERSION
        else
            echo "⚠️  未找到 checkpoint 目录"
        fi
        ;;
    3)
        echo "⚠️  清理操作需要谨慎，这里仅演示 dry-run"
        VERSION=$(ls -d logs/skrl/lidar_short_nav_v*_direct 2>/dev/null | head -1 | sed 's/.*lidar_short_nav_\(v[0-9a-z]*\)_direct/\1/')
        if [ -n "$VERSION" ]; then
            echo "运行: python scripts/checkpoint_manager.py cleanup --version $VERSION --keep-last 5"
            echo ""
            python scripts/checkpoint_manager.py cleanup --version $VERSION --keep-last 5
        else
            echo "⚠️  未找到 checkpoint 目录"
        fi
        ;;
esac
echo ""
read -p "按 Enter 继续..."

# ============================================================================
# Part 3: 可视化工具
# ============================================================================
clear
echo "=========================================="
echo "Part 3: 训练曲线可视化"
echo "=========================================="
echo ""
echo "用途：绘制训练曲线，对比不同版本"
echo ""
echo "示例命令："
echo "  python scripts/plot_training_curves.py v16 v17 v18 \\"
echo "    --metrics success collision avg_return \\"
echo "    --output training_comparison.png"
echo ""
echo "支持的指标："
echo "  - success (成功率)"
echo "  - collision (碰撞率)"
echo "  - timeout (超时率)"
echo "  - avg_return (平均回报)"
echo "  - avg_len (平均步长)"
echo "  - avg_goal_end_dist (终点距离)"
echo "  - avg_min_lidar (最小 LiDAR 距离)"
echo ""

# 检查 matplotlib
if python3 -c "import matplotlib" 2>/dev/null; then
    echo "✓ matplotlib 已安装"
    echo ""
    read -p "是否生成示例图表？(y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        VERSIONS=$(ls logs/skrl/short_nav_v*_*.log 2>/dev/null | sed 's/.*short_nav_\(v[0-9]*\).*/\1/' | sort -u | head -3 | tr '\n' ' ')
        if [ -n "$VERSIONS" ]; then
            OUTPUT="demos/training_curves_demo.png"
            mkdir -p demos
            echo "运行: python scripts/plot_training_curves.py $VERSIONS --output $OUTPUT"
            echo ""
            python scripts/plot_training_curves.py $VERSIONS --output $OUTPUT
            if [ -f "$OUTPUT" ]; then
                echo ""
                echo "✓ 图表已保存到: $OUTPUT"
                echo "  使用图片查看器打开: xdg-open $OUTPUT"
            fi
        else
            echo "⚠️  未找到训练日志"
        fi
    fi
else
    echo "⚠️  matplotlib 未安装"
    echo "   安装: pip install matplotlib"
fi
echo ""
read -p "按 Enter 继续..."

# ============================================================================
# Part 4: 自动监控工具
# ============================================================================
clear
echo "=========================================="
echo "Part 4: 自动监控工具（已内置）"
echo "=========================================="
echo ""
echo "用途：实时监控训练指标，自动告警"
echo ""
echo "启动监控："
echo "  make monitor-metrics MONITOR_LOG_ROOT=logs/skrl/lidar_short_nav_v17_direct"
echo ""
echo "监控内容："
echo "  - Policy entropy (策略熵)"
echo "  - KL divergence (KL 散度)"
echo "  - Explained variance (解释方差)"
echo "  - Gradient norm (梯度范数)"
echo ""
echo "告警阈值："
echo "  - Entropy < 0.10 (策略过早收敛)"
echo "  - KL divergence 不在 [0.005, 0.020] (更新步长异常)"
echo "  - Explained variance < 0.80 (值函数拟合差)"
echo "  - Grad norm > 5.0 (梯度爆炸)"
echo ""
echo "输出文件："
echo "  - logs/auto_monitor_v17/metrics_summary.csv"
echo "  - logs/auto_monitor_v17/metrics_summary.jsonl"
echo ""
echo "💡 提示：在训练时在另一个终端运行监控"
echo ""
read -p "按 Enter 继续..."

# ============================================================================
# Part 5: 自动评估工具
# ============================================================================
clear
echo "=========================================="
echo "Part 5: 自动评估工具（已内置）"
echo "=========================================="
echo ""
echo "用途：自动评估新生成的 checkpoint"
echo ""
echo "启动评估："
echo "  make auto-eval-checkpoints \\"
echo "    MONITOR_LOG_ROOT=logs/skrl/lidar_short_nav_v17_direct \\"
echo "    EXTRA_ARGS=\"--episodes 50 --num-envs 16\""
echo ""
echo "工作流程："
echo "  1. 每 10 分钟检查一次新 checkpoint"
echo "  2. 自动运行 50 个 episode 评估"
echo "  3. 生成性能报告和状态分析"
echo "  4. 保存到 CSV 文件"
echo ""
echo "输出文件："
echo "  - logs/auto_eval_v17/summary.csv"
echo "  - logs/auto_eval_v17/eval_logs/*.log"
echo "  - logs/auto_eval_v17/eval_reports/*_state_report.json"
echo ""
echo "💡 提示：在训练时在另一个终端运行评估"
echo ""
read -p "按 Enter 继续..."

# ============================================================================
# Part 6: Demo 工具
# ============================================================================
clear
echo "=========================================="
echo "Part 6: Demo 演示工具"
echo "=========================================="
echo ""
echo "用途：创建演示、录制视频、打包分发"
echo ""
echo "主要工具："
echo "  1. demo_trainer.py - 训练和打包"
echo "  2. demo_controller.py - 交互控制"
echo "  3. demo_visualizer.py - 视频录制"
echo "  4. quick_demo.sh - 一键启动"
echo ""
echo "常用命令："
echo "  # 训练 demo 策略（热启动）"
echo "  python scripts/demo_trainer.py train \\"
echo "    --base-checkpoint v18a_best.pt \\"
echo "    --iterations 300"
echo ""
echo "  # 快速测试"
echo "  python scripts/demo_trainer.py test --checkpoint best.pt"
echo ""
echo "  # 启动演示"
echo "  ./scripts/quick_demo.sh"
echo ""
echo "  # 录制视频"
echo "  python scripts/demo_visualizer.py record \\"
echo "    --checkpoint best.pt --episodes 5"
echo ""
echo "  # 打包分发"
echo "  python scripts/demo_trainer.py package --checkpoint best.pt"
echo ""
read -p "查看 Demo 详细文档？(y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    cat DEMO_QUICK_REF.md | head -50
    echo ""
    echo "... (查看完整文档: cat DEMO_QUICK_REF.md)"
fi
echo ""
read -p "按 Enter 继续..."

# ============================================================================
# Part 7: 完整工作流示例
# ============================================================================
clear
echo "=========================================="
echo "Part 7: 完整工作流示例"
echo "=========================================="
echo ""
echo "场景：训练新版本并进行完整分析"
echo ""
echo "步骤 1: 启动训练"
echo "  make train MODE=v17 NUM_ENVS=64 &"
echo ""
echo "步骤 2: 启动监控（另一个终端）"
echo "  make monitor-metrics MONITOR_POLL_SECONDS=600 &"
echo ""
echo "步骤 3: 启动自动评估（另一个终端）"
echo "  make auto-eval-checkpoints MONITOR_POLL_SECONDS=600 &"
echo ""
echo "步骤 4: 查看实时日志（另一个终端）"
echo "  tail -f logs/skrl/short_nav_v17_*.log | grep METRIC"
echo ""
echo "步骤 5: 训练完成后，对比结果"
echo "  python scripts/compare_training_runs.py v16 v17 v18"
echo ""
echo "步骤 6: 绘制训练曲线"
echo "  python scripts/plot_training_curves.py v16 v17 v18 --output comparison.png"
echo ""
echo "步骤 7: 找到最佳 checkpoint"
echo "  python scripts/checkpoint_manager.py best --version v17"
echo ""
echo "步骤 8: 评估最佳策略"
echo "  make play CHECKPOINT=logs/skrl/lidar_short_nav_v17_direct/*/checkpoints/best_agent.pt"
echo ""
read -p "按 Enter 继续..."

# ============================================================================
# Summary
# ============================================================================
clear
echo "=========================================="
echo "总结：所有可用的 Skills"
echo "=========================================="
echo ""
echo "📊 分析工具："
echo "  ✓ compare_training_runs.py - 对比训练结果"
echo "  ✓ plot_training_curves.py - 绘制训练曲线"
echo "  ✓ checkpoint_manager.py - 管理 checkpoint"
echo ""
echo "🔍 监控工具（已内置）："
echo "  ✓ monitor-metrics - 实时监控训练指标"
echo "  ✓ auto-eval-checkpoints - 自动评估 checkpoint"
echo ""
echo "🎬 Demo 工具："
echo "  ✓ demo_trainer.py - 训练和打包"
echo "  ✓ demo_controller.py - 交互控制"
echo "  ✓ demo_visualizer.py - 视频录制"
echo "  ✓ quick_demo.sh - 一键启动"
echo ""
echo "📚 文档："
echo "  ✓ SKILLS_GUIDE.md - 完整技能指南"
echo "  ✓ DEMO_SKILLS_GUIDE.md - Demo 详细指南"
echo "  ✓ DEMO_QUICK_REF.md - Demo 快速参考"
echo "  ✓ v17_training_plan.md - V17 训练计划"
echo "  ✓ v17_quickstart.md - V17 快速开始"
echo ""
echo "🚀 快速开始："
echo "  # 查看所有工具"
echo "  ls -lh scripts/*.py | grep -E '(compare|checkpoint|plot|demo)'"
echo ""
echo "  # 运行演示"
echo "  ./scripts/demo_skills.sh"
echo ""
echo "  # 查看文档"
echo "  cat SKILLS_GUIDE.md"
echo ""
echo "=========================================="
echo "教程完成！"
echo "=========================================="
echo ""
