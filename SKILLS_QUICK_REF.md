# Skills 使用速查表

## 🎯 最常用的命令（Top 10）

```bash
# 1. 对比训练结果
python scripts/compare_training_runs.py v16 v17 v18

# 2. 找最佳 checkpoint
python scripts/checkpoint_manager.py best --version v17

# 3. 绘制训练曲线
python scripts/plot_training_curves.py v16 v17 v18 --output comparison.png

# 4. 启动自动监控
make monitor-metrics MONITOR_LOG_ROOT=logs/skrl/lidar_short_nav_v17_direct

# 5. 启动自动评估
make auto-eval-checkpoints MONITOR_LOG_ROOT=logs/skrl/lidar_short_nav_v17_direct

# 6. 列出所有 checkpoint
python scripts/checkpoint_manager.py list --version v17

# 7. 清理旧 checkpoint
python scripts/checkpoint_manager.py cleanup --version v17 --keep-last 5 --no-dry-run

# 8. 训练 demo 策略
python scripts/demo_trainer.py train --base-checkpoint v18a_best.pt --iterations 300

# 9. 启动 demo 演示
./scripts/quick_demo.sh

# 10. 录制演示视频
python scripts/demo_visualizer.py record --checkpoint best.pt --episodes 5
```

---

## 📋 按场景分类

### 场景 1：训练完成后的分析

```bash
# 步骤 1: 对比多个版本
python scripts/compare_training_runs.py v14 v15 v16 v17 v18

# 步骤 2: 绘制训练曲线
python scripts/plot_training_curves.py v16 v17 v18 \
  --metrics success collision avg_return \
  --output results/training_comparison.png

# 步骤 3: 找到最佳 checkpoint
python scripts/checkpoint_manager.py best --version v17

# 步骤 4: 评估最佳策略
make play CHECKPOINT=<path_from_step3>
```

---

### 场景 2：训练过程中的监控

```bash
# 终端 1: 启动训练
make train MODE=v17 NUM_ENVS=64

# 终端 2: 实时监控指标
make monitor-metrics \
  MONITOR_LOG_ROOT=logs/skrl/lidar_short_nav_v17_direct \
  MONITOR_POLL_SECONDS=600

# 终端 3: 自动评估新 checkpoint
make auto-eval-checkpoints \
  MONITOR_LOG_ROOT=logs/skrl/lidar_short_nav_v17_direct \
  EXTRA_ARGS="--episodes 50 --num-envs 16"

# 终端 4: 查看实时日志
tail -f logs/skrl/short_nav_v17_*.log | grep METRIC
```

---

### 场景 3：创建演示

```bash
# 步骤 1: 训练 demo 策略（热启动，30 分钟）
python scripts/demo_trainer.py train \
  --base-checkpoint logs/skrl/lidar_short_nav_v18a_direct/*/checkpoints/best_agent.pt \
  --iterations 300

# 步骤 2: 测试性能
python scripts/demo_trainer.py test \
  --checkpoint logs/skrl/lidar_short_nav_demo_direct/*/checkpoints/best_agent.pt

# 步骤 3: 录制视频
python scripts/demo_visualizer.py record \
  --checkpoint logs/skrl/lidar_short_nav_demo_direct/*/checkpoints/best_agent.pt \
  --episodes 5 \
  --output demos/my_demo

# 步骤 4: 打包分发
python scripts/demo_trainer.py package \
  --checkpoint logs/skrl/lidar_short_nav_demo_direct/*/checkpoints/best_agent.pt \
  --output demos/navigation_demo_v1
```

---

### 场景 4：Checkpoint 管理

```bash
# 列出所有 checkpoint
python scripts/checkpoint_manager.py list --version v17

# 找最佳 checkpoint
python scripts/checkpoint_manager.py best --version v17

# 对比两个 checkpoint
python scripts/checkpoint_manager.py compare \
  --checkpoints ckpt1.pt ckpt2.pt

# 清理旧 checkpoint（dry-run）
python scripts/checkpoint_manager.py cleanup --version v17 --keep-last 5

# 实际删除
python scripts/checkpoint_manager.py cleanup --version v17 --keep-last 5 --no-dry-run
```

---

## 🔧 工具详解

### 1. compare_training_runs.py

**用途**：对比多个训练版本的性能

**基本用法**：
```bash
python scripts/compare_training_runs.py v16 v17 v18
```

**高级用法**：
```bash
# 导出详细数据到 JSON
python scripts/compare_training_runs.py v16 v17 --export comparison.json

# 指定日志目录
python scripts/compare_training_runs.py v16 v17 --log-dir custom_logs/
```

**输出示例**：
```
================================================================================
Training Runs Comparison
================================================================================

Version    Final Success   Peak Success    Final Collision   Iterations
--------------------------------------------------------------------------------
v16        43.0%           46.0% (iter 294)  53.0%             346
v17        58.2%           61.5% (iter 850)  38.5%             1200

🏆 Best performer: v17 (Final: 58.2%, Peak: 61.5%)
   v17 vs v16: +15.2% success rate
```

---

### 2. checkpoint_manager.py

**用途**：查找、对比和管理 checkpoint

**子命令**：

#### list - 列出所有 checkpoint
```bash
python scripts/checkpoint_manager.py list --version v17
```

#### best - 找最佳 checkpoint
```bash
python scripts/checkpoint_manager.py best --version v17
```

#### compare - 对比多个 checkpoint
```bash
python scripts/checkpoint_manager.py compare \
  --checkpoints path/to/ckpt1.pt path/to/ckpt2.pt
```

#### cleanup - 清理旧 checkpoint
```bash
# Dry-run（默认）
python scripts/checkpoint_manager.py cleanup --version v17 --keep-last 5

# 实际删除
python scripts/checkpoint_manager.py cleanup --version v17 --keep-last 5 --no-dry-run
```

---

### 3. plot_training_curves.py

**用途**：绘制训练曲线

**基本用法**：
```bash
python scripts/plot_training_curves.py v16 v17 v18
```

**指定指标**：
```bash
python scripts/plot_training_curves.py v16 v17 v18 \
  --metrics success collision avg_return avg_len
```

**保存到文件**：
```bash
python scripts/plot_training_curves.py v16 v17 v18 \
  --output results/training_curves.png
```

**支持的指标**：
- `success` - 成功率
- `collision` - 碰撞率
- `timeout` - 超时率
- `avg_return` - 平均回报
- `avg_len` - 平均步长
- `avg_goal_end_dist` - 终点距离
- `avg_min_lidar` - 最小 LiDAR 距离

---

### 4. demo_trainer.py

**用途**：训练和打包 demo 策略

**子命令**：

#### train - 训练 demo 策略
```bash
# 从头训练
python scripts/demo_trainer.py train --iterations 500

# 热启动（推荐）
python scripts/demo_trainer.py train \
  --base-checkpoint v18a_best.pt \
  --iterations 300
```

#### test - 快速测试
```bash
python scripts/demo_trainer.py test \
  --checkpoint best.pt \
  --episodes 10
```

#### package - 打包分发
```bash
python scripts/demo_trainer.py package \
  --checkpoint best.pt \
  --output demos/navigation_demo_v1
```

---

### 5. demo_controller.py

**用途**：交互控制和基准测试

**子命令**：

#### launch - 启动演示
```bash
# 带 GUI
python scripts/demo_controller.py launch --checkpoint best.pt

# Headless
python scripts/demo_controller.py launch --checkpoint best.pt --headless
```

#### benchmark - 性能基准
```bash
python scripts/demo_controller.py benchmark \
  --checkpoint best.pt \
  --runs 10
```

#### create-script - 创建独立脚本
```bash
python scripts/demo_controller.py create-script \
  --checkpoint best.pt \
  --output demos/run_demo.sh
```

---

### 6. demo_visualizer.py

**用途**：录制和对比视频

**子命令**：

#### record - 录制视频
```bash
python scripts/demo_visualizer.py record \
  --checkpoint best.pt \
  --episodes 5 \
  --output demos/my_demo
```

#### compare - 对比多个 checkpoint
```bash
python scripts/demo_visualizer.py compare \
  --checkpoints ckpt1.pt ckpt2.pt ckpt3.pt \
  --output demos/comparison.html
```

#### analyze - 性能分析
```bash
python scripts/demo_visualizer.py analyze \
  --checkpoint best.pt \
  --episodes 50
```

---

## 🚨 常见问题

### Q1: matplotlib 未安装
```bash
pip install matplotlib seaborn
```

### Q2: 找不到训练日志
```bash
# 检查日志目录
ls -lh logs/skrl/short_nav_v*_*.log

# 如果没有，先运行训练
make train MODE=v17 NUM_ENVS=64
```

### Q3: 找不到 checkpoint
```bash
# 检查 checkpoint 目录
ls -lh logs/skrl/lidar_short_nav_v*/*/checkpoints/

# 如果没有，训练还未生成 checkpoint
```

### Q4: 视频录制失败
```bash
# 安装 ffmpeg
sudo apt install ffmpeg

# 检查安装
which ffmpeg
```

---

## 📚 完整文档

- **SKILLS_GUIDE.md** - 完整技能指南
- **DEMO_SKILLS_GUIDE.md** - Demo 详细指南
- **DEMO_QUICK_REF.md** - Demo 快速参考
- **v17_training_plan.md** - V17 训练计划
- **v17_quickstart.md** - V17 快速开始

---

## 🎓 学习路径

### 初学者
1. 运行交互式教程：`./scripts/skills_tutorial.sh`
2. 查看演示：`./scripts/demo_skills.sh`
3. 阅读快速参考：`cat SKILLS_QUICK_REF.md`

### 进阶用户
1. 阅读完整指南：`cat SKILLS_GUIDE.md`
2. 尝试完整工作流（见"场景 1"）
3. 自定义工具脚本

### 高级用户
1. 阅读源码：`cat scripts/compare_training_runs.py`
2. 扩展工具功能
3. 集成 Weights & Biases 或 Optuna

---

## ⚡ 快速测试

```bash
# 测试所有工具是否可用
./scripts/demo_skills.sh

# 运行交互式教程
./scripts/skills_tutorial.sh

# 查看可用的脚本
ls -lh scripts/*.py | grep -E '(compare|checkpoint|plot|demo)'
```

---

**最后更新**: 2026-03-18
**版本**: 1.0
