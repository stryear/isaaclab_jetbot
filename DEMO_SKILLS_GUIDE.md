# Short Nav Demo - Skills Guide

## 概述

`short_nav_demo` 是一个特殊的训练和演示配置，专门用于：
- **可视化展示**：单环境、确定性场景
- **快速过拟合**：在固定场景上达到近 100% 成功率
- **架构验证**：测试 CNN 架构处理 LiDAR 数据
- **演示录制**：生成高质量的演示视频

## 配置特点

### 环境配置
```python
# 单环境（适合演示）
num_envs: 1

# 确定性场景
planner_state_prob_explore: 1.0  # 100% 走廊场景
linear_open_area_prob: 0.0       # 无开阔区域
linear_corridor_half_width: 2.0m # 固定宽度

# 固定起始位置
linear_state_spawn_distance: 2.0m
linear_state_spawn_lateral_jitter: 0.0

# 较高速度限制
speed_cap_explore: 0.30 m/s
```

### 网络架构（CNN）
```yaml
# LiDAR 处理分支（2D 卷积）
Input: [batch, 1, 90, 1]  # 90 个 LiDAR 束
Conv2D: 32 filters, kernel [5,1], stride [2,1]
Conv2D: 64 filters, kernel [3,1], stride [2,1]
Flatten + FC: 512

# 融合分支
Concatenate: [lidar_features(512), base_features(16)]
FC: 512 → 256 → actions
```

### 训练参数
```yaml
rollouts: 256
learning_rate: 3e-5
ratio_clip: 0.12
timesteps: 300,000
```

---

## Demo Skills 工具

### 1. 训练 Demo 策略 ⭐⭐⭐

#### 从头训练
```bash
# 使用 Makefile
make train MODE=demo NUM_ENVS=1

# 或直接使用脚本
python scripts/demo_trainer.py train --iterations 500
```

#### 从已有 checkpoint 热启动（推荐）
```bash
# 使用 v18a 的 checkpoint 作为起点
python scripts/demo_trainer.py train \
  --base-checkpoint logs/skrl/lidar_short_nav_v18a_direct/*/checkpoints/best_agent.pt \
  --iterations 500
```

**优势**：
- 从通用策略开始，快速收敛到特定场景
- 通常 100-200 iterations 即可达到 >95% 成功率
- 节省训练时间（约 30 分钟 vs 2 小时）

---

### 2. 快速测试 ⭐⭐

```bash
# 测试 checkpoint 性能（10 episodes）
python scripts/demo_trainer.py test \
  --checkpoint logs/skrl/lidar_short_nav_demo_direct/*/checkpoints/best_agent.pt \
  --episodes 10
```

**输出示例**：
```
🧪 Quick test (10 episodes)...

============================================================
Quick Test Results
============================================================
Success rate: 9/10 (90.0%)
Collision rate: 1/10 (10.0%)
Average episode length: 245 steps
```

---

### 3. 交互式演示 ⭐⭐⭐

#### 启动可视化演示
```bash
# 带 GUI（推荐用于演示）
python scripts/demo_controller.py launch \
  --checkpoint logs/skrl/lidar_short_nav_demo_direct/*/checkpoints/best_agent.pt

# Headless 模式（用于测试）
python scripts/demo_controller.py launch \
  --checkpoint best_agent.pt \
  --headless
```

#### 性能基准测试
```bash
# 运行 10 次，统计成功率和平均时间
python scripts/demo_controller.py benchmark \
  --checkpoint best_agent.pt \
  --runs 10
```

---

### 4. 录制演示视频 ⭐⭐⭐

#### 录制单个 checkpoint
```bash
# 录制 5 个 episode
python scripts/demo_visualizer.py record \
  --checkpoint best_agent.pt \
  --episodes 5 \
  --output demos/my_demo
```

**输出**：
- `demos/my_demo/recording.log` - 录制日志
- `videos/*.mp4` - 生成的视频文件

#### 对比多个 checkpoint
```bash
# 并排对比不同版本
python scripts/demo_visualizer.py compare \
  --checkpoints v17_best.pt v18_best.pt demo_best.pt \
  --output demos/comparison.html
```

**输出**：
- HTML 页面，包含并排视频对比
- 在浏览器中打开查看

---

### 5. 创建独立演示脚本 ⭐⭐

```bash
# 生成可分发的演示脚本
python scripts/demo_controller.py create-script \
  --checkpoint best_agent.pt \
  --output demos/run_demo.sh

# 使用生成的脚本
./demos/run_demo.sh
./demos/run_demo.sh --video  # 录制视频
```

---

### 6. 打包演示 ⭐⭐

```bash
# 创建完整的演示包（包含 checkpoint、脚本、文档）
python scripts/demo_trainer.py package \
  --checkpoint best_agent.pt \
  --output demos/navigation_demo_v1
```

**包含内容**：
```
demos/navigation_demo_v1/
├── policy.pt          # 训练好的策略
├── run_demo.sh        # 启动脚本
├── README.md          # 使用说明
└── config.json        # 环境配置
```

**分发**：
```bash
# 压缩打包
tar -czf navigation_demo_v1.tar.gz demos/navigation_demo_v1/

# 接收方解压后直接运行
cd navigation_demo_v1 && ./run_demo.sh
```

---

## 完整工作流示例

### 场景 1：从零开始创建演示

```bash
# 1. 训练 demo 策略（从 v18a 热启动）
python scripts/demo_trainer.py train \
  --base-checkpoint logs/skrl/lidar_short_nav_v18a_direct/*/checkpoints/best_agent.pt \
  --iterations 300

# 2. 找到最佳 checkpoint
python scripts/checkpoint_manager.py best --version demo

# 3. 快速测试
python scripts/demo_trainer.py test \
  --checkpoint logs/skrl/lidar_short_nav_demo_direct/*/checkpoints/best_agent.pt

# 4. 录制演示视频
python scripts/demo_visualizer.py record \
  --checkpoint logs/skrl/lidar_short_nav_demo_direct/*/checkpoints/best_agent.pt \
  --episodes 3

# 5. 打包分发
python scripts/demo_trainer.py package \
  --checkpoint logs/skrl/lidar_short_nav_demo_direct/*/checkpoints/best_agent.pt
```

---

### 场景 2：对比不同架构

```bash
# 1. 准备 3 个 checkpoint（v17 全连接、v18 CNN、demo 过拟合）
V17_CKPT=logs/skrl/lidar_short_nav_v17_direct/*/checkpoints/best_agent.pt
V18_CKPT=logs/skrl/lidar_short_nav_v18_direct/*/checkpoints/best_agent.pt
DEMO_CKPT=logs/skrl/lidar_short_nav_demo_direct/*/checkpoints/best_agent.pt

# 2. 在 demo 环境上测试所有 checkpoint
for ckpt in $V17_CKPT $V18_CKPT $DEMO_CKPT; do
    echo "Testing $ckpt..."
    python scripts/demo_trainer.py test --checkpoint $ckpt --episodes 20
done

# 3. 创建视频对比
python scripts/demo_visualizer.py compare \
  --checkpoints $V17_CKPT $V18_CKPT $DEMO_CKPT \
  --output demos/architecture_comparison.html
```

---

### 场景 3：现场演示准备

```bash
# 1. 创建独立演示脚本
python scripts/demo_controller.py create-script \
  --checkpoint best_agent.pt \
  --output ~/Desktop/isaac_demo.sh

# 2. 测试演示脚本
~/Desktop/isaac_demo.sh

# 3. 准备备用视频（以防现场问题）
python scripts/demo_visualizer.py record \
  --checkpoint best_agent.pt \
  --episodes 5 \
  --output ~/Desktop/demo_videos
```

---

## 性能目标

### Demo 环境预期性能

| 指标 | 目标 | 说明 |
|------|------|------|
| 成功率 | >95% | 确定性场景，应接近完美 |
| 碰撞率 | <3% | 偶尔的边界碰撞 |
| 超时率 | <2% | 极少数卡住情况 |
| 平均步长 | 200-300 | 取决于目标距离 |

### 训练收敛时间

| 起点 | 迭代次数 | 时间 | 最终成功率 |
|------|----------|------|------------|
| 随机初始化 | 800-1000 | 2-3 小时 | 90-95% |
| v18a 热启动 | 200-300 | 30-45 分钟 | 95-98% |
| v18a + 微调 | 100-150 | 15-20 分钟 | 98-100% |

---

## 故障排查

### 问题 1：成功率低于 90%

**可能原因**：
- 训练不足（需要更多 iterations）
- 学习率过高（导致不稳定）
- 网络容量不足

**解决方案**：
```bash
# 继续训练
python scripts/demo_trainer.py train \
  --base-checkpoint current_best.pt \
  --iterations 200

# 或降低学习率（修改 yaml 中的 learning_rate: 1e-5）
```

---

### 问题 2：视频录制失败

**可能原因**：
- 缺少视频编码器
- 磁盘空间不足

**解决方案**：
```bash
# 检查 ffmpeg
which ffmpeg

# 安装（如果缺失）
sudo apt install ffmpeg

# 检查磁盘空间
df -h
```

---

### 问题 3：演示时机器人不动

**可能原因**：
- Checkpoint 路径错误
- 观测空间不匹配

**解决方案**：
```bash
# 验证 checkpoint
python scripts/checkpoint_manager.py list --version demo

# 检查观测维度
python -c "
import torch
ckpt = torch.load('best_agent.pt', map_location='cpu')
print('Policy input shape:', list(ckpt['policy'].values())[0].shape)
"
```

---

## 高级技巧

### 1. 多场景 Demo

虽然 demo 配置默认是单一走廊，但可以修改配置测试不同场景：

```python
# 在 isaac_lab_tutorial_env_cfg.py 中修改
linear_corridor_half_width_choices: tuple[float, ...] = (1.5, 2.0, 2.5)  # 多种宽度
planner_state_prob_junction: float = 0.2  # 添加路口场景
```

### 2. 实时性能分析

```bash
# 使用 nvidia-smi 监控 GPU 使用
watch -n 1 nvidia-smi

# 使用 htop 监控 CPU
htop
```

### 3. 批量生成演示

```bash
# 为多个 checkpoint 批量生成演示
for ckpt in logs/skrl/*/checkpoints/best_agent.pt; do
    version=$(echo $ckpt | cut -d'/' -f3)
    python scripts/demo_visualizer.py record \
      --checkpoint $ckpt \
      --episodes 3 \
      --output demos/$version
done
```

---

## 快速参考

| 任务 | 命令 |
|------|------|
| 训练 demo | `python scripts/demo_trainer.py train --iterations 300` |
| 热启动训练 | `python scripts/demo_trainer.py train --base-checkpoint v18a_best.pt` |
| 快速测试 | `python scripts/demo_trainer.py test --checkpoint best.pt` |
| 启动演示 | `python scripts/demo_controller.py launch --checkpoint best.pt` |
| 录制视频 | `python scripts/demo_visualizer.py record --checkpoint best.pt` |
| 对比视频 | `python scripts/demo_visualizer.py compare --checkpoints ckpt1 ckpt2` |
| 创建脚本 | `python scripts/demo_controller.py create-script --checkpoint best.pt` |
| 打包演示 | `python scripts/demo_trainer.py package --checkpoint best.pt` |

---

## 总结

Demo 模式提供了：
- ✅ 快速训练（30 分钟达到 95% 成功率）
- ✅ 确定性场景（便于演示和调试）
- ✅ CNN 架构（验证卷积处理 LiDAR）
- ✅ 完整工具链（训练、测试、录制、打包）
- ✅ 可分发包（独立运行，无需源码）

适用于：
- 🎬 项目演示和展示
- 🧪 架构验证和消融实验
- 📹 视频录制和对比
- 🎓 教学和培训材料
