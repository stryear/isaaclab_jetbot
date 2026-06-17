# Skills 使用指南 - 完整索引

## 🚀 快速开始（3 步）

```bash
# 1. 查看可用工具
./scripts/demo_skills.sh

# 2. 运行交互式教程
./scripts/skills_tutorial.sh

# 3. 查看速查表
cat SKILLS_QUICK_REF.md
```

---

## 📚 文档导航

### 核心文档
| 文档 | 用途 | 适合人群 |
|------|------|----------|
| **SKILLS_QUICK_REF.md** | 速查表，最常用命令 | 所有用户 ⭐⭐⭐ |
| **SKILLS_GUIDE.md** | 完整技能指南 | 进阶用户 |
| **DEMO_QUICK_REF.md** | Demo 快速参考 | Demo 用户 ⭐⭐⭐ |
| **DEMO_SKILLS_GUIDE.md** | Demo 详细指南 | Demo 用户 |

### 训练相关
| 文档 | 用途 |
|------|------|
| **v17_quickstart.md** | V17 快速开始 |
| **v17_training_plan.md** | V17 训练计划 |

### 交互式工具
| 脚本 | 用途 |
|------|------|
| **scripts/skills_tutorial.sh** | 交互式教程 ⭐⭐⭐ |
| **scripts/demo_skills.sh** | 工具演示 |

---

## 🛠️ 工具清单

### 分析工具
| 工具 | 功能 | 命令示例 |
|------|------|----------|
| **compare_training_runs.py** | 对比训练结果 | `python scripts/compare_training_runs.py v16 v17` |
| **plot_training_curves.py** | 绘制训练曲线 | `python scripts/plot_training_curves.py v16 v17 --output plot.png` |
| **checkpoint_manager.py** | 管理 checkpoint | `python scripts/checkpoint_manager.py best --version v17` |

### 监控工具（内置）
| 工具 | 功能 | 命令示例 |
|------|------|----------|
| **monitor-metrics** | 实时监控指标 | `make monitor-metrics` |
| **auto-eval-checkpoints** | 自动评估 | `make auto-eval-checkpoints` |

### Demo 工具
| 工具 | 功能 | 命令示例 |
|------|------|----------|
| **demo_trainer.py** | 训练和打包 | `python scripts/demo_trainer.py train --iterations 300` |
| **demo_controller.py** | 交互控制 | `python scripts/demo_controller.py launch --checkpoint best.pt` |
| **demo_visualizer.py** | 视频录制 | `python scripts/demo_visualizer.py record --checkpoint best.pt` |
| **quick_demo.sh** | 一键启动 | `./scripts/quick_demo.sh` |

---

## 📖 使用场景索引

### 我想...

#### 对比不同版本的训练结果
→ 查看 [SKILLS_QUICK_REF.md - 场景 1](#)
```bash
python scripts/compare_training_runs.py v16 v17 v18
```

#### 找到最佳的 checkpoint
→ 查看 [SKILLS_QUICK_REF.md - Checkpoint 管理](#)
```bash
python scripts/checkpoint_manager.py best --version v17
```

#### 绘制训练曲线图
→ 查看 [SKILLS_QUICK_REF.md - plot_training_curves](#)
```bash
python scripts/plot_training_curves.py v16 v17 --output curves.png
```

#### 实时监控训练过程
→ 查看 [SKILLS_QUICK_REF.md - 场景 2](#)
```bash
make monitor-metrics MONITOR_LOG_ROOT=logs/skrl/lidar_short_nav_v17_direct
```

#### 创建演示视频
→ 查看 [DEMO_QUICK_REF.md - 录制视频](#)
```bash
python scripts/demo_visualizer.py record --checkpoint best.pt --episodes 5
```

#### 打包演示给别人
→ 查看 [DEMO_QUICK_REF.md - 打包分发](#)
```bash
python scripts/demo_trainer.py package --checkpoint best.pt
```

#### 清理旧的 checkpoint
→ 查看 [SKILLS_QUICK_REF.md - Checkpoint 管理](#)
```bash
python scripts/checkpoint_manager.py cleanup --version v17 --keep-last 5 --no-dry-run
```

#### 快速训练一个演示策略
→ 查看 [DEMO_QUICK_REF.md - 训练](#)
```bash
python scripts/demo_trainer.py train --base-checkpoint v18a_best.pt --iterations 300
```

---

## 🎯 按用户类型推荐

### 初学者
**推荐阅读顺序**：
1. 运行 `./scripts/skills_tutorial.sh`（交互式教程）
2. 阅读 `SKILLS_QUICK_REF.md`（速查表）
3. 尝试 `./scripts/demo_skills.sh`（工具演示）

**推荐命令**：
```bash
# 对比训练结果
python scripts/compare_training_runs.py v16 v17

# 找最佳 checkpoint
python scripts/checkpoint_manager.py best --version v17

# 启动 demo
./scripts/quick_demo.sh
```

---

### 进阶用户
**推荐阅读顺序**：
1. 阅读 `SKILLS_GUIDE.md`（完整指南）
2. 阅读 `DEMO_SKILLS_GUIDE.md`（Demo 详细指南）
3. 查看源码 `scripts/*.py`

**推荐工作流**：
```bash
# 完整训练监控流程
# 终端 1: 训练
make train MODE=v17 NUM_ENVS=64

# 终端 2: 监控
make monitor-metrics

# 终端 3: 自动评估
make auto-eval-checkpoints

# 训练完成后分析
python scripts/compare_training_runs.py v16 v17
python scripts/plot_training_curves.py v16 v17 --output results.png
```

---

### 高级用户
**推荐扩展**：
1. 集成 Weights & Biases
2. 使用 Optuna 进行超参数优化
3. 自定义工具脚本
4. 添加新的分析指标

**推荐阅读**：
- 源码：`scripts/compare_training_runs.py`
- 源码：`scripts/checkpoint_manager.py`
- 文档：`SKILLS_GUIDE.md` 的"未来可扩展的 Skills"部分

---

## 🔍 快速查找

### 按关键词查找

| 关键词 | 相关工具/文档 |
|--------|---------------|
| 对比 | compare_training_runs.py, SKILLS_QUICK_REF.md |
| 可视化 | plot_training_curves.py, demo_visualizer.py |
| checkpoint | checkpoint_manager.py, SKILLS_QUICK_REF.md |
| 监控 | monitor-metrics, SKILLS_QUICK_REF.md |
| 评估 | auto-eval-checkpoints, demo_trainer.py |
| 演示 | demo_*.py, DEMO_QUICK_REF.md |
| 视频 | demo_visualizer.py, DEMO_SKILLS_GUIDE.md |
| 打包 | demo_trainer.py, DEMO_QUICK_REF.md |

---

## 📊 工具对比

| 特性 | compare_training | plot_curves | checkpoint_manager | demo_trainer |
|------|------------------|-------------|-------------------|--------------|
| 对比版本 | ✅ | ✅ | ❌ | ❌ |
| 可视化 | ❌ | ✅ | ❌ | ❌ |
| 管理文件 | ❌ | ❌ | ✅ | ✅ |
| 训练策略 | ❌ | ❌ | ❌ | ✅ |
| 录制视频 | ❌ | ❌ | ❌ | ❌ |
| 打包分发 | ❌ | ❌ | ❌ | ✅ |

---

## 🎓 学习路径

### 路径 1: 快速上手（30 分钟）
1. ✅ 运行 `./scripts/demo_skills.sh`（5 分钟）
2. ✅ 阅读 `SKILLS_QUICK_REF.md`（10 分钟）
3. ✅ 尝试 3 个最常用命令（15 分钟）

### 路径 2: 深入学习（2 小时）
1. ✅ 运行 `./scripts/skills_tutorial.sh`（30 分钟）
2. ✅ 阅读 `SKILLS_GUIDE.md`（30 分钟）
3. ✅ 阅读 `DEMO_SKILLS_GUIDE.md`（30 分钟）
4. ✅ 实践完整工作流（30 分钟）

### 路径 3: 精通掌握（1 天）
1. ✅ 完成路径 2
2. ✅ 阅读所有源码（2 小时）
3. ✅ 尝试所有工具（2 小时）
4. ✅ 自定义扩展（2 小时）

---

## 🆘 获取帮助

### 命令行帮助
```bash
# 查看工具帮助
python scripts/compare_training_runs.py --help
python scripts/checkpoint_manager.py --help
python scripts/demo_trainer.py --help
```

### 文档帮助
```bash
# 查看速查表
cat SKILLS_QUICK_REF.md

# 查看完整指南
cat SKILLS_GUIDE.md

# 查看 Demo 指南
cat DEMO_QUICK_REF.md
```

### 交互式帮助
```bash
# 运行教程
./scripts/skills_tutorial.sh

# 查看演示
./scripts/demo_skills.sh
```

---

## 📝 更新日志

### 2026-03-18 - v1.0
- ✅ 创建 6 个核心工具脚本
- ✅ 创建 5 份完整文档
- ✅ 创建 2 个交互式脚本
- ✅ 所有工具已测试可用

---

## 🎉 总结

你现在拥有：
- ✅ **6 个核心工具**（分析、监控、Demo）
- ✅ **5 份完整文档**（指南、速查表、教程）
- ✅ **2 个交互式脚本**（教程、演示）
- ✅ **完整的工作流**（训练→监控→分析→演示）

**立即开始**：
```bash
# 最快的方式
./scripts/skills_tutorial.sh

# 或查看速查表
cat SKILLS_QUICK_REF.md

# 或查看所有工具
ls -lh scripts/*.py | grep -E '(compare|checkpoint|plot|demo)'
```

---

**创建日期**: 2026-03-18
**版本**: 1.0
**状态**: ✅ 完整可用
