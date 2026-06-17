# Recommended Skills for DRL Navigation Project

## 已实现的工具 (Built-in)

### 1. 自动化监控与评估
```bash
# 实时监控训练指标（每 10 分钟检查一次）
make monitor-metrics MONITOR_LOG_ROOT=logs/skrl/lidar_short_nav_v17_direct

# 自动评估新生成的 checkpoint
make auto-eval-checkpoints MONITOR_LOG_ROOT=logs/skrl/lidar_short_nav_v17_direct
```

**功能**：
- 监控关键指标（熵、KL 散度、梯度范数等）
- 阈值告警（自动检测训练异常）
- 自动评估新 checkpoint 的性能
- 生成 CSV/JSON 报告

**输出**：
- `logs/auto_monitor_v17/metrics_summary.csv`
- `logs/auto_eval_v17/summary.csv`

---

## 新增工具 (New Scripts)

### 2. 训练对比工具 ⭐
```bash
# 对比多个版本的训练结果
python scripts/compare_training_runs.py v16 v17 v18

# 导出详细数据到 JSON
python scripts/compare_training_runs.py v16 v17 --export comparison.json
```

**功能**：
- 自动解析训练日志
- 对比最终/峰值成功率
- 识别最佳版本
- 计算改进幅度

**输出示例**：
```
================================================================================
Training Runs Comparison
================================================================================

Version    Final Success   Peak Success    Final Collision   Iterations
--------------------------------------------------------------------------------
v16        43.0%           46.0% (iter 294)  53.0%             346
v17        58.2%           61.5% (iter 850)  38.5%             1200
v18        62.1%           64.0% (iter 920)  35.2%             1200

🏆 Best performer: v18 (Final: 62.1%, Peak: 64.0%)
   v17 vs v16: +15.2% success rate
   v18 vs v16: +19.1% success rate
```

---

### 3. Checkpoint 管理工具 ⭐
```bash
# 列出所有 checkpoint
python scripts/checkpoint_manager.py list --version v17

# 找到最佳 checkpoint
python scripts/checkpoint_manager.py best --version v17

# 对比多个 checkpoint
python scripts/checkpoint_manager.py compare --checkpoints ckpt1.pt ckpt2.pt

# 清理旧 checkpoint（保留最新 5 个）
python scripts/checkpoint_manager.py cleanup --version v17 --keep-last 5 --no-dry-run
```

**功能**：
- 查找和列出所有 checkpoint
- 提取 checkpoint 元数据（大小、参数量、时间步）
- 对比不同 checkpoint
- 自动清理旧文件（节省磁盘空间）

---

### 4. 可视化工具 ⭐
```bash
# 绘制训练曲线
python scripts/plot_training_curves.py v16 v17 v18 --metrics success collision avg_return

# 保存到文件
python scripts/plot_training_curves.py v17 v18 --output training_comparison.png
```

**功能**：
- 绘制多版本对比曲线
- 支持多种指标（成功率、碰撞率、回报等）
- 自动添加目标线（如 55% 成功率）
- 导出高质量图片

---

### 5. 超参数搜索工具
```bash
# 网格搜索学习率
python scripts/hyperparam_sweep.py \
  --base-config v17 \
  --param learning_rate \
  --values 3e-5 5e-5 1e-4 \
  --num-envs 32

# 仅生成配置文件（不训练）
python scripts/hyperparam_sweep.py \
  --base-config v17 \
  --param rollouts \
  --values 128 256 512 \
  --dry-run
```

**功能**：
- 自动生成超参数变体配置
- 批量启动训练任务
- 记录实验元数据
- 支持嵌套参数（如 `agent.learning_rate`）

---

## 推荐的外部工具

### 6. TensorBoard（已支持）
```bash
# 启动 TensorBoard
tensorboard --logdir logs/skrl/lidar_short_nav_v17_direct/

# 对比多个版本
tensorboard --logdir_spec v16:logs/skrl/lidar_short_nav_v16_direct,v17:logs/skrl/lidar_short_nav_v17_direct
```

**功能**：
- 实时可视化训练曲线
- 对比多个实验
- 查看网络结构
- 分析分布统计

---

### 7. Weights & Biases (推荐集成)
```bash
# 安装
pip install wandb

# 在 train.py 中添加
import wandb
wandb.init(project="isaaclab-navigation", name="v17_experiment")
wandb.log({"success_rate": success, "collision_rate": collision})
```

**优势**：
- 云端存储实验记录
- 强大的可视化和对比功能
- 超参数搜索（Sweeps）
- 团队协作

---

### 8. Optuna（自动超参数优化）
```bash
# 安装
pip install optuna

# 创建优化脚本
python scripts/optuna_optimize.py --version v17 --trials 20
```

**功能**：
- 贝叶斯优化（比网格搜索更高效）
- 自动剪枝（提前停止差的试验）
- 可视化参数重要性
- 并行试验

---

## 工作流建议

### 日常训练流程
```bash
# 1. 启动训练
make train MODE=v17 NUM_ENVS=64 &

# 2. 启动自动监控（另一个终端）
make monitor-metrics MONITOR_POLL_SECONDS=600 &

# 3. 启动自动评估（另一个终端）
make auto-eval-checkpoints MONITOR_POLL_SECONDS=600 &

# 4. 查看实时日志
tail -f logs/skrl/short_nav_v17_*.log | grep METRIC
```

### 实验对比流程
```bash
# 1. 训练完成后，对比结果
python scripts/compare_training_runs.py v16 v17 v18

# 2. 绘制曲线
python scripts/plot_training_curves.py v16 v17 v18 --output comparison.png

# 3. 找到最佳 checkpoint
python scripts/checkpoint_manager.py best --version v17

# 4. 评估最佳策略
make play CHECKPOINT=logs/skrl/lidar_short_nav_v17_direct/*/checkpoints/best_agent.pt
```

### 超参数调优流程
```bash
# 1. 快速网格搜索（小规模）
python scripts/hyperparam_sweep.py \
  --base-config v17 \
  --param learning_rate \
  --values 3e-5 5e-5 1e-4 \
  --num-envs 32

# 2. 对比结果
python scripts/compare_training_runs.py sweep_*

# 3. 选择最佳参数，完整训练
make train MODE=v17 NUM_ENVS=64
```

---

## 快速参考

| 任务 | 命令 |
|------|------|
| 对比训练结果 | `python scripts/compare_training_runs.py v16 v17 v18` |
| 绘制训练曲线 | `python scripts/plot_training_curves.py v17 v18` |
| 找最佳 checkpoint | `python scripts/checkpoint_manager.py best --version v17` |
| 清理旧 checkpoint | `python scripts/checkpoint_manager.py cleanup --version v17` |
| 监控训练指标 | `make monitor-metrics` |
| 自动评估 checkpoint | `make auto-eval-checkpoints` |
| 超参数搜索 | `python scripts/hyperparam_sweep.py --base-config v17 --param lr --values ...` |

---

## 未来可扩展的 Skills

### 9. 策略可视化工具
- 可视化策略的决策边界
- 绘制 LiDAR 扫描 + 动作热图
- 录制 episode 视频

### 10. 失败案例分析
- 自动识别失败模式（碰撞、超时、停滞）
- 聚类相似失败场景
- 生成失败报告

### 11. 迁移学习工具
- 从 v16 checkpoint 初始化 v17
- 冻结部分网络层
- 渐进式网络扩展

### 12. 分布式训练支持
- 多 GPU 并行训练
- 分布式超参数搜索
- 云端训练管理

---

## 安装依赖

```bash
# 基础可视化
pip install matplotlib seaborn

# 高级工具（可选）
pip install wandb optuna tensorboard

# 使脚本可执行
chmod +x scripts/*.py
```

---

## 总结

**立即可用的核心工具**：
1. ✅ 自动监控与评估（已内置）
2. ✅ 训练对比工具（新增）
3. ✅ Checkpoint 管理（新增）
4. ✅ 可视化工具（新增）
5. ✅ 超参数搜索（新增）

**推荐集成**：
- TensorBoard（实时可视化）
- Weights & Biases（云端实验管理）
- Optuna（智能超参数优化）

这些工具覆盖了 DRL 项目的完整生命周期：训练 → 监控 → 评估 → 对比 → 优化。
