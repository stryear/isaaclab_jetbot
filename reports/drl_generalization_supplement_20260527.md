# 局部 DRL 策略泛化补充实验汇总

日期：2026-05-27

## 说明

沙箱内直接运行 `nvidia-smi` 会失败，但在沙箱外运行正常，GPU 为 NVIDIA GeForce RTX 4060 Ti，driver 版本为 `580.126.09`。因此已使用 Isaac Lab launcher 完成本次补充评估。

本次复跑结果位于：

- `tmp/drl_generalization_supplement_20260527/formal_eval_summary.md`
- `tmp/drl_generalization_supplement_20260527/formal_eval_summary.csv`
- `tmp/drl_generalization_supplement_20260527/formal_eval_summary.json`

注意：三个 seed 的结果完全一致，说明这些固定失效场景在当前评估配置下基本为确定性重置。因此论文中可报告“3 次重复运行结果”，但不宜夸大为随机场景分布上的统计泛化。

下文第 0 节为本次真实复跑结果，建议论文优先采用。第 1-3 节为仓库已有历史评估记录，仅作为扩展对照。

## 0. 本次复跑的正式补充结果

评估设置：

- seeds：`42, 43, 44`
- episodes：每个 seed `100`
- num_envs：`32`
- 评估脚本：`scripts/skrl/eval_lidar_nav.py`

| 场景 | 场景类型 | 成功率 | 碰撞率 | 超时率 |
| --- | --- | ---: | ---: | ---: |
| door_w068_a25_o025 | 窄门参数扰动：门宽 0.68 m、线角 25 deg、目标横向偏移 0.25 m | 97.00% +/- 0.00% | 2.00% +/- 0.00% | 1.00% +/- 0.00% |
| dwb_baseline | DWB 振荡原始场景 | 98.00% +/- 0.00% | 1.00% +/- 0.00% | 1.00% +/- 0.00% |
| dwb_combined_harder | DWB 组合扰动：短通道、窄间距、横向偏移和 x 抖动叠加 | 18.00% +/- 0.00% | 68.00% +/- 0.00% | 14.00% +/- 0.00% |
| unseen_mppi_corner | 未见 MPPI 转角失效场景 | 0.00% +/- 0.00% | 0.00% +/- 0.00% | 100.00% +/- 0.00% |
| unseen_u_shape_trap | 未见 U 形陷阱场景 | 100.00% +/- 0.00% | 0.00% +/- 0.00% | 0.00% +/- 0.00% |

结论：同类窄门参数扰动和原始 DWB 振荡场景表现稳定，但 DWB 组合扰动下成功率从 98% 降至 18%，MPPI 转角未见场景完全超时，说明策略泛化边界明显。U 形陷阱未见场景表现良好，说明跨场景泛化并非完全失败，而是依赖几何拓扑和任务动力学相似性。

## 1. 历史窄门场景参数化评估

数据来源：

- `logs/door_best_eval/20260405_214153_door_offcenter_candidate_grid_selection/summary.json`
- `logs/door_best_eval/20260405_214153_door_offcenter_candidate_grid_selection/models.tsv`

评估设置：

- 任务：`Isaac-Lab-Tutorial-LidarShortNavDefectDoorDeadlock-TurtleBot3-Direct-v0`
- episodes：每个参数组合 `100`
- 参数网格：门宽 `0.68/0.72 m`，通道线角 `20/25 deg`，目标横向偏移 `0/0.15/0.25 m`
- 模型：
  - `low_collision`: `logs/skrl/lidar_short_nav_defect_door_deadlock_direct/door_offcenter_candidates_20260405_212709/best_agent_low_collision.pt`
  - `low_timeout`: `logs/skrl/lidar_short_nav_defect_door_deadlock_direct/door_offcenter_candidates_20260405_212709/agent_112000_low_timeout.pt`

| 模型 | 参数组合数 | 平均成功率 | 平均碰撞率 | 平均超时率 |
| --- | ---: | ---: | ---: | ---: |
| low_collision | 12 | 92.58% | 6.25% | 1.17% |
| low_timeout | 12 | 93.08% | 5.83% | 1.08% |

代表性结果（`low_collision`）：

| 门宽/角度/偏移 | 成功率 | 碰撞率 | 超时率 |
| --- | ---: | ---: | ---: |
| w0.68/a20/o0.00 | 93% | 6% | 1% |
| w0.68/a25/o0.15 | 85% | 14% | 1% |
| w0.68/a25/o0.25 | 91% | 7% | 2% |
| w0.72/a20/o0.15 | 99% | 1% | 0% |
| w0.72/a25/o0.25 | 92% | 5% | 3% |

结论：窄门策略在同类场景的参数扰动下保持较高成功率，尤其对门宽、通道角度和目标横向偏移具有一定鲁棒性。

## 2. 历史 DWB 振荡场景泛化与组合扰动

数据来源：

- `logs/dwb_generalization_eval/20260405_151031_candidate_best_grid_v2/summary.json`

评估设置：

- 任务：`Isaac-Lab-Tutorial-LidarShortNavDefectDwbOscillation-TurtleBot3-Direct-v0`
- episodes：每个场景 `100`

| 场景 | 关键扰动 | 成功率 | 碰撞率 | 超时率 |
| --- | --- | ---: | ---: | ---: |
| baseline_stage0 | 原始振荡预设 | 88% | 1% | 11% |
| gap_medium | 障碍间距 1.0 m | 5% | 77% | 18% |
| gap_narrow | 障碍间距 0.9 m | 0% | 66% | 34% |
| offset_pair | 障碍对横移 0.18 m + x 抖动 0.12 m | 76% | 5% | 19% |
| combined_harder | 更短通道 + 1.0 m 间距 + 横移 + x 抖动 + 较短目标距离 | 14% | 70% | 16% |

结论：振荡策略对横移和 x 抖动仍有一定适应性，但对通道缩窄和组合扰动非常敏感。该结果能明确给出泛化边界，适合在论文中作为“组合失效场景下性能下降”的补充分析。

## 3. 历史未见场景零样本测试

数据来源：

- `outputs/zero_shot_eval_2026-03-25_door_policy/dwb_oscillation.json`
- `outputs/zero_shot_eval_2026-03-25_door_policy/mppi_corner_fail.json`
- `outputs/zero_shot_eval_2026-03-25_door_policy/u_shape_trap.json`

评估设置：

- checkpoint：`logs/skrl/lidar_short_nav_v18a_direct/2026-03-24_17-33-14_ppo_torch/checkpoints/best_agent.pt`
- episodes：每个场景 `150`

| 未见/跨场景任务 | 成功率 | 碰撞率 | 超时率 | 平均步长 |
| --- | ---: | ---: | ---: | ---: |
| DWB oscillation | 0.00% | 0.00% | 100.00% | 299.00 |
| MPPI corner fail | 0.00% | 100.00% | 0.00% | 16.17 |
| U-shape trap | 52.67% | 33.33% | 14.00% | 192.30 |

结论：零样本跨场景能力有限，不同未见场景表现差异明显。该结果支持专家意见：仅凭振荡和窄门两类预设场景，不能充分说明局部 DRL 策略的实际泛化能力。

## 可写入论文的实验结论

补充实验表明，局部 DRL 策略在同类窄门参数扰动下仍能保持较高成功率，在门宽 0.68 m、通道线角 25 deg、目标横向偏移 0.25 m 的窄门场景中成功率为 97%。在原始 DWB 振荡场景中，策略成功率为 98%；但当短通道、障碍窄间距、横向偏移和 x 向抖动叠加后，成功率下降至 18%，碰撞率升至 68%。此外，在未见 MPPI 转角失效场景中策略出现完全超时，而在 U 形陷阱场景中仍能成功通过。上述结果表明，局部 DRL 策略具备一定同类参数扰动鲁棒性，但对部分组合失效和未见拓扑的泛化能力仍存在明显边界。

## GPU 机器复跑命令模板

DWB 组合扰动示例：

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH= \
  /home/cs/IsaacLab/isaaclab.sh -p scripts/skrl/eval_lidar_nav.py \
  --task Isaac-Lab-Tutorial-LidarShortNavDefectDwbOscillation-TurtleBot3-Direct-v0 \
  --algorithm PPO \
  --ml_framework torch \
  --checkpoint logs/skrl/lidar_short_nav_defect_dwb_finetune_direct/dwb_candidate_best_20260405_145329/best_agent.pt \
  --num_envs 32 \
  --episodes 200 \
  --seed 42 \
  --headless \
  --defect_dwb_corridor_width 3.0 \
  --defect_dwb_corridor_half_length 2.4 \
  --defect_dwb_obstacle_spacing 1.0 \
  --defect_dwb_obstacle_pair_shift_y 0.18 \
  --defect_dwb_obstacle_x_jitter 0.12 \
  --defect_goal_distance_max 1.6 \
  --state_report_json logs/dwb_generalization_eval/rerun_combined_harder_seed42.json
```

窄门参数化示例：

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH= \
  /home/cs/IsaacLab/isaaclab.sh -p scripts/skrl/eval_lidar_nav.py \
  --task Isaac-Lab-Tutorial-LidarShortNavDefectDoorDeadlock-TurtleBot3-Direct-v0 \
  --algorithm PPO \
  --ml_framework torch \
  --checkpoint logs/skrl/lidar_short_nav_defect_door_deadlock_direct/door_offcenter_candidates_20260405_212709/best_agent_low_collision.pt \
  --num_envs 32 \
  --episodes 200 \
  --seed 42 \
  --headless \
  --defect_door_width 0.68 \
  --defect_door_line_angle_max_deg 25 \
  --defect_door_goal_lateral_offset_max 0.25 \
  --state_report_json logs/door_best_eval/rerun_w068_a25_o025_seed42.json
```

建议最终论文结果至少复跑 `seed=42/43/44`，并报告成功率、碰撞率、超时率的均值和标准差。
