# Isaac Lab JetBot 与 TurtleBot3 导航强化学习

本仓库包含一组基于 Isaac Lab 的移动机器人强化学习任务和配套工具：JetBot 球体跟随、TurtleBot3
LiDAR 导航、短程局部导航课程、ROS2/Gazebo 部署探针，以及 demo 训练、回放、录制和打包工具。

项目基于 **NVIDIA Isaac Sim** 与 **Isaac Lab**，训练流程使用 skrl 的 PPO/SAC 等算法，并提供评估、
checkpoint 选择、曲线绘图和 ROS2 集成脚本。

## 总览

| 组件 | 说明 |
|------|------|
| **Isaac Lab 任务** | 直接式 RL 环境位于 `source/isaac_lab_tutorial/.../tasks/direct/isaac_lab_tutorial/` |
| **训练 / 回放** | skrl 入口位于 `scripts/skrl/train.py`、`scripts/skrl/play.py`、`scripts/skrl/eval_lidar_nav.py` |
| **机器人配置** | JetBot 与 TurtleBot3 Burger articulation 配置，USD 资产通过 manifest 同步 |
| **短程导航实验** | TurtleBot3 LiDAR 局部导航预设、课程版本、缺陷场景和 demo 场景 |
| **ROS2 / Gazebo 工具** | 策略运行器、成功率评估器、桥接循环、rosbag 采集和 Nav2 探针脚本 |
| **Isaac Sim 球体跟随回放** | 原始 JetBot sphere-follow 策略的 standalone 脚本和 Isaac Sim extension |

## 环境要求

- **NVIDIA Isaac Sim 4.5.0+** ([安装文档](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/index.html))
- **Isaac Lab** framework ([安装文档](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html))
- 支持 CUDA 的 **NVIDIA GPU**
- **Python 3.10+**

## 文档索引

| 文件 | 用途 |
|------|------|
| `README.md` | 项目总入口和命令索引 |
| `v17_quickstart.md` | V17 训练快速开始 |
| `v17_training_plan.md` | V17 训练计划和推荐流程 |
| `DEMO_QUICK_REF.md` | 短程导航 demo 命令速查 |
| `DEMO_SKILLS_GUIDE.md` | demo 训练、启动、录制、打包的详细指南 |
| `SKILLS_INDEX.md` | 本地辅助脚本和训练工具索引 |
| `reports/*.md` | 实验汇总和论文相关报告 |

## 资产同步

机器人 USD 文件不会进入 Git（`*.usd` 已被忽略）。训练或评估前，先从 mapping manifest 下载资产：

```bash
# 下载全部机器人资产（JetBot + TurtleBot3）
./launch.sh fetch-assets --robot all

# 或只下载单个 profile
./launch.sh fetch-assets --robot jetbot
./launch.sh fetch-assets --robot turtlebot3_burger
```

下载脚本读取：

- `source/isaac_lab_tutorial/isaac_lab_tutorial/assets/Jetbot/.collect.mapping.json`
- `source/isaac_lab_tutorial/isaac_lab_tutorial/assets/Turtlebot/.collect.mapping.json`

注意：当前 4.5 分支中的 JetBot manifest 使用 `Assets/Isaac/4.5/...` 路径。

---

## 第一部分：Isaac Lab 训练

### 快速开始

```bash
# 1. 下载 USD 资产
./launch.sh fetch-assets --robot all

# 2. 以 editable 模式安装包
./launch.sh install

# 3. 确认环境注册成功
./launch.sh list-envs

# 4. 用默认 JetBot sphere-follow 任务做可视化 smoke test
./launch.sh random-agent --num_envs 10

# 5. 训练默认 JetBot sphere-follow PPO 策略
./launch.sh train --algorithm PPO --num_envs 100

# 6. 回放训练后的 sphere-follow checkpoint
./launch.sh play --checkpoint logs/skrl/sphere_follow_direct/<run>/checkpoints/best_agent.pt
```

TurtleBot3 LiDAR 导航常用入口：

```bash
# 基础 LiDAR 目标导航
./launch.sh train --robot turtlebot3_burger --mode lidar_nav --num_envs 100 --algorithm PPO

# 短程局部导航预设
make train ROBOT=turtlebot3_burger MODE=short_nav_v18a NUM_ENVS=64 ALGORITHM=PPO
make evaluate ROBOT=turtlebot3_burger MODE=short_nav_v18a \
  CHECKPOINT=logs/skrl/lidar_short_nav_v18a_direct/<run>/checkpoints/best_agent.pt \
  EPISODES=100 NUM_ENVS=16

# Demo 策略流程
python scripts/demo_trainer.py train \
  --base-checkpoint logs/skrl/lidar_short_nav_v18a_direct/*/checkpoints/best_agent.pt \
  --iterations 300
python scripts/demo_trainer.py test --checkpoint <checkpoint.pt> --episodes 10
python scripts/demo_controller.py launch --checkpoint <checkpoint.pt>
```

### 环境

| 环境 | 观测 | 动作 | 任务 |
|------|------|------|------|
| `Template-Isaac-Lab-Tutorial-Direct-v0` | 3D | 2D | 跟随随机方向指令 |
| `Isaac-Lab-Tutorial-SphereFollow-Direct-v0` | 4D | 2D | 追踪绿色球体目标 |
| `Isaac-Lab-Tutorial-SphereFollow-TurtleBot3-Direct-v0` | 4D | 2D | 使用 TurtleBot3 Burger 追踪绿色球体 |
| `Isaac-Lab-Tutorial-LidarNav-TurtleBot3-Direct-v0` | 42D | 2D | LiDAR 障碍规避目标导航 |
| `Isaac-Lab-Tutorial-LidarCorridor-TurtleBot3-Direct-v0` | 42D | 2D | 直线走廊障碍规避 |
| `Isaac-Lab-Tutorial-LidarJunction-TurtleBot3-Direct-v0` | 58D | 2D | 直线、L/T/X 路口混合导航 |
| `Isaac-Lab-Tutorial-LidarShortNavJunction-TurtleBot3-Direct-v0` | 58D | 2D | 面向直线和 L/T/X 路口的短程障碍规避预设 |
| `Isaac-Lab-Tutorial-LidarShortNavUnified-TurtleBot3-Direct-v0` | 51D | 2D | 不使用 planner-state/turn-command 特权输入的统一短程局部导航 |

完整的课程、demo 和缺陷场景环境请运行：

```bash
./launch.sh list-envs
```

### 常用模式

| 模式 | 机器人 | 入口 | 用途 |
|------|--------|------|------|
| `sphere_follow` | JetBot 或 TurtleBot3 | `./launch.sh train --mode sphere_follow` | 原始绿色球体追踪 baseline |
| `lidar_nav` | TurtleBot3 | `./launch.sh train --robot turtlebot3_burger --mode lidar_nav` | 42D LiDAR 目标导航 |
| `lidar_junction` | TurtleBot3 | `./launch.sh train --robot turtlebot3_burger --mode lidar_junction` | 直线、L/T/X 路口混合场景 |
| `short_nav_junction` | TurtleBot3 | `make train ROBOT=turtlebot3_burger MODE=short_nav_junction` | 带路口标签的短程局部导航 |
| `short_nav_unified` | TurtleBot3 | `make train ROBOT=turtlebot3_burger MODE=short_nav_unified` | 不使用特权 planner-state 输入的短程局部导航 |
| `short_nav_v14` ... `short_nav_v18a` | TurtleBot3 | `make train ROBOT=turtlebot3_burger MODE=short_nav_v18a` | 版本化短程导航课程和 reward 变体 |
| `short_nav_demo*` | TurtleBot3 | `python scripts/demo_trainer.py ...` 或 `make train MODE=short_nav_demo` | 确定性 demo 场景和展示包 |
| `single_obstacle_*`、`mppi_corner_fail`、`door_deadlock`、`u_shape_trap` | TurtleBot3 | `make train MODE=<mode>` | 针对失败模式和恢复能力的实验 |

版本化 short-nav 实验优先使用 `make train MODE=<mode>`，因为 Makefile 会把本地模式别名映射到注册环境 ID。
`./launch.sh` 更适合安装、资产同步、baseline 训练、回放和 ROS2 bridge。

### Sphere-Following 环境

| 属性 | 值 |
|------|----|
| **观测空间** | 4D：dot product、cross_z、归一化距离、前向速度 |
| **动作空间** | 2D：左右轮速度目标 |
| **Reward** | approach + alignment + reach_bonus(5.0) + time_penalty(-0.01) |
| **Episode 长度** | 20 秒 |
| **物理频率** | 120 Hz（decimation=2，控制频率 60 Hz） |
| **并行环境数** | 默认 100 |
| **目标到达阈值** | 0.3m，到达后球体在 0.5-1.5m 范围内重生 |

### TurtleBot3 LiDAR 障碍导航环境

| 属性 | 值 |
|------|----|
| **观测空间** | 42D：`[goal_dir_x, goal_dir_y, goal_dist_norm, forward_speed, yaw_rate, min_lidar, lidar_36]` |
| **动作空间** | 2D：归一化左右轮命令（`[-1, 1]`，内部再缩放） |
| **LiDAR** | 2D 虚拟扫描，36 beams，270° FOV，0.05-3.0m |
| **障碍物** | 每个环境 6 个红色圆柱障碍物，reset 时随机化 |
| **终止条件** | 到达目标、LiDAR 最小距离触发碰撞、或超时 |

Reward 项：

- `progress`：目标距离减少量
- `heading`：机器人朝向与目标方向的对齐度
- `clearance`：归一化 LiDAR 最小距离
- `smoothness`：动作变化惩罚
- `time_penalty`：每步时间惩罚
- `success_bonus` 和 `collision_penalty`

TurtleBot3 LiDAR 变体：

- `Isaac-Lab-Tutorial-LidarCorridor-TurtleBot3-Direct-v0`：带墙和随机障碍的直线走廊导航。
- `Isaac-Lab-Tutorial-LidarJunction-TurtleBot3-Direct-v0`：线段和 L/T/X 路口混合短程导航。
- `Isaac-Lab-Tutorial-LidarShortNavJunction-TurtleBot3-Direct-v0`：针对 1.0m 到 2.5m 局部导航和更紧成功半径调过的预设。
- `Isaac-Lab-Tutorial-LidarShortNavUnified-TurtleBot3-Direct-v0`：统一短程局部规划预设，51D 纯感知观测。
- `short_nav_junction`：需要可重复短程行为时推荐使用，也固定了混合 planner-state 训练中的 relay-goal 距离。
- `short_nav_unified`：部署端无法稳定提供 planner-state/junction 标签时推荐使用。

短程路口预设示例：

```bash
./launch.sh train \
  --robot turtlebot3_burger \
  --mode short_nav_junction \
  --num_envs 64 \
  --algorithm PPO

./launch.sh train \
  --robot turtlebot3_burger \
  --mode short_nav_unified \
  --num_envs 64 \
  --algorithm PPO
```

### 启动命令

| 命令 | 说明 |
|------|------|
| `./launch.sh install` | 以 editable 模式安装 package |
| `./launch.sh fetch-assets --robot all` | 从 mapping manifest 下载机器人 USD 资产 |
| `./launch.sh list-envs` | 列出所有注册环境 |
| `./launch.sh random-agent` | 使用随机动作运行环境 |
| `./launch.sh zero-agent` | 使用零动作运行 baseline |
| `./launch.sh train` | 使用 skrl 训练 RL agent |
| `./launch.sh play` | 回放训练后的 checkpoint |
| `./launch.sh select-ckpt` | 选择最近的健康 LiDAR-nav checkpoint |
| `./launch.sh check` | 检查 Python/GPU/Isaac Lab 前置条件 |
| `./launch.sh help` | 显示完整命令帮助 |
| `./launch.sh train --robot turtlebot3_burger` | 使用 TurtleBot3 Burger profile 训练 |
| `./launch.sh train --robot turtlebot3_burger --mode lidar_nav` | 训练 LiDAR 障碍导航策略 |
| `./launch.sh train --robot turtlebot3_burger --mode short_nav_junction` | 训练短程直线 + L/T/X 导航预设 |
| `./launch.sh train --robot turtlebot3_burger --mode short_nav_unified` | 训练不使用 planner-state/turn-command 特权观测的统一短程导航预设 |
| `./launch.sh ros2-bridge --robot turtlebot3_burger --mode lidar_nav` | 启动 ROS2 `/cmd_vel`、`/scan`、`/odom` bridge loop |

Make 快捷命令：

```bash
./launch.sh fetch-assets --robot all
make fetch-assets ROBOT=all
make train ALGORITHM=PPO NUM_ENVS=100
make train ROBOT=turtlebot3_burger NUM_ENVS=100
make train ROBOT=turtlebot3_burger MODE=lidar_nav NUM_ENVS=100
make train ROBOT=turtlebot3_burger MODE=short_nav_junction NUM_ENVS=64
make train ROBOT=turtlebot3_burger MODE=short_nav_unified NUM_ENVS=64
make train ROBOT=turtlebot3_burger MODE=short_nav_v18a NUM_ENVS=64
make evaluate MODE=short_nav_v18a CHECKPOINT=logs/skrl/.../checkpoints/best_agent.pt EPISODES=100 NUM_ENVS=16
make monitor-metrics MONITOR_LOG_ROOT=logs/skrl/lidar_short_nav_v17_direct
make auto-eval-checkpoints MONITOR_LOG_ROOT=logs/skrl/lidar_short_nav_v17_direct
make ros2-bridge
make play CHECKPOINT=logs/skrl/sphere_follow_direct/<run>/checkpoints/best_agent.pt
make train-ppo
```

### LiDAR 导航训练命令

```bash
# 训练 LiDAR 障碍规避导航策略
./launch.sh train --robot turtlebot3_burger --mode lidar_nav \
  --algorithm PPO --num_envs 100 --headless

# 评估 LiDAR 导航 checkpoint
./launch.sh play --robot turtlebot3_burger --mode lidar_nav \
  --checkpoint logs/skrl/lidar_nav_direct/<run>/checkpoints/best_agent.pt --headless
```

### ROS2 Bridge（cmd_vel / scan / odom）

```bash
# 使用 LiDAR nav 环境启动单环境 bridge loop
./launch.sh ros2-bridge --robot turtlebot3_burger --mode lidar_nav --headless

# 可选 topic 覆盖
./launch.sh ros2-bridge --robot turtlebot3_burger --mode lidar_nav \
  --topic_cmd /cmd_vel --topic_scan /scan --topic_odom /odom
```

### Demo 流程

Demo 工具封装了短程导航训练、评估、回放、录制和打包。完整命令 cookbook 见 `DEMO_QUICK_REF.md` 和
`DEMO_SKILLS_GUIDE.md`。

```bash
# 从较强的 v18a checkpoint warm-start 一个 demo 策略
python scripts/demo_trainer.py train \
  --base-checkpoint logs/skrl/lidar_short_nav_v18a_direct/*/checkpoints/best_agent.pt \
  --iterations 300

# 快速评估和交互式启动
python scripts/demo_trainer.py test --checkpoint <checkpoint.pt> --episodes 10
python scripts/demo_controller.py launch --checkpoint <checkpoint.pt>

# 基准测试、录制和打包
python scripts/demo_controller.py benchmark --checkpoint <checkpoint.pt> --runs 10
python scripts/demo_visualizer.py record --checkpoint <checkpoint.pt> --episodes 5 --output demos/my_demo
python scripts/demo_trainer.py package --checkpoint <checkpoint.pt> --output demos/navigation_demo_v1
```

### PPO 训练配置

| 参数 | 值 |
|------|----|
| Network | [64, 64] shared backbone + ELU |
| Rollout length | 48 steps |
| Learning rate | 3e-4（KL adaptive） |
| Mini-batches | 8 |
| Learning epochs | 8 |
| Entropy coeff | 0.01 |
| Total timesteps | 24,000 |
| Discount factor | 0.99 |

### ROS2 和 Gazebo 工具

| 脚本 | 用途 |
|------|------|
| `scripts/ros2_bridge_lidar_nav.py` | 单环境 ROS2 bridge，发布/订阅 `/cmd_vel`、`/scan`、`/odom` |
| `scripts/gazebo_policy_runner.py` | 在 ROS2/Gazebo 中运行训练后的 LiDAR-nav 策略 |
| `scripts/gazebo_eval_success.py` | 在 Gazebo 中评估导航成功率 |
| `scripts/run_nav2_online_probe.sh` | 检查 live ROS2 设置中的 Nav2 topic/action wiring |
| `scripts/run_gazebo_eval_multiscene.sh` | 跨多个 Gazebo 场景做批量评估 |
| `scripts/gazebo_smoke_test_v8.sh` | Smoke test / 失败案例采集 harness |

代表命令：

```bash
./launch.sh ros2-bridge --robot turtlebot3_burger --mode lidar_nav

# 直接运行 Gazebo/ROS2 脚本前需要先 source ROS2。
python scripts/gazebo_policy_runner.py --help
python scripts/gazebo_eval_success.py --help
./scripts/run_nav2_online_probe.sh --help
```

---

## 第二部分：Isaac Sim Standalone

该 standalone 脚本用于回放训练后的 JetBot sphere-follow checkpoint，不依赖 Isaac Lab 运行环境。

### 使用方式

```bash
python scripts/standalone_sphere_follow.py \
    --checkpoint logs/skrl/sphere_follow_direct/<run>/checkpoints/best_agent.pt
```

### API 映射（Isaac Lab -> Isaac Sim）

| Isaac Lab | Isaac Sim Standalone |
|-----------|---------------------|
| `robot.data.root_pos_w` | `jetbot.get_world_pose()[0]` |
| `robot.data.root_link_quat_w` | `jetbot.get_world_pose()[1]`（wxyz） |
| `robot.data.root_com_lin_vel_b[:,0]` | `quat_rotate_inverse(q, get_linear_velocity())[0]` |
| `math_utils.quat_apply(q, v)` | `quat_rotate(q, v)` |
| `robot.set_joint_velocity_target(a)` | `articulation_view.set_joint_velocity_targets(a)` |

### 脚本功能

- 从 checkpoint 加载 PolicyNetwork（4->64->64->2）和 observation normalizer
- 创建与训练匹配的 physics/render rate（120Hz/60Hz）
- 生成 JetBot、地面、dome light 和绿色球体
- 循环执行策略推理：读取状态 -> 构造 obs -> normalize -> forward pass -> 应用动作
- JetBot 到达球体后重新放置球体，实现连续追踪

---

## 第三部分：Isaac Sim Extension

这是注册在 **Examples Browser** 中 **Policy > JetBot Sphere Follow** 下的 Isaac Sim extension。

### 安装

extension 文件安装到 Isaac Sim interactive examples：

```text
isaacsim/exts/isaacsim.examples.interactive/
  isaacsim/examples/interactive/sphere_follow/
    __init__.py
    sphere_follow.py                # SphereFollow(BaseSample)
    sphere_follow_extension.py      # UI + Extension registration
```

在 `extension.toml` 中添加：

```toml
[[python.module]]
name = "isaacsim.examples.interactive.sphere_follow"
```

### 架构

| 类 | 作用 |
|-------|------|
| `SphereFollow(BaseSample)` | 场景搭建、策略加载、physics callback 推理 |
| `SphereFollowUI(BaseSampleUITemplate)` | checkpoint 路径输入和 live status 显示 |
| `SphereFollowExtension(omni.ext.IExt)` | 注册到 Examples Browser 的 `Policy` 分类 |

### UI 功能

- **World Controls**：标准 Load / Reset 按钮
- **Policy Configuration**：可编辑 checkpoint 路径，需在 Load 前设置
- **Live Status**：显示到达球体次数、step 计数、4D 观测、轮速动作，约每 0.5 秒更新一次

### 工作流程

1. 在 Examples Browser 中点击 **Load**
2. extension 加载 checkpoint，生成 JetBot 和球体，并注册 physics callback
3. 策略以 60Hz 运行（120Hz physics，decimation=2）
4. JetBot 追踪球体；到达后球体重新放置
5. 点击 **Reset** 重置场景

---

## 项目结构

```text
IsaacLabTutorial/
├── launch.sh                           # 主入口脚本
├── Makefile                            # Make 快捷命令
├── blog_generate_pdf.py                # PDF 文档生成器
├── JetBot_Sphere_Following_RL_Blog.pdf # 已生成文档
├── README.md                           # 项目总览和命令索引
├── DEMO_QUICK_REF.md                   # Demo 速查
├── DEMO_SKILLS_GUIDE.md                # Demo workflow 指南
├── v17_quickstart.md                   # V17 快速开始
├── v17_training_plan.md                # V17 训练计划
├── reports/                            # 实验汇总
├── scripts/
│   ├── download_assets.py              # 根据 .collect.mapping.json 下载 USD 资产
│   ├── ros2_bridge_lidar_nav.py         # ROS2 bridge loop (/cmd_vel, /scan, /odom)
│   ├── gazebo_policy_runner.py          # ROS2/Gazebo 策略运行器
│   ├── gazebo_eval_success.py           # Gazebo 成功率评估器
│   ├── demo_trainer.py                  # Demo 训练和打包
│   ├── demo_controller.py               # 交互式 demo 启动和 benchmark
│   ├── demo_visualizer.py               # Demo 录制和对比
│   ├── standalone_sphere_follow.py      # Isaac Sim standalone 回放
│   ├── list_envs.py                     # 列出注册环境
│   ├── random_agent.py                  # 随机动作 baseline
│   ├── zero_agent.py                    # 零动作 baseline
│   └── skrl/
│       ├── train.py                     # RL 训练（PPO/AMP/SAC）
│       ├── play.py                      # Checkpoint 回放
│       ├── eval_lidar_nav.py            # LiDAR-nav 评估指标
│       └── export_policy_onnx.py        # 策略导出辅助脚本
├── exts/
│   └── isaacsim.examples.interactive.sphere_follow/
│       └── sphere_follow/              # Isaac Sim extension 的仓库副本
│           ├── __init__.py
│           ├── sphere_follow.py
│           └── sphere_follow_extension.py
└── source/
    └── isaac_lab_tutorial/
        ├── setup.py
        └── isaac_lab_tutorial/
            ├── __init__.py
            ├── ui_extension_example.py
            ├── sphere_follow_extension.py  # 本地 Isaac Sim extension
            ├── assets/
            │   ├── Jetbot/.collect.mapping.json
            │   └── Turtlebot/.collect.mapping.json
            ├── robots/
            │   ├── jetbot.py              # JetBot ArticulationCfg（本地 USD）
            │   └── turtlebot3_burger.py   # TurtleBot3 Burger ArticulationCfg
            └── tasks/
                └── direct/
                    └── isaac_lab_tutorial/
                        ├── __init__.py                    # Gym 注册
                        ├── isaac_lab_tutorial_env.py      # Template + SphereFollow + LidarNav 环境
                        ├── isaac_lab_tutorial_env_cfg.py  # JetBot/TurtleBot3/LiDAR Nav 环境配置
                        └── agents/
                            ├── skrl_ppo_cfg.yaml
                            ├── skrl_amp_cfg.yaml
                            ├── skrl_sphere_follow_ppo_cfg.yaml
                            ├── skrl_lidar_nav_ppo_cfg.yaml
                            ├── skrl_lidar_junction_ppo_cfg.yaml
                            ├── skrl_lidar_short_nav_junction_ppo_cfg.yaml
                            ├── skrl_lidar_short_nav_unified_ppo_cfg.yaml
                            ├── skrl_lidar_short_nav_v14_ppo_cfg.yaml
                            ├── skrl_lidar_short_nav_v15_ppo_cfg.yaml
                            ├── skrl_lidar_short_nav_v15a_ppo_cfg.yaml
                            ├── skrl_lidar_short_nav_v15b_ppo_cfg.yaml
                            ├── skrl_lidar_short_nav_v16_ppo_cfg.yaml
                            ├── skrl_lidar_short_nav_v17_ppo_cfg.yaml
                            ├── skrl_lidar_short_nav_v18_ppo_cfg.yaml
                            ├── skrl_lidar_short_nav_v18a_ppo_cfg.yaml
                            └── demo / defect / curriculum configs
```

## 训练日志

```text
logs/skrl/sphere_follow_direct/
└── <timestamp>_ppo_torch/
    ├── checkpoints/
    │   ├── best_agent.pt      # 按 episode return 选择的最佳 checkpoint
    │   └── agent_24000.pt     # 最终 checkpoint
    └── params/
        ├── env.yaml           # 环境配置快照
        └── agent.yaml         # agent 超参数
```

以下常见输出不会进入 Git：

```text
logs/
tmp/
demos/
policy_*.onnx
policy_*.json
policy_*.pt
nohup.out
```

## 许可证

Apache 2.0，详见 [LICENSE](LICENSE)。
