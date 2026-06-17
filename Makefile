# ==============================================================================
# IsaacLabTutorial - Makefile
# ==============================================================================

SHELL := /bin/bash
TASK_JETBOT := Isaac-Lab-Tutorial-SphereFollow-Direct-v0
TASK_TURTLEBOT3 := Isaac-Lab-Tutorial-SphereFollow-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_LIDAR := Isaac-Lab-Tutorial-LidarNav-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_LIDAR_JUNCTION := Isaac-Lab-Tutorial-LidarJunction-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_JUNCTION := Isaac-Lab-Tutorial-LidarShortNavJunction-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_UNIFIED := Isaac-Lab-Tutorial-LidarShortNavUnified-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_CURRICULUM_P1 := Isaac-Lab-Tutorial-LidarShortNavCurriculumP1-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_V14 := Isaac-Lab-Tutorial-LidarShortNavV14-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_V15 := Isaac-Lab-Tutorial-LidarShortNavV15-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_V15A := Isaac-Lab-Tutorial-LidarShortNavV15A-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_V15B := Isaac-Lab-Tutorial-LidarShortNavV15B-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_V16 := Isaac-Lab-Tutorial-LidarShortNavV16-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_V17 := Isaac-Lab-Tutorial-LidarShortNavV17-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_V18 := Isaac-Lab-Tutorial-LidarShortNavV18-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_V18A := Isaac-Lab-Tutorial-LidarShortNavV18A-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_SYMMETRIC_GAP := Isaac-Lab-Tutorial-LidarShortNavDefectSingleObstacleSymmetricGap-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_DEFECT_SMALL_OBSTACLE_PRETRAIN := Isaac-Lab-Tutorial-LidarShortNavDefectSmallObstaclePretrain-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_WAYPOINT_BRIDGE := Isaac-Lab-Tutorial-LidarShortNavDefectSingleObstacleWaypointBridge-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_REAR_OFFSET_BRIDGE := Isaac-Lab-Tutorial-LidarShortNavDefectSingleObstacleRearOffsetBridge-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_REAR_CENTERING_BRIDGE := Isaac-Lab-Tutorial-LidarShortNavDefectSingleObstacleRearCenteringBridge-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_FINAL_CURRICULUM := Isaac-Lab-Tutorial-LidarShortNavDefectSingleObstacleFinalCurriculum-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_DEMO := Isaac-Lab-Tutorial-LidarShortNavDemo-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_DEMO_HARD := Isaac-Lab-Tutorial-LidarShortNavDemoHard-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_DEMO_HARD_V1 := Isaac-Lab-Tutorial-LidarShortNavDemoHardV1-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_DEMO_MULTI_V1 := Isaac-Lab-Tutorial-LidarShortNavDemoMultiV1-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_DEMO_MULTI_V1_BRIDGE := Isaac-Lab-Tutorial-LidarShortNavDemoMultiV1Bridge-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_DEMO_MULTI_V1_BRIDGE_LSTM := Isaac-Lab-Tutorial-LidarShortNavDemoMultiV1BridgeLSTM-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_DEMO_SCENE4_EASY := Isaac-Lab-Tutorial-LidarShortNavDemoScene4Easy-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_DEMO_SCENE4_MID := Isaac-Lab-Tutorial-LidarShortNavDemoScene4Mid-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_DEMO_SCENE4_MID2 := Isaac-Lab-Tutorial-LidarShortNavDemoScene4Mid2-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_DEMO_SCENE4_MID2_OBS1 := Isaac-Lab-Tutorial-LidarShortNavDemoScene4Mid2Obs1-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_DEMO_SCENE4_MID2_OBS1_OPEN := Isaac-Lab-Tutorial-LidarShortNavDemoScene4Mid2Obs1Open-TurtleBot3-Direct-v0
TASK_TURTLEBOT3_SHORT_NAV_DEMO_SCENE4_WALL_ONLY := Isaac-Lab-Tutorial-LidarShortNavDemoScene4WallOnly-TurtleBot3-Direct-v0
ROBOT ?= jetbot
MODE ?= sphere_follow
TASK ?= $(TASK_JETBOT)
ifneq (,$(filter $(MODE),short_nav_junction short_junction short_nav))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_JUNCTION)
else ifneq (,$(filter $(MODE),short_nav_unified short_unified unified))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_UNIFIED)
else ifneq (,$(filter $(MODE),short_nav_curriculum_p1 curriculum_p1 short_curriculum_p1))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_CURRICULUM_P1)
else ifneq (,$(filter $(MODE),short_nav_v14 v14 short_v14))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_V14)
else ifneq (,$(filter $(MODE),short_nav_v15 v15 short_v15))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_V15)
else ifneq (,$(filter $(MODE),short_nav_v15a v15a short_v15a))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_V15A)
else ifneq (,$(filter $(MODE),short_nav_v15b v15b short_v15b))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_V15B)
else ifneq (,$(filter $(MODE),short_nav_v16 v16 short_v16))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_V16)
else ifneq (,$(filter $(MODE),short_nav_v17 v17 short_v17))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_V17)
else ifneq (,$(filter $(MODE),short_nav_v18 v18 short_v18))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_V18)
else ifneq (,$(filter $(MODE),short_nav_v18a v18a short_v18a short_nav_v18_1 v18_1 short_v18_1))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_V18A)
else ifneq (,$(filter $(MODE),short_nav_demo demo short_demo))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_DEMO)
else ifneq (,$(filter $(MODE),short_nav_demo_hard demo_hard short_demo_hard))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_DEMO_HARD)
else ifneq (,$(filter $(MODE),short_nav_demo_hard_v1 demo_hard_v1 short_demo_hard_v1))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_DEMO_HARD_V1)
else ifneq (,$(filter $(MODE),short_nav_demo_multi_v1 demo_multi_v1 short_demo_multi_v1))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_DEMO_MULTI_V1)
else ifneq (,$(filter $(MODE),short_nav_demo_multi_v1_bridge demo_multi_v1_bridge short_demo_multi_v1_bridge))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_DEMO_MULTI_V1_BRIDGE)
else ifneq (,$(filter $(MODE),short_nav_demo_multi_v1_bridge_lstm demo_multi_v1_bridge_lstm short_demo_multi_v1_bridge_lstm short_nav_demo_multi_v1_lstm demo_multi_v1_lstm short_demo_multi_v1_lstm))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_DEMO_MULTI_V1_BRIDGE_LSTM)
else ifneq (,$(filter $(MODE),short_nav_demo_scene4_easy demo_scene4_easy short_demo_scene4_easy))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_DEMO_SCENE4_EASY)
else ifneq (,$(filter $(MODE),short_nav_demo_scene4_mid demo_scene4_mid short_demo_scene4_mid))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_DEMO_SCENE4_MID)
else ifneq (,$(filter $(MODE),short_nav_demo_scene4_mid2 demo_scene4_mid2 short_demo_scene4_mid2))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_DEMO_SCENE4_MID2)
else ifneq (,$(filter $(MODE),short_nav_demo_scene4_mid2_obs1 demo_scene4_mid2_obs1 short_demo_scene4_mid2_obs1))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_DEMO_SCENE4_MID2_OBS1)
else ifneq (,$(filter $(MODE),short_nav_demo_scene4_mid2_obs1_open demo_scene4_mid2_obs1_open short_demo_scene4_mid2_obs1_open))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_DEMO_SCENE4_MID2_OBS1_OPEN)
else ifneq (,$(filter $(MODE),short_nav_demo_scene4_wall_only demo_scene4_wall_only short_demo_scene4_wall_only))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_DEMO_SCENE4_WALL_ONLY)
else ifneq (,$(filter $(MODE),single_obstacle_symmetric_gap short_nav_defect_single_obstacle_symmetric_gap single_obstacle_gap single_gap defect_single_gap))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_SYMMETRIC_GAP)
else ifneq (,$(filter $(MODE),small_obstacle_pretrain short_nav_defect_small_obstacle_pretrain small_obstacle_gap_pretrain easy_single_gap))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_DEFECT_SMALL_OBSTACLE_PRETRAIN)
else ifneq (,$(filter $(MODE),single_obstacle_waypoint_bridge short_nav_defect_single_obstacle_waypoint_bridge single_obstacle_bridge waypoint_bridge))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_WAYPOINT_BRIDGE)
else ifneq (,$(filter $(MODE),single_obstacle_rear_offset_bridge short_nav_defect_single_obstacle_rear_offset_bridge rear_offset_bridge rear_bridge))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_REAR_OFFSET_BRIDGE)
else ifneq (,$(filter $(MODE),single_obstacle_rear_centering_bridge short_nav_defect_single_obstacle_rear_centering_bridge rear_centering_bridge centering_bridge))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_REAR_CENTERING_BRIDGE)
else ifneq (,$(filter $(MODE),single_obstacle_final_curriculum short_nav_defect_single_obstacle_final_curriculum final_curriculum_bridge final_curriculum))
TASK := $(TASK_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_FINAL_CURRICULUM)
else ifneq (,$(filter $(MODE),lidar_junction junction))
TASK := $(TASK_TURTLEBOT3_LIDAR_JUNCTION)
else ifneq (,$(filter $(MODE),lidar_nav lidar nav))
TASK := $(TASK_TURTLEBOT3_LIDAR)
else ifneq (,$(filter $(ROBOT),turtlebot3_burger turtlebot3 turtlebot tb3 burger))
TASK := $(TASK_TURTLEBOT3)
endif
NUM_ENVS ?= 100
EPISODES ?= 200
ALGORITHM ?= PPO
DEVICE ?= cuda:0
ML_FRAMEWORK ?= torch
ISAACLAB_SH ?= $(HOME)/IsaacLab/isaaclab.sh
ISAACSIM_PY ?= $(HOME)/IsaacLab/_isaac_sim/python.sh

.PHONY: help install fetch-assets list-envs random-agent zero-agent train play evaluate eval ros2-bridge monitor-metrics auto-eval-checkpoints lint format check

help: ## Show this help message
	@echo "IsaacLabTutorial - Mobile Robot Navigation RL Environment"
	@echo ""
	@echo "Usage: make <target> [VAR=value]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables:"
	@echo "  NUM_ENVS=$(NUM_ENVS)        Number of parallel environments"
	@echo "  EPISODES=$(EPISODES)       Evaluation episodes (for make evaluate)"
	@echo "  ALGORITHM=$(ALGORITHM)          RL algorithm (PPO, AMP, IPPO, MAPPO, SAC)"
	@echo "  ROBOT=$(ROBOT)            Robot profile (jetbot, turtlebot3_burger)"
	@echo "  MODE=$(MODE)             Task mode (sphere_follow, lidar_nav, lidar_junction, short_nav_junction, short_nav_unified, short_nav_curriculum_p1, short_nav_v14, short_nav_v15, short_nav_v15a, short_nav_v15b, short_nav_v16, short_nav_v17, short_nav_v18, short_nav_v18a, dwb_oscillation, single_obstacle_symmetric_gap, small_obstacle_pretrain, single_obstacle_waypoint_bridge, mppi_corner_fail, door_deadlock, u_shape_trap, short_nav_demo, short_nav_demo_hard, short_nav_demo_hard_v1, short_nav_demo_multi_v1, short_nav_demo_multi_v1_bridge, short_nav_demo_multi_v1_bridge_lstm, short_nav_demo_scene4_easy, short_nav_demo_scene4_mid, short_nav_demo_scene4_mid2, short_nav_demo_scene4_mid2_obs1, short_nav_demo_scene4_mid2_obs1_open, short_nav_demo_scene4_wall_only)"
	@echo "  TASK=$(TASK)              Task selected from ROBOT or explicit override"
	@echo "  DEVICE=$(DEVICE)        Compute device"
	@echo "  ML_FRAMEWORK=$(ML_FRAMEWORK)     ML framework (torch, jax, jax-numpy)"
	@echo ""
	@echo "Examples:"
	@echo "  make fetch-assets ROBOT=all"
	@echo "  make install"
	@echo "  make train ALGORITHM=PPO NUM_ENVS=100"
	@echo "  make train ROBOT=turtlebot3_burger NUM_ENVS=100"
	@echo "  make train ROBOT=turtlebot3_burger MODE=lidar_nav NUM_ENVS=100"
	@echo "  make train ROBOT=turtlebot3_burger MODE=lidar_junction NUM_ENVS=64"
	@echo "  make train ROBOT=turtlebot3_burger MODE=short_nav_junction NUM_ENVS=64"
	@echo "  make train ROBOT=turtlebot3_burger MODE=short_nav_unified NUM_ENVS=64"
	@echo "  make train ROBOT=turtlebot3_burger MODE=short_nav_curriculum_p1 NUM_ENVS=64"
	@echo "  make train ROBOT=turtlebot3_burger MODE=short_nav_v14 NUM_ENVS=64"
	@echo "  make train ROBOT=turtlebot3_burger MODE=short_nav_v15 NUM_ENVS=64"
	@echo "  make train ROBOT=turtlebot3_burger MODE=short_nav_v15a NUM_ENVS=64"
	@echo "  make train ROBOT=turtlebot3_burger MODE=short_nav_v15b NUM_ENVS=64"
	@echo "  make train ROBOT=turtlebot3_burger MODE=short_nav_v16 NUM_ENVS=64"
	@echo "  make train ROBOT=turtlebot3_burger MODE=short_nav_v17 NUM_ENVS=64"
	@echo "  make train ROBOT=turtlebot3_burger MODE=short_nav_v18 NUM_ENVS=64"
	@echo "  make train ROBOT=turtlebot3_burger MODE=short_nav_v18a NUM_ENVS=64"
	@echo "  make train ROBOT=turtlebot3_burger MODE=single_obstacle_symmetric_gap NUM_ENVS=100"
	@echo "  make train ROBOT=turtlebot3_burger MODE=small_obstacle_pretrain NUM_ENVS=100"
	@echo "  make train ROBOT=turtlebot3_burger MODE=single_obstacle_waypoint_bridge NUM_ENVS=100"
	@echo "  make train ROBOT=turtlebot3_burger MODE=short_nav_demo NUM_ENVS=1"
	@echo "  make train ROBOT=turtlebot3_burger MODE=short_nav_demo_hard NUM_ENVS=64"
	@echo "  make train ROBOT=turtlebot3_burger MODE=short_nav_demo_hard_v1 NUM_ENVS=64"
	@echo "  make train ROBOT=turtlebot3_burger MODE=short_nav_demo_multi_v1 NUM_ENVS=64"
	@echo "  make train ROBOT=turtlebot3_burger MODE=short_nav_demo_multi_v1_bridge NUM_ENVS=64"
	@echo "  make train ROBOT=turtlebot3_burger MODE=short_nav_demo_multi_v1_bridge_lstm NUM_ENVS=64"
	@echo "  make train ROBOT=turtlebot3_burger MODE=short_nav_demo_scene4_easy NUM_ENVS=64"
	@echo "  make train ROBOT=turtlebot3_burger MODE=short_nav_demo_scene4_mid NUM_ENVS=64"
	@echo "  make train ROBOT=turtlebot3_burger MODE=short_nav_demo_scene4_mid2 NUM_ENVS=64"
	@echo "  make train ROBOT=turtlebot3_burger MODE=short_nav_demo_scene4_mid2_obs1 NUM_ENVS=64"
	@echo "  make train ROBOT=turtlebot3_burger MODE=short_nav_demo_scene4_mid2_obs1_open NUM_ENVS=64"
	@echo "  make train ROBOT=turtlebot3_burger MODE=short_nav_demo_scene4_wall_only NUM_ENVS=64"
	@echo "  make play CHECKPOINT=logs/skrl/.../best_agent.pt"
	@echo "  make evaluate MODE=short_nav_demo CHECKPOINT=logs/skrl/.../best_agent.pt EPISODES=100 NUM_ENVS=16"
	@echo "  make random-agent NUM_ENVS=64"
	@echo "  make monitor-metrics MONITOR_LOG_ROOT=logs/skrl/lidar_short_nav_v17_direct"
	@echo "  make auto-eval-checkpoints MONITOR_LOG_ROOT=logs/skrl/lidar_short_nav_v17_direct"

# ---- Setup -------------------------------------------------------------------

install: ## Install the isaac_lab_tutorial package in editable mode
	cd source/isaac_lab_tutorial && pip install -e .
	@echo "Installation complete. Run 'make fetch-assets ROBOT=all' then 'make list-envs'."

fetch-assets: ## Download USD assets from .collect.mapping.json manifests
	python3 scripts/download_assets.py --robot $(ROBOT) $(if $(FORCE),--force,)

# ---- Environment Inspection --------------------------------------------------

list-envs: ## List all registered Isaac Lab environments
	python3 scripts/list_envs.py

check: ## Verify prerequisites (Python, GPU, Isaac Lab)
	./launch.sh check

# ---- Run Agents --------------------------------------------------------------

random-agent: ## Run environment with random actions
	python3 scripts/random_agent.py \
		--task $(TASK) \
		--num_envs $(NUM_ENVS) \
		--device $(DEVICE)

zero-agent: ## Run environment with zero actions (no control)
	python3 scripts/zero_agent.py \
		--task $(TASK) \
		--num_envs $(NUM_ENVS) \
		--device $(DEVICE)

# ---- Training ----------------------------------------------------------------

train: ## Train RL agent with skrl (use ALGORITHM=PPO|AMP|IPPO|MAPPO|SAC)
	env PYTHONNOUSERSITE=1 PYTHONPATH= \
		"$(ISAACLAB_SH)" -p scripts/skrl/train.py \
		--task $(TASK) \
		--algorithm $(ALGORITHM) \
		--num_envs $(NUM_ENVS) \
		--ml_framework $(ML_FRAMEWORK) \
		$(if $(MAX_ITERATIONS),--max_iterations $(MAX_ITERATIONS),) \
		$(if $(SEED),--seed $(SEED),) \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		$(EXTRA_ARGS)

train-ppo: ## Train with PPO algorithm (shortcut)
	$(MAKE) train ALGORITHM=PPO

train-amp: ## Train with AMP algorithm (shortcut)
	$(MAKE) train ALGORITHM=AMP

# ---- Evaluation --------------------------------------------------------------

play: ## Evaluate a trained agent checkpoint
	env PYTHONNOUSERSITE=1 PYTHONPATH= \
		"$(ISAACLAB_SH)" -p scripts/skrl/play.py \
		--task $(TASK) \
		--algorithm $(ALGORITHM) \
		--num_envs $(NUM_ENVS) \
		--ml_framework $(ML_FRAMEWORK) \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		$(EXTRA_ARGS)

play-realtime: ## Evaluate trained agent in real-time
	$(MAKE) play EXTRA_ARGS="--real-time"

evaluate: ## Evaluate checkpoint with episode metrics (LiDAR nav tasks)
	env PYTHONNOUSERSITE=1 PYTHONPATH= \
		"$(ISAACLAB_SH)" -p scripts/skrl/eval_lidar_nav.py \
		--task $(TASK) \
		--algorithm $(ALGORITHM) \
		--num_envs $(NUM_ENVS) \
		--ml_framework $(ML_FRAMEWORK) \
		--episodes $(EPISODES) \
		$(if $(SEED),--seed $(SEED),) \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		$(EXTRA_ARGS)

eval: ## Alias of evaluate
	$(MAKE) evaluate

ros2-bridge: ## Run ROS2 bridge loop for LiDAR nav env (num_envs forced to 1)
	python3 scripts/ros2_bridge_lidar_nav.py \
		--task $(TASK_TURTLEBOT3_LIDAR) \
		--num_envs 1 \
		$(EXTRA_ARGS)

MONITOR_LOG_ROOT ?= logs/skrl/lidar_short_nav_v17_direct
MONITOR_POLL_SECONDS ?= 600

monitor-metrics: ## Monitor TensorBoard metrics and threshold alerts
	"$(ISAACSIM_PY)" scripts/skrl/monitor_training_metrics.py \
		--log-root "$(MONITOR_LOG_ROOT)" \
		--poll-seconds "$(MONITOR_POLL_SECONDS)" \
		$(EXTRA_ARGS)

auto-eval-checkpoints: ## Evaluate newly generated checkpoints periodically
	env PYTHONNOUSERSITE=1 PYTHONPATH= \
		python3 -u scripts/skrl/auto_eval_checkpoints.py \
		--log-root "$(MONITOR_LOG_ROOT)" \
		--poll-seconds "$(MONITOR_POLL_SECONDS)" \
		$(EXTRA_ARGS)

# ---- Code Quality ------------------------------------------------------------

lint: ## Run linting (flake8)
	flake8 source/ scripts/

format: ## Format code (black + isort)
	black source/ scripts/
	isort source/ scripts/
