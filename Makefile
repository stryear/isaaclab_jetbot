# ==============================================================================
# IsaacLabTutorial - Makefile
# ==============================================================================

SHELL := /bin/bash
TASK_JETBOT := Isaac-Lab-Tutorial-SphereFollow-Direct-v0
TASK_TURTLEBOT3 := Isaac-Lab-Tutorial-SphereFollow-TurtleBot3-Direct-v0
ROBOT ?= jetbot
TASK ?= $(TASK_JETBOT)
ifneq (,$(filter $(ROBOT),turtlebot3_burger turtlebot3 turtlebot tb3 burger))
TASK := $(TASK_TURTLEBOT3)
endif
NUM_ENVS ?= 100
ALGORITHM ?= PPO
DEVICE ?= cuda:0
ML_FRAMEWORK ?= torch

.PHONY: help install fetch-assets list-envs random-agent zero-agent train play lint format check

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
	@echo "  ALGORITHM=$(ALGORITHM)          RL algorithm (PPO, AMP, IPPO, MAPPO)"
	@echo "  ROBOT=$(ROBOT)            Robot profile (jetbot, turtlebot3_burger)"
	@echo "  TASK=$(TASK)              Task selected from ROBOT or explicit override"
	@echo "  DEVICE=$(DEVICE)        Compute device"
	@echo "  ML_FRAMEWORK=$(ML_FRAMEWORK)     ML framework (torch, jax, jax-numpy)"
	@echo ""
	@echo "Examples:"
	@echo "  make fetch-assets ROBOT=all"
	@echo "  make install"
	@echo "  make train ALGORITHM=PPO NUM_ENVS=100"
	@echo "  make train ROBOT=turtlebot3_burger NUM_ENVS=100"
	@echo "  make play CHECKPOINT=logs/skrl/.../best_agent.pt"
	@echo "  make random-agent NUM_ENVS=64"

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

train: ## Train RL agent with skrl (use ALGORITHM=PPO|AMP)
	python3 scripts/skrl/train.py \
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
	python3 scripts/skrl/play.py \
		--task $(TASK) \
		--algorithm $(ALGORITHM) \
		--num_envs $(NUM_ENVS) \
		--ml_framework $(ML_FRAMEWORK) \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		$(EXTRA_ARGS)

play-realtime: ## Evaluate trained agent in real-time
	$(MAKE) play EXTRA_ARGS="--real-time"

# ---- Code Quality ------------------------------------------------------------

lint: ## Run linting (flake8)
	flake8 source/ scripts/

format: ## Format code (black + isort)
	black source/ scripts/
	isort source/ scripts/
