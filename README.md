# JetBot Sphere-Following RL on Isaac Lab & Isaac Sim

A complete reinforcement learning pipeline where a **NVIDIA JetBot** learns to chase a green sphere using **PPO** - trained in **Isaac Lab**, deployed in **Isaac Sim standalone** and as an **Isaac Sim extension**.

Built on **NVIDIA Isaac Sim** and **Isaac Lab**, trained on a **Dell Pro Max** with **NVIDIA RTX PRO 6000 Blackwell** (96 GB).

## Overview

| Component | Description |
|-----------|-------------|
| **Isaac Lab Environment** | GPU-accelerated parallel RL training (100 envs, ~10 min) |
| **Isaac Sim Standalone** | Single-script policy playback using Isaac Sim APIs |
| **Isaac Sim Extension** | Full UI extension in the Examples Browser (Load/Reset/Live Status) |

## Prerequisites

- **NVIDIA Isaac Sim 4.5.0+** ([installation guide](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/index.html))
- **Isaac Lab** framework ([installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html))
- **NVIDIA GPU** with CUDA support
- **Python 3.10+**

## Asset Sync (USD Not Tracked in Git)

Robot USD files are intentionally ignored by Git (`*.usd`).  
Before training or evaluation, download assets from mapping manifests:

```bash
# Download all robot assets (JetBot + TurtleBot3)
./launch.sh fetch-assets --robot all

# Or download a single profile
./launch.sh fetch-assets --robot jetbot
./launch.sh fetch-assets --robot turtlebot3_burger
```

The downloader reads:
- `source/isaac_lab_tutorial/isaac_lab_tutorial/assets/Jetbot/.collect.mapping.json`
- `source/isaac_lab_tutorial/isaac_lab_tutorial/assets/Turtlebot/.collect.mapping.json`

> Note: JetBot manifest uses public S3 asset URLs under `Assets/Isaac/5.1/...`.

---

## Chapter I: Isaac Lab - Training

### Quick Start

```bash
# 1. Download USD assets
./launch.sh fetch-assets --robot all

# 2. Install the package
./launch.sh install

# 3. Verify environments are registered
./launch.sh list-envs
# Output:
#   Template-Isaac-Lab-Tutorial-Direct-v0
#   Isaac-Lab-Tutorial-SphereFollow-Direct-v0

# 4. Visual smoke test
./launch.sh random-agent --num_envs 10

# 5. Train PPO agent (100 parallel envs, ~10 min)
./launch.sh train --algorithm PPO --num_envs 100

# 6. Evaluate the trained agent
./launch.sh play --checkpoint logs/skrl/sphere_follow_direct/<run>/checkpoints/best_agent.pt
```

### Environments

| Environment | Obs | Action | Task |
|------------|-----|--------|------|
| `Template-Isaac-Lab-Tutorial-Direct-v0` | 3D | 2D | Follow random direction command |
| `Isaac-Lab-Tutorial-SphereFollow-Direct-v0` | 4D | 2D | Chase a green sphere target |
| `Isaac-Lab-Tutorial-SphereFollow-TurtleBot3-Direct-v0` | 4D | 2D | Chase a green sphere target (TurtleBot3 Burger) |

### Sphere-Following Environment

| Property | Value |
|----------|-------|
| **Observation Space** | 4D: dot product, cross_z, normalized distance, forward speed |
| **Action Space** | 2D: left/right wheel velocity targets |
| **Reward** | approach + alignment + reach_bonus(5.0) + time_penalty(-0.01) |
| **Episode Length** | 20 seconds |
| **Physics Rate** | 120 Hz (decimation=2, control at 60 Hz) |
| **Parallel Envs** | 100 (default) |
| **Sphere Reach** | 0.3m threshold, respawns 0.5-1.5m away |

### Launch Commands

| Command | Description |
|---------|-------------|
| `./launch.sh install` | Install package in editable mode |
| `./launch.sh fetch-assets --robot all` | Download robot USD assets from mapping manifests |
| `./launch.sh list-envs` | List all registered environments |
| `./launch.sh random-agent` | Run with random actions |
| `./launch.sh zero-agent` | Run with zero actions (baseline) |
| `./launch.sh train` | Train RL agent with skrl |
| `./launch.sh play` | Evaluate trained checkpoint |
| `./launch.sh check` | Verify prerequisites |
| `./launch.sh help` | Show full usage details |
| `./launch.sh train --robot turtlebot3_burger` | Train with TurtleBot3 Burger profile |

Make shortcuts:

```bash
./launch.sh fetch-assets --robot all
make fetch-assets ROBOT=all
make train ALGORITHM=PPO NUM_ENVS=100
make train ROBOT=turtlebot3_burger NUM_ENVS=100
make play CHECKPOINT=logs/skrl/sphere_follow_direct/<run>/checkpoints/best_agent.pt
make train-ppo           # Shortcut for PPO training
```

### Training Configuration (PPO)

| Parameter | Value |
|-----------|-------|
| Network | [64, 64] shared backbone + ELU |
| Rollout length | 48 steps |
| Learning rate | 3e-4 (KL adaptive) |
| Mini-batches | 8 |
| Learning epochs | 8 |
| Entropy coeff | 0.01 |
| Total timesteps | 24,000 |
| Discount factor | 0.99 |

---

## Chapter II: Isaac Sim Standalone

Deploy the trained checkpoint in a standalone Isaac Sim script - no Isaac Lab required.

### Usage

```bash
python scripts/standalone_sphere_follow.py \
    --checkpoint logs/skrl/sphere_follow_direct/<run>/checkpoints/best_agent.pt
```

### API Mapping (Isaac Lab -> Isaac Sim)

| Isaac Lab | Isaac Sim Standalone |
|-----------|---------------------|
| `robot.data.root_pos_w` | `jetbot.get_world_pose()[0]` |
| `robot.data.root_link_quat_w` | `jetbot.get_world_pose()[1]` (wxyz) |
| `robot.data.root_com_lin_vel_b[:,0]` | `quat_rotate_inverse(q, get_linear_velocity())[0]` |
| `math_utils.quat_apply(q, v)` | `quat_rotate(q, v)` |
| `robot.set_joint_velocity_target(a)` | `articulation_view.set_joint_velocity_targets(a)` |

### What the Script Does

- Loads the PolicyNetwork (4->64->64->2) and observation normalizer from the checkpoint
- Creates a World with matching physics/render rates (120Hz/60Hz)
- Spawns JetBot, ground plane, dome light, and green sphere
- Runs policy inference in a loop: read state -> compute obs -> normalize -> forward pass -> apply action
- Repositions the sphere when the JetBot reaches it (continuous tracking)

---

## Chapter III: Isaac Sim Extension

A full Isaac Sim extension registered in the **Examples Browser** under **Policy > JetBot Sphere Follow**.

### Installation

The extension files are installed into the Isaac Sim interactive examples:

```
isaacsim/exts/isaacsim.examples.interactive/
  isaacsim/examples/interactive/sphere_follow/
    __init__.py
    sphere_follow.py                # SphereFollow(BaseSample)
    sphere_follow_extension.py      # UI + Extension registration
```

Add to `extension.toml`:

```toml
[[python.module]]
name = "isaacsim.examples.interactive.sphere_follow"
```

### Architecture

| Class | Role |
|-------|------|
| `SphereFollow(BaseSample)` | Scene setup, policy loading, physics callback for inference |
| `SphereFollowUI(BaseSampleUITemplate)` | Checkpoint path field + live status display |
| `SphereFollowExtension(omni.ext.IExt)` | Registers in Examples Browser under "Policy" |

### UI Features

- **World Controls**: Standard Load / Reset buttons
- **Policy Configuration**: Editable checkpoint path (set before Load)
- **Live Status**: Spheres reached, step count, 4D observations, wheel actions (updated every ~0.5s)

### How It Works

1. Click **Load** in the Examples Browser
2. Extension loads checkpoint, spawns JetBot + sphere, registers physics callback
3. Policy runs at 60Hz (decimation=2 from 120Hz physics)
4. JetBot tracks sphere; sphere repositions on reach
5. Click **Reset** to restart the scene

---

## Project Structure

```
IsaacLabTutorial/
├── launch.sh                          # Main launch script
├── Makefile                           # Make-based shortcuts
├── blog_generate_pdf.py               # PDF documentation generator
├── JetBot_Sphere_Following_RL_Blog.pdf # Generated documentation
├── scripts/
│   ├── download_assets.py            # Download USD assets via .collect.mapping.json
│   ├── standalone_sphere_follow.py    # Isaac Sim standalone deployment
│   ├── list_envs.py                   # List registered environments
│   ├── random_agent.py                # Random action baseline
│   ├── zero_agent.py                  # Zero action baseline
│   └── skrl/
│       ├── train.py                   # RL training (PPO/AMP)
│       └── play.py                    # Checkpoint evaluation
├── exts/
│   └── isaacsim.examples.interactive.sphere_follow/
│       └── sphere_follow/             # Isaac Sim extension (repo copy)
│           ├── __init__.py
│           ├── sphere_follow.py
│           └── sphere_follow_extension.py
└── source/
    └── isaac_lab_tutorial/
        ├── setup.py
        └── isaac_lab_tutorial/
            ├── __init__.py
            ├── ui_extension_example.py
            ├── sphere_follow_extension.py  # Local Isaac Sim extension
            ├── assets/
            │   ├── Jetbot/.collect.mapping.json
            │   └── Turtlebot/.collect.mapping.json
            ├── robots/
            │   ├── jetbot.py              # JetBot ArticulationCfg (local USD)
            │   └── turtlebot3_burger.py   # TurtleBot3 Burger ArticulationCfg
            └── tasks/
                └── direct/
                    └── isaac_lab_tutorial/
                        ├── __init__.py                    # Gym registration
                        ├── isaac_lab_tutorial_env.py      # IsaacLabTutorialEnv + SphereFollowEnv
                        ├── isaac_lab_tutorial_env_cfg.py  # Environment configs
                        └── agents/
                            ├── skrl_ppo_cfg.yaml
                            ├── skrl_amp_cfg.yaml
                            └── skrl_sphere_follow_ppo_cfg.yaml
```

## Training Logs

```
logs/skrl/sphere_follow_direct/
└── <timestamp>_ppo_torch/
    ├── checkpoints/
    │   ├── best_agent.pt      # Best by episode return
    │   └── agent_24000.pt     # Final checkpoint
    └── params/
        ├── env.yaml           # Environment config snapshot
        └── agent.yaml         # Agent hyperparameters
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
