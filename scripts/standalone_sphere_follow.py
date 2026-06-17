# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone Isaac Sim script to play a trained JetBot sphere-following policy.
#
# Workflow reference (Isaac Lab -> Isaac Sim standalone):
#   Isaac Lab                              Isaac Sim Standalone
#   ---------------------------------------------------------------
#   robot.data.root_pos_w                  jetbot.get_world_pose()[0]
#   robot.data.root_link_quat_w            jetbot.get_world_pose()[1]  (wxyz)
#   robot.data.root_com_lin_vel_b[:,0]     quat_rotate_inverse(quat, get_linear_velocity())[0]
#   math_utils.quat_apply(q, FORWARD_B)   quat_rotate(quat, [1,0,0])
#   robot.set_joint_velocity_target(a)     articulation_view.set_joint_velocity_targets(a)
#   sim dt=1/120, decimation=2             physics_dt=1/120, rendering_dt=1/60
#
# Usage:
#   python scripts/standalone_sphere_follow.py \
#       --checkpoint logs/skrl/sphere_follow_direct/<run>/checkpoints/best_agent.pt

import argparse

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Play trained JetBot sphere-following policy in Isaac Sim.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="logs/skrl/sphere_follow_direct/2026-02-21_19-46-44_ppo_torch/checkpoints/best_agent.pt",
    help="Path to the trained agent checkpoint (.pt file)",
)
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode (single step)")
args, unknown = parser.parse_known_args()

simulation_app = SimulationApp({"headless": False})

# ---- Imports after SimulationApp (Isaac Sim requirement) ----
import math

import carb
import numpy as np
import torch
import torch.nn as nn
from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade


# ==============================================================================
# Quaternion utilities (matching Isaac Lab's math_utils, wxyz convention)
# ==============================================================================

def quat_rotate(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vector by quaternion. Equivalent to Isaac Lab's quat_apply().

    Args:
        quat_wxyz: quaternion [w, x, y, z]
        vec: 3D vector
    Returns:
        Rotated 3D vector
    """
    xyz = quat_wxyz[1:4]
    t = 2.0 * np.cross(xyz, vec)
    return vec + quat_wxyz[0] * t + np.cross(xyz, t)


def quat_rotate_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vector by inverse quaternion. Equivalent to Isaac Lab's quat_apply_inverse().

    Used to convert world-frame velocity to body-frame velocity.

    Args:
        quat_wxyz: quaternion [w, x, y, z]
        vec: 3D vector in world frame
    Returns:
        3D vector in body frame
    """
    xyz = quat_wxyz[1:4]
    t = 2.0 * np.cross(xyz, vec)
    return vec - quat_wxyz[0] * t + np.cross(xyz, t)


# ==============================================================================
# Policy network (matches skrl training architecture exactly)
# ==============================================================================

class PolicyNetwork(nn.Module):
    """Policy network: 4 obs -> [64, ELU, 64, ELU] -> 2 actions.

    Matches the skrl GaussianMixin shared model from training:
        net_container.0 = Linear(4, 64)
        net_container.1 = ELU
        net_container.2 = Linear(64, 64)
        net_container.3 = ELU
        policy_layer    = Linear(64, 2)
    """

    def __init__(self):
        super().__init__()
        self.net_container = nn.Sequential(
            nn.Linear(4, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
        )
        self.policy_layer = nn.Linear(64, 2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy_layer(self.net_container(obs))


def load_policy(checkpoint_path: str):
    """Load policy weights and observation normalizer from skrl checkpoint.

    The checkpoint contains:
        policy:             model state dict (shared backbone + policy/value heads)
        state_preprocessor: RunningStandardScaler {running_mean, running_variance, current_count}

    We extract only the policy-relevant weights and the observation normalizer.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Rebuild policy network with matching keys
    policy = PolicyNetwork()
    state_dict = {
        key: ckpt["policy"][key]
        for key in [
            "net_container.0.weight", "net_container.0.bias",
            "net_container.2.weight", "net_container.2.bias",
            "policy_layer.weight", "policy_layer.bias",
        ]
    }
    policy.load_state_dict(state_dict)
    policy.eval()

    # Observation normalizer (RunningStandardScaler from training)
    obs_mean = ckpt["state_preprocessor"]["running_mean"].float()  # shape (4,)
    obs_var = ckpt["state_preprocessor"]["running_variance"].float()  # shape (4,)
    obs_std = torch.sqrt(obs_var + 1e-8)

    return policy, obs_mean, obs_std


# ==============================================================================
# Scene setup helpers
# ==============================================================================

def create_green_sphere(stage, position: np.ndarray, radius: float = 0.1):
    """Create a green sphere visual prim (no physics, just visual target)."""
    sphere_path = "/World/TargetSphere"
    sphere_prim = UsdGeom.Sphere.Define(stage, sphere_path)
    sphere_prim.GetRadiusAttr().Set(radius)
    sphere_prim.AddTranslateOp().Set(Gf.Vec3f(*position.tolist()))

    # Green material
    mat_path = "/World/TargetSphere/GreenMaterial"
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 1.0, 0.0))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(sphere_prim.GetPrim()).Bind(material)

    return sphere_prim


def set_sphere_position(sphere_prim, position: np.ndarray):
    """Update sphere translate op to new position."""
    for op in sphere_prim.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3f(*position.tolist()))
            return


def create_dome_light(stage, intensity: float = 2000.0):
    """Create dome light matching Isaac Lab environment."""
    light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    light.CreateIntensityAttr().Set(intensity)
    light.CreateColorAttr().Set(Gf.Vec3f(0.75, 0.75, 0.75))


def random_sphere_position(robot_pos: np.ndarray, spawn_min=0.5, spawn_max=1.5, z=0.1) -> np.ndarray:
    """Random position at [spawn_min, spawn_max] distance from robot, matching _spawn_sphere_positions()."""
    angle = np.random.uniform(0, 2 * math.pi)
    dist = np.random.uniform(spawn_min, spawn_max)
    return np.array([
        robot_pos[0] + dist * math.cos(angle),
        robot_pos[1] + dist * math.sin(angle),
        z,
    ])


# ==============================================================================
# Observation computation (exact match to SphereFollowEnv._get_observations)
# ==============================================================================

FORWARD_VEC_B = np.array([1.0, 0.0, 0.0])  # body-frame forward direction


def compute_observations(
    robot_pos: np.ndarray,
    robot_quat_wxyz: np.ndarray,
    robot_lin_vel_w: np.ndarray,
    sphere_pos: np.ndarray,
    max_distance: float = 3.0,
) -> np.ndarray:
    """Compute 4D observation vector, matching Isaac Lab SphereFollowEnv exactly.

    Isaac Lab reference (env.py lines 234-253):
        forwards = quat_apply(root_link_quat_w, FORWARD_VEC_B)
        dir_to_sphere = sphere_pos_w - root_pos_w;  dir_to_sphere[:, 2] = 0
        dist = norm(dir_to_sphere[:, :2]).clamp(1e-6)
        dir_norm = dir_to_sphere / dist
        dot = sum(forwards * dir_norm)
        cross_z = forwards[:,0]*dir_norm[:,1] - forwards[:,1]*dir_norm[:,0]
        dist_norm = (dist / max_distance).clamp(0, 1)
        forward_speed = root_com_lin_vel_b[:, 0]

    Returns: [dot, cross_z, dist_norm, forward_speed] as float32
    """
    # Forward direction in world frame: quat_apply(quat, [1,0,0])
    forward_w = quat_rotate(robot_quat_wxyz, FORWARD_VEC_B)

    # Direction to sphere (XY plane only)
    dir_to_sphere = sphere_pos - robot_pos
    dir_to_sphere[2] = 0.0
    dist = max(np.linalg.norm(dir_to_sphere[:2]), 1e-6)
    dir_norm = dir_to_sphere / dist

    # 1. Dot product: alignment (+1 = facing sphere)
    dot = np.dot(forward_w, dir_norm)

    # 2. Cross product Z: steering signal (+left, -right)
    cross_z = forward_w[0] * dir_norm[1] - forward_w[1] * dir_norm[0]

    # 3. Normalized distance
    dist_norm = min(dist / max_distance, 1.0)

    # 4. Forward speed in body frame: quat_apply_inverse(quat, world_vel)[0]
    #    Matches Isaac Lab's root_com_lin_vel_b[:, 0]
    lin_vel_b = quat_rotate_inverse(robot_quat_wxyz, robot_lin_vel_w)
    forward_speed = lin_vel_b[0]

    return np.array([dot, cross_z, dist_norm, forward_speed], dtype=np.float32)


# ==============================================================================
# Main
# ==============================================================================

# Load trained policy
print(f"[INFO] Loading checkpoint: {args.checkpoint}")
policy, obs_mean, obs_std = load_policy(args.checkpoint)
print(f"[INFO] Policy loaded. Obs normalizer mean={obs_mean.numpy()}, std={obs_std.numpy()}")

# Create world (matching training: physics at 120Hz, render at 60Hz = decimation 2)
my_world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)

# Spawn JetBot
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
jetbot_asset_path = assets_root_path + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
my_jetbot = my_world.scene.add(
    WheeledRobot(
        prim_path="/World/Jetbot",
        name="my_jetbot",
        wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
        create_robot=True,
        usd_path=jetbot_asset_path,
        position=np.array([0, 0.0, 0.05]),
    )
)

# Ground plane
my_world.scene.add_default_ground_plane()

# Dome light (matching Isaac Lab env)
stage = simulation_app.context.get_stage()
create_dome_light(stage, intensity=2000.0)

# Green target sphere
initial_sphere_pos = np.array([0.8, 0.3, 0.1])
sphere_prim = create_green_sphere(stage, initial_sphere_pos, radius=0.1)

# Reset world to initialize physics
my_world.reset()

# Get wheel joint indices for velocity target control
left_wheel_idx = my_jetbot.dof_names.index("left_wheel_joint")
right_wheel_idx = my_jetbot.dof_names.index("right_wheel_joint")

# ---- Config (matching SphereFollowEnvCfg) ----
SPHERE_REACH_THRESHOLD = 0.3
SPHERE_SPAWN_MIN = 0.5
SPHERE_SPAWN_MAX = 1.5
SPHERE_RADIUS = 0.1
MAX_DISTANCE = 3.0

# ---- State ----
sphere_pos = initial_sphere_pos.copy()
step_count = 0
spheres_reached = 0
reset_needed = False

print("[INFO] Running trained policy. Close the window to stop.")

while simulation_app.is_running():
    my_world.step(render=True)

    if my_world.is_stopped() and not reset_needed:
        reset_needed = True

    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            reset_needed = False
            step_count = 0
            spheres_reached = 0
            robot_pos, _ = my_jetbot.get_world_pose()
            sphere_pos = random_sphere_position(robot_pos, SPHERE_SPAWN_MIN, SPHERE_SPAWN_MAX, SPHERE_RADIUS)
            set_sphere_position(sphere_prim, sphere_pos)
            continue

        # ---- Read robot state ----
        robot_pos, robot_quat = my_jetbot.get_world_pose()  # quat is wxyz
        robot_lin_vel_w = my_jetbot.get_linear_velocity()    # world frame

        # ---- Compute 4D observation (same as Isaac Lab env) ----
        obs_np = compute_observations(robot_pos, robot_quat, robot_lin_vel_w, sphere_pos, MAX_DISTANCE)
        obs_tensor = torch.from_numpy(obs_np).unsqueeze(0)  # (1, 4)

        # ---- Normalize observation (RunningStandardScaler from training) ----
        obs_normalized = (obs_tensor - obs_mean) / obs_std

        # ---- Run policy forward pass ----
        with torch.no_grad():
            action = policy(obs_normalized).squeeze(0).numpy()  # (2,) = [left_vel, right_vel]

        # ---- Apply action as joint velocity targets ----
        # In Isaac Lab: robot.set_joint_velocity_target(actions)
        # In Isaac Sim: use articulation_view.set_joint_velocity_targets()
        vel_targets = np.zeros(my_jetbot.num_dof)
        vel_targets[left_wheel_idx] = float(action[0])
        vel_targets[right_wheel_idx] = float(action[1])
        my_jetbot._articulation_view.set_joint_velocity_targets(
            vel_targets.reshape(1, -1)
        )

        # ---- Check if sphere is reached ----
        dist_to_sphere = np.linalg.norm(sphere_pos[:2] - robot_pos[:2])
        if dist_to_sphere < SPHERE_REACH_THRESHOLD:
            spheres_reached += 1
            sphere_pos = random_sphere_position(robot_pos, SPHERE_SPAWN_MIN, SPHERE_SPAWN_MAX, SPHERE_RADIUS)
            set_sphere_position(sphere_prim, sphere_pos)
            print(
                f"[REACHED #{spheres_reached}] New sphere at "
                f"[{sphere_pos[0]:.2f}, {sphere_pos[1]:.2f}]"
            )

        step_count += 1

        # Periodic status
        if step_count % 300 == 0:
            print(
                f"[Step {step_count:>5d}] "
                f"Robot=[{robot_pos[0]:+.2f}, {robot_pos[1]:+.2f}]  "
                f"Sphere=[{sphere_pos[0]:+.2f}, {sphere_pos[1]:+.2f}]  "
                f"Dist={dist_to_sphere:.3f}  "
                f"Act=[{action[0]:+.3f}, {action[1]:+.3f}]  "
                f"Obs=[{obs_np[0]:+.2f}, {obs_np[1]:+.2f}, {obs_np[2]:.2f}, {obs_np[3]:+.2f}]  "
                f"Reached={spheres_reached}"
            )

    if args.test:
        break

my_world.stop()
simulation_app.close()
