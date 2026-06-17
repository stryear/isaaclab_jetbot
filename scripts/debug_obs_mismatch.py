#!/usr/bin/env python3
"""Diagnostic tool to check observation space mismatch between training and inference."""

import argparse
import torch
import numpy as np

def load_checkpoint_stats(checkpoint_path: str):
    """Load running mean/variance from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_pre = ckpt.get("state_preprocessor", {})

    if "running_mean" in state_pre and "running_variance" in state_pre:
        mean = state_pre["running_mean"].numpy()
        var = state_pre["running_variance"].numpy()
        std = np.sqrt(var)
        return mean, std
    else:
        print("WARNING: No running statistics found in checkpoint")
        return None, None

def print_obs_analysis(mean, std, obs_dim: int):
    """Print analysis of observation statistics."""
    if mean is None:
        return

    print(f"\n{'='*80}")
    print(f"Observation Dimension: {obs_dim}")
    print(f"Checkpoint Statistics Shape: mean={mean.shape}, std={std.shape}")
    print(f"{'='*80}\n")

    # Observation layout for LidarNav (42D) and LidarJunction (45D)
    if obs_dim == 42:
        labels = [
            "goal_dir_x", "goal_dir_y", "goal_dist_norm",
            "forward_speed", "yaw_rate", "min_lidar"
        ] + [f"lidar_{i:02d}" for i in range(36)]
    elif obs_dim == 45:
        labels = [
            "goal_dir_x", "goal_dir_y", "goal_dist_norm",
            "forward_speed", "yaw_rate", "min_lidar",
            "turn_left", "turn_straight", "turn_right"
        ] + [f"lidar_{i:02d}" for i in range(36)]
    else:
        labels = [f"obs_{i:02d}" for i in range(obs_dim)]

    print(f"{'Index':<6} {'Feature':<18} {'Mean':<12} {'Std':<12} {'Notes'}")
    print("-" * 80)

    for i in range(min(len(labels), len(mean))):
        notes = ""
        # Flag suspicious values
        if abs(mean[i]) > 10:
            notes += "⚠️ Large mean "
        if std[i] < 0.01:
            notes += "⚠️ Low variance "
        if std[i] > 10:
            notes += "⚠️ High variance "

        print(f"{i:<6} {labels[i]:<18} {mean[i]:<12.4f} {std[i]:<12.4f} {notes}")

    print("\n" + "="*80)
    print("Expected ranges (for sanity check):")
    print("  goal_dir_x/y:     mean≈0, std≈0.5 (unit vector components)")
    print("  goal_dist_norm:   mean≈0.3-0.5, std≈0.2-0.3 (normalized to [0,1])")
    print("  forward_speed:    mean≈0.05-0.10, std≈0.03-0.05 (m/s)")
    print("  yaw_rate:         mean≈0, std≈0.3-0.6 (rad/s)")
    print("  min_lidar:        mean≈0.4-0.6, std≈0.2-0.3 (normalized)")
    print("  lidar_XX:         mean≈0.5-0.7, std≈0.2-0.3 (normalized)")
    print("="*80 + "\n")

def simulate_gazebo_obs(mean, std):
    """Simulate a typical Gazebo observation and check normalization."""
    if mean is None:
        return

    print("\n" + "="*80)
    print("Simulated Gazebo Observation Test")
    print("="*80 + "\n")

    # Typical Gazebo observation (before normalization)
    obs_raw = np.array([
        0.707, 0.707,  # goal 45° ahead, unit vector
        0.6,           # goal 3m away, normalized by 5m
        0.08,          # forward speed 0.08 m/s
        0.0,           # no turning
        0.5,           # min lidar 1.5m, normalized by 3m
    ] + [0.6] * 36)  # lidar beams around 1.8m average

    obs_raw = obs_raw[:len(mean)]  # match dimension

    # Apply checkpoint normalization
    obs_normalized = (obs_raw - mean) / (std + 1e-8)
    obs_clipped = np.clip(obs_normalized, -5.0, 5.0)

    print("Raw observation (first 10 dims):")
    print(obs_raw[:10])
    print("\nNormalized observation (first 10 dims):")
    print(obs_normalized[:10])
    print("\nClipped observation (first 10 dims):")
    print(obs_clipped[:10])

    # Check for anomalies
    out_of_range = np.sum(np.abs(obs_clipped) > 4.0)
    print(f"\nDimensions with |normalized_obs| > 4.0: {out_of_range}/{len(obs_clipped)}")

    if out_of_range > len(obs_clipped) * 0.3:
        print("⚠️  WARNING: >30% of observations are near clipping bounds!")
        print("   This suggests training/inference distribution mismatch.")
    else:
        print("✓  Normalization looks reasonable.")

    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Diagnose observation space mismatch")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_agent.pt")
    parser.add_argument("--obs_dim", type=int, default=42, help="Expected observation dimension")
    args = parser.parse_args()

    print(f"\nLoading checkpoint: {args.checkpoint}")
    mean, std = load_checkpoint_stats(args.checkpoint)

    if mean is not None:
        print_obs_analysis(mean, std, args.obs_dim)
        simulate_gazebo_obs(mean, std)
    else:
        print("ERROR: Could not load normalization statistics from checkpoint.")
        print("The checkpoint may not have been trained with observation normalization.")

if __name__ == "__main__":
    main()
