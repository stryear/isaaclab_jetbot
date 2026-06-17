#!/usr/bin/env python3
"""Migrate a feedforward LiDAR checkpoint to LSTM-compatible checkpoint format."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from lstm_runner_patch import LidarLSTMDeterministicMixin, LidarLSTMGaussianMixin


def _build_models(args) -> tuple[LidarLSTMGaussianMixin, LidarLSTMDeterministicMixin]:
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(args.obs_dim,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(args.action_dim,), dtype=np.float32)

    common_kwargs = {
        "state_dim": args.state_dim,
        "lidar_dim": args.lidar_dim,
        "lidar_feature_dim": args.lidar_feature_dim,
        "fuse_hidden_dims": tuple(args.fuse_hidden_dims),
        "rnn_hidden_size": args.rnn_hidden_size,
        "rnn_num_layers": args.rnn_num_layers,
        "sequence_length": args.sequence_length,
        "rnn_num_envs": 1,
    }

    policy = LidarLSTMGaussianMixin(
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        clip_actions=True,
        clip_log_std=True,
        min_log_std=-20.0,
        max_log_std=2.0,
        initial_log_std=0.0,
        **common_kwargs,
    )
    value = LidarLSTMDeterministicMixin(
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        clip_actions=False,
        **common_kwargs,
    )

    # Initialize lazy modules once before reading state dicts.
    policy.init_state_dict(role="policy")
    value.init_state_dict(role="value")
    return policy, value


def _copy_matching(src: dict[str, torch.Tensor], dst: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
    copied = 0
    out = {k: v.detach().clone() for k, v in dst.items()}
    for key, dst_tensor in dst.items():
        src_tensor = src.get(key, None)
        if src_tensor is None:
            continue
        if tuple(src_tensor.shape) != tuple(dst_tensor.shape):
            continue
        out[key] = src_tensor.detach().clone()
        copied += 1
    return out, copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert feedforward LiDAR PPO checkpoint to LSTM checkpoint.")
    parser.add_argument("--source", required=True, help="Path to source (MLP) checkpoint.")
    parser.add_argument("--output", required=True, help="Path to output (LSTM) checkpoint.")
    parser.add_argument("--obs-dim", type=int, default=106, help="Flat observation dimension.")
    parser.add_argument("--action-dim", type=int, default=2, help="Action dimension.")
    parser.add_argument("--state-dim", type=int, default=16, help="State prefix dimension in observations.")
    parser.add_argument("--lidar-dim", type=int, default=90, help="LiDAR slice dimension in observations.")
    parser.add_argument("--lidar-feature-dim", type=int, default=512, help="LiDAR encoder output size.")
    parser.add_argument(
        "--fuse-hidden-dims",
        type=int,
        nargs="+",
        default=[512, 256],
        help="Hidden dimensions for fused MLP layers.",
    )
    parser.add_argument("--rnn-hidden-size", type=int, default=256, help="LSTM hidden size.")
    parser.add_argument("--rnn-num-layers", type=int, default=1, help="LSTM layer count.")
    parser.add_argument("--sequence-length", type=int, default=16, help="RNN sequence length used during PPO updates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not source_path.is_file():
        raise FileNotFoundError(f"Source checkpoint does not exist: {source_path}")

    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint format in {source_path}")

    source_policy = payload.get("policy")
    source_value = payload.get("value")
    if not isinstance(source_policy, dict):
        raise ValueError("Source checkpoint has no 'policy' state_dict.")
    if not isinstance(source_value, dict):
        source_value = source_policy

    policy_model, value_model = _build_models(args)
    target_policy = policy_model.state_dict()
    target_value = value_model.state_dict()

    migrated_policy, copied_policy = _copy_matching(source_policy, target_policy)
    migrated_value, copied_value = _copy_matching(source_value, target_value)

    migrated = {
        "policy": migrated_policy,
        "value": migrated_value,
    }
    if "state_preprocessor" in payload:
        migrated["state_preprocessor"] = payload["state_preprocessor"]
    if "value_preprocessor" in payload:
        migrated["value_preprocessor"] = payload["value_preprocessor"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, output_path)

    print(f"[MIGRATE] source: {source_path}")
    print(f"[MIGRATE] output: {output_path}")
    print(f"[MIGRATE] policy tensors copied: {copied_policy}/{len(target_policy)}")
    print(f"[MIGRATE] value tensors copied: {copied_value}/{len(target_value)}")


if __name__ == "__main__":
    main()
