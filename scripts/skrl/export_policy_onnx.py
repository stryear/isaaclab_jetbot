#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Export a skrl PPO checkpoint to policy.onnx for ROS2 deployment.

This exporter intentionally mirrors the deployment contract used by
``turtlebot_rl_planner``:

- input: raw observation tensor with shape ``[batch, obs_dim]``
- preprocessing: embedded RunningStandardScaler normalization
- output: deterministic policy mean action, clamped to ``[-1, 1]``

The exported ONNX model can therefore be consumed directly by the runtime node
without any extra normalization logic outside the graph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class SharedPolicyModel(nn.Module):
    """Policy network matching the skrl shared Actor-Critic backbone."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net_container = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
        )
        self.policy_layer = nn.Linear(128, action_dim)
        # Unused in deterministic deployment, but kept for checkpoint compatibility.
        self.log_std_parameter = nn.Parameter(torch.zeros(action_dim), requires_grad=True)
        self.value_layer = nn.Linear(128, 1)

    def forward_policy(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.net_container(obs)
        return self.policy_layer(features)


class ExportablePolicy(nn.Module):
    """Deployment graph: normalize observations, run policy mean, clamp output."""

    def __init__(
        self,
        policy_state_dict: dict[str, torch.Tensor],
        running_mean: torch.Tensor,
        running_variance: torch.Tensor,
        action_dim: int = 2,
        clip_output: bool = True,
    ) -> None:
        super().__init__()

        if "net_container.0.weight" not in policy_state_dict:
            raise RuntimeError("Unsupported checkpoint: missing 'net_container.0.weight' in policy state dict.")

        obs_dim = int(policy_state_dict["net_container.0.weight"].shape[1])
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.clip_output = clip_output

        self.policy = SharedPolicyModel(obs_dim=obs_dim, action_dim=action_dim)
        self.policy.load_state_dict(policy_state_dict, strict=True)
        self.policy.eval()

        self.register_buffer("running_mean", running_mean.to(dtype=torch.float32).reshape(obs_dim))
        self.register_buffer("running_variance", running_variance.to(dtype=torch.float32).reshape(obs_dim))
        self.register_buffer("eps", torch.tensor(1.0e-8, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(dtype=torch.float32)
        norm_obs = torch.clamp(
            (obs - self.running_mean) / (torch.sqrt(self.running_variance) + self.eps),
            min=-5.0,
            max=5.0,
        )
        action = self.policy.forward_policy(norm_obs)
        if self.clip_output:
            action = torch.clamp(action, min=-1.0, max=1.0)
        return action


def _find_latest_selected_checkpoint(project_root: Path) -> Path | None:
    summary_files = sorted(
        project_root.glob("logs/eval_screen_*/selection_summary_v2.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for summary_path in summary_files:
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        selected = data.get("selected") or {}
        checkpoint = selected.get("checkpoint")
        if checkpoint:
            candidate = Path(checkpoint).expanduser().resolve()
            if candidate.exists():
                return candidate
    return None


def _find_latest_best_checkpoint(project_root: Path) -> Path | None:
    best_candidates = sorted(
        project_root.glob("logs/skrl/*/*/checkpoints/best_agent.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return best_candidates[0].resolve() if best_candidates else None


def _resolve_checkpoint(project_root: Path, checkpoint_arg: str | None) -> Path:
    if checkpoint_arg:
        checkpoint_path = Path(checkpoint_arg).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    checkpoint_path = _find_latest_selected_checkpoint(project_root)
    if checkpoint_path is not None:
        return checkpoint_path

    checkpoint_path = _find_latest_best_checkpoint(project_root)
    if checkpoint_path is not None:
        return checkpoint_path

    raise FileNotFoundError("Could not auto-discover a checkpoint under logs/.")


def _load_checkpoint(checkpoint_path: Path) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "policy" not in checkpoint:
        raise RuntimeError(f"Unsupported checkpoint format: missing 'policy' in {checkpoint_path}")

    state_preprocessor = checkpoint.get("state_preprocessor")
    if not isinstance(state_preprocessor, dict):
        raise RuntimeError(f"Unsupported checkpoint format: missing 'state_preprocessor' in {checkpoint_path}")
    if "running_mean" not in state_preprocessor or "running_variance" not in state_preprocessor:
        raise RuntimeError(f"Unsupported state preprocessor format in {checkpoint_path}")

    running_mean = state_preprocessor["running_mean"]
    running_variance = state_preprocessor["running_variance"]
    if not isinstance(running_mean, torch.Tensor) or not isinstance(running_variance, torch.Tensor):
        raise RuntimeError(f"Invalid state preprocessor tensors in {checkpoint_path}")

    return checkpoint, running_mean, running_variance


def _default_output_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name("policy.onnx")


def _export_model(
    model: ExportablePolicy,
    output_path: Path,
    opset: int,
    obs_dim: int,
    dynamic_batch: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)
    export_kwargs: dict[str, Any] = {
        "input_names": ["obs"],
        "output_names": ["action"],
        "opset_version": opset,
        "do_constant_folding": True,
    }
    if dynamic_batch:
        export_kwargs["dynamic_axes"] = {
            "obs": {0: "batch"},
            "action": {0: "batch"},
        }

    with torch.no_grad():
        torch.onnx.export(model, dummy_obs, str(output_path), **export_kwargs)


def _validate_onnx(model: ExportablePolicy, output_path: Path, obs_dim: int) -> dict[str, float]:
    import onnxruntime as ort

    rng = np.random.default_rng(42)
    test_obs = rng.normal(size=(4, obs_dim)).astype(np.float32)

    with torch.inference_mode():
        torch_out = model(torch.from_numpy(test_obs)).cpu().numpy()

    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: test_obs})[0]

    abs_diff = np.abs(torch_out - onnx_out)
    return {
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
    }


def _write_metadata(
    metadata_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    obs_dim: int,
    validation: dict[str, float],
    clip_output: bool,
) -> None:
    payload = {
        "checkpoint": str(checkpoint_path),
        "onnx_path": str(output_path),
        "obs_dim": obs_dim,
        "action_dim": 2,
        "clip_output": clip_output,
        "validation": validation,
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a skrl checkpoint to deployment ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (.pt). If omitted, auto-pick latest selected checkpoint from logs/eval_screen_*/.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX path. Defaults to <checkpoint_dir>/policy.onnx.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--no-dynamic-batch",
        action="store_true",
        default=False,
        help="Disable dynamic batch axis and export fixed batch=1 input.",
    )
    parser.add_argument(
        "--no-clip-output",
        action="store_true",
        default=False,
        help="Export raw policy mean action without clamping to [-1, 1].",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        default=False,
        help="Skip ONNXRuntime numeric validation after export.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    project_root = Path(__file__).resolve().parents[2]

    checkpoint_path = _resolve_checkpoint(project_root, args.checkpoint)
    output_path = Path(args.output).expanduser().resolve() if args.output else _default_output_path(checkpoint_path)

    checkpoint, running_mean, running_variance = _load_checkpoint(checkpoint_path)
    model = ExportablePolicy(
        policy_state_dict=checkpoint["policy"],
        running_mean=running_mean,
        running_variance=running_variance,
        clip_output=not args.no_clip_output,
    )
    model.eval()

    _export_model(
        model=model,
        output_path=output_path,
        opset=args.opset,
        obs_dim=model.obs_dim,
        dynamic_batch=not args.no_dynamic_batch,
    )

    validation = {"max_abs_diff": float("nan"), "mean_abs_diff": float("nan")}
    if not args.skip_validate:
        validation = _validate_onnx(model, output_path, model.obs_dim)

    metadata_path = output_path.with_suffix(".json")
    _write_metadata(
        metadata_path=metadata_path,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        obs_dim=model.obs_dim,
        validation=validation,
        clip_output=not args.no_clip_output,
    )

    print(f"[EXPORT] checkpoint={checkpoint_path}")
    print(f"[EXPORT] onnx={output_path}")
    print(f"[EXPORT] obs_dim={model.obs_dim} action_dim={model.action_dim}")
    print(f"[EXPORT] clip_output={not args.no_clip_output}")
    if not args.skip_validate:
        print(
            "[EXPORT] validation "
            f"max_abs_diff={validation['max_abs_diff']:.6e} "
            f"mean_abs_diff={validation['mean_abs_diff']:.6e}"
        )
    print(f"[EXPORT] metadata={metadata_path}")


if __name__ == "__main__":
    main()
