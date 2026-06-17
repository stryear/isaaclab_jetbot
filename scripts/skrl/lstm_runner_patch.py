"""Runtime patch to extend skrl Runner with LiDAR LSTM models and PPO_RNN."""

from __future__ import annotations

import copy
import inspect
import textwrap
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


def _parse_hidden_dims(value: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        dims = tuple(int(v) for v in value)
        return dims if dims else (256,)
    return (int(value),)


class _LidarLSTMBase(Model):
    """Common LiDAR + MLP + LSTM backbone for recurrent navigation policies."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        *,
        state_dim: int = 16,
        lidar_dim: int = 90,
        lidar_feature_dim: int = 512,
        fuse_hidden_dims: int | Sequence[int] = (512, 256),
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 1,
        sequence_length: int = 16,
        rnn_num_envs: int = 1,
    ):
        super().__init__(observation_space, action_space, device)

        self.state_dim = int(state_dim)
        self.lidar_dim = int(lidar_dim)
        self.rnn_hidden_size = int(rnn_hidden_size)
        self.rnn_num_layers = int(rnn_num_layers)
        self.sequence_length = max(1, int(sequence_length))
        self.rnn_num_envs = max(1, int(rnn_num_envs))

        self.lidar_net_container = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 1), stride=(2, 1)),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1)),
            nn.ELU(),
            nn.Flatten(),
            nn.LazyLinear(int(lidar_feature_dim)),
            nn.ELU(),
        )

        fuse_dims = _parse_hidden_dims(fuse_hidden_dims)
        fuse_layers: list[nn.Module] = []
        in_features = int(lidar_feature_dim) + self.state_dim
        for hidden_dim in fuse_dims:
            fuse_layers.append(nn.Linear(in_features, hidden_dim))
            fuse_layers.append(nn.ELU())
            in_features = hidden_dim
        self.fuse_container = nn.Sequential(*fuse_layers)

        self.rnn = nn.LSTM(
            input_size=in_features,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            batch_first=True,
        )

    def get_specification(self):
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (self.rnn_num_layers, self.rnn_num_envs, self.rnn_hidden_size),
                    (self.rnn_num_layers, self.rnn_num_envs, self.rnn_hidden_size),
                ],
            }
        }

    def _prepare_states(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if states.ndim != 2:
            states = states.view(states.shape[0], -1)
        if states.shape[1] < self.state_dim + self.lidar_dim:
            raise ValueError(
                f"Observation size {states.shape[1]} is smaller than state_dim + lidar_dim "
                f"({self.state_dim + self.lidar_dim})."
            )
        return states

    def _encode_features(self, states: torch.Tensor) -> torch.Tensor:
        states = self._prepare_states(states)
        base = states[:, : self.state_dim]
        lidar = states[:, self.state_dim : self.state_dim + self.lidar_dim]
        lidar = lidar.view(-1, 1, self.lidar_dim, 1)
        lidar_feat = self.lidar_net_container(lidar)
        fused = torch.cat((lidar_feat, base), dim=1)
        return self.fuse_container(fused)

    def _initial_rnn_states(self, batch_size: int, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (self.rnn_num_layers, batch_size, self.rnn_hidden_size)
        h = torch.zeros(shape, dtype=ref.dtype, device=ref.device)
        c = torch.zeros(shape, dtype=ref.dtype, device=ref.device)
        return h, c

    def _select_rnn_states(
        self,
        rnn_states: Sequence[torch.Tensor] | None,
        batch_size: int,
        ref: torch.Tensor,
        sequence_mode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rnn_states and len(rnn_states) >= 2:
            h = rnn_states[0].to(device=ref.device, dtype=ref.dtype).contiguous()
            c = rnn_states[1].to(device=ref.device, dtype=ref.dtype).contiguous()
            if sequence_mode and h.shape[1] >= self.sequence_length and h.shape[1] % self.sequence_length == 0:
                h = h[:, :: self.sequence_length, :].contiguous()
                c = c[:, :: self.sequence_length, :].contiguous()
            if h.shape[1] != batch_size:
                if h.shape[1] == 1:
                    h = h.expand(-1, batch_size, -1).contiguous()
                    c = c.expand(-1, batch_size, -1).contiguous()
                elif h.shape[1] > batch_size:
                    h = h[:, :batch_size, :].contiguous()
                    c = c[:, :batch_size, :].contiguous()
                else:
                    repeat = (batch_size + h.shape[1] - 1) // h.shape[1]
                    h = h.repeat(1, repeat, 1)[:, :batch_size, :].contiguous()
                    c = c.repeat(1, repeat, 1)[:, :batch_size, :].contiguous()
            return h, c
        return self._initial_rnn_states(batch_size=batch_size, ref=ref)

    def _forward_recurrent(self, features: torch.Tensor, inputs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        terminated = inputs.get("terminated", None)
        sequence_mode = (
            terminated is not None
            and self.sequence_length > 1
            and features.shape[0] >= self.sequence_length
            and features.shape[0] % self.sequence_length == 0
        )

        if sequence_mode:
            total = features.shape[0]
            batch_size = total // self.sequence_length
            rnn_states = inputs.get("rnn", None)
            h, c = self._select_rnn_states(rnn_states, batch_size, features, sequence_mode=True)

            sequence = features.reshape(batch_size, self.sequence_length, -1)
            terminated_view = terminated
            if terminated_view.ndim > 1:
                terminated_view = terminated_view.reshape(terminated_view.shape[0], -1)
                terminated_view = terminated_view[:, 0]
            terminated_seq = terminated_view.to(torch.bool).reshape(batch_size, self.sequence_length)

            outputs = []
            for t in range(self.sequence_length):
                if t > 0:
                    done_prev = terminated_seq[:, t - 1]
                    if bool(done_prev.any().item()):
                        keep = (~done_prev).to(h.dtype).view(1, batch_size, 1)
                        h = h * keep
                        c = c * keep
                out_t, (h, c) = self.rnn(sequence[:, t : t + 1, :], (h, c))
                outputs.append(out_t)

            recurrent_features = torch.cat(outputs, dim=1).reshape(total, self.rnn_hidden_size)
            return recurrent_features, [h, c]

        batch_size = features.shape[0]
        rnn_states = inputs.get("rnn", None)
        h, c = self._select_rnn_states(rnn_states, batch_size, features, sequence_mode=False)
        out, (h, c) = self.rnn(features.unsqueeze(1), (h, c))
        recurrent_features = out.squeeze(1)
        return recurrent_features, [h, c]


class LidarLSTMGaussianMixin(GaussianMixin, _LidarLSTMBase):
    """LiDAR recurrent Gaussian policy compatible with skrl PPO_RNN."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = True,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        initial_log_std: float = 0.0,
        **kwargs,
    ):
        _LidarLSTMBase.__init__(self, observation_space, action_space, device, **kwargs)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )
        self.policy_layer = nn.Linear(self.rnn_hidden_size, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), float(initial_log_std)))

    def compute(self, inputs, role):
        features = self._encode_features(inputs["states"])
        recurrent_features, rnn_states = self._forward_recurrent(features, inputs)
        mean_actions = self.policy_layer(recurrent_features)
        return mean_actions, self.log_std_parameter, {"rnn": rnn_states}


class LidarLSTMDeterministicMixin(DeterministicMixin, _LidarLSTMBase):
    """LiDAR recurrent value model compatible with skrl PPO_RNN."""

    def __init__(self, observation_space, action_space, device, clip_actions: bool = False, **kwargs):
        _LidarLSTMBase.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.value_layer = nn.Linear(self.rnn_hidden_size, 1)

    def compute(self, inputs, role):
        features = self._encode_features(inputs["states"])
        recurrent_features, rnn_states = self._forward_recurrent(features, inputs)
        values = self.value_layer(recurrent_features)
        return values, {"rnn": rnn_states}


def _source_for(cls) -> str:
    return textwrap.dedent(inspect.getsource(cls))


def lidar_lstm_gaussian_model(
    observation_space,
    action_space,
    device,
    return_source: bool = False,
    **kwargs,
):
    if return_source:
        return _source_for(LidarLSTMGaussianMixin)
    return LidarLSTMGaussianMixin(observation_space=observation_space, action_space=action_space, device=device, **kwargs)


def lidar_lstm_deterministic_model(
    observation_space,
    action_space,
    device,
    return_source: bool = False,
    **kwargs,
):
    if return_source:
        return _source_for(LidarLSTMDeterministicMixin)
    return LidarLSTMDeterministicMixin(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        **kwargs,
    )


def apply_lstm_runner_patch() -> None:
    """Patch skrl Runner to resolve custom LiDAR LSTM model and PPO_RNN classes."""

    from skrl.utils.runner.torch import Runner
    from skrl import logger

    if getattr(Runner, "_isaaclab_lstm_patch", False):
        return

    original_component = Runner._component
    original_generate_agent = Runner._generate_agent

    def _component_with_lstm(self, name: str):
        key = str(name).lower()

        if key in {"lidarlstmgaussianmixin", "lidar_lstm_gaussian", "lidar_lstm_gaussian_mixin"}:
            return lidar_lstm_gaussian_model
        if key in {"lidarlstmdeterministicmixin", "lidar_lstm_deterministic", "lidar_lstm_deterministic_mixin"}:
            return lidar_lstm_deterministic_model
        if key in {"ppo_rnn", "ppornn"}:
            from skrl.agents.torch.ppo import PPO_RNN

            return PPO_RNN
        if key in {"ppo_rnn_default_config", "ppornn_default_config"}:
            from skrl.agents.torch.ppo.ppo_rnn import PPO_DEFAULT_CONFIG as PPO_RNN_DEFAULT_CONFIG

            return PPO_RNN_DEFAULT_CONFIG
        return original_component(self, name)

    def _generate_agent_with_lstm(self, env, cfg, models):
        from skrl.envs.wrappers.torch import MultiAgentEnvWrapper

        multi_agent = isinstance(env, MultiAgentEnvWrapper)
        device = env.device
        num_envs = env.num_envs
        possible_agents = env.possible_agents if multi_agent else ["agent"]
        state_spaces = env.state_spaces if multi_agent else {"agent": env.state_space}
        observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
        action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}

        agent_class = cfg.get("agent", {}).get("class", "").lower()
        if not agent_class:
            raise ValueError("No 'class' field defined in 'agent' cfg")

        # Fall back to original implementation for unaffected classes.
        if agent_class not in {"ppo_rnn"}:
            return original_generate_agent(self, env, cfg, models)

        # check for memory configuration (backward compatibility)
        if "memory" not in cfg:
            logger.warning(
                "Deprecation warning: No 'memory' field defined in cfg. Using the default generated configuration"
            )
            cfg["memory"] = {"class": "RandomMemory", "memory_size": -1}
        # get memory class and remove 'class' field
        try:
            memory_class = self._component(cfg["memory"]["class"])
            del cfg["memory"]["class"]
        except KeyError:
            memory_class = self._component("RandomMemory")
            logger.warning("No 'class' field defined in 'memory' cfg. 'RandomMemory' will be used as default")
        memories = {}
        # instantiate memory
        if cfg["memory"]["memory_size"] < 0:
            cfg["memory"]["memory_size"] = cfg["agent"]["rollouts"]
        for agent_id in possible_agents:
            memories[agent_id] = memory_class(num_envs=num_envs, device=device, **self._process_cfg(cfg["memory"]))

        if agent_class in ["a2c", "cem", "ddpg", "ddqn", "dqn", "ppo", "ppo_rnn", "rpo", "sac", "td3", "trpo"]:
            agent_id = possible_agents[0]
            agent_cfg = self._component(f"{agent_class}_DEFAULT_CONFIG").copy()
            agent_cfg.update(self._process_cfg(cfg["agent"]))
            agent_cfg.get("state_preprocessor_kwargs", {}).update(
                {"size": observation_spaces[agent_id], "device": device}
            )
            agent_cfg.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})
            if agent_cfg.get("exploration", {}).get("noise", None):
                agent_cfg["exploration"].get("noise_kwargs", {}).update({"device": device})
                agent_cfg["exploration"]["noise"] = agent_cfg["exploration"]["noise"](
                    **agent_cfg["exploration"].get("noise_kwargs", {})
                )
            if agent_cfg.get("smooth_regularization_noise", None):
                agent_cfg.get("smooth_regularization_noise_kwargs", {}).update({"device": device})
                agent_cfg["smooth_regularization_noise"] = agent_cfg["smooth_regularization_noise"](
                    **agent_cfg.get("smooth_regularization_noise_kwargs", {})
                )
            agent_kwargs = {
                "models": models[agent_id],
                "memory": memories[agent_id],
                "observation_space": observation_spaces[agent_id],
                "action_space": action_spaces[agent_id],
            }
            return self._component(agent_class)(cfg=agent_cfg, device=device, **agent_kwargs)

        # If class changes in future, preserve original behavior.
        return original_generate_agent(self, env, cfg, models)

    Runner._component = _component_with_lstm
    Runner._generate_agent = _generate_agent_with_lstm
    Runner._isaaclab_lstm_patch = True


__all__ = [
    "LidarLSTMGaussianMixin",
    "LidarLSTMDeterministicMixin",
    "apply_lstm_runner_patch",
    "lidar_lstm_gaussian_model",
    "lidar_lstm_deterministic_model",
]
