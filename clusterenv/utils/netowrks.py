from typing import Any
from tianshou.highlevel.module.intermediate import (
    IntermediateModule,
    IntermediateModuleFactory,
)
from tianshou.highlevel.env import Environments
from tianshou.highlevel.module.core import (
    TDevice,
)
from tianshou.utils.net.common import NetBase, TRecurrentState
from clusterenv.envs.base import ClusterEnv
from collections.abc import Callable, Sequence
import numpy as np
from torch import nn
import torch
import gymnasium as gym


class DQN(NetBase[Any]):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int] | int,
        device: str | int | torch.device = "cpu",
        features_only: bool = False,
        output_dim_added_layer: int | None = None,
        layer_init: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> None:
        if not features_only and output_dim_added_layer is not None:
            raise ValueError(
                "Should not provide explicit output dimension using `output_dim_added_layer` when `features_only` is true.",
            )
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            base_cnn_output_dim = int(
                np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:])
            )
        if not features_only:
            action_dim = int(np.prod(action_shape))
            self.net = nn.Sequential(
                self.net,
                layer_init(nn.Linear(base_cnn_output_dim, 512)),
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(512, action_dim)),
            )
            self.output_dim = action_dim
        elif output_dim_added_layer is not None:
            self.net = nn.Sequential(
                self.net,
                layer_init(nn.Linear(base_cnn_output_dim, output_dim_added_layer)),
                nn.ReLU(inplace=True),
            )
            self.output_dim = output_dim_added_layer
        else:
            self.output_dim = base_cnn_output_dim

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.net(obs), state


class IntermediateModuleFactoryClusterDQN(IntermediateModuleFactory):
    def __init__(self, features_only: bool = False, net_only: bool = False) -> None:
        self.features_only = features_only
        self.net_only = net_only

    def create_intermediate_module(
        self, envs: Environments, device: TDevice
    ) -> IntermediateModule:
        obs_shape = envs.get_observation_shape()
        if isinstance(obs_shape, int):
            obs_shape = [obs_shape]
        assert len(obs_shape) == 3
        c, h, w = obs_shape
        action_shape = envs.get_action_shape()
        if isinstance(action_shape, np.int64):
            action_shape = int(action_shape)
        dqn = DQN(
            c=c,
            h=h,
            w=w,
            action_shape=action_shape,
            device=device,
            features_only=self.features_only,
        ).to(device)
        module = dqn.net if self.net_only else dqn
        return IntermediateModule(module, dqn.output_dim)
