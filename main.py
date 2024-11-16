import gymnasium as gym
import tianshou as ts
import clusterenv.envs
import torch
import numpy as np
from torch import nn

import logging

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s [%(levelname)s] %(filename)s: %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )
# logging.getLogger("tianshou").setLevel(logging.WARNING)
# logging.getLogger("numba").setLevel(logging.WARNING)


class Net(nn.Module):
    def __init__(self, machine_shape, job_shape, action_shape):
        super().__init__()
        # Number of channels from the resource dimension
        machine_channels = machine_shape[1]
        job_channels = job_shape[1]

        # Sub-networks for machines and jobs
        self.machine_net = nn.Sequential(
            nn.Conv2d(
                machine_channels,
                32,
                kernel_size=(
                    3,
                    3),
                stride=1,
                padding=1),
            nn.ReLU(
                inplace=True),
            nn.Flatten(),
            nn.Linear(
                32 *
                machine_shape[0] *
                machine_shape[2],
                128),
            nn.ReLU(
                inplace=True),
        )
        self.job_net = nn.Sequential(
            nn.Conv2d(
                job_channels,
                32,
                kernel_size=(
                    3,
                    3),
                stride=1,
                padding=1),
            nn.ReLU(
                inplace=True),
            nn.Flatten(),
            nn.Linear(
                32 *
                job_shape[0] *
                job_shape[2],
                128),
            nn.ReLU(
                inplace=True),
        )
        # Combine both outputs
        self.combined_net = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_shape),
        )

    def forward(self, obs, state=None, info={}):
        # Extract machines and jobs from the observation dictionary
        if not isinstance(obs["machines"], torch.Tensor):
            machines = torch.tensor(obs["machines"], dtype=torch.float)
        else:
            machines = obs["machines"]

        if not isinstance(obs["jobs"], torch.Tensor):
            jobs = torch.tensor(obs["jobs"], dtype=torch.float)
        else:
            jobs = obs["jobs"]

        # Ensure proper shape for the convolutional layers
        # (batch, channels, height, width)
        machines = machines.permute(0, 2, 1, 3)
        jobs = jobs.permute(0, 2, 1, 3)  # (batch, channels, height, width)

        # Process machines and jobs separately
        machine_features = self.machine_net(machines)
        job_features = self.job_net(jobs)

        # Combine features and pass to final layer
        combined_features = torch.cat([machine_features, job_features], dim=-1)
        logits = self.combined_net(combined_features)

        return logits, state


def main() -> None:
    env_id = "Cluster-discrete-v0"
    train_envs = ts.env.DummyVectorEnv(
        [lambda: gym.make(env_id) for _ in range(10)])
    test_envs = ts.env.DummyVectorEnv(
        [lambda: gym.make(env_id) for _ in range(100)])
    state_shape = train_envs.observation_space[0]
    action_shape = train_envs.action_space[0]
    net = Net(
        state_shape["machines"].shape,
        state_shape["jobs"].shape,
        action_shape.n)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    policy = ts.policy.DQNPolicy(
        model=net,
        optim=optim,
        action_space=action_shape,
        discount_factor=0.9,
        estimation_step=3,
        target_update_freq=320,
    )
    train_collector = ts.data.Collector(
        policy,
        train_envs,
        ts.data.VectorReplayBuffer(2_000, 200),
        exploration_noise=True,
    )
    test_collector = ts.data.Collector(
        policy, test_envs, exploration_noise=True)
    result = ts.trainer.OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=10,
        step_per_epoch=10_000,
        step_per_collect=10,
        update_per_step=0.1,
        episode_per_test=100,
        batch_size=64,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
    ).run()
    print(f'Finished training! Use {result["duration"]}')


if __name__ == "__main__":
    main()
