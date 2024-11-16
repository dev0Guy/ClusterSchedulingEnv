import numpy as np
from gymnasium import ActionWrapper
import gymnasium as gym
from clusterenv.envs import ClusterEnv


class DiscreteActionWrapper(ActionWrapper):
    """Wrap Cluster env and allow for discrite action by mapping to (n_jobs * n_machines + 1) added one for skip time action"""

    def __init__(self, env: ClusterEnv):
        super().__init__(env)
        queue_size = getattr(self,"queue_size", self.n_jobs)
        number_of_actions: int = (self.n_machines * queue_size) + 1 # add null oprtation (skip time)
        self.action_space = gym.spaces.Discrete(number_of_actions, start=0)

    def action(self, action: int) -> np.ndarray:
        skip_time: bool = action == 0
        if skip_time:
            return np.array([1, 0, 0])
        action -= 1
        m_idx, j_idx = action % self.n_machines, action // self.n_machines

        return np.array([0, m_idx, j_idx])
