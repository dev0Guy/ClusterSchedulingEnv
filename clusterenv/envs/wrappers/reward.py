from typing import SupportsFloat

import numpy as np
from gymnasium import RewardWrapper
from clusterenv.envs.base import ClusterEnv, logger
from clusterenv.envs.common import JobStatus


class AverageSlowDownReward(RewardWrapper):

    def __init__(self, env: ClusterEnv):
        super().__init__(env)
        self._slowdown = np.zeros(self.env.n_jobs)
        self._prev_tick = self.env.current_clock_tick

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        if self.env.current_clock_tick != self._prev_tick:
            self._prev_tick = self.env.current_clock_tick
            current_status = self.env.jobs.status
            self._slowdown += (current_status == JobStatus.Pending)
            slowdown = self._slowdown[self._slowdown > 0]
            reward = - (1 / slowdown).sum()
            logger.debug(f"Slowdown reward: {reward}")
            return reward
        return 0
