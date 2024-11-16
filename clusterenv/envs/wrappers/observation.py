from gymnasium import ObservationWrapper
from clusterenv.envs.base import ClusterEnv
import numpy as np
import gymnasium as gym


class CombineMachinJobWrapper(ObservationWrapper):

    def __init__(self, env: ClusterEnv):
        super(CombineMachinJobWrapper, self).__init__(env)
        jobs_space = env.observation_space.spaces['jobs']
        machines_space = env.observation_space.spaces['machines']
        jobs_shape = jobs_space.shape  # (n_jobs, resource, ticks)
        machines_shape = machines_space.shape  # (n_machines, resource, ticks)
        combined_shape = (jobs_shape[0] + machines_shape[0],) + jobs_shape[1:]
        combined_low = np.concatenate([jobs_space.low, machines_space.low], axis=0)
        combined_high = np.concatenate([jobs_space.high, machines_space.high], axis=0)

        # Define the new observation space as a single Box space
        self.observation_space = gym.spaces.Box(
            low=combined_low,
            high=combined_high,
            shape=combined_shape,
            dtype=jobs_space.dtype
        )

    def observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        combined_observation = np.concatenate([observation['jobs'], observation['machines']], axis=0)
        return combined_observation
