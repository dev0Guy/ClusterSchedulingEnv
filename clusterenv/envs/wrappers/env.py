import numpy as np
from gymnasium import Wrapper
from pydantic import PositiveInt

from clusterenv.envs import ClusterEnv
import typing as tp
import gymnasium as gym

from clusterenv.envs.common import ClusterObservation, CLUSTER_CELL_HIGH
from clusterenv.envs.common.typing import (
    ClusterBaseObservation,
    MachineIndex,
    JobIndex,
    IsSucceed,
)


class ScheduleFromSelectedTimeWrapper(Wrapper):
    def __init__(self, env: ClusterEnv):
        super().__init__(env)
        env.schedule = self.schedule

    def step(
        self, action: np.ndarray
    ) -> tuple[ClusterBaseObservation, tp.SupportsFloat, bool, bool, dict[str, tp.Any]]:
        return super().step(action)

    def schedule(
        self,
        m: MachineIndex,
        j: JobIndex,
    ) -> IsSucceed:
        if not self.validate_job_status(j):
            return False
        job_arrival_time = max(self.jobs.arrival_time[j], self.current_clock_tick)
        job_length = self.jobs.length[j]
        job_utlization_block = self.jobs.utlization[
            j, :, job_arrival_time : job_arrival_time + job_length
        ]
        # Extract machine availability for the duration from current tick to max time allowed
        machine_free_space = self.machines.free_space[
            m, :, self.current_clock_tick : (self.current_clock_tick + job_length)
        ]
        # Slide over the free space array to check blocks of size job_length for resource sufficiency
        free_space_slices = np.lib.stride_tricks.sliding_window_view(
            machine_free_space, window_shape=job_utlization_block.shape[1:], axis=1
        )
        # Check if there's any time block where all resources meet the job's requirements
        sufficient_space = np.all(
            free_space_slices >= job_utlization_block, axis=(1, 2)
        )
        if np.any(sufficient_space):
            # Get the first time block that fits, adjust to absolute time by adding current tick
            start_time = self.current_clock_tick + np.argmax(sufficient_space)
            self.machines.free_space[
                m, :, start_time : start_time + job_length
            ] -= job_utlization_block
            return True
        return False


class QueueWrapper(Wrapper):
    """Wrapp jobs and select [m] according to the best status (pending)"""

    def __init__(self, env: ClusterEnv, *, queue_size: PositiveInt):
        if queue_size <= 0:
            raise ValueError(
                f"{type(self).__name__} queue size should be postive bigger than zero. not {queue_size}"
            )
        super().__init__(env)
        self.queue_size = queue_size
        self.action_space = gym.spaces.MultiDiscrete(
            np.array([self.tick_action_option, self.n_machines, self.queue_size])
        )
        self.observation_space["jobs"] = gym.spaces.Box(  # type: ignore
            0,
            CLUSTER_CELL_HIGH,
            (self.queue_size, self.n_resources, self.n_ticks),
            dtype=self.observation_space["jobs"].dtype,  # type: ignore
        )
        self.observation_space["status"] = gym.spaces.Box(  # type: ignore
            0,
            CLUSTER_CELL_HIGH,
            (self.queue_size,),
            dtype=self.observation_space["status"].dtype,  # type: ignore
        )
        self.job_indexer = np.arange(env.n_jobs)

    def modify_action(self, action: np.ndarray) -> np.ndarray:
        skip, m_idx, j_idx = action
        j_idx = self.job_indexer[j_idx]
        return np.array([skip, m_idx, j_idx])

    def modify_observation(
        self, observation: ClusterBaseObservation
    ) -> ClusterBaseObservation:
        """Remove jobs that are not inside the"""
        observation = observation.copy()
        status: np.ndarray = observation["status"]  # type: ignore
        self.job_indexer: np.ndarray = np.argsort(status)
        observation["jobs"] = observation["jobs"][self.job_indexer][: self.queue_size]  # type: ignore
        observation["status"] = observation["status"][self.job_indexer][
            : self.queue_size
        ]
        return observation

    def step(
        self, action: np.ndarray
    ) -> tuple[ClusterObservation, tp.SupportsFloat, bool, bool, dict[str, tp.Any]]:
        action = self.modify_action(action)  # type: ignore[no-redef]
        observation, *extra = super().step(action)
        observation = self.modify_observation(observation)
        return observation, *extra  # type: ignore

    def reset(
        self, *, seed: int | None = None, options: dict[str, tp.Any] | None = None
    ) -> tuple[ClusterObservation, dict[str, tp.Any]]:
        observation, *extra = super().reset(seed=seed, options=options)
        observation = self.modify_observation(observation)
        return observation, *extra  # type: ignore
