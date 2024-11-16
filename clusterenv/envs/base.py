import logging
import typing as tp
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame
from pydantic import Field, NonNegativeInt, PositiveInt
from typing_extensions import Doc

from clusterenv.envs.common import CLUSTER_CELL_HIGH, Jobs, Machines
from clusterenv.envs.common.typing import IsSucceed, JobIndex, JobStatus, MachineIndex

logger = logging.getLogger(__name__)


@dataclass
class ClusterEnv(gym.Env[dict, np.ndarray]):
    n_machines: tp.Annotated[PositiveInt, Doc("Number of machines in the cluster")]
    n_jobs: tp.Annotated[PositiveInt, Doc("Total number of jobs in the cluster")]
    n_resources: tp.Annotated[PositiveInt, Doc("Number of resource in each machine (cpu, ram, disk, netowrk, etc...)")]
    n_ticks: tp.Annotated[
        PositiveInt, Doc("Maximum number of clock tick allowed in the system. In other word the maximum job size")]
    render_mode: tp.Optional[str] = None#Field(kw_only=True, default=None)
    tick_action_option: tp.Annotated[PositiveInt, Doc("Either 0 for no time skip and 1 otherwise")] = field(init=False,
                                                                                                            default=2)
    current_clock_tick: tp.Annotated[NonNegativeInt, Doc("Total ticks from start of the cluster")] = field(init=False,
                                                                                                           default=0)

    @property
    def observation(self) -> dict:
        obs = dict(
            machines=self.machines.free_space,
            jobs=self.jobs.utlization,
            status=self.jobs.status
        )
        return obs

    @property
    def extra_info(self) -> dict:
        return dict(
            n_not_created=(self.jobs.status == JobStatus.NotCreated).sum(),
            n_pending=(self.jobs.status == JobStatus.Pending).sum(),
            n_running=(self.jobs.status == JobStatus.Running).sum(),
            n_completed=(self.jobs.status == JobStatus.Completed).sum(),
            n_ticks=self.current_clock_tick
        )

    @property
    def is_done(self) -> bool:
        is_jobs_allocated: np.array = (self.jobs.status == JobStatus.Running) | (
                    self.jobs.status == JobStatus.Completed)
        return bool(is_jobs_allocated.all())

    @property
    def is_truncated(self) -> bool:
        return self.current_clock_tick >= self.n_ticks

    def __post_init__(self):
        super(ClusterEnv, self).__init__()
        logger.info(f"Cluster env created with {self.n_machines=} {self.n_jobs=} {self.n_resources=} {self.n_ticks=}")
        self.action_space = gym.spaces.MultiDiscrete(np.array([self.tick_action_option, self.n_machines, self.n_jobs]))
        self.observation_space = self.generate_cluster_observation_shape(CLUSTER_CELL_HIGH)

    def reset(self, *, seed: int | None = None, options: dict[str, tp.Any] | None = None) -> tuple[
        dict, dict[str, tp.Any]]:
        super().reset(seed=seed,options=options)
        np.random.seed(seed)
        self.machines = Machines(self.n_machines, self.n_resources, self.n_ticks)
        self.jobs = Jobs(self.n_jobs, self.n_resources, self.n_ticks)
        self.current_clock_tick = 0
        return self.observation, self.extra_info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        super().render()
        pass

    def generate_cluster_observation_shape(self, high: float) -> gym.spaces.Dict:
        jobs_shape = gym.spaces.Box(low=0, high=high, shape=(self.n_jobs, self.n_resources, self.n_ticks),
                                    dtype=np.uint8)
        job_status_shape = gym.spaces.Box(low=0, high=high, shape=(self.n_jobs,), dtype=np.uint8)
        machines_shape = gym.spaces.Box(low=0, high=high, shape=(self.n_machines, self.n_resources, self.n_ticks),
                                        dtype=np.uint8)
        return gym.spaces.Dict(
            dict(machines=machines_shape, jobs=jobs_shape, status=job_status_shape)
        )

    def validate_job_status(self, j: JobIndex) -> IsSucceed:
        job_status = self.jobs.status[j]
        arrival_time = self.jobs.arrival_time[j]
        if job_status != JobStatus.Pending:
            logger.debug(f"Failed Scheduling job index {j} with status {job_status} should be {JobStatus.Pending}")
            return False
        if self.current_clock_tick < arrival_time:
            logger.error(
                f"Failed Scheduling job index {j} with arrival time {arrival_time} while tick number is smaller with {self.current_clock_tick}")
            return False
        return True

    def schedule(
            self,
            m: MachineIndex,
            j: JobIndex,
    ) -> IsSucceed:
        if not self.validate_job_status(j):
            return False

        job_length = self.jobs.length[j]
        current_tick = self.current_clock_tick
        job_start_tick = self.jobs.arrival_time[j]
        job_end_tick = job_start_tick + job_length
        machine_end_tick = current_tick + job_length

        if machine_end_tick <= self.n_ticks:
            free_space = self.machines.free_space[m, :, current_tick:machine_end_tick]
            job_utlization = self.jobs.utlization[j, :, job_start_tick:job_end_tick]

            if np.all((free_space - job_utlization) >= 0):
                self.machines.free_space[m, :, current_tick:current_tick + job_length] -= job_utlization
                return True
        return False

    def step(self, action: np.ndarray) -> tuple[dict, tp.SupportsFloat, bool, bool, dict[str, tp.Any]]:
        """tp.Tuple[TickAction, MacineIndex, JobIndex]"""
        should_tick, machine_idx, job_idx = action
        # Handle clock ticking
        if should_tick:
            logger.debug(f"Foward time, {self.current_clock_tick} -> {self.current_clock_tick + 1}")
            self.current_clock_tick += 1
            self.jobs.on_tick(self.current_clock_tick)
            return self.observation, -1.0, self.is_done, self.is_truncated, self.extra_info
        # Attempt job allocation
        if not self.schedule(machine_idx, job_idx):
            logger.debug(f"Cannot allocate job {job_idx} to machine {machine_idx} not enough free space")
            return self.observation, 0.0, self.is_done, self.is_truncated, self.extra_info
        # Successful allocation
        self.jobs.allocate(job_idx)
        logger.debug(f"Schedule machine.{machine_idx}, job.{job_idx} on {self.current_clock_tick} tick")
        return self.observation, -1.0, self.is_done, self.is_truncated, self.extra_info
