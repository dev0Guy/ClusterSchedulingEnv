import logging
import typing as tp
from dataclasses import dataclass, field

import numpy as np
from pydantic import NonNegativeInt, PositiveInt
from typing_extensions import Doc

from clusterenv.envs.common.typing import JobStatus, JobIndex

CLUSTER_CELL_HIGH: float = 255.0

logger = logging.getLogger(__name__)


def get_first_none_zero_value(arr: np.ndarray):
    assert len(arr.shape) == 3
    non_zero_mask = np.any(arr != 0, axis=1)
    first_non_zero_times = np.argmax(non_zero_mask, axis=1)
    no_non_zero = ~np.any(non_zero_mask, axis=1)
    # TODO: raise Exception
    first_non_zero_times[no_non_zero] = -1
    return first_non_zero_times


class ClusterObservation(tp.TypedDict):
    machines: np.ndarray
    jobs: np.ndarray


@dataclass
class Machines:
    n_machines: tp.Annotated[PositiveInt, Doc("Number of machines in the cluster")]
    n_resources: tp.Annotated[
        PositiveInt,
        Doc("Number of resource in each machine (cpu, ram, disk, netowrk, etc...)"),
    ]
    n_ticks: tp.Annotated[
        PositiveInt,
        Doc(
            "Maximum number of clock tick allowed in the system. In other word the maximum job size"
        ),
    ]
    free_space: tp.Annotated[
        np.ndarray, Doc("Unutlize computing power, size of [machines, resource, ticks]")
    ] = field(init=False)

    def __post_init__(self):
        self.free_space = np.full(
            shape=(self.n_machines, self.n_resources, self.n_ticks),
            fill_value=CLUSTER_CELL_HIGH,
            dtype=np.uint8,
        )


@dataclass
class Jobs:
    n_jobs: tp.Annotated[PositiveInt, Doc("Total number of jobs in the cluster")]
    n_resources: tp.Annotated[
        PositiveInt,
        Doc("Number of resource in each machine (cpu, ram, disk, netowrk, etc...)"),
    ]
    n_ticks: tp.Annotated[
        PositiveInt,
        Doc(
            "Maximum number of clock tick allowed in the system. In other word the maximum job size"
        ),
    ]
    utlization: tp.Annotated[
        np.ndarray, Doc("Space utlization of job, size of [jobs, resource, ticks]")
    ] = field(init=False)
    arrival_time: tp.Annotated[
        np.ndarray, Doc("One dim arrray of job arrival time")
    ] = field(init=False)
    length: tp.Annotated[np.ndarray, Doc("Length of job")] = field(init=False)
    status: tp.Annotated[np.ndarray, Doc("Status of jobs")] = field(init=False)
    runnning_time: tp.Annotated[
        np.ndarray, Doc("Running time for each job (1d array)")
    ] = field(init=False)
    _job_arrival_rate: float = 0.5

    def __post_init__(self):
        self.utlization = self.generate_jobs_utilization()
        self.arrival_time = get_first_none_zero_value(self.utlization)
        jobs_end_time = self.n_ticks - get_first_none_zero_value(
            np.flip(self.utlization, axis=-1)
        )
        self.length = jobs_end_time - self.arrival_time
        self.status = np.full(
            shape=(self.n_jobs), fill_value=JobStatus.NotCreated, dtype=np.uint8
        )
        self.runnning_time = np.zeros(shape=(self.n_jobs))
        self.on_tick(0)
        logger.debug(f"Jobs Arrival: {self.arrival_time}, Length: {self.length}")

    def generate_jobs_utilization(self) -> np.ndarray:
        # Initialize a 3D array to hold job resource demands over time, as float for initial calculations
        jobs_array = np.zeros(
            (self.n_jobs, self.n_resources, self.n_ticks), dtype=np.float32
        )

        # Job arrival times based on Bernoulli process
        enter_times = np.cumsum(
            np.random.binomial(1, self._job_arrival_rate, size=self.n_jobs)
        )

        # Job durations: 80% short jobs (1 to 3), 20% long jobs (10 to 15)
        job_durations = np.where(
            np.random.rand(self.n_jobs) < 0.8,
            np.random.randint(1, 4, size=self.n_jobs),  # Short jobs (1 to 3)
            np.random.randint(10, 16, size=self.n_jobs),  # Long jobs (10 to 15)
        )

        # Randomly select the dominant resource for each job (either 0 or 1)
        dominant_resource = np.random.randint(0, self.n_resources, size=self.n_jobs)

        # Resource demands, scaled to fit within uint8
        dominant_demand = np.random.uniform(0.25, 0.5, size=self.n_jobs) * (
            255 / 0.5
        )  # Scale to [127.5, 255]
        secondary_demand = np.random.uniform(0.05, 0.1, size=self.n_jobs) * (
            255 / 0.5
        )  # Scale to [25.5, 51]
        res_demand = np.zeros((self.n_jobs, self.n_resources), dtype=np.float32)
        res_demand[np.arange(self.n_jobs), dominant_resource] = dominant_demand
        res_demand[np.arange(self.n_jobs), 1 - dominant_resource] = secondary_demand

        # Create an array for the time indices of each job based on `enter_times` and `job_durations`
        time_indices = np.arange(self.n_ticks)
        mask = (time_indices >= enter_times[:, None]) & (
            time_indices < (enter_times + job_durations)[:, None]
        )

        # Use broadcasting to place resource demands over the job durations
        jobs_array[:, :, :] = np.where(mask[:, None, :], res_demand[:, :, None], 0)

        # Convert the jobs_array to uint8
        jobs_array = jobs_array.astype(np.uint8)

        return jobs_array

    def allocate(self, j: JobIndex) -> None:
        self.status[j] = JobStatus.Running
        self.utlization[j] = 0

    def on_tick(self, tick: NonNegativeInt) -> None:
        self.status[self.runnning_time == self.length] = JobStatus.Completed
        self.runnning_time[self.status == JobStatus.Running] += 1
        self.status[self.arrival_time == tick] = JobStatus.Pending
