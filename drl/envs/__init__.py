from gymnasium.envs.registration import register
from .base import ClusterEnv
from .wrappers import ScheduleFromSelectedTimeWrapper, DiscreteActionWrapper, QueueWrapper, CombineMachinJobWrapper


SIMULATION_LENGTH: int = 10
DEFUALT_PARAMS: dict = dict(
    n_machines=1,
    n_jobs=10,
    n_resources=2,
    n_ticks=SIMULATION_LENGTH
)


def create_cluster_v0(**kwargs):
    return ClusterEnv(**kwargs) # type: ignore


def create_cluster_v1(queue_size, **kwargs):
    env = create_cluster_v0(**kwargs)
    return QueueWrapper(env, queue_size=queue_size)# type: ignore


def create_cluster_discrite_v0(**kwargs):
    return DiscreteActionWrapper(create_cluster_v0(**kwargs)) # type: ignore


def create_cluster_discrite_v1(queue_size, **kwargs):
    return DiscreteActionWrapper(create_cluster_v1(queue_size,**kwargs)) # type: ignore

def create_cluster_with_schedule_v0(**kwargs):
    return ScheduleFromSelectedTimeWrapper(create_cluster_v0(**kwargs))


register(
    id="Cluster-v0",
    entry_point=create_cluster_v0,
    kwargs=DEFUALT_PARAMS
)
register(
    id="Cluster-scheduled-v0",
    entry_point=create_cluster_with_schedule_v0,
    kwargs=DEFUALT_PARAMS
)
register(
    id="Cluster-discrete-v0",
    entry_point=create_cluster_discrite_v0,
    kwargs=DEFUALT_PARAMS
)
register(
    id="Cluster-v1",
    entry_point=create_cluster_v1,
    kwargs={
        "queue_size": 1,
        **DEFUALT_PARAMS
    }
)
register(
    id="Cluster-discrete-v1",
    entry_point=create_cluster_discrite_v1,
    kwargs={
        "queue_size": 1,
        **DEFUALT_PARAMS
    }
)
