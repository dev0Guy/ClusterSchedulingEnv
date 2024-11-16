import typing as tp
from enum import IntEnum, auto

from pydantic import NonNegativeInt

MachineIndex: tp.TypeAlias = NonNegativeInt
JobIndex: tp.TypeAlias = NonNegativeInt
TickAction: tp.TypeAlias = tp.Literal[0, 1]
IsSucceed: tp.TypeAlias = bool
ObsType = tp.NewType('ObsType', dict[str, tp.Any])
ClusterBaseObservation: tp.TypeAlias = dict


class JobStatus(IntEnum):
    Pending = auto()
    Running = auto()
    Completed = auto()
    NotCreated = auto()

