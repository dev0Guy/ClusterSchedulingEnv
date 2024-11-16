import typing as tp
from enum import IntEnum, auto

from pydantic import NonNegativeInt

MacineIndex: tp.TypeAlias = NonNegativeInt
JobIndex: tp.TypeAlias = NonNegativeInt
TickAction: tp.TypeAlias = tp.Literal[0,1]
IsSucessed: tp.TypeAlias = bool
ObsType = tp.NewType('ObsType', dict[str, tp.Any])
ClusterBaseOvservation: tp.TypeAlias = dict

class JobStatus(IntEnum):
    Pending = auto()
    Running = auto()
    Completed = auto()
    NotCreated = auto()

