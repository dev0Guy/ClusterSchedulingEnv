import typing as tp

# import clusterenv.common.types as ctp
import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, WrapperObsType

DilatedMachineIndex: tp.TypeAlias = ctp.MachineIndex


class DilationWrapper(gym.Wrapper):
    _view: tp.Annotated[
        list[np.ndarray],
        "list mean machine usage layer, where each layer is a new dilation step",
    ]
    _layer_idx: tp.Annotated[
        int, "Current index of dilation layer, each layer has a diffrenct shape"
    ]
    _start_view_idx: tp.Annotated[
        int, "Current view start index in the previous dilation layer"
    ]

    @staticmethod
    def _dilation(machines: np.ndarray, kernel: int) -> np.ndarray:
        """
        Perform max pooling on the given ndarray of shape [n_machines, n_resource, n_time]
        using a kernel size of [k, n_resource, n_time], with no overlap between machines.

        If the number of machines is not a multiple of k, it pads with zeros.

        Parameters:
        - machines: ndarray of shape (n_machines, n_resource, n_time)
        - k: int, the kernel size (number of machines to pool over)

        Returns:
        - pooled: ndarray, the result of the max pooling operation
        """
        n_machines, n_resource, n_time = machines.shape

        if (padding_size := (kernel - (n_machines % kernel)) % kernel) > 0:
            machines = np.pad(
                machines, ((0, padding_size), (0, 0), (0, 0)), mode="constant"
            )

        reshaped = machines.reshape(
            machines.shape[0] // kernel, kernel, n_resource, n_time
        )
        return np.max(reshaped, axis=1)

    @classmethod
    def _create_view(cls, machines: np.ndarray, kernel: int) -> list[np.ndarray]:
        """Create a list of views of the `machines` array by applying the dilation process iteratively.

        This method generates a list of NumPy arrays, starting with the original `machines` array.
        In each iteration, reduce the previous state by applying running window sum along
        the first axis. The process continues until the number of
        machines in the array is less than or equal to the specified `kernel` size.
        """
        return [
            machines,
            *[
                machines := cls._dilation(machines, kernel)
                for _ in range(machines.shape[0])
                if machines.shape[0] > kernel
            ],
        ]

    @property
    def view(self) -> list[np.ndarray]:
        return self._view

    @view.setter
    def view(self, machines: np.ndarray) -> None:
        self._view = self._create_view(machines, self._kernel)
        self._layer_idx = len(self._view) - 1
        self._start_view_idx = 0

    def __init__(self, env: gym.Env, kernel: int):
        super(DilationWrapper, self).__init__(env)
        self._kernel = kernel
        # inilize with inner value
        self.observation_space = env.observation_space
        # custome space for usage
        self.action_space = gym.spaces.MultiDiscrete(
            [2, self._kernel, env.action_space.nvec[-1]], dtype=np.int32  # type: ignore[attr-defined]
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, tp.Any] | None = None
    ) -> tuple[ObsType, dict[str, tp.Any]]:
        """
        Resets the environment and updates the observation with a specific machine layer.

        This method resets the environment to its initial state by calling the `reset` method
        of the parent class. It then modifies the observation by selecting a specific layer
        from the "machines" space in the observation, based on the current layer index
        (`self._layer_idx`). The previous observation and information are stored for
        future reference.
        """
        obs, info = super().reset(seed=seed, options=options)
        self.view = obs["machines"]["space"]
        obs["machines"]["space"] = self.view[self._layer_idx]
        self._prev_obs = obs
        self._prev_info = info
        return obs, info

    def step(
        self, action: tuple[ctp.TimeAction, DilatedMachineIndex, ctp.JobIndex]
    ) -> tuple[WrapperObsType, tp.SupportsFloat, bool, bool, dict[str, tp.Any]]:
        """
        Perform a step in the environment, updating the observation based on the action taken.
        This method processes a single step in the environment by either executing the action
        directly or updating the dilation view index when navigating through the machine layers.
        If the `skip_time` or the current layer index (`self._layer_idx`) is 0 (real action),
        the action is passed to the parent `step` method. Otherwise, the observation is updated
        based on the dilation index and layer.
        """
        # TODO: add action of going back in dilation
        skip_time, dilation_idx, job_idx = action
        assert (
            0 <= self._layer_idx < len(self.view)
        ), f"'_dilation_index' attribute should be in range [0, {len(self.view)}), not {self._layer_idx}"
        is_real_action: bool = self._layer_idx == 0
        if skip_time or is_real_action:
            obs, *extra, info = super().step(action)
            self.view = obs["machines"]["space"]
            obs["machines"]["space"] = self.view[self._layer_idx]
            self._prev_obs = obs
            self._prev_info = info
            return obs, *extra, info
        self._start_view_idx = (self._start_view_idx + dilation_idx) * self._kernel
        _start: int = self._start_view_idx
        _end: int = _start + self._kernel
        self._layer_idx -= 1
        self._prev_obs["machines"]["space"] = self.view[self._layer_idx][_start:_end]
        return self._prev_obs, 0, False, False, self._prev_info
