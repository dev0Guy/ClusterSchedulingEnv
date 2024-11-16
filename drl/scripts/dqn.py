from typing import Any
from tianshou.highlevel.experiment import DQNExperimentBuilder, ExperimentConfig
from tianshou.highlevel.env import (
    EnvFactoryRegistered,
    VectorEnvType,
)
from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.params.policy_params import DQNParams
from tianshou.highlevel.trainer import (
    EpochStopCallbackRewardThreshold,
    EpochTestCallbackDQNSetEps,
    EpochTrainCallbackDQNSetEps,
)
import gymnasium as gym
import logging
import drl.envs
from tianshou.utils.net.common import NetBase, TRecurrentState
import torch, numpy as np
from drl.utils.netowrks import IntermediateModuleFactoryClusterDQN

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(filename)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main(
    
):
    env_id = "Cluster-discrete-v0"
    env = gym.make(env_id)
    print(env.observation_space)
    print(env.action_space)
    experiment = (
        DQNExperimentBuilder(
            env_factory=EnvFactoryRegistered(
                task=env_id,
                venv_type=VectorEnvType.DUMMY,
                train_seed=0,
                test_seed=10,
            ),
            experiment_config=ExperimentConfig(
                persistence_enabled=True,
                watch=False,
                # watch_render=1 / 35,
                # watch_num_episodes=100,
            ),
            sampling_config=SamplingConfig(
                num_epochs=10,
                step_per_epoch=10000,
                batch_size=64,
                num_train_envs=10,
                num_test_envs=100,
                buffer_size=20000,
                step_per_collect=10,
                update_per_step=1 / 10,
            ),
        )
        .with_model_factory(IntermediateModuleFactoryClusterDQN())
        .with_dqn_params(
            DQNParams(
                lr=1e-3,
                discount_factor=0.9,
                estimation_step=3,
                target_update_freq=320,
            ),
        )
        .with_epoch_train_callback(EpochTrainCallbackDQNSetEps(0.3))
        .with_epoch_test_callback(EpochTestCallbackDQNSetEps(0.0))
        .with_epoch_stop_callback(EpochStopCallbackRewardThreshold(195))
        .build()
    )
    experiment.run()



if __name__ == "__main__":
    main()