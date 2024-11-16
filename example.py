import logging

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import clusterenv.envs
from clusterenv.envs.wrappers.reward import AverageSlowDownReward

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(filename)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

id = "Cluster-discrete-v1"
env = AverageSlowDownReward(gym.make(id))
# check_env(env)

obs, info = env.reset()
for _ in range(10_000):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action=action)
    logging.info(info)
    if done or truncated:
        print("FINISHED")
        break
