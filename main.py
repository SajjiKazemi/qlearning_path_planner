import sys
sys.path.append("./cube_gym")

import gym
from cube_gym.envs import CubeGym

env = gym.make("cube_gym/CubeGym-v0", render_mode="human", size=5)
observation, info = env.reset()

for _ in range(10000):
    action = env.action_space.sample()
    observation, reward, terminated, x, y = env.step(action)

    if terminated:
        observation, info = env.reset()

env.close()