from stable_baselines3 import DQN, PPO
import gymnasium as gym
# import gym

from src.RL import RL

import sys
sys.path.append("./cube_gym")
from cube_gym.envs import CubeGym

env = gym.make("cube_gym/CubeGym-v0", size=9)
env.reset()
model = DQN('MultiInputPolicy', env=env, verbose=1, tensorboard_log="./logs/")
model.learn(total_timesteps=100000, progress_bar=True)
model.save("dqn_cube_gym_test")