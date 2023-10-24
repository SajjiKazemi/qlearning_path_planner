import gym
import numpy as np

from src.q_learning.RL import RL

import sys
sys.path.append("./cube_gym/src/")
from cube_gym.envs import cube_gym

model_name = "sarsa"
env = gym.make("cube_gym/CubeGym-v0", size=9)
observation, info = env.reset()
model = RL(env, env.nStates, env.nActions, discount=0.9)

nEpisodes = 30

if model_name == "sarsa":
    [Q, policy] = model.sarsa(current_state=info["current_state"], initialQ=np.zeros([env.nActions, env.nStates]),
                    nEpisodes=nEpisodes, nSteps=500, epsilon=0.5)

elif model_name == "qlearning":
    [Q, policy] = model.qLearning(current_state=info["current_state"], initialQ=np.zeros([env.nActions, env.nStates]),
                    nEpisodes=nEpisodes, nSteps=500, epsilon=0.5)

reward, path = model.run_model(Q, info["current_state"])
print(reward)
legal_moves = {0: 'right', 1: 'up', 2: 'left', 3: 'down', 4: 'forward'}
with open(model_name + f"-{0.5}" + f'-{nEpisodes}' + '.txt', 'w') as f:
    for item in path:
        f.write("%s" % str(item) + '    ' + '---->' + '     ' + legal_moves[item[1]] + '\n')
print("\nQ-learning results")