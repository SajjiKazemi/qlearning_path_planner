import gym
import numpy as np

from src.RL import RL

import sys
sys.path.append("./cube_gym")
from cube_gym.envs import CubeGym


env = gym.make("cube_gym/CubeGym-v0", size=9)
observation, info = env.reset()
model = RL(env, env.nStates, env.nActions, discount=0.9)

[Q, policy] = model.qLearning(current_state=info["current_state"], initialQ=np.zeros([env.nActions, env.nStates]),
                nEpisodes=100000, nSteps=100000, epsilon=0.5)

reward, path = model.run_model(Q, info["current_state"])
print(reward)
legal_moves = {0: 'right', 1: 'up', 2: 'left', 3: 'down', 4: 'forward'}
with open('qlearning' + f"{0.5}" + '.txt', 'w') as f:
    for item in path:
        f.write("%s" % str(item) + '    ' + '---->' + '     ' + legal_moves[item[1]] + '\n')
print("\nQ-learning results")