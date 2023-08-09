import numpy as np
import gym
import time

class RL:
    def __init__(self, env: gym, nStates: int, nActions: int, discount=0.9):
        '''Constructor for the RL class

        Inputs:
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''
        self.env = env
        self.discount = discount
        self.nStates = nStates
        self.nActions = nActions


    def qLearning(self, current_state, initialQ, nEpisodes, nSteps, epsilon=0):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
        are combined in one procedure by sampling a random action with 
        probabilty epsilon and performing Boltzmann exploration otherwise.  
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        initialQ = np.zeros([self.nActions, self.nStates])
        Q = initialQ
        current_state = current_state
        reward_list = np.zeros(nEpisodes)
        visit_numbers = np.zeros([self.nActions, self.nStates], dtype=float)
        _ , info = self.env.reset(seed=np.random.randint(0, 100000))
        for i in range(1):
            cum_reward = np.zeros(0)
            Q = initialQ
            visit_numbers = np.zeros([self.nActions, self.nStates], dtype=float)
            for n in range(nEpisodes):
                total_reward = 0
                for t in range(nSteps):
                    if np.random.rand(1) < epsilon:
                        action = np.random.randint(0, self.nActions)
                    else:
                        action = np.argmax(Q[:, current_state-1])

                    observation, reward, terminated, reached_goal, info = self.env.step(action)

                    next_state = info['next_state']
                    visit_numbers[action, current_state-1] = visit_numbers[action, current_state-1] + 1
                    learning_rate = 1 / visit_numbers[action, current_state-1]
                    Q[action, current_state-1] = Q[action, current_state-1] + learning_rate *\
                        (reward + self.discount * np.max(Q[:, next_state-1]) - Q[action, current_state-1])
                    current_state1 = current_state
                    current_state = next_state
                    total_reward = total_reward + (self.discount**(t))*reward
                    
                    if terminated:
                      observation, info = self.env.reset(seed=np.random.randint(0, 100000))
                      t = nSteps
                      current_state = info['current_state']
                      break
                cum_reward = np.append(cum_reward,total_reward)
            reward_list = reward_list + cum_reward

        policy = np.argmax(Q, axis=0)
        #average_reward = reward_list / 100
        #with open('qlearning' + f"{epsilon}" + '.txt', 'w') as f:
        #    for item in average_reward:
        #        f.write("%s\n" % item)
        return [Q,policy]
    
    def run_model(self, Q, initial_state):
        path = []
        observation, info = self.env.reset(seed=np.random.randint(0, 100000))
        current_state = info['current_state']
        total_reward = 0
        done = False
        while not done:
            action = np.argmax(Q[:, current_state-1])
            observation, reward, terminated, done, info = self.env.step(action)
            #self.env.render(mode='3d')
            time.sleep(1)
            current_state = info['next_state']
            total_reward = total_reward + reward
            path.append((info['current_state'], action, self.env.get_agent_location()))
        return total_reward, path

        