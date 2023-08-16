import numpy as np
import gym
import time
import matplotlib.pyplot as plt

from src.helpers import *


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
            number_of_steps = []
            test_rewards = []
            for n in range(nEpisodes):
                total_reward = 0
                _ , info = self.env.reset(seed=np.random.randint(0, 100000))
                current_state = info['current_state']
                for t in range(nSteps):
                    # if n <= 6*nEpisodes/10:
                    #     epsilon = epsilon
                    # elif n <= 9*nEpisodes/10:
                    #     epsilon = epsilon/2
                    # elif n <= nEpisodes:
                    #     epsilon = epsilon/4
                    action = self.get_action(Q, current_state, epsilon)
                    observation, reward, terminated, reached_goal, info = self.env.step(action)

                    next_state = info['next_state']
                    visit_numbers[action, current_state-1] = visit_numbers[action, current_state-1] + 1
                    learning_rate = 1 / visit_numbers[action, current_state-1]
                    Q[action, current_state-1] = Q[action, current_state-1] + learning_rate *\
                        (reward + self.discount * np.max(Q[:, next_state-1]) - Q[action, current_state-1])
                    current_state = next_state
                    total_reward = total_reward + (self.discount**(t))*reward
                    
                    if terminated:
                      observation, info = self.env.reset(seed=np.random.randint(0, 100000))
                      #t = nSteps
                      current_state = info['current_state']
                      break
                test_reward, test_steps = self.check_convegence(Q)
                cum_reward = np.append(cum_reward,total_reward)
                number_of_steps.append(test_steps)
                test_rewards.append(test_reward)

            reward_list = reward_list + cum_reward
        avg_reward = reward_list / 1

        get_moving_avg(test_rewards, 100, 'test_rewards')
        write_data(number_of_steps, 'number_of_steps.txt')
        get_moving_avg(number_of_steps, 100, 'number_of_steps')

        policy = np.argmax(Q, axis=0)
        #average_reward = reward_list / 100
        #with open('qlearning' + f"{epsilon}" + '.txt', 'w') as f:
        #    for item in average_reward:
        #        f.write("%s\n" % item)
        return [Q,policy]
    
    def get_action(self, Q, current_state, epsilon):
        if np.random.rand(1) < epsilon:
            action = np.random.randint(0, self.nActions)
        else:
            action = np.argmax(Q[:, current_state-1])
        return action

    def run_model(self, Q, initial_state):
        path = []
        observation, info = self.env.reset(seed=np.random.randint(0, 100000))
        current_state = info['current_state']
        total_reward = 0
        done = False
        while not done:
            action = np.argmax(Q[:, current_state-1])
            observation, reward, terminated, done, info = self.env.step(action)
            self.env.render(mode='3d')
            time.sleep(1)
            current_state = info['next_state']
            total_reward = total_reward + reward
            path.append((info['current_state'], action, info['current_location']))
        path.append((info['next_state'], action, info['next_location']))
        return total_reward, path

    def check_convegence(self, Q):
        observation, info = self.env.reset(seed=np.random.randint(0, 100000))
        current_state = info['current_state']
        total_reward = 0
        done = False
        steps = 0
        while not done and steps < 500:
            steps += 1
            action = np.argmax(Q[:, current_state-1])
            observation, reward, terminated, done, info = self.env.step(action)
            current_state = info['next_state']
            total_reward = total_reward + (self.discount**(steps))*reward
            #total_reward = total_reward + reward
        return total_reward, steps

    def check_convegence2(self, visit_numbers):
        visit_numbers = visit_numbers[:,:-1]
        if np.min(visit_numbers) > 100:
            return True
        else:
            return False

    def sarsa(self, current_state, initialQ, nEpisodes, nSteps, epsilon=0):

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
            number_of_steps = []
            test_rewards = []
            for n in range(nEpisodes):
            #     if n <= 6*nEpisodes/10:
            #         epsilon = epsilon
            #     elif n <= 9*nEpisodes/10:
            #         epsilon = epsilon/2
            #     elif n <= nEpisodes:
            #         epsilon = epsilon/4
                total_reward = 0
                _ , info = self.env.reset(seed=np.random.randint(0, 100000))
                current_state = info['current_state']
            #trial = 0
            #while not self.check_convegence2(visit_numbers):
            #    trial = trial + 1
                for t in range(nSteps):
                    if np.random.rand(1) < epsilon:
                        action = np.random.randint(0, self.nActions)
                    else:
                        action = np.argmax(Q[:, current_state-1])

                    observation, reward, terminated, reached_goal, info = self.env.step(action)

                    next_state = info['next_state']
                    visit_numbers[action, current_state-1] = visit_numbers[action, current_state-1] + 1
                    learning_rate = 1 / visit_numbers[action, current_state-1]
                    if np.random.rand(1) < epsilon:
                        next_action = np.random.randint(0, self.nActions)
                    else:
                        next_action = np.argmax(Q[:, next_state-1])
                    Q[action, current_state-1] = Q[action, current_state-1] + learning_rate *\
                        (reward + self.discount * Q[next_action, next_state-1] - Q[action, current_state-1])

                    current_state = next_state
                    total_reward = total_reward + (self.discount**(t))*reward

                    if terminated:
                      observation, info = self.env.reset(seed=np.random.randint(0, 100000))
                      #t = nSteps
                      current_state = info['current_state']
                      break
                test_reward, test_steps = self.check_convegence(Q)
                cum_reward = np.append(cum_reward,total_reward)
                number_of_steps.append(test_steps)
                test_rewards.append(test_reward)
            reward_list = reward_list + cum_reward
        avg_reward = reward_list / 1

        get_moving_avg(test_rewards, 100, 'cum_test_rewards_sarsa')
        write_data(number_of_steps, 'number_of_steps_sarsa.txt')
        get_moving_avg(number_of_steps, 100, 'number_of_test_steps_sarsa')

        policy = np.argmax(Q, axis=0)
        #average_reward = reward_list / 100
        #with open('qlearning' + f"{epsilon}" + '.txt', 'w') as f:
        #    for item in average_reward:
        #        f.write("%s\n" % item)
        return [Q,policy]