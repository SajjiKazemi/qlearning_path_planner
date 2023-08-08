import numpy as np
import MDP

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
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
        Q = initialQ
        reward_list = np.zeros(nEpisodes)
        visit_numbers = np.zeros([self.mdp.nActions, self.mdp.nStates], dtype=float)
        for i in range(100):
            cum_reward = np.zeros(0)
            Q = initialQ
            visit_numbers = np.zeros([self.mdp.nActions, self.mdp.nStates], dtype=float)
            for n in range(nEpisodes):
                total_reward = 0
                for t in range(nSteps):
                    if np.random.rand(1) < epsilon:
                        action = np.random.randint(0, self.mdp.nActions)
                    else:
                        action = np.argmax(Q[:, s0])
                    reward, nextState = self.sampleRewardAndNextState(s0, action)
                    visit_numbers[action, s0] = visit_numbers[action, s0] + 1
                    learning_rate = 1 / visit_numbers[action, s0]
                    Q[action, s0] = Q[action, s0] + learning_rate * (reward + self.mdp.discount * np.max(Q[:, nextState]) - Q[action, s0])
                    s0 = nextState
                    total_reward = total_reward + (self.mdp.discount**(t))*reward
                cum_reward = np.append(cum_reward,total_reward)
            reward_list = reward_list + cum_reward

        policy = np.argmax(Q, axis=0)
        average_reward = reward_list / 100
        with open('qlearning' + f"{epsilon}" + '.txt', 'w') as f:
            for item in average_reward:
                f.write("%s\n" % item)
        return [Q,policy]    