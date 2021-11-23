import numpy as np
from gym import Env
import pickle
from numpy.core.fromnumeric import size

from typing import Any

class RL_Algorithm():
    def __init__(self):
        pass

    def loadz(self, file: str) -> Any:
        with open(file, 'rb') as f:
            return pickle.load(f)

    def savez(self, obj: Any, file: str):
        with open(file, 'wb') as f:
            pickle.dump(obj, f)

        


class QLearning(RL_Algorithm):

    #TODO: Make size of Q_table dependable on observation_space.
    #TODO: Setup Q_table according to size not upscaling and rounding.
    def __init__(self, env: Env, discretize: list):
        self.env = env
        self.env.reset()
        self.discretize = discretize

        self.Q_table = {}

    def load(self):
        self.Q_table = self.loadz(f'./models/Q_Table_{self.env.spec.id}')

    def save(self):
        self.savez(self.Q_table, f'./models/Q_Table_{self.env.spec.id}')

    def discretizeState(self, state):
        state_adj = state*np.array(self.discretize)
        state_adj = np.round(state_adj, 0).astype(int)
        if not (tuple(state_adj) in self.Q_table):
            self.Q_table[tuple(state_adj)] = np.random.uniform(low=-1.0, high=1.0, size=self.env.action_space.n)
        return state_adj

    def do_action(self, state_adj, epsilon=0):
        if np.random.random() > epsilon:
            return np.argmax(self.Q_table[tuple(state_adj)])
        else:
            return self.env.action_space.sample()   

    def generate_trajectories_and_actions(self, n=10, render=False):
        states = []
        actions = []

        for i in range(n):
            total_reward = 0
            #Reset Environmnet
            state = self.env.reset()
            state = self.discretizeState(state)
            done = False
            while done == False:
                if render:
                    self.env.render()
                action = self.do_action(state)

        
                #perform action in environment
                state, reward, done, info = self.env.step(action)
        
                #Save state and action to memory
                states.append(state)
                actions.append(action)
        
                state = self.discretizeState(state)
                total_reward+=reward
            self.env.close()
            #save trajectory and actions to large memory
        return states, actions

    def train(self, epochs, lr, epsilon, discount):
        reward_list = []
        ave_reward_list = []

        reduction = epsilon / (epochs-epochs/4)
        print('Reduction: ', reduction)

        for i in range(epochs):
            done = False
            tot_reward, reward = 0, 0
            state = self.env.reset()
            state_adj = self.discretizeState(state)

            while done != True:
                # Randomly do a random action, this will increase exploration in the beginning.
                action = self.do_action(state_adj, epsilon=epsilon)

                state2, reward, done, info = self.env.step(action)
                state2_adj = self.discretizeState(state2)

                #Allow for terminal states (if done in time -> maximize reward)
                if done and state2[0] >= 0.5:
                    self.Q_table[tuple(list(state_adj))][action] = reward
                else:
                    delta = lr*(reward + discount*np.max(self.Q_table[tuple(state2_adj)]) - self.Q_table[tuple(state_adj)][action])
                    self.Q_table[tuple(state_adj)][action] += delta

                #Update total_reward & update state for new action.
                tot_reward += reward
                state_adj = state2_adj

            # Decay epsilon
            if epsilon > 0:
                epsilon -= reduction
            
            # Keep track of rewards (No influence on workings of the algorithm)
            reward_list.append(tot_reward)
            if (i+1) % 100 == 0:
                ave_reward = np.mean(reward_list)
                ave_reward_list.append(ave_reward)
                reward_list = []
                print('Episode {} Q_table Size: {} Average Reward: {}'.format(i+1, len(self.Q_table), ave_reward))
                
        self.env.close()
        print("Finished training!")