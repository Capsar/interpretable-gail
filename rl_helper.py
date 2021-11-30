import numpy as np
from gym import Env
import pickle
from numpy.core.fromnumeric import size
from typing import Any
from sklearn.tree import DecisionTreeClassifier

class RL_Algorithm():
    def __init__(self, env: Env):
        pass

    def loadz(self, file: str) -> Any:
        try:
            with open(file, 'rb') as f:
                return pickle.load(f)
        except:
            print('file not found:', file)

    def savez(self, obj: Any, file: str):
        with open(file, 'wb') as f:
            pickle.dump(obj, f)

    def generate_trajectories(self, do_action, env:Env, n=10, render=False):
        trajectories = []

        for i in range(n):
            #Reset Environmnet
            state = env.reset()
            done = False
            while done == False:
                if render:
                    env.render()
                action = do_action(state)

                #perform action in environment
                next_state, reward, done, info = env.step(action)
        
                #Save state and action to memory
                trajectories.append((state, action))

                state = next_state
                
            env.close()
            #save trajectory and actions to large memory
        return trajectories

    def get_average_reward(self, do_action, env:Env, number_of_tests, actor, render=False):
        total_reward = 0
        for ii in range(number_of_tests):
            state = env.reset()
            done = False
            while done == False:
                if render:
                    env.render()
                action = do_action(state)
                state, reward, done, info = env.step(action)
                total_reward+=reward
        if number_of_tests == 1:
            print(f'Total average {actor} reward: {total_reward / number_of_tests}')
        else:
            print(f'Total average {actor} reward over {number_of_tests} games is: {total_reward / number_of_tests}')
        return total_reward

class DecisionTree(RL_Algorithm):

    def __init__(self, env: Env, max_depth=5, max_features=2):
        self.env = env
        self.env.reset()
        self.max_depth = max_depth
        self.max_features = max_features
        self.decision_tree = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features)
        self.decision_tree.fit([env.reset()], [-1])

    def load(self):
        self.decision_tree = self.loadz(f'./models/DecisionTree_{self.env.spec.id}_{self.max_depth}_{self.max_features}')

    def save(self):
        self.savez(self.decision_tree, f'./models/DecisionTree_{self.env.spec.id}_{self.max_depth}_{self.max_features}')

    def do_action(self, state):
        return self.decision_tree.predict([state])[0]

    def fit(self, new_generator_s, new_generator_a):
        self.decision_tree.fit(new_generator_s, new_generator_a)

    def generate_trajectories(self, n=10, render=False):
        return super().generate_trajectories(self.do_action, self.env, n, render)

    def get_average_reward(self, number_of_tests, render=False):
        return super().get_average_reward(self.do_action, self.env, number_of_tests, "DecisionTree", render=False)

class QLearning(RL_Algorithm):

    def __init__(self, env: Env, discretize: list):
        self.env = env
        self.env.reset()
        self.discretize = discretize

        self.Q_table = {}

    def load(self):
        discretize = '_'.join(map(str, self.discretize))
        self.Q_table = self.loadz(f'./models/Q_Table_{self.env.spec.id}_{discretize}')

    def save(self):
        discretize = '_'.join(map(str, self.discretize))
        self.savez(self.Q_table, f'./models/Q_Table_{self.env.spec.id}_{discretize}')

    def discretizeState(self, state):
        state_adj = state*np.array(self.discretize)
        state_adj = np.round(state_adj, 0).astype(int)
        if not (tuple(state_adj) in self.Q_table):
            self.Q_table[tuple(state_adj)] = np.random.uniform(low=-1.0, high=1.0, size=self.env.action_space.n)
        return state_adj

    def do_action(self, state, epsilon=0):
        if np.random.random() > epsilon:
            return np.argmax(self.Q_table[tuple(self.discretizeState(state))])
        else:
            return self.env.action_space.sample() 

    def predict_q(self, state):
        return np.max(self.Q_table[tuple(self.discretizeState(state))])

    def generate_trajectories(self, n=10, render=False):
        return super().generate_trajectories(self.do_action, self.env, n, render)

    def get_average_reward(self, number_of_tests, render=False):
        return super().get_average_reward(self.do_action, self.env, number_of_tests, "Q_Learning", render=False)

    def train(self, epochs, lr, epsilon, discount):
        reward_list = []
        ave_reward_list = []

        reduction = epsilon / (epochs-epochs/4)
        print('Reduction: ', reduction)

        for i in range(epochs):
            done = False
            tot_reward, reward = 0, 0
            state = self.env.reset()
            while done != True:
                # Randomly do a random action, this will increase exploration in the beginning.
                action = self.do_action(state, epsilon=epsilon)

                state2, reward, done, info = self.env.step(action)

                delta = lr*(reward + discount*np.max(self.Q_table[tuple(self.discretizeState(state2))]) - self.Q_table[tuple(self.discretizeState(state))][action])
                self.Q_table[tuple(self.discretizeState(state))][action] += delta

                #Update total_reward & update state for new action.
                tot_reward += reward
                state = state2

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