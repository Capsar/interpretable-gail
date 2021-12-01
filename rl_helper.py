import numpy as np
from gym import Env
import pickle
from numpy.core.fromnumeric import size
from typing import Any
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

from keras.models import Sequential
from keras.layers import Dense

class RL_Agent():
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

    def get_average_reward(self, do_action, env:Env, number_of_tests, actor, start_state=[], render=False, printt=False):
        rewards = []
        for ii in range(number_of_tests):
            state = env.reset()
            if len(start_state) != 0:
                state = start_state

            game_reward = 0
            done = False
            while done == False:
                if render:
                    env.render()
                action = do_action(state)
                state, reward, done, info = env.step(action)
                game_reward += reward
            rewards.append(game_reward)
            
        rewards = np.asarray(rewards)
        if printt:
                print(f'Reward of {actor} with {number_of_tests} tests -> mean: {rewards.mean()}, std: {rewards.std()}, min: {rewards.min()}, & max: {rewards.max()}')
        return rewards.mean(), rewards.std()

class DecisionTree(RL_Agent):

    def __init__(self, env: Env, max_depth=5, max_features=2):
        self.env = env
        self.env.reset()
        self.max_depth = max_depth
        self.max_features = max_features
        self.decision_tree = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features)
        self.decision_tree.fit([env.reset()], [0])

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

    def get_average_reward(self, number_of_tests, state=[], render=False, print=False):
        return super().get_average_reward(self.do_action, self.env, number_of_tests, "DecisionTree", start_state=state, render=render, printt=print)

class QLearning(RL_Agent):

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

    def generate_trajectories(self, n=10, render=False):
        return super().generate_trajectories(self.do_action, self.env, n, render)

    def get_average_reward(self, number_of_tests, state=[], render=False, print=False):
        return super().get_average_reward(self.do_action, self.env, number_of_tests, "Q_Learning", start_state=state, render=render, printt=print)

    def train(self, epochs, lr, epsilon, discount):
        rewards = []

        reduction = epsilon / (epochs-epochs/4)
        print('Reduction: ', reduction)

        for i in range(epochs):
            done = False
            cum_reward = 0
            state = self.env.reset()
            while done != True:
                # Randomly do a random action, this will increase exploration in the beginning.
                action = self.do_action(state, epsilon=epsilon)

                state2, reward, done, info = self.env.step(action)

                delta = lr*(reward + discount*np.max(self.Q_table[tuple(self.discretizeState(state2))]) - self.Q_table[tuple(self.discretizeState(state))][action])
                self.Q_table[tuple(self.discretizeState(state))][action] += delta

                #Update total_reward & update state for new action.
                cum_reward += reward
                state = state2

            # Decay epsilon
            if epsilon > 0:
                epsilon -= reduction
            
            # Keep track of rewards (No influence on workings of the algorithm)
            rewards.append(cum_reward)
            if (i+1) % 100 == 0:
                rewards = np.asarray(rewards)
                print(f'Episode {i+1} Q_table Size: {len(self.Q_table)} Reward -> mean: {rewards.mean()} , std: {rewards.std()}, min: {rewards.min()}, & max: {rewards.max()}')
                rewards = []
                
        self.env.close()
        print("Finished training!")





class LogRegression:
    def __init__(self, env: Env):
        self.discriminator = LogisticRegression()
        self.discriminator.fit([tuple(list(env.reset()) + [0]), tuple(list(env.reset()) + [1])], [0, 1])

    def predict(self, state_action):
        return self.discriminator.predict_proba(state_action)[:,1]

    def fit(self, sample_state_actions, sample_labels):
        self.discriminator.fit(sample_state_actions, sample_labels)

class NeuralNetwork:
    def __init__(self, env: Env, hidden_dims=[128, 128, 128], learning_rate=0.1):
        self.env = env
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate

        self.model = Sequential()
        self.model.add(Dense(hidden_dims[0], input_dim=self.env.observation_space.shape[0]+1, activation='relu'))
        for dim in hidden_dims[1:]:
            self.model.add(Dense(dim, activation='relu'))
        self.model.add(Dense(1))

        self.model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.KLDivergence()])

    def predict(self, state_action):
        return self.model.predict(state_action)

    def fit(self, sample_state_actions, sample_labels):
        return self.model.fit(sample_state_actions, sample_labels, epochs=10, verbose=0)
