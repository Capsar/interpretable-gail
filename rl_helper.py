from keras.layers.advanced_activations import ReLU
import numpy as np
from gym import Env, envs
import pickle
from numpy.core.fromnumeric import size
from typing import Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam


from tensorforce import Agent, Environment
from tensorforce.execution import Runner


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

    def generate_trajectories(self, do_action, env:Env, n=10, render=False, min_reward=-10000):
        trajectories = []
        i = 0
        while i < n:
            #Reset Environmnet
            state = env.reset()
            done = False
            trajectory=[]
            total_reward=0
            while done == False:
                if render:
                    env.render()
                action = do_action(state)

                #perform action in environment
                next_state, reward, done, info = env.step(action)
        
                total_reward+=reward
                #Save state and action to memory
                trajectory.append((state, action))

                state = next_state

            if total_reward > min_reward:
                trajectories.extend(trajectory)
                i+=1

            env.close()
            #save trajectory and actions to large memory
        return trajectories

    def get_average_reward(self, do_action, env:Env, number_of_tests, actor, start_state=[], start_action=-1, render=False, printt=False):
        rewards = []
        for ii in range(number_of_tests):
            state = env.reset()
            if len(start_state) != 0:
                env.state = env.unwrapped.state = start_state
                state = start_state
            do_start_action = False
            if start_action != -1:
                do_start_action = True

            game_reward = 0
            done = False
            while done == False:
                if render:
                    env.render()
                action = do_action(state)
                if do_start_action:
                    action = start_action
                state, reward, done, info = env.step(action)
                game_reward += reward
            rewards.append(game_reward)
        env.close()
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
        self.decision_tree.fit([env.reset(), env.reset()], [0, 1])

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

    def get_average_reward(self, number_of_tests, state=[], action=-1, render=False, print=False):
        return super().get_average_reward(self.do_action, self.env, number_of_tests, "DecisionTree", start_state=state, start_action=action, render=render, printt=print)

class DQN(RL_Agent):

    def __init__(self, env:Env, memory_size):
        self.env = env
        self.env.reset()

        self.environment = Environment.create(environment='gym', level=self.env.spec.id)

        self.agent = Agent.create(
            agent='ppo',
            environment=self.environment,
            memory='minimum',
            batch_size=12
        )

    def load(self):
        self.agent = Agent.load(directory=f'./models/Tensorforce_{self.env.spec.id}', filename=f'Tensorforce_ {self.env.spec.id}', format='checkpoint')

    def save(self):
        self.agent.save(directory=f'./models/Tensorforce_{self.env.spec.id}', filename=f'Tensorforce_ {self.env.spec.id}', format='checkpoint')

    def do_action(self, state):
        return self.agent.act(state, independent=True)

    def generate_trajectories(self, n=10, render=False):
        return super().generate_trajectories(self.do_action, self.env, n, render)

    def get_average_reward(self, number_of_tests, state=[], action=-1, render=False, print=False):
        return super().get_average_reward(self.do_action, self.env, number_of_tests, "TensorForce", start_state=state, start_action=action, render=render, printt=print)

    def train(self, epochs, number=10):
        total_reward = 0
        for i in range(1, epochs):
            
            states = self.environment.reset()
            terminal = False
            while not terminal:
                actions = self.agent.act(states=states)
                states, terminal, reward = self.environment.execute(actions=actions)
                self.agent.observe(terminal=terminal, reward=reward)
                total_reward += reward

            if i % number == 0:
                print('episode:', i, '/', epochs, "total reward:", total_reward/number)
                total_reward = 0

class QLearning(RL_Agent):

    def __init__(self, env: Env, discretize: list, discount: float):
        self.env = env
        self.env.reset()
        self.discretize = discretize
        self.discount = discount
        self.Q_table = {}

    def load(self):
        discretize = '_'.join(map(str, self.discretize))
        discount = str(self.discount).replace(".", '-')
        loadedQ = self.loadz(f'./models/Q_Table_{self.env.spec.id}_{discretize}_{discount}')
        if isinstance(loadedQ, dict):
            self.Q_table = loadedQ

    def save(self):
        discretize = '_'.join(map(str, self.discretize))
        discount = str(self.discount).replace(".", '-')
        self.savez(self.Q_table, f'./models/Q_Table_{self.env.spec.id}_{discretize}_{discount}')

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

    def generate_trajectories(self, n=10, render=False, min_reward=-10000):
        return super().generate_trajectories(self.do_action, self.env, n, render, min_reward)

    def get_average_reward(self, number_of_tests, state=[], action=-1, render=False, print=False):
        return super().get_average_reward(self.do_action, self.env, number_of_tests, "Q_Learning", start_state=state, start_action=action, render=render, printt=print)

    def train(self, epochs, lr, epsilon, max_steps=-1):
        rewards = []

        reduction = epsilon / epochs
        print('Reduction: ', reduction)

        for i in range(epochs):
            done = False
            cum_reward = 0
            state = self.env.reset()
            step = 1
            while done != True:
                step+=1
                # Randomly do a random action, this will increase exploration in the beginning.
                action = self.do_action(state, epsilon=epsilon)

                state2, reward, done, info = self.env.step(action)

                delta = lr*(reward + self.discount*np.max(self.Q_table[tuple(self.discretizeState(state2))]) - self.Q_table[tuple(self.discretizeState(state))][action])
                self.Q_table[tuple(self.discretizeState(state))][action] += delta

                #Update total_reward & update state for new action.
                cum_reward += reward
                state = state2

                if step > max_steps and max_steps != -1:
                    done = True

            # Decay epsilon
            if epsilon > 0.02:
                epsilon -= reduction
            
            # Keep track of rewards (No influence on workings of the algorithm)
            rewards.append(cum_reward)
            if (i+1) % 100 == 0:
                rewards = np.asarray(rewards)
                print(f'Episode {i+1}: Epsilon: {round(epsilon, 3)} Q_table Size: {len(self.Q_table)} Reward -> mean: {rewards.mean()} , std: {rewards.std()}, min: {rewards.min()}, & max: {rewards.max()}')
                rewards = []
            if (i+1) % 1000 == 0:
                self.save()
                print(f'Episode {i+1}: Saved the Q_table.')

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

class DiscriminatorNN:
    def __init__(self, env: Env, hidden_dims=[128, 128, 128], epochs=10, learning_rate=0.1, discriminateWithQ=False):
        self.env = env
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.learning_rate = learning_rate
        extraQ = 2 if discriminateWithQ else 1

        self.model = Sequential()
        self.model.add(Dense(hidden_dims[0], input_dim=self.env.observation_space.shape[0]+extraQ, activation='relu'))
        for dim in hidden_dims[1:]:
            self.model.add(Dense(dim))
            self.model.add(LeakyReLU(alpha=0.05))
            self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation='sigmoid'))

        self.optimizer = tf.keras.optimizers.Adam()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy()
        
        # self.model.compile(optimizer=self.optimizer, loss=self.discriminator_loss, metrics=[tf.keras.metrics.KLDivergence()])

    def predict(self, state_action):
        return self.model(state_action)

    def discriminator_loss(self, expert_output, generator_output):
        expert_loss = self.cross_entropy(tf.ones_like(expert_output), expert_output)
        generator_loss = self.cross_entropy(tf.zeros_like(generator_output), generator_output)
        total_loss = expert_loss + generator_loss
        return total_loss

    def fit(self, expert_state_actions, generator_state_actions):
        for i in range(self.epochs):
            with tf.GradientTape() as disc_tape:
                real_output = self.model(expert_state_actions, training=True)
                fake_output = self.model(generator_state_actions, training=True)

                disc_loss = self.discriminator_loss(real_output, fake_output)

                gradients_of_discriminator = disc_tape.gradient(disc_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients_of_discriminator, self.model.trainable_variables))

    ## Yes, successive calls to fit will incrementally train the model.
    # def fit_label(self, sample_state_actions, sample_labels):
    #     for i in range(self.epochs):
    #         with tf.GradientTape() as disc_tape:
    #             output = self.model(sample_state_actions, training=True)
    #             disc_loss = self.cross_entropy(sample_labels, output)

    #             gradients_of_discriminator = disc_tape.gradient(disc_loss, self.model.trainable_variables)
    #             self.optimizer.apply_gradients(zip(gradients_of_discriminator, self.model.trainable_variables))
