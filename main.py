import sklearn
import tensorflow as tf
from tensorflow import keras
import gym
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression

from rl_helper import QLearning

print("Tensforflow version:", tf.__version__)
print('keras version:', keras.__version__)

env = gym.make("Acrobot-v1")
# env = gym.make("MountainCar-v0")
print('Action space: ', env.action_space)
print('Observation space:', env.observation_space)
print('Metadata: ', env.metadata)

# Setup Qlearning to create expert model

q_learner = QLearning(env, discretize=[5, 5, 5, 5, 2, 2])
try:
    q_learner.load()
except:
    print('file not found.')
q_learner.train(50000, lr=0.5, epsilon=0.5, discount=0.99)
q_learner.save()

# Generate expert data from expert model
q_agent = QLearning(env, discretize=[5, 5, 5, 5, 2, 2])
q_agent.load()
# q_agent.generate_trajectories(1, render=True)
expert_trajectories = q_agent.generate_trajectories(200)
expert_state_actions = [tuple(list(s) + [a]) for s,a in expert_trajectories]
expert_labels = [1 for i in expert_trajectories]
print(len(expert_trajectories), expert_trajectories[0:5], expert_state_actions[0:5], expert_labels[0:5])

# Setup gail from expert data with decision tree as generative model and mlp as descriminative model
epochs = 10
max_sample_size = 5000
number_of_tests = 50
print('------------------------------------------------------------------------------')
#init generator and discriminator policy parameters theta & w
generator = DecisionTreeClassifier()
generator.fit([env.reset()], [0])

discriminator = LogisticRegression()
discriminator.fit([tuple(list(env.reset()) + [0]), tuple(list(env.reset()) + [1])], [0, 1])

for i in range(epochs):
    total_reward = 0
    for ii in range(number_of_tests):
        state = env.reset()
        done = False
        while done == False:
            action = generator.predict([state])[0]
            state, reward, done, info = env.step(action)
            total_reward+=reward
    print('Total average reward:', total_reward / number_of_tests)

    #Generate state action pairs from generator (expert states used as input)
    generator_state_actions = [tuple(list(s) + list(generator.predict([s]))) for s,_ in list(expert_trajectories)]
    generator_labels = [0 for i in generator_state_actions]
    # print(generator_trajectories[0:5], generator_labels[0:5])


    sample_state_actions, sample_labels = [], []
    sample_state_actions.extend(expert_state_actions)
    sample_labels.extend(expert_labels)
    sample_state_actions.extend(generator_state_actions)
    sample_labels.extend(generator_labels)
    # print(len(sample_state_actions), len(sample_labels), len(expert_state_actions), len(expert_labels), len(generator_state_actions), len(generator_labels))
    # print(list(zip(sample_state_actions, sample_labels))[0:5], list(zip(expert_state_actions, expert_labels))[0:5], list(zip(generator_state_actions, generator_labels))[0:5])


    #Train the discriminator with full sample trajectories and labels.
    discriminator.fit(sample_state_actions, sample_labels)
    print('Discriminator score on expert & generator data:', discriminator.score(expert_state_actions, expert_labels), discriminator.score(generator_state_actions, generator_labels))
    print('Discriminator prediction on expert & generator data:', np.array(discriminator.predict(expert_state_actions)).mean(), np.array(discriminator.predict(generator_state_actions)).mean())

    #Sample random amount of state_action and labels.
    idx = np.random.choice(len(sample_state_actions), size=min(len(sample_state_actions), max_sample_size))
    sample_state_actions, sample_labels = [sample_state_actions[i] for i in idx], [sample_labels[i] for i in idx]

    #Get all state action pairs classified as expert data by the discriminator
    new_generator_s, new_generator_a = [], []
    for s_a, p_expert in list(zip(sample_state_actions, discriminator.predict_proba(sample_state_actions))):
        if np.random.random() < p_expert[1]:
            new_generator_s.append(np.array(s_a[:-1], dtype=float))
            new_generator_a.append(s_a[-1])
    generator.fit(new_generator_s, new_generator_a)

total_reward = 0
for ii in range(3):
    state = env.reset()
    done = False
    while done == False:
        env.render()
        action = generator.predict([state])[0]
        state, reward, done, info = env.step(action)
        total_reward+=reward
print('Total average reward:', total_reward / 3)


# Evaluate the new generative model in terms of interpretability (size, average path length, compared to optimal)

# Recommendation for increasing interpretability of the generative model.

print('Finished!')