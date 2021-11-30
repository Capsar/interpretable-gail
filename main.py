import tensorflow as tf
from tensorflow import keras
import gym
import numpy as np
from sklearn.linear_model import LogisticRegression
from rl_helper import QLearning, DecisionTree
from scipy.spatial import distance


print("Tensforflow version:", tf.__version__)
print('keras version:', keras.__version__)

env, discretize = gym.make("Acrobot-v1"), [5, 5, 5, 5, 2, 2]
# env = gym.make("CartPole-v1")
# env = gym.make("MountainCar-v0")
print('Action space: ', env.action_space)
print('Observation space:', env.observation_space)
print('Metadata: ', env.metadata)
print('')

# Setup Qlearning to create expert model
q_learner = QLearning(env, discretize)
q_learner.load()
q_learner.train(1000, lr=0.5, epsilon=0.05, discount=0.99)
q_learner.save()

# Generate expert data from expert model
q_agent = QLearning(env, discretize)
q_agent.load()
q_agent.get_average_reward(1, render=True)
expert_trajectories = q_agent.generate_trajectories(200)
expert_state_actions = [tuple(list(s) + [a]) for s,a in expert_trajectories]
expert_labels = [1 for i in expert_trajectories]

# Setup gail from expert data with decision tree as generative model and mlp as descriminative model
epochs = 3
max_sample_size = 1000
number_of_tests = 50

#init generator and discriminator policy parameters theta & w
generator = DecisionTree(env, max_depth=3, max_features=2)

discriminator = LogisticRegression()
discriminator.fit([tuple(list(env.reset()) + [0]), tuple(list(env.reset()) + [1])], [0, 1])

generator.get_average_reward(number_of_tests)
print('------------------------------------------------------------------------------')

for i in range(epochs):

    print('Epoch: ', i)
    #Generate state action pairs from generator (expert states used as input)
    generator_state_actions = [tuple(list(s) + [generator.do_action(s)]) for s,_ in list(expert_trajectories)]
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
    print('Discriminator prediction probability on expert & generator data:', np.array(discriminator.predict_proba(expert_state_actions))[:,1].mean(), np.array(discriminator.predict_proba(generator_state_actions))[:,1].mean())
    print('The Jensen-Shannon distance between expert and generator data:', distance.jensenshannon([a[-1] for a in expert_state_actions], [a[-1] for a in generator_state_actions]))
    
    #Sample random amount of state_action and labels.
    # idx = np.random.choice(len(sample_state_actions), size=min(len(sample_state_actions), max_sample_size))
    # sample_state_actions, sample_labels = [sample_state_actions[i] for i in idx], [sample_labels[i] for i in idx]

    #Get all state action pairs classified as expert data by the discriminator
    new_generator_s, new_generator_a = [], []
    for s_a, p_expert in list(zip(sample_state_actions, discriminator.predict_proba(sample_state_actions))):
        if np.random.random() < p_expert[1]:
            new_generator_s.append(np.array(s_a[:-1], dtype=float))
            new_generator_a.append(s_a[-1])
    generator.fit(new_generator_s, new_generator_a)

    #Get average reward of the generator.
    generator.get_average_reward(number_of_tests)

    print('')



#Final test expert vs Generator:
total_tests = 500
q_agent.get_average_reward(total_tests)
generator.get_average_reward(total_tests)

# Evaluate the new generative model in terms of interpretability (size, average path length, compared to optimal)

# Recommendation for increasing interpretability of the generative model.

print('Finished!')