import tensorflow as tf
from tensorflow import keras
import gym
import numpy as np
from rl_helper import LogRegression, NeuralNetwork, QLearning, DecisionTree
from scipy.spatial import distance


print("Tensforflow version:", tf.__version__)
print('keras version:', keras.__version__)

env, discretize = gym.make("CartPole-v1"), [10, 10, 10, 10]
# env = gym.make("CartPole-v1")
# env = gym.make("MountainCar-v0")
print('Action space: ', env.action_space)
print('Observation space:', env.observation_space)
print('Metadata: ', env.metadata)
print('')

# Setup Qlearning to create expert model
q_learner = QLearning(env, discretize)
q_learner.load()
q_learner.train(50000, lr=0.5, epsilon=0.5, discount=0.99)
q_learner.save()

# Generate expert data from expert model
q_agent = QLearning(env, discretize)
q_agent.load()
q_agent.get_average_reward(1, render=False, print=True)
expert_trajectories = q_agent.generate_trajectories(100)
# expert_state_actions = [tuple(list(s) + [a]) for s,a in expert_trajectories]
expert_state_actions = np.asarray([np.asarray(list(s) + [a]) for s,a in expert_trajectories])
expert_labels = [1 for i in expert_trajectories]
print('Number of expert state-action pairs:', len(expert_trajectories))

# Setup gail from expert data with decision tree as generative model and mlp as descriminative model
epochs = 2
# max_sample_size = 1000
number_of_tests = 500

#init generator and discriminator policy parameters theta & w
generator = DecisionTree(env, max_depth=300, max_features=env.observation_space.shape[0])

discriminator = NeuralNetwork(env)
# discriminator = LogRegression(env)
q_agent.get_average_reward(500, print=True)
print('------------------------------------------------------------------------------')

for i in range(epochs):

    print('Epoch: ', i+1)

    generator.get_average_reward(number_of_tests, print=True)

    #Generate state action pairs from generator (expert states used as input)
    # generator_state_actions = [tuple(list(s) + [generator.do_action(s)]) for s,_ in list(expert_trajectories)]
    generator_state_actions = np.asarray([(list(s) + [generator.do_action(s)]) for s,_ in list(expert_trajectories)])
    generator_labels = [0 for i in generator_state_actions]

    sample_state_actions, sample_labels = [], []
    sample_state_actions.extend(expert_state_actions)
    sample_labels.extend(expert_labels)
    sample_state_actions.extend(generator_state_actions)
    sample_labels.extend(generator_labels)
    sample_state_actions = np.asarray(sample_state_actions)
    sample_labels = np.asarray(sample_labels)

    #Train the discriminator with full sample trajectories and labels.
    print('Training discriminator:')
    discriminator.fit(sample_state_actions, sample_labels)
    print('Discriminator prediction on expert & generator data:', np.array(discriminator.predict(expert_state_actions)).mean(), np.array(discriminator.predict(generator_state_actions)).mean())
    print('The Jensen-Shannon distance between expert and generator data:', distance.jensenshannon([a[-1] for a in expert_state_actions], [a[-1] for a in generator_state_actions]))
    
    #Sample random amount of state_action and labels. (This is in Viper)
    ## expert_qs = [q_agent.get_average_reward(1, s) for s,_ in expert_trajectories]
    ## generator_qs = [generator.get_average_reward(1, s) for s,_ in expert_trajectories]
    ## sample_qs = []
    ## sample_qs.extend(expert_qs)
    ## sample_qs.extend(generator_qs)
    ## ps = sample_qs / np.sum(sample_qs)
    ## idx = np.random.choice(len(sample_state_actions), size=min(len(sample_state_actions), max_sample_size), p=ps)
    ## sample_state_actions, sample_labels = [sample_state_actions[i] for i in idx], [sample_labels[i] for i in idx]

    #Get all state action pairs classified as expert data by the discriminator
    new_generator_s, new_generator_a = [], []
    for s_a, p_expert in list(zip(sample_state_actions, discriminator.predict(sample_state_actions))):
        if np.random.random() < p_expert:
            new_generator_s.append(np.array(s_a[:-1], dtype=float))
            new_generator_a.append(int(s_a[-1]))
    generator.fit(new_generator_s, new_generator_a)
    print(len(new_generator_a), 'out of the', len(sample_state_actions), 'were chosen as new generator data.')

    print('')



#Final test expert vs Generator:
total_tests = 1000
generator.get_average_reward(total_tests, print=True)
generator.save()
q_agent.get_average_reward(total_tests, print=True)
# Evaluate the new generative model in terms of interpretability (size, average path length, compared to optimal)

# Recommendation for increasing interpretability of the generative model.

print('Finished!')