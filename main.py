import sklearn
import tensorflow as tf
from tensorflow import keras
import gym
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from rl_helper import QLearning

print("Tensforflow version:", tf.__version__)
print('keras version:', keras.__version__)

env = gym.make("MountainCar-v0")
print('Action space: ', env.action_space)
print('Observation space:', env.observation_space)
print('Metadata: ', env.metadata)

# Setup Qlearning to create expert model

q_learner = QLearning(env, discretize=[30, 300])
q_learner.train(5000, lr=0.3, epsilon=0.0, discount=0.99)
# q_learner.save()

# Generate expert data from expert model
# q_agent = QLearning(env, discretize=[30, 300])
# q_agent.load('./models/MountainCar-v0.npy')
# expert_s, expert_a = q_agent.generate_trajectories_and_actions(20)
# # print(expert_s, expert_a)

# expert_s_a = [(x[0][0], x[0][1], x[1]) for x in list(zip(expert_s, expert_a))]
# expert_labels = [1 for i in expert_s]

# # Setup gail from expert data with decision tree as generative model and mlp as descriminative model
# epochs = 2
# print('------------------------------------------------------------------------------')
# #init generator and discriminator policy parameters theta & w
# generator = DecisionTreeClassifier()
# generator.fit([env.reset()], [0])

# discriminator = DecisionTreeRegressor()
# discriminator.fit([(0,0,0)], [0])

# for i in range(epochs):
#     #Generate state action pairs from generator (expert states used as input)
#     generator_s, generator_a = expert_s, list(generator.predict(expert_s))
#     # print(generator_s, generator_a)

#     generator_s_a = [(x[0][0], x[0][1], x[1]) for x in list(zip(generator_s, generator_a))]
#     generator_labels = [0 for i in generator_s]

#     total_s_a = []
#     total_s_a.extend(expert_s_a)
#     total_s_a.extend(generator_s_a)

#     total_labels = []
#     total_labels.extend(expert_labels)
#     total_labels.extend(generator_labels)

#     #Train the discriminator
#     discriminator.fit(total_s_a, total_labels)
#     print(np.array(discriminator.predict(expert_s_a)).mean(), np.array(discriminator.predict(generator_s_a)).mean())

#     #Get all state action pairs classified as expert data by the discriminator
#     new_generator_s, new_generator_a = [], []
#     for s_a in list(zip(total_s_a, discriminator.predict(total_s_a))):
#         if np.random.random() < s_a[1]:
#             new_generator_s.append(np.array([s_a[0][0], s_a[0][1]]))
#             new_generator_a.append(s_a[0][2])
#     # print(new_generator_s, new_generator_a)

#     generator.fit(new_generator_s, new_generator_a)

#     state = env.reset()
#     done = False
#     while done == False:
#         # env.render()
#         action = generator.predict([state])[0]
#         state, reward, done, info = env.step(action)
#         # generator_s.append(state)
#         # generator_a.append(action)

# Evaluate the new generative model in terms of interpretability (size, average path length, compared to optimal)

# Recommendation for increasing interpretability of the generative model.

print('Finished!')