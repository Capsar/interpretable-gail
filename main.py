import tensorflow as tf
from tensorflow import keras
import gym
import numpy as np

from rl_helper import QLearning

print("Tensforflow version:", tf.__version__)
print('keras version:', keras.__version__)

env = gym.make("MountainCar-v0")
print('Action space: ', env.action_space)
print('Observation space:', env.observation_space)
print('Metadata: ', env.metadata)

# Setup Qlearning to create expert model
q_learner = QLearning(env, discretize=[10, 100])
q_learner.load('./models/MountainCar-v0.npy')
q_learner.train(5000, lr=0.2, epsilon=0.8, discount=0.9)
q_learner.save()

q_agent = QLearning(env, discretize=[10, 100])
q_agent.load('./models/MountainCar-v0.npy')

state = env.reset()
state = q_agent.discretizeState(state, env.observation_space.low)
done = False

total_reward = 0
while done == False:
    env.render()
    action = q_agent.do_action(state)
    state, reward, done, info = env.step(action)
    state = q_agent.discretizeState(state, env.observation_space.low)
    total_reward+=reward

print(total_reward)

# Generate expert data from expert model

# Setup gail from expert data with decision tree as generative model and mlp as descriminative model

# Evaluate the new generative model in terms of interpretability (size, average path length, compared to optimal)

#Recommendation for increasing interpretability of the generative model.

print('Finished!')