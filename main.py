import tensorflow as tf
from tensorflow import keras
import gym
from gail import do_gail
from sklearn import tree
from rl_helper import QLearning, DecisionTree, NeuralNetwork
from PongEnv import PongEnv
from matplotlib import pyplot as plt
from dtreeviz.trees import *


print("Tensforflow version:", tf.__version__)
print('keras version:', keras.__version__)

# env, discretize, discount = PongEnv(), [0.5, 0.5, 0.5, 0.5, 0.25, 0.25], 0.999
# env, discretize, discount = gym.make("Acrobot-v1"), [5, 5, 5, 5, 1, 1], 0.95
env, discretize, discount, feature_names, class_names = gym.make("CartPole-v1"), [10, 10, 10, 10], 0.99, ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity'], ['Left', 'Right']
# env, discretize, discount, feature_names, class_names = gym.make("MountainCar-v0"), [30, 300], 0.99, ['Cart Postion', 'Cart Velocity'], ['Left++', 'Do nothing', 'Right++']
print('Env:', env.spec.id)
print('Discretize:', discretize, 'Discount:', discount)
print('Action space: ', env.action_space)
# print('Observation space:', env.observation_space)
print('Metadata: ', env.metadata)
print('')

## Setup Qlearning to create expert model
# q_learner = QLearning(env, discretize, discount)
# q_learner.load()
# q_learner.get_average_reward(500, print=True)
# for i in range(200):
#     q_learner.train(30000, lr=0.1, epsilon=0.3)
#     q_learner.save()

doGAIL = True
if doGAIL:

    for depth in range(1, 5):
        expert = QLearning(env, discretize, discount)
        expert.load()

        ## Setup gail from expert data with decision tree as generative model and mlp as descriminative model
        generator = DecisionTree(env, max_depth=depth, max_features=env.observation_space.shape[0])
        discriminator = NeuralNetwork(env, hidden_dims=[32,32,32], epochs=100)
        generator = do_gail(expert, generator, discriminator, n_e_trajectories=100, epochs=5)

        ## Final test expert vs Generator:
        total_tests = 500
        print("Decision Tree depth:", generator.decision_tree.get_depth(), '#leave nodes:', generator.decision_tree.get_n_leaves())
        g_mean, g_std = generator.get_average_reward(total_tests, print=True)
        generator.save()
        e_mean, e_std = expert.get_average_reward(total_tests, print=True)

        total_trajectories = 1000
        trajectories = expert.generate_trajectories(n=total_trajectories)
        states = np.array([s for s,_ in trajectories])
        actions = np.array([a for _,a in trajectories])

        ## Evaluate the new generative model in terms of interpretability (size, average path length, compared to optimal)
        # fig = plt.figure(figsize=(50,50))
        # _ = tree.plot_tree(generator.decision_tree, feature_names=feature_names, class_names=class_names)
        # fig.savefig(f'./trees/DecistionTree_{env.spec.id}_{generator.max_depth}_{generator.max_features}.png')
        target_name = f'{env.spec.id}'
        viz = dtreeviz(generator.decision_tree, states, actions, feature_names=feature_names, class_names=class_names, target_name=target_name)
        viz.save(f'./trees/svg_DecistionTree_{env.spec.id}_{generator.max_depth}_{generator.max_features}.svg')

        ct = ctreeviz_bivar(generator.decision_tree, states, actions, feature_names=feature_names, class_names=class_names, target_name=target_name)
        plt.tight_layout()
        plt.savefig(f'./trees/png_DecistionTree_{env.spec.id}_{generator.max_depth}_{generator.max_features}.png')

    ## Recommendation for increasing interpretability of the generative model.

print('Finished!')