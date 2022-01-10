import tensorflow as tf
from tensorflow import keras
import gym
from gail import do_gail
from sklearn import tree
from rl_helper import DQN, QLearning, DecisionTree, DiscriminatorNN
from matplotlib import pyplot as plt
from dtreeviz.trees import *


print("Tensforflow version:", tf.__version__)
print('keras version:', keras.__version__)

# env, discretize, discount = PongEnv(), [0.5, 0.5, 0.5, 0.5, 0.25, 0.25], 0.999
# env, memory_size, discretize, discount, feature_names, class_names = gym.make("Acrobot-v1"), 50000, [10, 10, 10, 10, 2, 2], 0.95, ['cos(theta1)', 'sin(theta1)', 'cos(theta2)', 'sin(theta2)', 'thetaDOT1', 'thetaDOT2'], ['torque +1', 'torque 0', 'torque -1']
env, memory_size, discretize, discount, feature_names, class_names = gym.make("CartPole-v1"), 50000, [10, 10, 10, 10], 0.95, ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity'], ['Left', 'Right']
# env, memory_size, discretize, discount, feature_names, class_names = gym.make("MountainCar-v0"), 50000, [10, 100], 0.95, ['Cart Postion', 'Cart Velocity'], ['Left++', 'Do nothing', 'Right++']
print('Env:', env.spec.id)
print('Discretize:', discretize, 'Discount:', discount)
print('Action space: ', env.action_space)
# print('Observation space:', env.observation_space)
print('Metadata: ', env.metadata)
print('')

# Setup Qlearning to create expert model
# q_learner = QLearning(env, discretize, discount)
# q_learner.load()
# q_learner.train(10000, lr=0.1, epsilon=0.3)
# q_learner.save()

# tensorforce_agent = DQN(env, memory_size)
# tensorforce_agent.train(1000)
# tensorforce_agent.save()

doGAIL = True
if doGAIL:

    total_tests = 500
    n_e_trajectories = (5, 2)
    # for depth in range(1, 5):
    depth = 2
    # expert = QLearning(env, discretize, discount)
    # expert.load()
    expert = DQN(env, memory_size)
    expert.load()
    expert.generate_trajectories(n=1, render=True)


    ## Setup gail from expert data with decision tree as generative model and mlp as descriminative model
    generator = DecisionTree(env, max_depth=depth, max_features=env.observation_space.shape[0])
    discriminator = DiscriminatorNN(env, hidden_dims=[16,16], epochs=500, discriminateWithQ=True)
    generator, expert_trajectories = do_gail(expert, generator, discriminator, n_e_trajectories=n_e_trajectories, epochs=10, ownGeneratorTrajectories=False, hasAccessToExpert=True, sampleWithQ=False, discriminateWithQ=True, max_sample_size=100000)

    ## Final test expert vs Generator:

    print("BC Decision Tree depth:", generator.decision_tree.get_depth(), '#leave nodes:', generator.decision_tree.get_n_leaves())
    behaviour_cloning = DecisionTree(env, max_depth=depth, max_features=env.observation_space.shape[0])
    # expert_trajectories = expert.generate_trajectories(n_e_trajectories[0])
    expert_states = [s for s,_ in expert_trajectories]
    expert_actions = [a for _,a in expert_trajectories]
    behaviour_cloning.fit(expert_states, expert_actions)
    print('Behaviour Cloning on', len(expert_trajectories), 'state-action pairs')
    b_mean, b_std = behaviour_cloning.get_average_reward(total_tests, print=True)

    print("GAIL Decision Tree depth:", generator.decision_tree.get_depth(), '#leave nodes:', generator.decision_tree.get_n_leaves())
    g_mean, g_std = generator.get_average_reward(total_tests, print=True)
    generator.save()

    e_mean, e_std = expert.get_average_reward(total_tests, print=True)

    total_trajectories = 10
    trajectories = expert.generate_trajectories(n=total_trajectories)
    states = np.array([s for s,_ in trajectories])
    actions = np.array([a for _,a in trajectories])

    ## Evaluate the new generative model in terms of interpretability (size, average path length, compared to optimal)
    # fig = plt.figure(figsize=(50,50))
    # _ = tree.plot_tree(generator.decision_tree, feature_names=feature_names, class_names=class_names)
    # fig.savefig(f'./trees/DecistionTree_{env.spec.id}_{generator.max_depth}_{generator.max_features}.png')
    target_name = f'{env.spec.id}: {str(round(g_mean, 2))} & {str(round(g_std, 2))}'
    viz = dtreeviz(generator.decision_tree, states, actions, feature_names=feature_names, class_names=class_names, target_name=target_name)
    viz.save(f'./trees/svg_DecistionTree_{env.spec.id}_{generator.max_depth}_{generator.max_features}.svg')

    # ct = ctreeviz_bivar(generator.decision_tree, states[:,[1,3]], actions, feature_names=feature_names, class_names=class_names, target_name=target_name)
    # plt.tight_layout()
    # plt.savefig(f'./trees/png_DecistionTree_{env.spec.id}_{generator.max_depth}_{generator.max_features}.png')

    ## Recommendation for increasing interpretability of the generative model.

print('Finished!')