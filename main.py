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
env, discretize, discount, feature_names, class_names = gym.make("Acrobot-v1"), [10, 10, 10, 10, 2, 2], 0.95, ['cos(theta1)', 'sin(theta1)', 'cos(theta2)', 'sin(theta2)', 'thetaDOT1', 'thetaDOT2'], ['torque +1', 'torque 0', 'torque -1']
# env, discretize, discount, feature_names, class_names = gym.make("CartPole-v1"), [10, 10, 10, 10], 0.95, ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity'], ['Left', 'Right']
# env, discretize, discount, feature_names, class_names = gym.make("MountainCar-v0"), [30, 300], 0.95, ['Cart Postion', 'Cart Velocity'], ['Left++', 'Do nothing', 'Right++']
print('Env:', env.spec.id)
print('Discretize:', discretize, 'Discount:', discount)
print('Action space: ', env.action_space)
# print('Observation space:', env.observation_space)
print('Metadata: ', env.metadata)
print('')

# Setup Qlearning to create expert model
# q_learner = QLearning(env, discretize, discount)
# q_learner.load()
# q_learner.train(300000, lr=0.1, epsilon=0.3)
# q_learner.save()

memory_size, batch_size = 200000, 32
tensorforce_agent = DQN(env, memory_size, batch_size)
# tensorforce_agent.load()
# tensorforce_agent.train(2500)
# tensorforce_agent.save()

doGAIL = True
if doGAIL:

    # expert = QLearning(env, discretize, discount)
    # expert.load()
    expert = DQN(env, memory_size, batch_size)
    expert.load()

    total_tests = 100
    e_mean, e_std = expert.get_average_reward(total_tests, print=True)


    depth = 1
    n_e_trajectories = (1, 1)
    epochs = 5
    ownGeneratorTrajectories = False
    hasAccessToExpert = False
    sampleWithQ = False
    discriminateWithQ = True  

    for depth in [1, 2, 3]:
        # for n_e_trajectories in [(1,1), (3, 1), (5, 1), (7,1), (9, 1), (11, 1)]:

        behaviour_cloning = DecisionTree(env, max_depth=depth)
        expert_trajectories = expert.generate_trajectories(n_e_trajectories[0])
        expert_states = [s for s,_ in expert_trajectories]
        expert_actions = [a for _,a in expert_trajectories]
        behaviour_cloning.fit(expert_states, expert_actions)

        # print('Behaviour Cloning on', len(expert_trajectories), 'state-action pairs')
        b_mean, b_std = behaviour_cloning.get_average_reward(total_tests, print=False)


        expert_states = np.array(expert_states)
        expert_actions = np.array(expert_actions)
        target_name = f'{env.spec.id}: {str(round(b_mean, 2))} & {str(round(b_std, 2))}'
        file_name_extra = f'{env.spec.id}_{depth}_{n_e_trajectories[0]}_{str(round(b_mean, 2))}_{str(round(b_std, 2))}'
        viz = dtreeviz(behaviour_cloning.decision_tree, expert_states, expert_actions, feature_names=feature_names, class_names=class_names, target_name=target_name)
        viz.save(f'./trees/svg_BC_DecistionTree_{file_name_extra}.svg')

        # for epochs in [1, 2, 5, 10]:
        for ownGeneratorTrajectories in [False, True]:
            for hasAccessToExpert in [False, True]:
                for sampleWithQ in [False, True]:
                    for discriminateWithQ in [False, True]:

                        ## Setup gail from expert data with decision tree as generative model and mlp as descriminative model
                        generator = DecisionTree(env, max_depth=depth)
                        discriminator = DiscriminatorNN(env, hidden_dims=[32,32], epochs=500, discriminateWithQ=discriminateWithQ)
                        generator, gail_expert_trajectories, gail_epoch_results = do_gail(expert, generator, discriminator, n_e_trajectories=n_e_trajectories, epochs=epochs, 
                                                                ownGeneratorTrajectories=ownGeneratorTrajectories, hasAccessToExpert=hasAccessToExpert, 
                                                                sampleWithQ=sampleWithQ, discriminateWithQ=discriminateWithQ, pprint=False)
                        ## Final test expert vs Generator:
                        # print("BC Decision Tree depth:", generator.decision_tree.get_depth(), '#leave nodes:', generator.decision_tree.get_n_leaves())

                        # print("GAIL Decision Tree depth:", generator.decision_tree.get_depth(), '#leave nodes:', generator.decision_tree.get_n_leaves())
                        g_mean, g_std = generator.get_average_reward(total_tests, print=False)
                        generator.save(f'{str(round(g_mean, 2))}_{str(round(g_std, 2))}')

                        # total_trajectories = 10
                        states = np.array([s for s,_ in gail_expert_trajectories])
                        actions = np.array([a for _,a in gail_expert_trajectories])

                        ## Evaluate the new generative model in terms of interpretability (size, average path length, compared to optimal)
                        target_name = f'{env.spec.id}: {str(round(g_mean, 2))} & {str(round(g_std, 2))}'
                        file_name_extra = f'{env.spec.id}_{depth}_{n_e_trajectories[0]}_{n_e_trajectories[1]}_{epochs}_{ownGeneratorTrajectories}_{hasAccessToExpert}_{sampleWithQ}_{discriminateWithQ}_{len(gail_expert_trajectories)}_{str(round(e_mean, 2))}_{str(round(e_std, 2))}_{str(round(b_mean, 2))}_{str(round(b_std, 2))}_{str(round(g_mean, 2))}_{str(round(g_std, 2))}'
                        viz = dtreeviz(generator.decision_tree, states, actions, feature_names=feature_names, class_names=class_names, target_name=target_name)
                        viz.save(f'./trees/svg_GAIL_DecistionTree_{file_name_extra}.svg')

                        print(env.spec.id, depth, n_e_trajectories[0], n_e_trajectories[1], epochs, ownGeneratorTrajectories, hasAccessToExpert, sampleWithQ, discriminateWithQ, len(gail_expert_trajectories), e_mean, e_std, b_mean, b_std, g_mean, g_std, gail_epoch_results)

    # ct = ctreeviz_bivar(generator.decision_tree, states[:,[1,3]], actions, feature_names=feature_names, class_names=class_names, target_name=target_name)
    # plt.tight_layout()
    # plt.savefig(f'./trees/png_DecistionTree_{env.spec.id}_{generator.max_depth}_{generator.max_features}.png')

    ## Recommendation for increasing interpretability of the generative model.

print('Finished!')