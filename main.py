from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import gym
from gail import do_gail
from rl_helper import DQN, QLearning, DecisionTree, DiscriminatorNN
from dtreeviz.trees import *
import copy

print("Tensforflow version:", tf.__version__)
print('keras version:', keras.__version__)

# env, discretize, discount = PongEnv(), [0.5, 0.5, 0.5, 0.5, 0.25, 0.25], 0.999
env, discretize, discount, feature_names, class_names = gym.make("Acrobot-v1"), [10, 10, 10, 10, 2, 2], 0.95, ['cos(theta1)', 'sin(theta1)', 'cos(theta2)', 'sin(theta2)', 'thetaDOT1', 'thetaDOT2'], ['torque +1', 'torque 0', 'torque -1']
# env, discretize, discount, feature_names, class_names = gym.make("CartPole-v1"), [10, 10, 10, 10], 0.95, ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity'], ['Left', 'Right']
# env, discretize, discount, feature_names, class_names = gym.make("MountainCar-v0"), [30, 300], 0.95, ['Cart Postion', 'Cart Velocity'], ['Left++', 'Do nothing', 'Right++']
print('Time:', datetime.now())
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

memory_size, batch_size = 50000, 32
tensorforce_agent = DQN(env, memory_size, batch_size)
# tensorforce_agent.load()
# tensorforce_agent.train(2500, 499)
# tensorforce_agent.save()

doGAIL = True
if doGAIL:

    # expert = QLearning(env, discretize, discount)
    # expert.load()
    expert = DQN(env, memory_size, batch_size)
    expert.load()

    total_test_rollouts = 100

    test_e_mean, test_e_std, test_e_trajectories = expert.do_rollout(n=total_test_rollouts, print=True)
    test_e_states = np.array([s for s,_ in test_e_trajectories])
    test_e_actions = np.array([a for _,a in test_e_trajectories])

    depth = 1
    n_e_trajectories = (1, 1)
    epochs = 10
    ownGeneratorTrajectories = False
    hasAccessToExpert = False
    sampleWithQ = False
    discriminateWithQ = False  

    for xx in range(2):
        train_e_mean, train_e_std, train_e_trajectories = expert.do_rollout(n=n_e_trajectories[0])
        train_e_states = np.array([s for s,_ in train_e_trajectories])
        train_e_actions = np.array([a for _,a in train_e_trajectories])
        print('train_e_trajectories:', train_e_trajectories)
        print('train_e_mean:', train_e_mean, 'train_e_std:', train_e_std)

        for depth in [1, 2, 3]:
            # for n_e_trajectories in [(1,1), (3, 1), (5, 1), (7,1), (9, 1), (11, 1)]:

            b_means, b_stds, b_scores = [], [], []
            for i in range(3):
                behaviour_cloning = DecisionTree(env, max_depth=depth)
                behaviour_cloning.fit(train_e_states, train_e_actions)

                # print('Behaviour Cloning on', len(expert_trajectories), 'state-action pairs')
                b_mean, b_std, b_trajectories = behaviour_cloning.do_rollout(n=total_test_rollouts, print=False)
                b_score = behaviour_cloning.score(test_e_states, test_e_actions)

                b_means.append(b_mean)
                b_stds.append(b_std)
                b_scores.append(b_score)

                behaviour_cloning.save(f'BC_{str(round(b_mean, 3))}_{str(round(b_std, 3))}_{str(round(b_score, 3))}')

                target_name = f'{env.spec.id}: {str(round(b_mean, 2))} & {str(round(b_std, 2))}'
                file_name_extra = f'{env.spec.id}_{depth}_{i}_{n_e_trajectories[0]}_{str(round(b_mean, 2))}_{str(round(b_std, 2))}_{str(round(b_score, 2))}'
                viz = dtreeviz(behaviour_cloning.decision_tree, train_e_states, train_e_actions, feature_names=feature_names, class_names=class_names, target_name=target_name)
                viz.save(f'./trees/svg_BC_DecistionTree_{file_name_extra}.svg')
            b_mean, b_std, b_score = np.mean(b_means), np.mean(b_stds), np.mean(b_scores)
            print(f'Depth {depth} BC:', b_means, b_stds, b_scores, b_mean, b_std, b_score)

            for ownGeneratorTrajectories in [False, True]:
                for hasAccessToExpert in [False, True]:
                    for sampleWithQ in [False, True]:
                        for discriminateWithQ in [False, True]:

                            ## Setup gail from expert data with decision tree as generative model and mlp as discriminator model
                            generator = DecisionTree(env, max_depth=depth)
                            discriminator = DiscriminatorNN(env, hidden_dims=[32,32], epochs=500, discriminateWithQ=discriminateWithQ)
                            generator, gail_expert_trajectories, gail_epoch_results = do_gail(expert, generator, discriminator, expert_trajectories=copy.deepcopy(train_e_trajectories),
                                                                    n_e_trajectories=n_e_trajectories, epochs=epochs, ownGeneratorTrajectories=ownGeneratorTrajectories, 
                                                                    hasAccessToExpert=hasAccessToExpert, sampleWithQ=sampleWithQ, discriminateWithQ=discriminateWithQ, pprint=False)
                            ## Final test expert vs Generator:
                            # print("BC Decision Tree depth:", generator.decision_tree.get_depth(), '#leave nodes:', generator.decision_tree.get_n_leaves())

                            # print("GAIL Decision Tree depth:", generator.decision_tree.get_depth(), '#leave nodes:', generator.decision_tree.get_n_leaves())
                            g_mean, g_std, g_trajectories = generator.do_rollout(n=total_test_rollouts, print=False)
                            g_score = generator.score(test_e_states, test_e_actions)

                            generator.save(f'{str(round(g_mean, 3))}_{str(round(g_std, 3))}_{str(round(g_score, 3))}')
                            ## Evaluate the new generative model in terms of interpretability (size, average path length, compared to optimal)
                            target_name = f'{env.spec.id}: u: {str(round(g_mean, 2))} & SD: {str(round(g_std, 2))}'
                            file_name_extra = f'{env.spec.id}_{depth}_{n_e_trajectories[0]}_{n_e_trajectories[1]}_{epochs}_{ownGeneratorTrajectories}_{hasAccessToExpert}_{sampleWithQ}_{discriminateWithQ}_{len(gail_expert_trajectories)}_{str(round(train_e_mean, 3))}_{str(round(train_e_std, 3))}_{str(round(b_mean, 3))}_{str(round(b_std, 3))}_{str(round(g_mean, 3))}_{str(round(g_std, 3))}_{str(round(b_score, 3))}_{str(round(g_score, 3))}'
                            viz = dtreeviz(generator.decision_tree, test_e_states, test_e_actions, feature_names=feature_names, class_names=class_names, target_name=target_name)
                            viz.save(f'./trees/svg_GAIL_DecistionTree_{file_name_extra}.svg')

                            print(env.spec.id, depth, n_e_trajectories[0], n_e_trajectories[1], epochs, ownGeneratorTrajectories, hasAccessToExpert, sampleWithQ, discriminateWithQ, len(gail_expert_trajectories), round(train_e_mean, 3), round(train_e_std, 3), round(b_mean, 3), round(b_std, 3), round(g_mean, 3), round(g_std,3), round(b_score,3), round(g_score,3), gail_epoch_results)

    # ct = ctreeviz_bivar(generator.decision_tree, states[:,[1,3]], actions, feature_names=feature_names, class_names=class_names, target_name=target_name)
    # plt.tight_layout()
    # plt.savefig(f'./trees/png_DecistionTree_{env.spec.id}_{generator.max_depth}_{generator.max_features}.png')

    ## Recommendation for increasing interpretability of the generative model.

print('Finished!')