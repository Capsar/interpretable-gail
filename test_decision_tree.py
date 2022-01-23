from rl_helper import DecisionTree, DQN
import gym
from sklearn.tree import export_text
env, discretize, discount, feature_names, class_names = gym.make("Acrobot-v1"), [10, 10, 10, 10, 2, 2], 0.95, ['cos(theta1)', 'sin(theta1)', 'cos(theta2)', 'sin(theta2)', 'thetaDOT1', 'thetaDOT2'], ['torque +1', 'torque 0', 'torque -1']
# env, discretize, discount, feature_names, class_names = gym.make("CartPole-v1"), [10, 10, 10, 10], 0.95, ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity'], ['Left', 'Right']
# env, discretize, discount, feature_names, class_names = gym.make("MountainCar-v0"), [30, 300], 0.95, ['Cart Postion', 'Cart Velocity'], ['Left++', 'Do nothing', 'Right++']

depth = 1
n_e_trajectories = (1, 1)
epochs = 10
ownGeneratorTrajectories = False
hasAccessToExpert = False
sampleWithQ = False
discriminateWithQ = False

expert = DQN(env, 10000, 32)
expert.load()

# expert.do_rollout(1, render=True, print=True)



test_dt = DecisionTree(env, max_depth=depth)

# extra = '-83.6_13.183_0.808'
# extra = '-95.89_27.36'
# extra = '498.2_14.12'
# extra = '-116.88_1.27'

# # extra = '189.46_51.04'

# test_dt.load(extra)


mean, sd, trajectory = test_dt.do_rollout(10, render=False, print=True)

print(export_text(test_dt.decision_tree, feature_names=feature_names))
