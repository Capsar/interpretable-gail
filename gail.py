from rl_helper import QLearning, DecisionTree, NeuralNetwork
import numpy as np
from scipy.spatial import distance

def do_gail(expert: QLearning, generator: DecisionTree, discriminator: NeuralNetwork, n_e_trajectories=20, epochs=10, hasAccessToExpert=False, sampleWithQ=False, max_sample_size = 1000):
    
    # Generate expert data from expert model
    expert_trajectories = expert.generate_trajectories(n_e_trajectories)
    # expert_state_actions = [tuple(list(s) + [a]) for s,a in expert_trajectories]
    print('Number of expert state-action pairs:', len(expert_trajectories))

    number_of_tests = 100

    expert_mean, expert_std = expert.get_average_reward(50, print=True)
    print('------------------------------------------------------------------------------')

    for i in range(epochs):

        print('Epoch: ', i+1, 'with', len(expert_trajectories), 'expert state action pairs.')
        print("Decision Tree depth:", generator.decision_tree.get_depth(), '#leave nodes:', generator.decision_tree.get_n_leaves())
        generator_mean, generator_std = generator.get_average_reward(number_of_tests, print=True)
        if generator_mean >= expert_mean and generator_std <= expert_std:
            print("Found a better reward by generator.")
            break

        ## Generate the state action pairs for the expert from the expert trajectories.
        expert_state_actions = np.asarray([np.asarray(list(s) + [a]) for s,a in expert_trajectories])
        expert_labels = [1 for i in expert_trajectories]

        ## Generate state action pairs for the generator, using the states of the expert trajectories.
        generator_state_actions = np.asarray([(list(s) + [generator.do_action(s)]) for s,_ in expert_trajectories])
        generator_labels = [0 for i in generator_state_actions]

        sample_state_actions, sample_labels = [], []
        sample_state_actions.extend(expert_state_actions)
        sample_labels.extend(expert_labels)
        sample_state_actions.extend(generator_state_actions)
        sample_labels.extend(generator_labels)
        sample_state_actions = np.asarray(sample_state_actions)
        sample_labels = np.asarray(sample_labels)

        ## Train the discriminator with full sample trajectories and labels.
        print('Training discriminator:')
        # discriminator.fit(sample_state_actions, sample_labels)
        discriminator.fit(expert_state_actions, generator_state_actions)
        print('Discriminator prediction on expert & generator data:', np.array(discriminator.predict(expert_state_actions)).mean(), np.array(discriminator.predict(generator_state_actions)).mean())
        print('The Jensen-Shannon distance between expert and generator data:', distance.jensenshannon([a[-1] for a in expert_state_actions], [a[-1] for a in generator_state_actions]))
        
        ## Sample random amount of state_action and labels. (This is in Viper)
        if sampleWithQ:
            expert_qs = [expert.get_average_reward(1, s) for s,_ in expert_trajectories]
            generator_qs = [generator.get_average_reward(1, s) for s,_ in expert_trajectories]
            sample_qs = []
            sample_qs.extend(expert_qs)
            sample_qs.extend(generator_qs)
            ps = sample_qs / np.sum(sample_qs)
            idx = np.random.choice(len(sample_state_actions), size=min(len(sample_state_actions), max_sample_size), p=ps)
            sample_state_actions, sample_labels = [sample_state_actions[i] for i in idx], [sample_labels[i] for i in idx]

        ## Get all state action pairs classified as expert data by the discriminator
        new_generator_s, new_generator_a = [], []
        for s_a, p_expert in list(zip(sample_state_actions, discriminator.predict(sample_state_actions))):
            if np.random.random() < p_expert:
                new_generator_s.append(np.array(s_a[:-1], dtype=float))
                new_generator_a.append(int(s_a[-1]))
        if len(new_generator_a) > 0 and len(new_generator_s) > 0:
            generator.fit(new_generator_s, new_generator_a)
        print(len(new_generator_a), 'out of the', len(sample_state_actions), 'were chosen as new generator data.')

        ## If access to expert policy, generate new tranjectories with the generator to ask what the expert would do.
        if hasAccessToExpert:
            generator_trajectories = generator.generate_trajectories(5)
            expert_trajectories.extend([(s, expert.do_action(s)) for s,_ in generator_trajectories])
        print('')
    return generator