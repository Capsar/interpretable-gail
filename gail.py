from rl_helper import DQN, QLearning, DecisionTree, DiscriminatorNN
import numpy as np
from scipy.spatial import distance

def do_gail(expert: DQN or QLearning, generator: DecisionTree, discriminator: DiscriminatorNN, n_e_trajectories=(20,5), epochs=10, ownGeneratorTrajectories=False, hasAccessToExpert=False, sampleWithQ=False, discriminateWithQ=True, pprint=False):
    
    # Generate expert data from expert model
    expert_trajectories = expert.generate_trajectories(n_e_trajectories[0])
    # expert_state_actions = [tuple(list(s) + [a]) for s,a in expert_trajectories]
    val_ratio = 0.2
    if pprint:
        print('Number of expert state-action pairs:', len(expert_trajectories))

    number_of_tests = 100

    expert_mean, expert_std = expert.get_average_reward(number_of_tests, print=False)
    if pprint:
        print('------------------------------------------------------------------------------')

    for i in range(epochs):

        if pprint:
            print('Epoch: ', i+1, 'with', len(expert_trajectories), 'expert state action pairs.')
            print("Decision Tree depth:", generator.decision_tree.get_depth(), '#leave nodes:', generator.decision_tree.get_n_leaves())
        generator_mean, generator_std = generator.get_average_reward(number_of_tests, print=False)
        if generator_mean >= expert_mean and generator_std <= expert_std:
            if pprint:
                print("Found a better reward by generator.")
                print('')
            break

        ## Generate the state action pairs for the expert from the expert trajectories.
        expert_state_actions = np.asarray([np.asarray(list(s) + [a]) for s,a in expert_trajectories])
        len_e = int(len(expert_trajectories)*val_ratio)

        ## Generate state action pairs for the generator, using the states of the expert trajectories.
        if ownGeneratorTrajectories:
            generator_trajectories = []
            while len(generator_trajectories) < len(expert_trajectories):
                generator_trajectories.extend(generator.generate_trajectories(n_e_trajectories[0]))
            generator_trajectories = generator_trajectories[:len(expert_trajectories)]
            generator_state_actions = np.asarray([np.asarray(list(s) + [a]) for s,a in generator_trajectories])
        else:
            generator_trajectories = [(s, generator.do_action(s)) for s,_ in expert_trajectories]
            generator_state_actions = np.asarray([(list(s) + [generator.do_action(s)]) for s,_ in expert_trajectories])
    
        if discriminateWithQ:
            expert_qs = [expert.get_average_reward(1, s, a)[0] for s,a in expert_trajectories]
            generator_qs = [generator.get_average_reward(1, s, a)[0] for s,a in generator_trajectories]

            sample_qs = []
            sample_qs.extend(expert_qs)
            sample_qs.extend(generator_qs)
            sample_qs = np.asarray(sample_qs)

            sample_qs = sample_qs + min(sample_qs)
            sample_qs = sample_qs / max(sample_qs)

            expert_qs = sample_qs[:len(expert_qs)]
            generator_qs = sample_qs[len(expert_qs):]

            expert_state_actions = np.asarray([np.asarray(list(s) + [a] + [q]) for (s,a),q in list(zip(expert_trajectories, expert_qs))])
            generator_state_actions = np.asarray([np.asarray(list(s) + [a] + [q]) for (s,a),q in list(zip(generator_trajectories, generator_qs))])
            if pprint:
                print("Extended the expert and generator state-action pairs with their normalised cumulative reward, resulting in state-action-reward pairs.")
        if pprint:
            print('Training discriminator on expert=1 or generator=0. train:', (len(expert_trajectories)-len_e)*2, 'validation:', len_e*2)
        discriminator.fit(expert_state_actions[len_e:], generator_state_actions[len_e:])

        ## Train the discriminator with full sample trajectories and labels.
        if pprint:
            print('Discriminator prediction on expert & generator data train:', np.array(discriminator.predict(expert_state_actions[len_e:])).mean(), np.array(discriminator.predict(generator_state_actions[len_e:])).mean())
            print('Discriminator prediction on expert & generator data validation:', np.array(discriminator.predict(expert_state_actions[:len_e])).mean(), np.array(discriminator.predict(generator_state_actions[:len_e])).mean())
            # print('The Jensen-Shannon distance between expert and generator data train:', distance.jensenshannon([a[-1] for a in expert_state_actions[len_e:]], [a[-1] for a in generator_state_actions[len_e:]]))
            # print('The Jensen-Shannon distance between expert and generator data validation:', distance.jensenshannon([a[-1] for a in expert_state_actions[:len_e]], [a[-1] for a in generator_state_actions[:len_e]]))
        
        sample_state_actions = []
        sample_state_actions.extend(expert_state_actions)
        sample_state_actions.extend(generator_state_actions)
        sample_state_actions = np.asarray(sample_state_actions)

        ## Sample random amount of state_action and labels. (This is in Viper)
        if sampleWithQ:
            if pprint:
                print("Going to sample the expert and generator data on Cumulative Reward")
            expert_qs = [expert.get_average_reward(1, s, a)[0] for s,a in expert_trajectories]
            generator_qs = [generator.get_average_reward(1, s, a)[0] for s,a in generator_trajectories]

            sample_qs = []
            sample_qs.extend(expert_qs)
            sample_qs.extend(generator_qs)
            sample_qs = np.asarray(sample_qs)

            sample_qs = sample_qs + min(sample_qs)
            sample_qs = sample_qs / max(sample_qs)
            sample_qs = sample_qs / np.sum(sample_qs)

            idx = np.random.choice(len(sample_state_actions), size=len(sample_state_actions), p=sample_qs)
            sample_state_actions = np.asarray([sample_state_actions[i] for i in idx])


        ## Get all state action pairs classified as expert data by the discriminator
        new_generator_s, new_generator_a = [], []
        for s_a, p_expert in list(zip(sample_state_actions, discriminator.predict(sample_state_actions))):
            if np.random.random() < p_expert:
                if discriminateWithQ:
                    new_generator_s.append(np.array(s_a[:-2], dtype=float))
                    new_generator_a.append(int(s_a[-2]))
                else:
                    new_generator_s.append(np.array(s_a[:-1], dtype=float))
                    new_generator_a.append(int(s_a[-1]))
        if len(new_generator_a) > 0 and len(new_generator_s) > 0:
            generator.fit(new_generator_s, new_generator_a)
        if pprint:
            print(len(new_generator_a), 'out of the', len(sample_state_actions), 'were chosen as new generator data.')

        ## If access to expert policy, generate new tranjectories with the generator to ask what the expert would do.
        if hasAccessToExpert:
            generator_trajectories = generator.generate_trajectories(n_e_trajectories[1])
            expert_trajectories.extend([(s, expert.do_action(s)) for s,_ in generator_trajectories])
        if pprint:
            print('')
    return generator, expert_trajectories