from numpy.core.fromnumeric import argmax
from rl_helper import DQN, QLearning, DecisionTree, DiscriminatorNN
import numpy as np
import copy

def do_gail(expert:DQN, generator: DecisionTree, discriminator: DiscriminatorNN, expert_trajectories=[], n_e_trajectories=(1,1), epochs=10, ownGeneratorTrajectories=False, hasAccessToExpert=False, sampleWithQ=False, discriminateWithQ=False, pprint=False):
    
    # Generate expert data from expert model if none is provided.
    if len(expert_trajectories) == 0:
        _, _, expert_trajectories = expert.do_rollout(n=n_e_trajectories[0])

    results = []
    decisionTrees = []
    val_ratio = 0.2
    if pprint:
        print('Number of expert state-action pairs:', len(expert_trajectories))

    number_of_tests = 100

    expert_mean, expert_std, _ = expert.do_rollout(n=number_of_tests, print=pprint)
    if pprint:
        print('------------------------------------------------------------------------------')

    for i in range(epochs):

        if pprint:
            print('Epoch: ', i+1, 'with', len(expert_trajectories), 'expert state action pairs.')
            print("Decision Tree depth:", generator.decision_tree.get_depth(), '#leave nodes:', generator.decision_tree.get_n_leaves())
        generator_mean, generator_std, _ = generator.do_rollout(n=number_of_tests, print=pprint)
        results.append(generator_mean)
        decisionTrees.append(copy.deepcopy(generator))
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
                _, _, t = generator.do_rollout(n=n_e_trajectories[0])
                generator_trajectories.extend(t)
            generator_trajectories = generator_trajectories[:len(expert_trajectories)]
            generator_state_actions = np.asarray([np.asarray(list(s) + [a]) for s,a in generator_trajectories])
        else:
            generator_trajectories = [(s, generator.do_action(s)) for s,_ in expert_trajectories]
            generator_state_actions = np.asarray([(list(s) + [generator.do_action(s)]) for s,_ in expert_trajectories])

        sample_qs = []
        if discriminateWithQ:
            expert_qs = [expert.get_P(s) for s,_ in expert_trajectories]
            generator_qs = [expert.get_P(s) for s,_ in generator_trajectories]

            sample_qs.extend(expert_qs)
            sample_qs.extend(generator_qs)
            sample_qs = np.asarray(sample_qs)

            sample_qs = (sample_qs - min(sample_qs)) / (max(sample_qs) - min(sample_qs))

            expert_qs = sample_qs[:len(expert_qs)]
            generator_qs = sample_qs[len(expert_qs):]

            expert_state_actions = np.asarray([np.asarray(list(s) + [a] + [q]) for (s,a),q in list(zip(expert_trajectories, expert_qs))])
            generator_state_actions = np.asarray([np.asarray(list(s) + [a] + [q]) for (s,a),q in list(zip(generator_trajectories, generator_qs))])
            if pprint:
                print("Extended the expert and generator state-action pairs with their the difference in probability of best and worst action: action cost, resulting in state-action-actionCost pairs.")
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
                print("Going to sample the expert and generator data on p(s,a) - min(p(s))")
            if not discriminateWithQ:
                expert_qs = [expert.get_P(s) for s,_ in expert_trajectories]
                generator_qs = [expert.get_P(s) for s,_ in generator_trajectories]
                
                sample_qs = []
                sample_qs.extend(expert_qs)
                sample_qs.extend(generator_qs)
                sample_qs = np.asarray(sample_qs)
            
            try:
                sample_ps = sample_qs / np.sum(sample_qs)
                idx = np.random.choice(len(sample_state_actions), size=len(sample_state_actions), p=sample_ps)
                sample_state_actions = np.asarray([sample_state_actions[i] for i in idx])
            except:
                print(sample_qs, np.sum(sample_qs))
        
        length = len(expert_state_actions)
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
            _, _, generator_trajectories = generator.do_rollout(n=n_e_trajectories[1])
            expert_trajectories.extend([(s, expert.do_action(s)) for s,_ in generator_trajectories])
        if pprint:
            print('')
    if pprint:
        print("Decision Tree depth:", generator.decision_tree.get_depth(), '#leave nodes:', generator.decision_tree.get_n_leaves())
    generator_mean, generator_std, _ = generator.do_rollout(n=number_of_tests, print=pprint)
    results.append(generator_mean)
    decisionTrees.append(copy.deepcopy(generator))

    return decisionTrees[argmax(results)], expert_trajectories, results