import numpy as np
from gym import Env


class QLearning:

    #TODO: Make size of Q_table dependable on observation_space.
    #TODO: Setup Q_table according to size not upscaling and rounding.
    def __init__(self, env: Env, discretize: list):
        self.env = env
        self.env.reset()

        self.discretize = discretize

        self.num_states = (self.env.observation_space.high - self.env.observation_space.low)*np.array(self.discretize)
        self.num_states = np.round(self.num_states, 0).astype(int) + 1
        print(self.num_states)

        self.Q_table = np.random.uniform(low= -1, high=1, size=(list(self.num_states) + [self.env.action_space.n]))
        print("Initialized Q_table:", self.Q_table.shape)

    def load(self, file:str):
        self.Q_table = np.load(file=file)

    def save(self):
        np.save(file='./models/'+self.env.spec.id, arr=self.Q_table)

    def discretizeState(self, state, low):
        state_adj = (state - low)*np.array(self.discretize)
        state_adj = np.round(state_adj, 0).astype(int)
        return state_adj

    def do_action(self, state_adj, epsilon=0):
        if np.random.random() < 1 - epsilon:
            return np.argmax(self.Q_table[tuple(state_adj)])
        else:
            return self.env.action_space.sample()   

    def generate_trajectories_and_actions(self, n=10, render=False):
        states = []
        actions = []

        for i in range(n):
            total_reward = 0
            #Reset Environmnet
            state = self.env.reset()
            state = self.discretizeState(state, self.env.observation_space.low)
            done = False
            while done == False:
                if render:
                    self.env.render()
                action = self.do_action(state)

        
                #perform action in environment
                state, reward, done, info = self.env.step(action)
        
                #Save state and action to memory
                states.append(state)
                actions.append(action)
        
                state = self.discretizeState(state, self.env.observation_space.low)
                total_reward+=reward

            #save trajectory and actions to large memory
        return states, actions

    def train(self, epochs, lr, epsilon, discount):
        reward_list = []
        ave_reward_list = []

        reduction = epsilon / (epochs-epochs/4)
        print('Reduction: ', reduction)

        for i in range(epochs):
            done = False
            tot_reward, reward = 0, 0
            state = self.env.reset()
            state_adj = self.discretizeState(state, self.env.observation_space.low)

            while done != True:
                # Randomly do a random action, this will increase exploration in the beginning.
                action = self.do_action(state_adj, epsilon=epsilon)

                state2, reward, done, info = self.env.step(action)
                state2_adj = self.discretizeState(state2, self.env.observation_space.low)

                #Allow for terminal states (if done in time -> maximize reward)
                if done and state2[0] >= 0.5:
                    self.Q_table[state_adj[0], state_adj[1], action] = reward
                else:
                    delta = lr*(reward + discount*np.max(self.Q_table[tuple(state2_adj)]) - self.Q_table[tuple(list(state_adj) + [action])])
                    self.Q_table[tuple(list(state_adj) + [action])] += delta

                #Update total_reward & update state for new action.
                tot_reward += reward
                state_adj = state2_adj

            # Decay epsilon
            if epsilon > 0:
                epsilon -= reduction
            
            # Keep track of rewards (No influence on workings of the algorithm)
            reward_list.append(tot_reward)
            if (i+1) % 100 == 0:
                ave_reward = np.mean(reward_list)
                ave_reward_list.append(ave_reward)
                reward_list = []
                print('Episode {} Average Reward: {}'.format(i+1, ave_reward))
                
        self.env.close()
        print("Finished training!")