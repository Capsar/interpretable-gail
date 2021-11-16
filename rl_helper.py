import numpy as np
from gym import Env

class QLearning:

    def __init__(self, env: Env, discretize: list):
        self.env = env
        self.env.reset()

        self.discretize = discretize
        
        self.num_states = (self.env.observation_space.high - self.env.observation_space.low)*np.array(self.discretize)
        self.num_states = np.round(self.num_states, 0).astype(int) + 1

        self.Q_table = np.random.uniform(low= -1, high=1, size=(self.num_states[0], self.num_states[1], self.env.action_space.n))
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
            return np.argmax(self.Q_table[state_adj[0], state_adj[1]])
        else:
            return np.random.randint(0, self.env.action_space.n)   

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

                #Allow for terminal states
                if done and state2[0] >= 0.5:
                    self.Q_table[state_adj[0], state_adj[1], action] = reward
                else:
                    delta = lr*(reward + discount*np.max(self.Q_table[state2_adj[0], state2_adj[1]]) - self.Q_table[state_adj[0], state_adj[1], action])
                    self.Q_table[state_adj[0], state_adj[1], action] += delta

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