from tensorforce import Agent, Environment
from tensorforce.agents import TensorforceAgent
import gym

# Pre-defined or custom environment
environment = Environment.create(
    environment='gym', level='Acrobot', max_episode_timesteps=500
)

# Instantiate a Tensorforce agent
agent = Agent.create(
    agent='tensorforce',
    environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
    memory=50000,
    update=dict(unit='timesteps', batch_size=64),
    optimizer=dict(type='adam', learning_rate=3e-4),
    policy=dict(network='auto'),
    objective='policy_gradient',
    reward_estimation=dict(horizon=20)
)


# Train for 300 episodes
total_reward = 0
for i in range(300):
    
    # Initialize episode
    states = environment.reset()
    terminal = False
    while not terminal:
        # Episode timestep
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        total_reward += reward

    if i % 100 == 0:
        print('episode:', i, "total reward:", total_reward/100)
        total_reward=0

agent.save(directory='./models/', filename='tensorforce', format='hdf5')
print("trained for 300 episodes")
agent.close()
environment.close()


loaded_agent = TensorforceAgent.load(directory='./models/', filename='tensorforce', format='hdf5')

env = gym.make('CartPole-v1')

state = env.reset()
done = False
total_reward = 0
while done == False:
    env.render()
    action = loaded_agent.act(state, independent=True)
    print(state, action)
    state, reward, done, info = env.step(action)
    total_reward+=1
print(env, total_reward)

loaded_agent.close()
env.close()