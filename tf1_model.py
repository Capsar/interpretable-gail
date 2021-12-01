import tensorflow as tf
from gym import Env

class Discriminator:

    def model(x, hidden_dims, activation, output_activation):
        for h in hidden_dims:
            x = tf.layers.dense(inputs=x, units=h, activation=activation)
        return tf.layers.dense(inputs=x, units=1, activation=output_activation)

    def __init__(self, env: Env, hidden_dims=[128, 128], learning_rate=0.1):
        self.env = Env
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate

        with tf.variable_scope('discriminator'):
            self.scope = tf.get_variable_scope().name

            self.expert_state = tf.placeholder(tf.float32, shape=[None] + list(self.env.observation_space.shape))
            self.expert_action = tf.placeholder(tf.float32, shape=[None])
            expert_action_one_hot = tf.one_hot(self.expert_action, depth=self.env.action_space.n)
            # expert_action_one_hot += tf.random_normal(tf.shape(expert_action_one_hot), mean=0.2, stddev=0.1) / 1.2
            self.expert_concat = tf.concat([self.expert_state, expert_action_one_hot], axis=1)

            self.agent_state = tf.placeholder(tf.float32, shape=[None] + list(self.env.observation_space.shape))
            self.agent_action = tf.placeholder(tf.float32, shape=[None])
            agent_action_one_hot = tf.one_hot(self.agent_action, depth=self.env.action_space.n)
            # agent_action_one_hot += tf.random_normal(tf.shape(agent_action_one_hot), mean=0.2, stddev=0.1) / 1.2
            self.agent_concat = tf.concat([self.agent_state, agent_action_one_hot], axis=1)

            with tf.variable_scope('model'):
                self.prob_1 = self.model(self.expert_concat, self.hidden_dims, tf.nn.relu, tf.sigmoid)
            with tf.variable_scope('model', reuse=True):
                self.prob_2 = self.model(self.agent_concat, self.hidden_dims, tf.nn.relu, tf.sigmoid)

            with tf.variable_scope('loss'):
                loss_expert = tf.reduce_mean(tf.log(tf.clip_by_value(1 - self.prob_1, 0.01, 1.0)))
                loss_agent = tf.reduce_mean(tf.log(tf.clip_by_value(self.prob_2, 0.01, 1.0)))
                self.loss = - loss_expert - loss_agent

        ## Can also be self.prob1 and 1 - self.prob2
        # loss_expert = tf.reduce_mean(tf.log(tf.clip_by_value(prob_1, 0.01, 1)))
        # loss_agent = tf.reduce_mean(tf.log(tf.clip_by_value(1 - prob_2, 0.01, 1)))
        # loss = loss_expert + loss_agent
        # loss = -loss

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

        self.reward_1 = - tf.squeeze(tf.log(tf.clip_by_value(self.prob_1, 1e-10, 1.0)), axis=1)
        self.reward_2 = - tf.squeeze(tf.log(tf.clip_by_value(self.prob_2, 1e-10, 1.0)), axis=1)
        # self.rewards = tf.log(tf.clip_by_value(prob_2, 1e-10, 1))  # log(P(expert|s,a)) larger is better for agent

    def get_rewards(self, agent_state, agent_action):
        return tf.get_default_session().run(self.reward_2, feed_dict={self.agent_state: agent_state, self.agent_action: agent_action})

    def update(self, expert_state_actions, generator_state_actions):
        expert_state = [s[:-1] for s in expert_state_actions]
        expert_action = [a[-1] for a in expert_state_actions]
        agent_state = [s[:-1] for s in generator_state_actions]
        agent_action = [a[-1] for a in generator_state_actions]
        return tf.get_default_session().run([self.loss, self.train_op], feed_dict={self.expert_state: expert_state, self.expert_action: expert_action, self.agent_state: agent_state, self.agent_action: agent_action})
