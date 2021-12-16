from collections import namedtuple
import gym
import numpy as np
from gym import spaces, Env


'''Made by Otto Kaaij'''
class PongEnv:
    def __init__(self):
        self.env = gym.make('PongDeterministic-v0')
        self.spec = self.env.spec
        self.metadata = self.env.metadata
        self.previous_ball_pos = -1, -1
        self.total_reward = 0

    @staticmethod
    def _get_enemy_paddle(obs):
        relevant_column = obs[:, 16]
        paddle_color = np.array([213, 130, 74])
        paddle_indices = (np.unique(np.argwhere(relevant_column == paddle_color)[:, 0]))
        if len(paddle_indices) == 0:
            return 22
        if len(paddle_indices) < 16 and min(paddle_indices) == 34:
            # enemy paddle is in top border
            return max(paddle_indices) - 16
        return min(paddle_indices)

    @staticmethod
    def _get_our_paddle(obs):
        relevant_column = obs[:, 140]
        paddle_color = np.array([92, 186, 92])
        paddle_indices = (np.unique(np.argwhere(relevant_column == paddle_color)[:, 0]))
        if len(paddle_indices) == 0:
            return 22
        if len(paddle_indices) < 16 and min(paddle_indices) == 34:
            # our paddle is in top border
            return max(paddle_indices) - 16
        return min(paddle_indices)

    def _get_ball(self, obs):
        relevant_area = obs[34:194, :, 0]  # only need R value, because the 236 is unique in this area
        ball_color = 236
        pos = np.argwhere(relevant_area == ball_color)
        x, y = (pos[0][0] + 34, pos[0][1]) if len(pos) else (34, 34)
        prevx, prevy = self.previous_ball_pos
        self.previous_ball_pos = (x, y)
        if prevx == -1:
            return x, y, 0, 0
        vx, vy = prevx - x, prevy - y
        self.previous_ball_pos = (x, y)
        return x, y, vx, vy

    def get_pong_symbolic(self, obs):
        enemy_paddle = PongEnv._get_enemy_paddle(obs)
        our_paddle = PongEnv._get_our_paddle(obs)
        ballx, bally, ballvx, ballvy = self._get_ball(obs)
        return enemy_paddle, our_paddle, ballx, bally, ballvx, ballvy

    def reset(self):
        self.total_reward = 0
        return self.get_pong_symbolic(self.env.reset())

    def render(self):
        self.env.render()

    def step(self, action):
        new_action = 1
        if action == 0:
            new_action = 2
        elif action == 2:
            new_action = 3
        obs, rew, done, info = self.env.step(new_action)
        return self.get_pong_symbolic(obs), rew, done, info

    @property
    def observation_space(self):
        return spaces.Box(low=np.array([0, 0, 0, 0, -28, -28]), high=np.array([210, 210, 210, 210, 28, 28]), dtype=np.int32)
        
    @property
    def action_space(self):
        # return self.env.action_space
        return spaces.Discrete(3)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def close(self):
        self.env.close()


