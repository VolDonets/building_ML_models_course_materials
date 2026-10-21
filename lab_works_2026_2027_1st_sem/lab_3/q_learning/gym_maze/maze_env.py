import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from gymnasium import spaces
from maze.core import random_state, move, plot_maze_state


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["ascii", "human"]}

    def __init__(self, x_coord, y_coord, blocks, finish_state, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.blocks = blocks
        self.finish_state = finish_state
        self.state = random_state(x_coord, y_coord, blocks)

        # debug properties
        self._total_actions = 0
        self._total_reward = 0

    def step(self, action):
        prev_state = self.state
        next_state = move(self.state, action, self.x_coord, self.y_coord, self.blocks)
        reward = 0
        terminated = False
        truncated = False
        if next_state == self.finish_state:
            reward = 1000
            terminated = True
        elif prev_state == next_state:
            reward = -10
        elif prev_state != next_state:
            reward = -1
        self.state = next_state

        self._total_actions += 1
        self._total_reward += reward
        info = {}

        return self.state, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._total_actions = 0
        self._total_reward = 0
        not_allowed = self.blocks + [self.finish_state]
        self.state = random_state(self.x_coord, self.y_coord, not_allowed)
        info = {}
        return self.state, info

    def seed(self, seed=None):
        if seed is not None:
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)

    def render(self, mode="ascii"):
        plot_maze_state(
            self.x_coord,
            self.y_coord,
            self.blocks,
            self.state,
            self.finish_state,
            f'Total Reward: {self._total_reward} \n'
            f'Total Actions: {self._total_actions}'
        )

    def close(self):
        pass
