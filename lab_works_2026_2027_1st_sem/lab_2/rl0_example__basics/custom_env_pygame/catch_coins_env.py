import sys
from math import ceil
from random import random, randint
from time import sleep
import gymnasium as gym
from gymnasium import spaces
import collections
import os
import numpy as np


class CatchCoinsEnv(gym.Env):
    # Updated metadata key to "render_modes"
    metadata = {"render_modes": ["human", "ascii"]}

    def __init__(self, display_width=10, display_height=10, density=.8, render_mode=None):
        self.display_width = display_width
        self.display_height = display_height
        self.density = density
        self.render_mode = render_mode
        self.display = collections.deque(maxlen=display_height)

        self.last_action = None
        self.last_reward = None
        self.total_score = 0
        self.v_position = 0
        self.game_scr = None

        # Gymnasium REQUIRES defining action and observation spaces
        # Assuming actions are e.g., -1 (left), 0 (stay), 1 (right) -> mapped to Discrete(3)
        self.action_space = spaces.Discrete(3)

        # Defining a generic Dict space for the observation (deque + integer position)
        self.observation_space = spaces.Dict({
            "display": spaces.Box(low=0, high=3, shape=(display_height, display_width), dtype=np.int32),
            "v_position": spaces.Discrete(display_width)
        })

    def step(self, action):
        # Map Discrete(3) actions [0, 1, 2] to [-1, 0, 1] if needed for your logic
        # action_val = action - 1
        self.last_action = action
        self.v_position = min(max(self.v_position + action, 0), self.display_width - 1)
        reward = self.display[0][self.v_position]
        self.last_reward = reward
        self.total_score += reward
        self.display.append(self.line_generator())

        state = (list(self.display), self.v_position)

        # Gymnasium API: terminated, truncated
        terminated = False
        truncated = False
        info = {}

        # Must return 5 variables
        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.display.clear()
        for _ in range(self.display_height):
            self.display.append(self.line_generator())
        self.v_position = ceil(self.display_width / 2)

        state = (list(self.display), self.v_position)
        info = {}

        # Must return observation and info
        return state, info

    def line_generator(self):
        # Fixed syntax error
        line = [0] * self.display_width

        # Fixed syntax error
        if random() > (1 - self.density):
            r = random()
            if r < .6:
                v = 1
            elif r < .9:
                v = 2
            else:
                v = 3

            line[randint(0, self.display_width - 1)] = v
        return line

    def render(self):
        # Use self.render_mode instead of passing mode to render()
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "ascii":
            self._render_ascii()

    def _render_ascii(self):
        os.system('clear')
        outfile = sys.stdout
        area = []
        for i in range(self.display_height):
            # Fixed syntax error
            line = self.display[self.display_height - 1 - i]
            row = []

            for j in range(len(line)):
                p = line[j]
                if p > 0:
                    row.append(str(p))
                    if i > 0 and area[-1][j] == ' ':
                        area[-1][j] = '|'
                    if i > 1 and area[-2][j] == ' ':
                        area[-2][j] = '.'
                else:
                    row.append(' ')

            area.append(row)

        # Fixed syntax error
        pos_line = ['_'] * self.display_width
        pos_line[self.v_position] = str(self.last_reward) if self.last_reward else 'V'
        area.append(pos_line)
        outfile.write(f'\nTotal score: {self.total_score}\n')
        outfile.write('\n'.join(' '.join(line) for line in area) + '\n')

    def _render_human(self):
        from catch_coins_screen import CatchCoinsScreen
        if not self.game_scr:
            self.game_scr = CatchCoinsScreen(
                h=self.display_height,
                w=self.display_width
            )

        if self.last_reward:
            self.game_scr.plus()
            sleep(.1)

        self.game_scr.update(
            list(self.display),
            self.v_position,
            self.total_score
        )
