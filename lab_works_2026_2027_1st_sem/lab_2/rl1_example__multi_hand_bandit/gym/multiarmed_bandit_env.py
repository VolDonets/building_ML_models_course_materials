import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MultiArmedBanditEnv(gym.Env):
    metadata = {"render_modes": ["ascii"]}

    def __init__(self, bandits, render_mode=None):
        super().__init__()
        self.bandits = bandits
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(bandits))
        self.state = {}
        self.reset()

    def step(self, action):
        p = self.bandits[action]
        r = random.random()
        reward = 1 if r <= p else -1
        self.state[action].append(reward)
        terminated = False
        truncated = False
        info = {}
        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = {}
        for i in range(len(self.bandits)):
            self.state[i] = []
        info = {}
        return self.state, info

    def render(self, mode="ascii"):
        returns = {}
        trials = {}
        for i in range(len(self.bandits)):
            returns[i] = sum(self.state[i])
            trials[i] = len(self.state[i])

        for b, r in returns.items():
            t = trials[b]
            print(f'Bandit {b}| returns: {r}, trials: {t}')
        print(f'Total Trials: {sum(trials.values())}')
        print(f'Total Returns: {sum(returns.values())}')


def get_bandit_env_5():
    bandits = [.45, .45, .4, .6, .4]
    return MultiArmedBanditEnv(bandits)


if __name__ == '__main__':
    seed = 1
    random.seed(seed)
    np.random.seed(1)

    balance = 1_000
    env = get_bandit_env_5()

    state, info = env.reset(seed=seed)
    rewards = []

    for i in range(balance):
        random_bandit = random.randint(0, 4)
        state, reward, terminated, truncated, info = env.step(random_bandit)
        rewards.append(reward)

    env.render()
    env.close()
