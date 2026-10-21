import os
import sys



import random
import numpy as np

from policy import epsilon_greedy_policy
from td import q_learning
from utils import plot_q_map

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gym_maze.maze_env import MazeEnv
from maze.core import maze_states
from maze.map import map1



actions = ['U', 'R', 'D', 'L']
x_coord, y_coord, blocks, goal = map1()

seed = 1
random.seed(seed)
env = MazeEnv(x_coord, y_coord, blocks, goal)

states = maze_states(x_coord, y_coord)
Q = np.array(np.zeros([len(states), len(actions)]))

# Training
epsilon = .2
gamma = 0.9
alpha = 0.8
episodes = 50

for e in range(episodes):
    state, info = env.reset(seed=seed)
    i = 0
    while True:
        i += 1

        action_idx = epsilon_greedy_policy(epsilon, Q, states.index(state))
        action = actions[action_idx]

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        Q = q_learning(
            Q,
            states.index(state),
            states.index(next_state),
            action_idx,
            reward,
            gamma,
            alpha
        )

        state = next_state

        if done:
            if e < 5 or e % 10 == 9 or e == episodes - 1:
                plot_q_map(x_coord, y_coord, blocks, Q, goal,
                           title=f'After Episode {e + 1}')
            break

env.close()
