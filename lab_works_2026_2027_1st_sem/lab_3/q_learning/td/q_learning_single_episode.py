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

seed = 10
random.seed(seed)
env = MazeEnv(x_coord, y_coord, blocks, goal)

states = maze_states(x_coord, y_coord)

epsilon = .2
gamma = 0.9
alpha = 0.8

Q = np.array(np.zeros([len(states), len(actions)]))

state, info = env.reset(seed=seed)

i = 1
while True:

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
        plot_q_map(x_coord, y_coord, blocks, Q, goal,
                   title=f'After Final Action')
        break
    else:
        if i % 10 == 0 or i < 3:
            plot_q_map(x_coord, y_coord, blocks, Q, goal, title=f'After Action: {i}')
    i += 1

env.close()
