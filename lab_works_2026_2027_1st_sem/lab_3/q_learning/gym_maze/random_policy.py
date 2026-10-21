import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
from gym_maze.maze_env import MazeEnv
from maze.core import actions
from maze.map import map1


x_coord, y_coord, blocks, goal = map1()

seed = 0
random.seed(seed)
env = MazeEnv(x_coord, y_coord, blocks, goal)

state, info = env.reset(seed=seed)
action_history = []

for i in range(1000):
    if i % 10 == 0:
        env.render()
    action = random.choice(actions)
    print(f'action: {action}')
    action_history.append(action)
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        env.render()
        break

env.close()

print(f'Moves: {len(action_history)}')
