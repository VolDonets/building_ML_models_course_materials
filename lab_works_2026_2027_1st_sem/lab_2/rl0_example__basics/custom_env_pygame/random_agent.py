import random
from time import sleep
from catch_coins_env import CatchCoinsEnv

# 1. Pass render_mode during initialization
env = CatchCoinsEnv(render_mode='human')

# 2. reset() now returns two values: observation and info
init_state, info = env.reset()

# -1: left
#  0: stay
# +1: right
action_space = [-1, 0, 1]

for _ in range(1000):
    # 3. render() no longer takes the mode argument
    env.render()

    action = random.choice(action_space)

    # 4. step() returns 5 values instead of 4
    state, reward, terminated, truncated, info = env.step(action)

    # Standard practice: break the loop if the environment finishes
    done = terminated or truncated
    if done:
        break

    sleep(.2)

env.close()
